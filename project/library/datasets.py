import os
import pickle
import codecs
import sys
from urllib.error import URLError
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import scipy.io as sio
import torch
import torchvision

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import (
    download_and_extract_archive,
    verify_str_arg,
    check_integrity,
    download_url,
    _flip_byte_order,
)
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from PIL import Image


def collate_multi_domain_batch(batch):
    """Custom collate function for multi-domain batches.

    Handles batches where each sample is (image, (domain_id, domain_class_id))
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)  # Keep as tuple (domain_id, domain_class_id)

    # Stack images as usual
    images = torch.stack(images)

    # Keep targets as list of tuples
    return images, targets


class CombinedDataset(torch.utils.data.Dataset):
    """Combined dataset that merges multiple datasets with domain IDs.

    Args:
        datasets: List of datasets to combine
        domain_ids: List of domain IDs corresponding to each dataset
    """

    def __init__(self, datasets, domain_ids):
        assert len(datasets) == len(
            domain_ids
        ), "Number of datasets must match number of domain IDs"

        self.datasets = datasets
        self.domain_ids = domain_ids

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = []
        cumsum = 0
        for dataset in datasets:
            cumsum += len(dataset)
            self.cumulative_lengths.append(cumsum)

        self.total_length = cumsum

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_lengths):
            if idx < cumsum:
                dataset_idx = i
                break

        # Calculate the index within the specific dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Get the sample from the appropriate dataset
        image, domain_class_id = self.datasets[dataset_idx][local_idx]

        # Return (image, (domain_id, domain_class_id))
        return image, (self.domain_ids[dataset_idx], domain_class_id)


class CombinedDataModule(LightningDataModule):
    """Data module that combines multiple domain datasets for multi-domain training.

    Args:
        dataset_modules: List of dataset modules (each should have train/val/test splits)
        domain_ids: List of domain IDs corresponding to each dataset module
        batch_size: Batch size for data loaders
    """

    def __init__(self, dataset_modules, domain_ids, batch_size=64, num_workers=1):
        super().__init__()
        self.dataset_modules = dataset_modules
        self.domain_ids = domain_ids
        self.batch_size = batch_size
        self.num_workers = num_workers

        assert len(dataset_modules) == len(
            domain_ids
        ), "Number of dataset modules must match number of domain IDs"

    def prepare_data(self):
        # Prepare data for all dataset modules
        for dm in self.dataset_modules:
            dm.prepare_data()

    def setup(self, stage=None):
        # Setup all dataset modules
        for dm in self.dataset_modules:
            dm.setup(stage)

        # Create combined datasets
        if stage == "fit" or stage is None:
            train_datasets = [dm.train for dm in self.dataset_modules]
            val_datasets = [dm.val for dm in self.dataset_modules]

            self.train = CombinedDataset(train_datasets, self.domain_ids)
            self.val = CombinedDataset(val_datasets, self.domain_ids)

        if stage == "test" or stage is None:
            test_datasets = [dm.test for dm in self.dataset_modules]
            self.test = CombinedDataset(test_datasets, self.domain_ids)

        if stage == "predict" or stage is None:
            # Use test datasets for prediction
            predict_datasets = [dm.test for dm in self.dataset_modules]
            self.predict_dataset = CombinedDataset(predict_datasets, self.domain_ids)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_multi_domain_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_multi_domain_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_multi_domain_batch,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_multi_domain_batch,
        )


class Caltech101DataModule(LightningDataModule):
    def __init__(self, batch_size=64, split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        torchvision.datasets.Caltech101(root="datasets/caltech101", download=True)

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RGB(),
                v2.Resize(
                    size=(224, 224)
                ),  # Not all images are same size, so resize to uniform size
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = torchvision.datasets.Caltech101(
            root="datasets/caltech101", transform=transform
        )
        self.train, self.val, self.test = random_split(
            dataset, self.split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Caltech101MappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        torchvision.datasets.Caltech101(root="datasets/caltech101", download=True)

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RGB(),
                v2.Resize(
                    size=(224, 224)
                ),  # Not all images are same size, so resize to uniform size
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Use our custom Caltech101 implementation with target mapping
        dataset = Caltech101Mapped(
            root="datasets/caltech101", transform=transform, target_mapping=self.mapping
        )

        self.train, self.val, self.test = random_split(
            dataset, self.split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Caltech256DataModule(LightningDataModule):
    def __init__(self, batch_size=64, split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        torchvision.datasets.Caltech256(root="datasets/caltech256", download=True)

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RGB(),
                v2.Resize(
                    size=(224, 224)
                ),  # Not all images are same size, so resize to uniform size
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = torchvision.datasets.Caltech256(
            root="datasets/caltech256", transform=transform
        )
        self.train, self.val, self.test = random_split(
            dataset, self.split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class CIFAR100DataModule(LightningDataModule):
    def __init__(self, batch_size=64, split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.batch_size = batch_size
        self.split = split

    def prepare_data(self):
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=True
        )
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=False
        )

    def setup(self, stage):
        transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),  # CIFAR100 images are 32x32
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(32, padding=4),
                v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Add color jitter
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                v2.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                v2.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        # Load CIFAR100 datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root="datasets/cifar100", train=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="datasets/cifar100", train=False, transform=transform_test
        )

        # Split train dataset into train/val according to the split ratio
        train_size = int(self.split[0] * len(train_dataset))
        val_size = len(train_dataset) - train_size

        self.train, self.val = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Caltech256MappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        torchvision.datasets.Caltech256(root="datasets/caltech256", download=True)

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RGB(),
                v2.Resize(
                    size=(224, 224)
                ),  # Not all images are same size, so resize to uniform size
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Use our custom Caltech256 implementation with target mapping
        dataset = Caltech256Mapped(
            root="datasets/caltech256", transform=transform, target_mapping=self.mapping
        )

        self.train, self.val, self.test = random_split(
            dataset, self.split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Caltech256DomainShiftedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        domain_shift_transform=None,
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping
        self.domain_shift_transform = domain_shift_transform

    def prepare_data(self):
        torchvision.datasets.Caltech256(root="datasets/caltech256", download=True)

    def setup(self, stage):
        # Base transforms that are always applied
        base_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RGB(),
            v2.Resize(size=(224, 224)),
        ]

        # Add domain shift transform if provided
        if self.domain_shift_transform is not None:
            base_transforms.append(self.domain_shift_transform)

        # Add remaining transforms
        base_transforms.extend(
            [
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        transform = v2.Compose(base_transforms)

        # Use our custom Caltech256 implementation with target mapping
        dataset = Caltech256Mapped(
            root="datasets/caltech256", transform=transform, target_mapping=self.mapping
        )

        self.train, self.val, self.test = random_split(
            dataset, self.split, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class CIFAR100DomainShiftedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        domain_shift_transform=None,
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping
        self.domain_shift_transform = domain_shift_transform

    def prepare_data(self):
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=True
        )
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=False
        )

    def setup(self, stage):
        # Base training transforms
        base_train_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(32, 32)),  # CIFAR100 images are 32x32
        ]

        # Add domain shift transform if provided (before data augmentation)
        if self.domain_shift_transform is not None:
            base_train_transforms.append(self.domain_shift_transform)

        # Add remaining training transforms
        base_train_transforms.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(32, padding=4),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                v2.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        # Base test transforms
        base_test_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(32, 32)),
        ]

        # Add domain shift transform if provided (for test data too)
        if self.domain_shift_transform is not None:
            base_test_transforms.append(self.domain_shift_transform)

        # Add normalization for test
        base_test_transforms.append(
            v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        )

        transform_train = v2.Compose(base_train_transforms)
        transform_test = v2.Compose(base_test_transforms)

        # Use our custom CIFAR100 implementation with target mapping
        train_dataset = CIFAR100Mapped(
            root="datasets/cifar100",
            train=True,
            transform=transform_train,
            target_mapping=self.mapping,
        )
        test_dataset = CIFAR100Mapped(
            root="datasets/cifar100",
            train=False,
            transform=transform_test,
            target_mapping=self.mapping,
        )

        # Split train dataset into train/val
        train_size = int(self.split[0] * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train, self.val = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Country211MappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split: str = "train",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        # Download the Country211 dataset
        Country211Mapped(root="datasets", download=True)

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RGB(),
                v2.Resize(size=(224, 224)),  # Resize to uniform size
                v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Use our custom Country211 implementation with target mapping
        self.dataset = Country211Mapped(
            root="datasets",
            split=self.split,
            transform=transform,
            target_mapping=self.mapping,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


class Caltech256Mapped(VisionDataset):
    """`Caltech 256 <https://data.caltech.edu/records/20087>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            os.path.join(root, "caltech256"),
            transform=transform,
            target_transform=target_transform,
        )
        os.makedirs(self.root, exist_ok=True)
        self.target_mapping = target_mapping

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.categories = sorted(
            os.listdir(os.path.join(self.root, "256_ObjectCategories"))
        )
        self.index: List[int] = []
        self.y = []
        for i, c in enumerate(self.categories):
            n = len(
                [
                    item
                    for item in os.listdir(
                        os.path.join(self.root, "256_ObjectCategories", c)
                    )
                    if item.endswith(".jpg")
                ]
            )
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            for idx, target in enumerate(self.y):
                if target in self.target_mapping:
                    filtered_indices.append(self.index[idx])
                    filtered_targets.append(target)

            self.index = filtered_indices
            self.y = filtered_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            If target_mapping is provided, the target will be mapped using the mapping.
        """
        # Use the original target for building the image path
        original_target = self.y[index]

        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[original_target],
                f"{original_target + 1:03d}_{self.index[index]:04d}.jpg",
            )
        )

        # Apply mapping if provided, otherwise use the original target
        if self.target_mapping is not None and original_target in self.target_mapping:
            target = self.target_mapping[original_target]
        else:
            target = original_target

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )


class Caltech101Mapped(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``101_ObjectCategories`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            os.path.join(root, "caltech101"),
            transform=transform,
            target_transform=target_transform,
        )
        os.makedirs(self.root, exist_ok=True)
        self.target_mapping = target_mapping

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        self.categories = sorted(
            os.listdir(os.path.join(self.root, "101_ObjectCategories"))
        )
        # Remove the 'BACKGROUND_Google' category from the list
        if "BACKGROUND_Google" in self.categories:
            self.categories.remove("BACKGROUND_Google")

        self.index: List[int] = []
        self.y = []
        for i, c in enumerate(self.categories):
            image_files = [
                item
                for item in os.listdir(
                    os.path.join(self.root, "101_ObjectCategories", c)
                )
                if item.endswith(".jpg")
            ]
            n = len(image_files)
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            for idx, target in enumerate(self.y):
                if target in self.target_mapping:
                    filtered_indices.append(self.index[idx])
                    filtered_targets.append(target)

            self.index = filtered_indices
            self.y = filtered_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            If target_mapping is provided, the target will be mapped using the mapping.
        """
        # Use the original target for building the image path
        original_target = self.y[index]
        category_name = self.categories[original_target]

        # Find the image file in the category directory
        category_path = os.path.join(self.root, "101_ObjectCategories", category_name)
        image_files = [f for f in os.listdir(category_path) if f.endswith(".jpg")]

        # Use the index to find the specific image file
        # Caltech101 images are named differently than Caltech256
        image_file = f"image_{self.index[index]:04d}.jpg"
        if image_file not in image_files:
            # Fallback: use the first available image file
            image_file = image_files[0] if image_files else None

        if image_file is None:
            raise FileNotFoundError(f"No image found for category {category_name}")

        img_path = os.path.join(category_path, image_file)
        img = Image.open(img_path)

        # Apply mapping if provided, otherwise use the original target
        if self.target_mapping is not None and original_target in self.target_mapping:
            target = self.target_mapping[original_target]
        else:
            target = original_target

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # Check if the main directory exists
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            return

        download_and_extract_archive(
            "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.tar.gz",
            self.root,
            filename="caltech-101.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )


class Country211Mapped(ImageFolder):
    """`The Country211 Data Set
    <https://github.com/openai/CLIP/blob/main/data/country211.md>`_ from OpenAI.

    This dataset was built by filtering the images from the YFCC100m dataset
    that have GPS coordinate corresponding to a ISO-3166 country code. The
    dataset is balanced by sampling 150 train images, 50 validation images, and
    100 test images for each country.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split
            supports ``"train"`` (default), ``"valid"`` and ``"test"``.
        transform (callable, optional): A function/transform that takes in
            a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/country211/``. If dataset is already downloaded, it is not downloaded again.
    """

    _URL = "https://openaipublic.azureedge.net/clip/data/country211.tgz"
    _MD5 = "84988d7644798601126c29e9877aab6a"

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "valid", "test"))

        root = Path(root).expanduser()
        self.root = str(root)
        self._base_folder = root / "country211"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        super().__init__(
            str(self._base_folder / self._split),
            transform=transform,
            target_transform=target_transform,
        )
        self.root = str(root)
        self.target_mapping = target_mapping

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            for idx, target in enumerate(self.targets):
                if target in self.target_mapping:
                    filtered_indices.append(idx)
                    filtered_targets.append(self.target_mapping[target])

            self.samples = [self.samples[i] for i in filtered_indices]
            self.targets = filtered_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Apply target mapping if provided
        if self.target_mapping is not None:
            # Target is already mapped in __init__, so just return it
            target = self.targets[index]

        return sample, target

    def _check_exists(self) -> bool:
        return self._base_folder.exists() and self._base_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class CIFAR100MappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=True
        )
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", download=True, train=False
        )

    def setup(self, stage):
        transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),  # CIFAR100 images are 32x32
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(32, padding=4),
                v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Add color jitter
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                v2.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                v2.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        # Use our custom CIFAR100 implementation with target mapping
        train_dataset = CIFAR100Mapped(
            root="datasets/cifar100",
            train=True,
            transform=transform_train,
            target_mapping=self.mapping,
        )
        test_dataset = CIFAR100Mapped(
            root="datasets/cifar100",
            train=False,
            transform=transform_test,
            target_mapping=self.mapping,
        )

        # Split train dataset into train/val
        train_size = int(self.split[0] * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train, self.val = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class CIFAR100Mapped(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset with target mapping.

    This is a subclass of the VisionDataset that extends CIFAR100 functionality
    with the ability to map original targets to new targets.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-100-python`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.target_mapping = target_mapping

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            filtered_data = []
            for idx, target in enumerate(self.targets):
                if target in self.target_mapping:
                    filtered_indices.append(idx)
                    filtered_targets.append(target)
                    filtered_data.append(self.data[idx])

            self.data = np.array(filtered_data)
            self.targets = filtered_targets

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted. "
                "You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            If target_mapping is provided, the target will be mapped using the mapping.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # Apply mapping if provided, otherwise use the original target
        if self.target_mapping is not None and target in self.target_mapping:
            target = self.target_mapping[target]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class SVHNMappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        # Download SVHN train and test sets
        SVHNMapped(root="datasets/svhn", split="train", download=True)
        SVHNMapped(root="datasets/svhn", split="test", download=True)

    def setup(self, stage):
        transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),  # SVHN images are 32x32
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomCrop(32, padding=4),
                v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Add color jitter
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization for SVHN
            ]
        )

        transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Use our custom SVHN implementation with target mapping
        train_dataset = SVHNMapped(
            root="datasets/svhn",
            split="train",
            transform=transform_train,
            target_mapping=self.mapping,
        )
        test_dataset = SVHNMapped(
            root="datasets/svhn",
            split="test",
            transform=transform_test,
            target_mapping=self.mapping,
        )

        # Split train dataset into train/val
        train_size = int(self.split[0] * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train, self.val = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class SVHNMapped(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset with target mapping.

    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    This is a subclass of the VisionDataset that extends SVHN functionality
    with the ability to map original targets to new targets.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset where the data is stored.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        self.target_mapping = target_mapping

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np.ndarray of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            filtered_data = []
            for idx, target in enumerate(self.labels):
                if target in self.target_mapping:
                    filtered_indices.append(idx)
                    filtered_targets.append(target)
                    filtered_data.append(self.data[idx])

            self.data = np.array(filtered_data)
            self.labels = np.array(filtered_targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            If target_mapping is provided, the target will be mapped using the mapping.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        # Apply mapping if provided, otherwise use the original target
        if self.target_mapping is not None and target in self.target_mapping:
            target = self.target_mapping[target]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return f"Split: {self.__dict__}"


class MNISTMappedDataModule(LightningDataModule):
    def __init__(
        self,
        mapping: dict[int, int],
        batch_size=64,
        split=(0.8, 0.1, 0.1),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.mapping = mapping

    def prepare_data(self):
        # Download MNIST train and test sets
        MNISTMapped(root="datasets/mnist", train=True, download=True)
        MNISTMapped(root="datasets/mnist", train=False, download=True)

    def setup(self, stage):
        transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),  # Resize to match SVHN size
                v2.RGB(),  # Convert grayscale to RGB
                v2.RandomAffine(
                    degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Use ImageNet normalization
            ]
        )

        transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),  # Resize to match SVHN size
                v2.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x
                ),  # Convert grayscale to RGB
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Use ImageNet normalization
            ]
        )

        # Use our custom MNIST implementation with target mapping
        train_dataset = MNISTMapped(
            root="datasets/mnist",
            train=True,
            transform=transform_train,
            target_mapping=self.mapping,
        )
        test_dataset = MNISTMapped(
            root="datasets/mnist",
            train=False,
            transform=transform_test,
            target_mapping=self.mapping,
        )

        # Split train dataset into train/val
        train_size = int(self.split[0] * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train, self.val = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class MNISTMapped(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset with target mapping.

    This is a subclass of the VisionDataset that extends MNIST functionality
    with the ability to map original targets to new targets.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        target_mapping (dict[int, int], optional): A dictionary that maps original targets
            to new targets. If provided, the dataset will return the mapped target.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_mapping: Optional[dict[int, int]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.target_mapping = target_mapping

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
        else:
            if download:
                self.download()

            if not self._check_exists():
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it"
                )

            self.data, self.targets = self._load_data()

        # Convert targets to list for easier manipulation
        if hasattr(self.targets, "tolist"):
            self.targets = self.targets.tolist()

        # If target_mapping is provided, only use data that has targets in the mapping
        if self.target_mapping is not None:
            filtered_indices = []
            filtered_targets = []
            filtered_data = []
            for idx, target in enumerate(self.targets):
                if target in self.target_mapping:
                    filtered_indices.append(idx)
                    filtered_targets.append(target)
                    filtered_data.append(self.data[idx])

            self.data = (
                torch.stack(filtered_data) if filtered_data else torch.empty(0, 28, 28)
            )
            self.targets = filtered_targets

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file))
            for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(
            os.path.join(self.processed_folder, data_file), weights_only=True
        )

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = self._read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = self._read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def _read_sn3_pascalvincent_tensor(
        self, path: str, strict: bool = True
    ) -> torch.Tensor:
        """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh')."""

        def get_int(b: bytes) -> int:
            return int(codecs.encode(b, "hex"), 16)

        SN3_PASCALVINCENT_TYPEMAP = {
            8: torch.uint8,
            9: torch.int8,
            11: torch.int16,
            12: torch.int32,
            13: torch.float32,
            14: torch.float64,
        }

        # read
        with open(path, "rb") as f:
            data = f.read()

        # parse
        if sys.byteorder == "little":
            magic = get_int(data[0:4])
            nd = magic % 256
            ty = magic // 256
        else:
            nd = get_int(data[0:1])
            ty = (
                get_int(data[1:2])
                + get_int(data[2:3]) * 256
                + get_int(data[3:4]) * 256 * 256
            )

        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
        s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

        if sys.byteorder == "big":
            for i, val in enumerate(s):
                s[i] = int.from_bytes(
                    val.to_bytes(4, byteorder="little"), byteorder="big", signed=False
                )

        parsed = torch.frombuffer(
            bytearray(data), dtype=torch_type, offset=(4 * (nd + 1))
        )

        # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
        # that is little endian and the dtype has more than one byte, we need to flip them.
        if sys.byteorder == "little" and parsed.element_size() > 1:

            parsed = _flip_byte_order(parsed)

        assert parsed.shape[0] == np.prod(s) or not strict
        return parsed.view(*s)

    def _read_label_file(self, path: str) -> torch.Tensor:
        x = self._read_sn3_pascalvincent_tensor(path, strict=False)
        if x.dtype != torch.uint8:
            raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
        if x.ndimension() != 1:
            raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
        return x.long()

    def _read_image_file(self, path: str) -> torch.Tensor:
        x = self._read_sn3_pascalvincent_tensor(path, strict=False)
        if x.dtype != torch.uint8:
            raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
        if x.ndimension() != 3:
            raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
        return x

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            If target_mapping is provided, the target will be mapped using the mapping.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        # Apply mapping if provided, otherwise use the original target
        if self.target_mapping is not None and target in self.target_mapping:
            target = self.target_mapping[target]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            errors = []
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download_and_extract_archive(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.mirrors, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
