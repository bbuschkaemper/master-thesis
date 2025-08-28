import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import torch
import torchvision

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
)
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from PIL import Image


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


class CIFAR100DataModule(LightningDataModule):
    def __init__(self, batch_size=256, split=(0.8, 0.1, 0.1)):
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
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)


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


class CIFAR100ScaledDataModule(LightningDataModule):
    def __init__(self, batch_size=256, split=(0.8, 0.1, 0.1)):
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
                v2.Resize(size=(224, 224)),  # Scale CIFAR100 images to 224x224
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(
                    size=(224, 224), scale=(0.8, 1.0)
                ),  # Keep at 224x224
                v2.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Add color jitter
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],  # Use ImageNet normalization for consistency
                ),
            ]
        )

        transform_test = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224)),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],  # Use ImageNet normalization for consistency
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
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)


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
