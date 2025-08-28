import os
from typing import Any, Callable, List, Optional, Tuple
import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from PIL import Image


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
