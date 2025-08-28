import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import scipy.io as sio
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    verify_str_arg,
    check_integrity,
    download_url,
)
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from PIL import Image


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
