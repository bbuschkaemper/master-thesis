import os
import codecs
import sys
from urllib.error import URLError
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    _flip_byte_order,
)
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule
from PIL import Image


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
