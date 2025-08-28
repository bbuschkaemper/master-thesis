from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule


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
