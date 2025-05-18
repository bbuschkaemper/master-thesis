import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
import lightning as pl
from torchvision.transforms import v2
from synthetic_taxonomy import Deviation


class Caltech101DataModule(pl.LightningDataModule):
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


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", train=True, download=True
        )
        torchvision.datasets.CIFAR100(
            root="datasets/cifar100", train=False, download=True
        )

    def setup(self, stage):
        transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = torchvision.datasets.CIFAR100(
            root="datasets/cifar100", transform=transform, train=True
        )
        self.train, self.val = random_split(
            dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        dataset = torchvision.datasets.CIFAR100(
            root="datasets/cifar100", transform=transform, train=False
        )
        self.test = dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class Caltech256DataModule(pl.LightningDataModule):
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


class Caltech256MappedClassDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        split=(0.8, 0.1, 0.1),
        deviation=Deviation(),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.deviation = deviation

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

        mappings = {}
        for idx, deviation_class in enumerate(self.deviation):
            for class_idx in deviation_class:
                mappings[class_idx] = idx

        target_idxs = []
        for idx, target in enumerate(dataset.y):
            if target in mappings:
                target_idxs.append(idx)

        dataset = Subset(dataset, target_idxs)

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
