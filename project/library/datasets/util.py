import torch
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule


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
