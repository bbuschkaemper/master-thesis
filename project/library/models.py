from typing import TypeAlias, Literal
import lightning as pl
import torch
import numpy as np
from torchvision.models.resnet import (
    resnet50,
    ResNet50_Weights,
    resnet152,
    ResNet152_Weights,
)
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from .taxonomy import Taxonomy
from .types import DomainClass, UniversalClass


_Architecture: TypeAlias = Literal["resnet50", "resnet152"]
_Optimizer: TypeAlias = Literal["adamw", "sgd"]
_LRScheduler: TypeAlias = Literal["multistep", "step"]


class ResNetModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        architecture: _Architecture,
        optim: _Optimizer,
        optim_kwargs,
        lr_scheduler: _LRScheduler | None = None,
        lr_scheduler_kwargs=None,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.architecture = architecture
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        if self.architecture == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif self.architecture == "resnet152":
            self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(  # type: ignore
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes),
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)

        self.log_dict({"train_loss": loss, "train_accuracy": accuracy})

        return loss

    def configure_optimizers(self):  # type: ignore
        if self.optim == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        elif self.optim == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")

        if self.lr_scheduler == "multistep":
            scheduler = MultiStepLR(
                optimizer,
                **self.lr_scheduler_kwargs,  # type: ignore
            )
        elif self.lr_scheduler == "step":
            scheduler = StepLR(
                optimizer,
                **self.lr_scheduler_kwargs,  # type: ignore
            )
        elif self.lr_scheduler is None:
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler}")

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        return optimizer

    def test_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)

        self.log_dict({"eval_loss": loss, "eval_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def validation_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)

        self.log_dict({"val_loss": loss, "val_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def predict_step(self, batch):
        (inputs, _) = batch
        output = self(inputs)

        return output


class UniversalResNetModel(pl.LightningModule):
    def __init__(
        self,
        taxonomy: Taxonomy,
        architecture: _Architecture,
        optim: _Optimizer,
        optim_kwargs,
        lr_scheduler: _LRScheduler | None = None,
        lr_scheduler_kwargs=None,
    ):
        super().__init__()

        # Save hyperparameters (excluding taxonomy to avoid serialization issues)
        self.save_hyperparameters(ignore=["taxonomy"])
        self.taxonomy = taxonomy
        self.architecture = architecture
        self.optim = optim
        self.optim_kwargs = optim_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # Extract universal classes from taxonomy
        self.universal_classes = [
            node for node in taxonomy.get_nodes() if isinstance(node, UniversalClass)
        ]
        self.num_universal_classes = len(self.universal_classes)

        if self.num_universal_classes == 0:
            raise ValueError("Taxonomy must contain universal classes")

        # Create mapping from universal class to index
        self.universal_class_to_idx = {
            uc: idx for idx, uc in enumerate(self.universal_classes)
        }

        # Pre-calculate conversion matrices for all domains
        self.domain_to_universal_raw = (
            {}
        )  # domain_id -> (num_classes, num_universal) raw weights
        self._precompute_conversion_matrices()

        # Build ResNet model
        if self.architecture == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif self.architecture == "resnet152":
            self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Replace final layer with universal class output
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(  # type: ignore
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, num_features),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, num_features),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, num_features),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(num_features, self.num_universal_classes),
        )

        # Use KL Divergence loss for universal class activations
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def _precompute_conversion_matrices(self):
        """Pre-compute raw weight matrices for all domains for efficient batch processing."""
        # Get all domain classes grouped by domain
        all_domain_classes = [
            node for node in self.taxonomy.get_nodes() if isinstance(node, DomainClass)
        ]

        # Group by domain
        domains = {}
        for domain_class in all_domain_classes:
            domain_id = int(domain_class[0])
            if domain_id not in domains:
                domains[domain_id] = []
            domains[domain_id].append(domain_class)

        # Build raw weight matrices for each domain
        for domain_id, domain_classes_list in domains.items():
            # Sort by class_id to ensure consistent ordering
            domain_classes_list.sort(key=lambda x: int(x[1]))
            num_domain_classes = len(domain_classes_list)

            # Initialize matrix with raw weights
            domain_to_universal_raw = np.zeros(
                (num_domain_classes, self.num_universal_classes), dtype=np.float32
            )

            # Fill matrix with raw relationship weights
            for class_idx, domain_class in enumerate(domain_classes_list):
                relationships = self.taxonomy.get_relationships_from(domain_class)

                for relationship in relationships:
                    _, source_class, weight = relationship
                    if isinstance(source_class, UniversalClass):
                        universal_idx = self.universal_class_to_idx[source_class]
                        domain_to_universal_raw[class_idx, universal_idx] += weight

            # Convert to torch tensor and store
            self.domain_to_universal_raw[domain_id] = torch.from_numpy(
                domain_to_universal_raw
            )

    def _domain_class_to_universal_targets(
        self, domain_class_labels: torch.Tensor, domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """Convert domain class labels to universal class activation targets using raw weights with runtime normalization.

        Parameters
        ----------
        domain_class_labels : torch.Tensor
            Tensor of domain class labels with shape (batch_size,)
        domain_ids : torch.Tensor
            Tensor of domain IDs with shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Tensor of universal class targets with shape (batch_size, num_universal_classes)
        """
        batch_size = domain_class_labels.shape[0]
        device = domain_class_labels.device

        # Initialize output tensor
        universal_targets = torch.zeros(
            batch_size, self.num_universal_classes, device=device
        )

        # Process each unique domain for efficiency
        unique_domains = torch.unique(domain_ids)

        for domain_id in unique_domains:
            domain_id_int = domain_id.item()

            # Get the raw weight matrix for this domain
            if domain_id_int not in self.domain_to_universal_raw:
                raise ValueError(
                    f"No conversion matrix found for domain {domain_id_int}"
                )

            raw_matrix = self.domain_to_universal_raw[domain_id_int].to(device)

            # Get mask for samples belonging to this domain
            domain_mask = domain_ids == domain_id
            domain_labels = domain_class_labels[domain_mask]

            # Use matrix indexing to get raw targets for this domain
            raw_targets = raw_matrix[domain_labels.long()]

            # Normalize each row to sum to 1 (runtime normalization)
            row_sums = raw_targets.sum(dim=1, keepdim=True)
            # Avoid division by zero
            row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
            normalized_targets = raw_targets / row_sums

            # Place normalized targets back in the correct positions
            universal_targets[domain_mask] = normalized_targets

        return universal_targets

    def _universal_to_domain_predictions(
        self, universal_predictions: torch.Tensor, domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """Convert universal class predictions to domain class predictions using transposed raw weights.

        Parameters
        ----------
        universal_predictions : torch.Tensor
            Universal class predictions with shape (batch_size, num_universal_classes)
        domain_ids : torch.Tensor
            Tensor of domain IDs with shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Domain class predictions with shape (batch_size, max_domain_classes)
            Note: Output is padded to max_domain_classes across all domains in the batch
        """
        batch_size = universal_predictions.shape[0]
        device = universal_predictions.device

        # Find the maximum number of domain classes across all domains in this batch
        unique_domains = torch.unique(domain_ids)
        max_domain_classes = max(
            self.domain_to_universal_raw[domain_id.item()].shape[0]
            for domain_id in unique_domains
        )

        # Initialize output tensor
        domain_predictions = torch.zeros(batch_size, max_domain_classes, device=device)

        # Process each unique domain for efficiency
        for domain_id in unique_domains:
            domain_id_int = domain_id.item()

            # Get the raw weight matrix for this domain
            if domain_id_int not in self.domain_to_universal_raw:
                raise ValueError(
                    f"No conversion matrix found for domain {domain_id_int}"
                )

            raw_matrix = self.domain_to_universal_raw[domain_id_int].to(device)

            # Get mask for samples belonging to this domain
            domain_mask = domain_ids == domain_id
            domain_universal_preds = universal_predictions[domain_mask]

            # Transpose to get universal_to_domain shape: (num_universal, num_domain)
            raw_universal_to_domain = raw_matrix.T

            # Use matrix multiplication for efficient conversion
            # domain_universal_preds: (domain_batch_size, num_universal)
            # raw_universal_to_domain: (num_universal, num_domain)
            # result: (domain_batch_size, num_domain)
            domain_preds = torch.matmul(domain_universal_preds, raw_universal_to_domain)

            # Pad if necessary to match max_domain_classes
            num_domain_classes = domain_preds.shape[1]
            if num_domain_classes < max_domain_classes:
                padding = torch.zeros(
                    domain_preds.shape[0],
                    max_domain_classes - num_domain_classes,
                    device=device,
                )
                domain_preds = torch.cat([domain_preds, padding], dim=1)

            # Place domain predictions back in the correct positions
            domain_predictions[domain_mask] = domain_preds

        return domain_predictions

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, targets = batch
        output = self(inputs)

        # Targets must be tuples of (domain_id, domain_class_id)
        if not isinstance(targets[0], tuple):
            raise ValueError(
                "Targets must be tuples of (domain_id, domain_class_id). Use CombinedDataModule for multi-domain training."
            )

        domain_ids, domain_class_ids = zip(*targets)
        domain_ids = torch.tensor(domain_ids, device=output.device)
        domain_class_ids = torch.tensor(domain_class_ids, device=output.device)

        # Convert domain class targets to universal class targets (vectorized)
        universal_targets = self._domain_class_to_universal_targets(
            domain_class_ids, domain_ids
        )

        # For KL divergence, we need log probabilities of predictions
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        # For logging, compute accuracy based on domain class predictions (vectorized)
        domain_predictions = self._universal_to_domain_predictions(output, domain_ids)

        # Calculate accuracy per sample
        correct = 0
        total = len(domain_class_ids)

        # Process each unique domain for accuracy calculation (still need this for trimming)
        unique_domains = torch.unique(domain_ids)
        for domain_id in unique_domains:
            domain_mask = domain_ids == domain_id
            domain_preds = domain_predictions[domain_mask]
            domain_targets = domain_class_ids[domain_mask]

            # Get the actual number of classes for this domain
            num_classes = self.domain_to_universal_raw[domain_id.item()].shape[0]
            domain_preds_trimmed = domain_preds[:, :num_classes]

            pred = domain_preds_trimmed.argmax(dim=1, keepdim=True)
            correct += pred.eq(domain_targets.view_as(pred)).sum().item()

        accuracy = correct / total
        self.log_dict({"train_loss": loss, "train_accuracy": accuracy})

        return loss

    def configure_optimizers(self):  # type: ignore
        if self.optim == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        elif self.optim == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")

        if self.lr_scheduler == "multistep":
            scheduler = MultiStepLR(
                optimizer,
                **self.lr_scheduler_kwargs,  # type: ignore
            )
        elif self.lr_scheduler == "step":
            scheduler = StepLR(
                optimizer,
                **self.lr_scheduler_kwargs,  # type: ignore
            )
        elif self.lr_scheduler is None:
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.lr_scheduler}")

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        return optimizer

    def test_step(self, batch):
        inputs, targets = batch
        output = self(inputs)

        # Targets must be tuples of (domain_id, domain_class_id)
        if not isinstance(targets[0], tuple):
            raise ValueError(
                "Targets must be tuples of (domain_id, domain_class_id). Use CombinedDataModule for multi-domain training."
            )

        domain_ids, domain_class_ids = zip(*targets)
        domain_ids = torch.tensor(domain_ids, device=output.device)
        domain_class_ids = torch.tensor(domain_class_ids, device=output.device)

        # Convert to universal targets and compute loss (vectorized)
        universal_targets = self._domain_class_to_universal_targets(
            domain_class_ids, domain_ids
        )
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        # Convert to domain predictions for evaluation (vectorized)
        domain_predictions = self._universal_to_domain_predictions(output, domain_ids)

        # Calculate accuracy per sample
        correct = 0
        total = len(domain_class_ids)

        # Process each unique domain for accuracy calculation (still need this for trimming)
        unique_domains = torch.unique(domain_ids)
        for domain_id in unique_domains:
            domain_mask = domain_ids == domain_id
            domain_preds = domain_predictions[domain_mask]
            domain_targets = domain_class_ids[domain_mask]

            # Get the actual number of classes for this domain
            num_classes = self.domain_to_universal_raw[domain_id.item()].shape[0]
            domain_preds_trimmed = domain_preds[:, :num_classes]

            pred = domain_preds_trimmed.argmax(dim=1, keepdim=True)
            correct += pred.eq(domain_targets.view_as(pred)).sum().item()

        accuracy = correct / total
        self.log_dict({"eval_loss": loss, "eval_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def validation_step(self, batch):
        inputs, targets = batch
        output = self(inputs)

        # Targets must be tuples of (domain_id, domain_class_id)
        if not isinstance(targets[0], tuple):
            raise ValueError(
                "Targets must be tuples of (domain_id, domain_class_id). Use CombinedDataModule for multi-domain training."
            )

        domain_ids, domain_class_ids = zip(*targets)
        domain_ids = torch.tensor(domain_ids, device=output.device)
        domain_class_ids = torch.tensor(domain_class_ids, device=output.device)

        # Convert to universal targets and compute loss (vectorized)
        universal_targets = self._domain_class_to_universal_targets(
            domain_class_ids, domain_ids
        )
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        # Convert to domain predictions for evaluation (vectorized)
        domain_predictions = self._universal_to_domain_predictions(output, domain_ids)

        # Calculate accuracy per sample
        correct = 0
        total = len(domain_class_ids)

        # Process each unique domain for accuracy calculation (still need this for trimming)
        unique_domains = torch.unique(domain_ids)
        for domain_id in unique_domains:
            domain_mask = domain_ids == domain_id
            domain_preds = domain_predictions[domain_mask]
            domain_targets = domain_class_ids[domain_mask]

            # Get the actual number of classes for this domain
            num_classes = self.domain_to_universal_raw[domain_id.item()].shape[0]
            domain_preds_trimmed = domain_preds[:, :num_classes]

            pred = domain_preds_trimmed.argmax(dim=1, keepdim=True)
            correct += pred.eq(domain_targets.view_as(pred)).sum().item()

        accuracy = correct / total
        self.log_dict({"val_loss": loss, "val_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def predict_step(self, batch):
        inputs, targets = batch
        output = self(inputs)

        # Targets must be tuples of (domain_id, domain_class_id)
        if not isinstance(targets[0], tuple):
            raise ValueError(
                "Targets must be tuples of (domain_id, domain_class_id). Use CombinedDataModule for multi-domain training."
            )

        domain_ids, _ = zip(*targets)
        domain_ids = torch.tensor(domain_ids, device=output.device)

        # Convert to domain predictions (vectorized)
        domain_predictions = self._universal_to_domain_predictions(output, domain_ids)

        # Apply softmax normalization for probabilities
        return torch.softmax(domain_predictions, dim=1)
