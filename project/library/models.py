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

        # Domain for inference
        self.domain_id = None

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
            torch.nn.Linear(num_features, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, self.num_universal_classes),
        )

        # Use KL Divergence loss for universal class activations
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")

    def set_domain(self, domain_id: int):
        """Set the domain for inference predictions.

        Parameters
        ----------
        domain_id : int
            The domain ID to use for mapping universal class predictions
            back to domain class predictions during inference.
        """
        self.domain_id = domain_id

    def _domain_class_to_universal_targets(
        self, domain_class_labels: torch.Tensor
    ) -> torch.Tensor:
        """Convert domain class labels to universal class activation targets.

        Parameters
        ----------
        domain_class_labels : torch.Tensor
            Tensor of domain class labels with shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Tensor of universal class targets with shape (batch_size, num_universal_classes)
        """
        assert (
            self.domain_id is not None
        ), "Domain ID must be set using set_domain() before training"

        batch_size = domain_class_labels.size(0)
        targets = torch.zeros(
            batch_size, self.num_universal_classes, device=domain_class_labels.device
        )

        for i, domain_class_label in enumerate(domain_class_labels):
            # Find the domain class from the label (class id)
            domain_class = DomainClass(
                (np.intp(self.domain_id), np.intp(int(domain_class_label.item())))
            )

            # Get relationships from this domain class to universal classes
            relationships = self.taxonomy.get_relationships_from(domain_class)

            # Sum weights for each universal class (raw relationship weights)
            for relationship in relationships:
                _, source_class, weight = relationship
                if isinstance(source_class, UniversalClass):
                    universal_idx = self.universal_class_to_idx[source_class]
                    targets[i, universal_idx] += weight

        # Normalize each target vector to sum to 1 (across universal classes)
        for i in range(batch_size):
            target_sum = targets[i].sum()
            if target_sum > 0:
                targets[i] = targets[i] / target_sum

        return targets

    def _universal_to_domain_predictions(
        self, universal_predictions: torch.Tensor
    ) -> torch.Tensor:
        """Convert universal class predictions to domain class predictions.

        Parameters
        ----------
        universal_predictions : torch.Tensor
            Universal class predictions with shape (batch_size, num_universal_classes)

        Returns
        -------
        torch.Tensor
            Domain class predictions with shape (batch_size, num_domain_classes)
        """
        assert self.domain_id is not None, "Domain ID must be set for prediction"

        # Get all domain classes for the current domain
        all_domain_classes = [
            node
            for node in self.taxonomy.get_nodes()
            if isinstance(node, DomainClass) and node[0] == self.domain_id
        ]

        if not all_domain_classes:
            raise ValueError(f"No domain classes found for domain {self.domain_id}")

        # Sort by class_id to ensure consistent ordering
        all_domain_classes.sort(key=lambda x: int(x[1]))
        num_domain_classes = len(all_domain_classes)

        batch_size = universal_predictions.size(0)
        domain_predictions = torch.zeros(
            batch_size, num_domain_classes, device=universal_predictions.device
        )

        # For each domain class, sum the weighted universal class predictions
        for domain_idx, domain_class in enumerate(all_domain_classes):
            # Get relationships from this domain class to universal classes
            relationships = self.taxonomy.get_relationships_from(domain_class)

            for relationship in relationships:
                _, source_class, weight = relationship  # target_class not needed here
                if isinstance(source_class, UniversalClass):
                    universal_idx = self.universal_class_to_idx[source_class]
                    # Add weighted universal prediction to domain prediction
                    domain_predictions[:, domain_idx] += (
                        weight * universal_predictions[:, universal_idx]
                    )

        return domain_predictions

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        assert self.domain_id is not None, "Domain ID must be set for training"

        inputs, target = batch
        output = self(inputs)

        # Convert domain class targets to universal class targets
        universal_targets = self._domain_class_to_universal_targets(target)

        # For KL divergence, we need log probabilities of predictions
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        # For logging, compute accuracy based on domain class predictions
        domain_predictions = self._universal_to_domain_predictions(output)
        pred = domain_predictions.argmax(dim=1, keepdim=True)
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
        assert self.domain_id is not None, "Domain ID must be set for testing"

        inputs, target = batch
        output = self(inputs)

        # Convert to domain predictions for evaluation
        domain_predictions = self._universal_to_domain_predictions(output)
        pred = domain_predictions.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)

        # For loss, still use universal targets with KL divergence
        universal_targets = self._domain_class_to_universal_targets(target)
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        self.log_dict({"eval_loss": loss, "eval_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def validation_step(self, batch):
        assert self.domain_id is not None, "Domain ID must be set for validation"

        inputs, target = batch
        output = self(inputs)

        # Convert to domain predictions for evaluation
        domain_predictions = self._universal_to_domain_predictions(output)
        pred = domain_predictions.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)

        # For loss, still use universal targets with KL divergence
        universal_targets = self._domain_class_to_universal_targets(target)
        output_log_probs = torch.log_softmax(output, dim=1)
        loss = self.criterion(output_log_probs, universal_targets)

        self.log_dict({"val_loss": loss, "val_accuracy": accuracy})
        self.log("hp_metric", accuracy)

    def predict_step(self, batch):
        assert self.domain_id is not None, "Domain ID must be set for prediction"

        (inputs, _) = batch
        output = self(inputs)

        # Return domain predictions
        return self._universal_to_domain_predictions(output)
