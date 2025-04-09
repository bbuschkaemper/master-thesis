import lightning as pl
import torch
from torchvision.models.resnet import (
    resnet50,
    ResNet50_Weights,
    resnet152,
    ResNet152_Weights,
)
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, StepLR


class ResNetModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        architecture,
        optim,
        optim_kwargs,
        lr_scheduler=None,
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

        # Replace last layer to fit the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
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

    def configure_optimizers(self):
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
                **self.lr_scheduler_kwargs,
            )
        elif self.lr_scheduler == "step":
            scheduler = StepLR(
                optimizer,
                **self.lr_scheduler_kwargs,
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
