import lightning as pl
import torch
from torchvision.models.resnet import (
    resnet50,
    ResNet50_Weights,
    resnet152,
    ResNet152_Weights,
)


class ResNetModule(pl.LightningModule):
    def __init__(self, num_classes, architecture, optim, **optim_kwargs):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.architecture = architecture
        self.optim = optim
        self.optim_kwargs = optim_kwargs

        if self.architecture == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif self.architecture == "resnet152":
            self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Replace last layer to fit the number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

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
            return torch.optim.AdamW(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        elif self.optim == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                **self.optim_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim}")

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
