from typing import TypeAlias, Literal


_Architecture: TypeAlias = Literal["resnet50", "resnet152", "efficientnetv2"]
_EfficientNetVariant: TypeAlias = Literal["s", "m", "l"]
_Optimizer: TypeAlias = Literal["adamw", "sgd"]
_LRScheduler: TypeAlias = Literal["multistep", "step"]
