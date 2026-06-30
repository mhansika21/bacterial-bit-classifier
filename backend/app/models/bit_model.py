from __future__ import annotations


class BiTModel:
    """Thin wrapper around timm BiT backbones.

    The class imports torch/timm lazily so the project remains runnable in demo
    mode without large ML dependencies.
    """

    def __init__(self, backbone_name: str = "resnet50x1_bitm", n_classes: int = 22) -> None:
        self.backbone_name = backbone_name
        self.n_classes = n_classes
        self.model = None

    def load_backbone(self, pretrained: bool = True):
        try:
            import torch.nn as nn
            import timm
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to enable real BiT inference") from exc

        backbone = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=0)
        feature_dim = getattr(backbone, "num_features", 2048)
        self.model = nn.Sequential(backbone, nn.Linear(feature_dim, self.n_classes))
        return self.model

    def freeze_layers(self, frozen_prefixes: tuple[str, ...] = ("0.conv1", "0.layer1", "0.layer2")) -> None:
        if self.model is None:
            raise RuntimeError("Model must be loaded before freezing layers")
        for name, param in self.model.named_parameters():
            param.requires_grad = not name.startswith(frozen_prefixes)
