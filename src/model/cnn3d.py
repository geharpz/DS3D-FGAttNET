import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Literal


class Separable3DConv(nn.Module):
    """
    Separable 3D convolution: spatial (1,3,3) + temporal (3,1,1).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    norm_layer : Callable
        Normalization layer to apply after convolution.
    activationRGB : Callable
        Activation function for the RGB stream.
    activationFlow : Callable
        Activation function for the Flow stream.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
                 activationRGB: Callable[[], nn.Module] = nn.ReLU,
                 activationFlow: Callable[[], nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                 padding=(0, 1, 1), bias=False)
        self.temporal = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1),
                                  padding=(1, 0, 0), bias=False)
        self.bn = norm_layer(out_channels)
        self.activationRGB = activationRGB()
        self.activationFlow = activationFlow()

    def forward(self, x: Tensor) -> Tensor:
        x = self.activationRGB(self.spatial(x))
        x = self.activationFlow(self.bn(self.temporal(x)))
        return x

class GatedFusion(nn.Module):
    """
    Gated fusion module between RGB and Flow streams.

    Applies a learned gate to blend features from both modalities.

    Returns: (1 - gate) * rgb + gate * flow
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb: Tensor, flow: Tensor) -> Tensor:
        fusion_input = torch.cat([rgb, flow], dim=1)
        gate = self.gate(fusion_input)
        return (1 - gate) * rgb + gate * flow


class FlowGuidedFusion(nn.Module):
    """
    Partial Dense 3D Fusion (FGF) block with flow-guided self-learned temporal pooling.

    Multiplies the RGB stream by the sigmoid-activated Flow stream to generate
    an attention-weighted feature volume, followed by a temporal max pooling.

    Parameters
    ----------
    kernel_size_t : int
        Temporal window size for max pooling after fusion.
    """

    def __init__(self, kernel_size_t: int = 8) -> None:
        super().__init__()
        self.temporal_pool = nn.MaxPool3d(kernel_size=(kernel_size_t, 1, 1))

    def forward(self, rgb: Tensor, flow: Tensor) -> Tensor:
        flow = torch.sigmoid(flow)
        fused = rgb * flow
        pooled = self.temporal_pool(fused)
        return pooled
    
class ViolenceDualStreamNet(nn.Module):
    """
    Dual-stream 3D CNN with flexible fusion strategies for violence detection in videos.

    Supports multiply, concatenation, gated and FGF fusions.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dropout : float
        Dropout rate in classifier.
    classifier_hidden : int
        Hidden layer size.
    fusion_type : str
        Fusion method: "multiply", "concat", "gated", or "fgf".
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, classifier_hidden: int = 128,
                 fusion_type: Literal["multiply", "concat", "gated", "fgf"] = "fgf") -> None:
        super().__init__()
        self.fusion_type = fusion_type.lower()
        assert self.fusion_type in {"multiply", "concat", "gated", "fgf"}

        def build_stream(in_channels: int, is_flow: bool = False) -> nn.Sequential:
            normalization = nn.BatchNorm3d
            activationRGB = lambda: nn.ReLU(inplace=True)
            activationFlow = lambda: nn.ReLU(inplace=True)
            finalFlowActivation = nn.Sigmoid if is_flow else activationFlow
            return nn.Sequential(
                Separable3DConv(in_channels, 16, normalization, activationRGB, activationFlow),
                Separable3DConv(16, 16, normalization, activationRGB, activationFlow),
                nn.MaxPool3d((1, 2, 2)),
                Separable3DConv(16, 16, normalization, activationRGB, activationFlow),
                Separable3DConv(16, 16, normalization, activationRGB, activationFlow),
                nn.MaxPool3d((1, 2, 2)),
                Separable3DConv(16, 32, normalization, activationRGB, activationFlow),
                Separable3DConv(32, 32, normalization, activationRGB, activationFlow),
                nn.MaxPool3d((1, 2, 2)),
                Separable3DConv(32, 32, normalization, activationRGB, finalFlowActivation),
                Separable3DConv(32, 32, normalization, activationRGB, finalFlowActivation),
                nn.MaxPool3d((1, 2, 2))
            )

        self.rgb_stream = build_stream(3, is_flow=False)
        self.flow_stream = build_stream(2, is_flow=True)

        if self.fusion_type == "concat":
            self.fusion_conv = nn.Conv3d(64, 32, kernel_size=1)
        elif self.fusion_type == "gated":
            self.gated_fusion = GatedFusion(32)
        elif self.fusion_type == "fgf":
            self.fgf_fusion = FlowGuidedFusion(kernel_size_t=8)

        self.merge_block = nn.Sequential(
            Separable3DConv(32, 64),
            nn.AdaptiveAvgPool3d((1, 3, 3)),
            Separable3DConv(64, 128),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Dropout3d(p=0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 5 and x.shape[1] == 5, f"Expected input shape (B, 5, D, H, W), got {x.shape}"
        rgb = self.rgb_stream(x[:, 0:3])
        flow = self.flow_stream(x[:, 3:5])

        if self.fusion_type == "multiply":
            fused = rgb * flow
        elif self.fusion_type == "concat":
            fused = torch.cat((rgb, flow), dim=1)
            fused = self.fusion_conv(fused)
        elif self.fusion_type == "gated":
            fused = self.gated_fusion(rgb, flow)
        elif self.fusion_type == "fgf":
            fused = self.fgf_fusion(rgb, flow)

        out = self.merge_block(fused)
        return self.classifier(out)

