from typing import Callable, Literal
import torch
import torch.nn as nn
from torch import Tensor


class Separable3DConv(nn.Module):
    """
    Factorized 3D convolution: spatial 1x3x3 followed by temporal 3x1x1.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels for both spatial and temporal convs.
    norm_layer : Callable[[int], nn.Module], default=nn.BatchNorm3d
        Normalization applied after the temporal convolution.
    activation : Callable[[], nn.Module], default=nn.ReLU
        Activation used after spatial conv and after BN.

    Notes
    -----
    The factorization reduces parameters and FLOPs versus a full 3x3x3 kernel
    while preserving spatiotemporal modeling capacity.

    Shape
    -----
    Input:  (B, C_in, D, H, W)
    Output: (B, C_out, D, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 norm_layer: Callable[[int], nn.Module] = nn.BatchNorm3d,
                 activation: Callable[[], nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                 padding=(0, 1, 1), bias=False)
        self.temporal = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1),
                                  padding=(1, 0, 0), bias=False)
        self.bn = norm_layer(out_channels)
        self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spatial then temporal factorized convolutions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, D, H, W).
        """
        x = self.activation(self.spatial(x))
        x = self.bn(self.temporal(x))
        return self.activation(x)


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module (3D variant).

    Applies channel attention (global pooling + MLP + sigmoid),
    followed by spatial attention (conv over pooled channel descriptors).

    Parameters
    ----------
    channels : int
        Number of channels in the input.
    reduction_ratio : int, default=16
        Channel reduction ratio for the MLP in the channel attention.

    Notes
    -----
    - `channels` should be divisible by `reduction_ratio`.
    - Spatial attention takes as input the concatenation of channel-wise
      average and max maps (2 channels).

    Shape
    -----
    Input:  (B, C, D, H, W)
    Output: (B, C, D, H, W)
    """
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply channel then spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Refined tensor of shape (B, C, D, H, W).
        """
        ca = self.channel_attention(x)
        x = x * ca
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        return x * sa


class FlowGuidedFusion(nn.Module):
    """
    Partial Dense 3D fusion using flow as a multiplicative gate over RGB.

    The flow stream is passed through a sigmoid to obtain [0, 1] weights
    which modulate the RGB features. A temporal pooling reduces the temporal
    dimension to aggregate motion evidence.

    Parameters
    ----------
    kernel_size_t : int, default=8
        Temporal pooling kernel size (MaxPool3d over D).

    Notes
    -----
    `kernel_size_t` should not exceed the temporal length D of the incoming
    features (otherwise the output may collapse undesirably).

    Shape
    -----
    rgb :  (B, C, D, H, W)
    flow : (B, C, D, H, W)
    return: (B, C, D_out, H, W) where D_out depends on pooling.
    """
    def __init__(self, kernel_size_t: int = 8) -> None:
        super().__init__()
        self.temporal_pool = nn.MaxPool3d(kernel_size=(kernel_size_t, 1, 1))

    def forward(self, rgb: Tensor, flow: Tensor) -> Tensor:
        """
        Fuse RGB and flow features with temporal pooling.

        Parameters
        ----------
        rgb : torch.Tensor
            RGB feature tensor (B, C, D, H, W).
        flow : torch.Tensor
            Flow feature tensor (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Fused features after temporal pooling, shape (B, C, D_out, H, W).
        """
        flow = torch.sigmoid(flow)
        fused = rgb * flow
        return self.temporal_pool(fused)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 3D tensors.

    Parameters
    ----------
    channels : int
        Number of channels in the input.
    reduction : int, default=16
        Reduction ratio for the bottleneck MLP.

    Shape
    -----
    Input:  (B, C, D, H, W)
    Output: (B, C, D, H, W)
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Channel re-calibration via global context.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Channel-weighted tensor with the same shape as input.
        """
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ViolenceDualStreamNet(nn.Module):
    """
    Dual-stream 3D CNN for violence detection (RGB + optical flow).

    The network processes RGB (3 ch) and flow (2 ch) streams with shared
    structural motifs (factorized 3D convs, attention), fuses them with a
    simple FGF-style multiplicative gating, aggregates spatiotemporal evidence,
    and classifies with a compact MLP head.

    Parameters
    ----------
    num_classes : int, default=2
        Number of output classes. For binary classification, 2 logits are
        returned (apply `torch.softmax` or `torch.sigmoid` as needed).
    dropout : float, default=0.3
        Dropout probability used in late 3D and 2D stages.
    classifier_hidden : int, default=128
        Hidden dimension of the first FC layer in the classifier.

    Notes
    -----
    - Input must be shaped as (B, 5, D, H, W) with channel order [R,G,B,flow_x,flow_y].
    - The head returns raw logits; apply an activation externally for probabilities.

    Shape
    -----
    Input:   (B, 5, D, H, W)
    Logits:  (B, num_classes)

    Examples
    --------
    >>> model = ViolenceDualStreamNet()
    >>> x = torch.randn(2, 5, 32, 112, 112)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([2, 2])
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 classifier_hidden: int = 128) -> None:
        super().__init__()

        def build_stream(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                Separable3DConv(in_channels, 16),
                Separable3DConv(16, 16),
                nn.MaxPool3d((1, 2, 2)),
                Separable3DConv(16, 32),
                Separable3DConv(32, 32),
                nn.MaxPool3d((1, 2, 2)),
                CBAM3D(32),
                Separable3DConv(32, 32),
                Separable3DConv(32, 32),
                nn.MaxPool3d((1, 2, 2))
            )

        self.rgb_stream = build_stream(3)
        self.flow_stream = build_stream(2)
        self.fgf_fusion = FlowGuidedFusion(kernel_size_t=8)

        self.merge_block = nn.Sequential(
            Separable3DConv(32, 64),
            nn.AdaptiveAvgPool3d((1, 3, 3)),
            SEBlock(64),
            Separable3DConv(64, 128),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
            nn.Dropout3d(p=dropout)
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
        """
        Initialize 3D convolution weights with Kaiming normal (ReLU).

        Parameters
        ----------
        m : nn.Module
            Module to initialize (called via `apply`).
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through dual streams, fusion, aggregation, and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Video tensor of shape (B, 5, D, H, W), channels ordered as
            [R, G, B, flow_x, flow_y].

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).

        Raises
        ------
        AssertionError
            If input tensor does not have shape (B, 5, D, H, W).
        """
        assert x.ndim == 5 and x.shape[1] == 5, f"Expected shape (B, 5, D, H, W), got {x.shape}"
        rgb = self.rgb_stream(x[:, 0:3])
        flow = self.flow_stream(x[:, 3:5])
        fused = self.fgf_fusion(rgb, flow)
        out = self.merge_block(fused)
        return self.classifier(out)