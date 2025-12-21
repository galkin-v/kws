from typing import List

import torch
import torch.nn.functional as F


class Conv1dNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        kernels: List[int],
        strides: List[int],
        channels: List[int],
        activation: torch.nn.Module,
        hidden_size: int,
    ):
        super().__init__()

        features = in_features

        module_list = []

        for kernel_size, stride, chs in zip(kernels, strides, channels):

            module_list.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=chs,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=chs,
                    ),
                    activation,
                    torch.nn.Conv1d(in_channels=chs, out_channels=chs, kernel_size=1),
                    torch.nn.BatchNorm1d(num_features=chs),
                    activation,
                    torch.nn.MaxPool1d(kernel_size=stride),
                ]
            )

            features = chs

        module_list.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(features, hidden_size),
                activation,
                torch.nn.Linear(hidden_size, n_classes),
                torch.nn.LogSoftmax(-1),
            ]
        )

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


class DepthwiseConv1dNet(torch.nn.Module):
    """Lightweight depthwise-separable 1D conv model for KWS."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        channels: List[int],
        activation: torch.nn.Module = torch.nn.SiLU(),
        kernel_size: int = 5,
        stride: int = 2,
    ):
        super().__init__()
        features = in_features
        layers: list[torch.nn.Module] = []

        for ch in channels:
            padding = kernel_size // 2
            layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=features,
                    ),
                    torch.nn.BatchNorm1d(features),
                    activation,
                    torch.nn.Conv1d(features, ch, kernel_size=1),
                    torch.nn.BatchNorm1d(ch),
                    activation,
                ]
            )
            features = ch

        layers.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(features, n_classes),
                torch.nn.LogSoftmax(dim=-1),
            ]
        )

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SubSpectralNorm(torch.nn.Module):
    """SubSpectral Normalization (SSN) that splits the frequency axis into
    sub-bands and applies BatchNorm per sub-band as described in the paper.
    This is a lightweight approximation suitable for small KWS models.
    """

    def __init__(self, channels: int, n_subbands: int = 5, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.n_subbands = n_subbands
        # We use a single BatchNorm over the concatenated (C * n_subbands)
        # feature maps after reshaping.
        self.bn = torch.nn.BatchNorm2d(channels * n_subbands, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        b, c, f, t = x.shape
        s = self.n_subbands
        if f % s != 0:
            # pad frequency dim so it's divisible by s
            pad = s - (f % s)
            x = F.pad(x, (0, 0, 0, pad))
            f = f + pad

        f_sub = f // s
        # reshape to (B, C*s, f_sub, T)
        x = x.view(b, c, s, f_sub, t).permute(0, 1, 2, 4, 3).contiguous()
        # now (B, C, s, T, f_sub) -> (B, C*s, T, f_sub) after reshape/permute
        x = x.view(b, c * s, t, f_sub)
        x = self.bn(x)
        # invert reshape: (B, C*s, T, f_sub) -> (B, C, s, T, f_sub)
        x = x.view(b, c, s, t, f_sub).permute(0, 1, 4, 3, 2).contiguous()
        # (B, C, f_sub, T, s) -> reshape back to (B, C, F, T)
        x = x.view(b, c, f, t)
        return x


class BCResBlock(torch.nn.Module):
    """Broadcasted Residual Block (simplified to be compact and efficient).

    Structure (roughly following the paper):
      - f2: frequency-depthwise conv (3x1 depthwise) + SSN
      - avg pool across frequency -> temporal feature
      - f1: temporal depthwise conv (1x3 depthwise) + pointwise conv
      - broadcast temporal residual back to 2D and add auxiliary 2D residual
    """

    def __init__(
        self,
        channels: int,
        kernel_freq: int = 3,
        kernel_time: int = 3,
        ssn_subbands: int = 5,
        dropout: float = 0.1,
        activation=torch.nn.ReLU(),
        stride_freq: int = 1,
        dilation_time: int = 1,
        transition: bool = False,
        out_channels: int | None = None,
    ):
        super().__init__()
        out_ch = out_channels or channels
        self.transition = transition

        # f2: frequency-depthwise conv (depthwise over channels)
        # frequency conv (use group=1 to avoid runtime groups mismatch)
        self.freq_dw = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_freq, 1),
            padding=(kernel_freq // 2, 0),
            groups=1,
            stride=(stride_freq, 1),
        )
        # project f2 output to out_ch so additions match when transition=True
        self.f2_pw = torch.nn.Conv2d(channels, out_ch, kernel_size=1)
        self.f2_bn = torch.nn.BatchNorm2d(out_ch)
        self.ssn = SubSpectralNorm(channels, n_subbands=ssn_subbands)

        # temporal branch f1: operates on (B, out_ch, 1, T)
        # temp is derived from f2_out after projecting to out_ch
        self.temp_dw = torch.nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=(1, kernel_time),
            padding=(0, kernel_time // 2),
            groups=1,
            dilation=(1, dilation_time),
        )
        self.pw = torch.nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.bn_pw = torch.nn.BatchNorm2d(out_ch)
        self.activation = activation
        self.dropout = torch.nn.Dropout2d(dropout)

        # if channel count changes we need a pointwise conv for identity
        if transition or out_ch != channels:
            self.res_conv = torch.nn.Conv2d(channels, out_ch, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        identity = x
        if self.res_conv is not None:
            identity = self.res_conv(identity)

        # f2 path
        f2_out = self.freq_dw(x)
        f2_out = self.ssn(f2_out)

        # project f2 to out_ch to match identity/broadcast channels
        f2_out = self.f2_pw(f2_out)
        f2_out = self.f2_bn(f2_out)

        # avgpool frequency -> (B, out_ch, 1, T)
        temp = f2_out.mean(dim=2, keepdim=True)

        # f1: temporal operations
        temp = self.temp_dw(temp)
        temp = self.activation(temp)
        temp = self.pw(temp)
        temp = self.bn_pw(temp)
        temp = self.activation(temp)
        temp = self.dropout(temp)

        # broadcast back to frequency dimension
        b, c, _, t = temp.shape
        f = f2_out.shape[2]
        # expand temporal residual to (B, C, F, T)
        bc = temp.expand(-1, -1, f, -1)

        if self.transition:
            # For transition blocks we don't use the identity shortcut (freq/chan mismatch)
            out = f2_out + bc
        else:
            out = identity + f2_out + bc
        return out


class BCResNet(torch.nn.Module):
    """A compact BC-ResNet family implementation.

    This is a simplified but faithful implementation that follows the
    broadcasted residual learning idea and is easy to scale by width.
    """

    def __init__(
        self,
        n_mels: int = 40,
        n_classes: int = 12,
        base_width: int = 16,
        width_mult: float = 1.0,
        stages: List[int] = [2, 2, 4, 4],
        ssn_subbands: int = 5,
        dropout: float = 0.1,
        activation=torch.nn.ReLU(),
    ):
        super().__init__()
        self.n_mels = n_mels
        widths = [int(w * width_mult) for w in [8, 12, 16, 20]]
        # front conv: (B, 1, F, T) -> (B, base_width, F', T)
        self.front = torch.nn.Sequential(
            torch.nn.Conv2d(1, base_width, kernel_size=(5, 5), padding=(2, 2), stride=(2, 1)),
            torch.nn.BatchNorm2d(base_width),
            activation,
        )

        blocks = []
        in_ch = base_width
        for stage_idx, n_blocks in enumerate(stages):
            out_ch = widths[stage_idx]
            for bidx in range(n_blocks):
                is_transition = bidx == 0 and in_ch != out_ch
                blocks.append(
                    BCResBlock(
                        channels=in_ch,
                        ssn_subbands=ssn_subbands,
                        dropout=dropout,
                        activation=activation,
                        transition=is_transition,
                        out_channels=out_ch if is_transition else in_ch,
                        stride_freq=2 if is_transition else 1,
                    )
                )
                in_ch = out_ch if is_transition else in_ch

        self.blocks = torch.nn.Sequential(*blocks)

        # final projection
        # keep final conv small enough for low freq resolution after downsampling
        self.final_dw = torch.nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding=(1, 1), groups=in_ch)
        self.final_pw = torch.nn.Conv2d(in_ch, base_width * 2, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(base_width * 2, 128),
            activation,
            torch.nn.Linear(128, n_classes),
            torch.nn.LogSoftmax(dim=-1),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, F, T) or (B, 1, F, T)
        if spec.dim() == 3:
            x = spec.unsqueeze(1)
        else:
            x = spec

        x = self.front(x)
        x = self.blocks(x)
        x = self.final_dw(x)
        x = self.final_pw(x)
        return self.classifier(x)
