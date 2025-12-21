from typing import List

import torch
import torch.nn.functional as F
from torch import nn


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


class CRNNNet(torch.nn.Module):
    """Tiny CRNN with depthwise conv front-end and BiGRU backend."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        conv_channels: List[int],
        rnn_hidden: int,
        kernel_size: int = 5,
        stride: int = 2,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        features = in_features
        conv_layers: list[nn.Module] = []
        for ch in conv_channels:
            padding = kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(
                        features,
                        features,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=features,
                    ),
                    nn.BatchNorm1d(features),
                    activation,
                    nn.Conv1d(features, ch, kernel_size=1),
                    nn.BatchNorm1d(ch),
                    activation,
                ]
            )
            features = ch
        self.conv = nn.Sequential(*conv_layers)
        self.rnn = nn.GRU(
            input_size=features, hidden_size=rnn_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            activation,
            nn.Linear(rnn_hidden, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = self.conv(x)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        y, _ = self.rnn(x)
        # pool over time
        y = y.mean(dim=1)
        return self.head(y)


class SE1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).unsqueeze(-1)
        return x * scale


class DSResBlock1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: float = 1.5,
        se_reduction: int = 8,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        hidden = max(out_ch, int(in_ch * expand_ratio))
        self.pw_expand = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, stride=stride, groups=hidden)
        self.se = SE1d(hidden, reduction=se_reduction)
        self.pw_proj = nn.Conv1d(hidden, out_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.act = activation
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_expand(x)
        out = self.act(self.bn1(out))
        out = self.dw(out)
        out = self.act(self.bn2(out))
        out = self.se(out)
        out = self.pw_proj(out)
        out = self.bn3(out)
        if self.use_res:
            out = out + x
        return self.act(out)


class DSResNetLite(nn.Module):
    """Depthwise-separable ResNet-lite with SE."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        widths: List[int],
        strides: List[int],
        stem_width: int = 16,
        activation: nn.Module = nn.SiLU(),
        expand_ratio: float = 1.5,
        se_reduction: int = 8,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, stem_width, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(stem_width),
            activation,
        )
        layers: list[nn.Module] = []
        in_ch = stem_width
        for w, s in zip(widths, strides):
            layers.append(
                DSResBlock1d(
                    in_ch,
                    w,
                    stride=s,
                    expand_ratio=expand_ratio,
                    se_reduction=se_reduction,
                    activation=activation,
                )
            )
            in_ch = w
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, head_hidden),
            activation,
            nn.Linear(head_hidden, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class DSResNet10(nn.Module):
    """Compact DS-ResNet10-inspired model (no residuals, ~10k params)."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        channels: int = 22,
        n_layers: int = 7,
        dilation_schedule: List[int] | None = None,
        se_reduction: int = 8,
        head_pool_kernel: int = 4,
        head_pool_stride: int = 2,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(channels),
            activation,
        )
        self.se = SE1d(channels, reduction=se_reduction)
        self.head_pool = nn.AvgPool1d(kernel_size=head_pool_kernel, stride=head_pool_stride)
        dilations = dilation_schedule or [1, 1, 2, 2, 4, 4, 8][:n_layers]
        layers: list[nn.Module] = []
        for d in dilations[:n_layers]:
            padding = (3 - 1) * d // 2
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=d, groups=channels, bias=False),
                    nn.BatchNorm1d(channels),
                    activation,
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                    activation,
                ]
            )
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.se(x)
        x = self.head_pool(x)
        x = self.blocks(x)
        return self.head(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 5, activation: nn.Module = nn.SiLU()):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=channels
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.act(self.bn(out))
        out = self.pw(out)
        return self.act(out + x)


class TCNNet(nn.Module):
    """Dilated depthwise TCN stack."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        channels: int = 32,
        dilations: List[int] = [1, 2, 4, 8],
        activation: nn.Module = nn.SiLU(),
        head_hidden: int = 48,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(channels),
            activation,
        )
        self.blocks = nn.Sequential(
            *[TCNBlock(channels, d, kernel_size=5, activation=activation) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, head_hidden),
            activation,
            nn.Linear(head_hidden, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class TinyConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        ff_mult: int = 2,
        conv_kernel: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=d_model,
            ),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.ln1(x)
        y, _ = self.mha(y, y, y, need_weights=False)
        x = x + y
        y = self.ffn(self.ln2(x))
        x = x + y
        y = self.conv(self.ln3(x).transpose(1, 2)).transpose(1, 2)
        return x + y


class TinyConformerNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_classes: int,
        d_model: int = 48,
        n_heads: int = 2,
        ff_mult: int = 2,
        conv_kernel: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.SiLU(),
            nn.Conv2d(16, d_model, kernel_size=(3, 3), stride=(2, 1), padding=1),
            nn.SiLU(),
        )
        self.block = TinyConformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            ff_mult=ff_mult,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = x.unsqueeze(1)  # (B,1,F,T)
        x = self.subsample(x)  # (B, C, F', T')
        x = x.mean(dim=2)  # pool freq -> (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        x = self.block(x)
        x = x.mean(dim=1)
        return self.head(x)


class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expansion: int, kernel_size: int, stride, se_reduction: int = 8):
        super().__init__()
        hidden = in_ch * expansion
        self.use_res = (stride == 1 or stride == (1, 1)) and in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.se_reduce = nn.Conv2d(hidden, max(1, hidden // se_reduction), kernel_size=1)
        self.se_expand = nn.Conv2d(max(1, hidden // se_reduction), hidden, kernel_size=1)
        self.pw = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.dw(y)
        se = y.mean(dim=[2, 3], keepdim=True)
        se = self.se_reduce(se)
        se = self.act(se)
        se = self.se_expand(se).sigmoid()
        y = y * se
        y = self.pw(y)
        if self.use_res:
            y = y + x
        return self.act(y)


class MobileNetV3Small2D(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_classes: int,
        widths: List[int] = [12, 16, 20],
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
        )
        blocks = []
        in_ch = widths[0]
        strides = [(2, 2), (2, 1), (1, 1)]
        for out_ch, s in zip(widths[1:], strides):
            blocks.append(MBConv(in_ch, out_ch, expansion=2, kernel_size=3, stride=s[0], se_reduction=8))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

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
        widths: List[int] | None = None,
        ssn_subbands: int = 5,
        dropout: float = 0.1,
        activation=torch.nn.ReLU(),
        classifier_hidden: int = 128,
        final_pw_channels: int | None = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        widths = widths or [int(w * width_mult) for w in [8, 12, 16, 20]]
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
        pw_out = final_pw_channels or base_width * 2
        self.final_pw = torch.nn.Conv2d(in_ch, pw_out, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(pw_out, classifier_hidden),
            activation,
            torch.nn.Linear(classifier_hidden, n_classes),
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


# Reference BC-ResNet implementation (closer to Qualcomm original)
class ConvBNReLURef(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(k_size, use_dil):
            rate = 1
            padding_len = (k_size - 1) // 2
            if use_dil and k_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        layers = [
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)
        ]
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlockRef(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        layers = []
        if self.transition_block:
            layers.append(ConvBNReLURef(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLURef(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        self.f1 = nn.Sequential(
            ConvBNReLURef(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def _bcblock_stage_ref(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlockRef(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNetRef(nn.Module):
    def __init__(self, base_c: int = 8, num_classes: int = 5, n_classes: int | None = None):
        super().__init__()
        self.num_classes = n_classes or num_classes
        self.n = [2, 2, 4, 4]
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.s = [1, 2]
        self._build_network()

    def _build_network(self):
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(_bcblock_stage_ref(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        self.classifier = nn.Sequential(
            nn.Conv2d(self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return F.log_softmax(x, dim=-1)
