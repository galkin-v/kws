import torch
from src.model import BCResNet


def test_bcresnet_forward():
    # small smoke test - forward pass and parameter count
    n_mels = 64
    n_classes = 5
    m = BCResNet(n_mels=n_mels, n_classes=n_classes)

    # synthetic input: (B, F, T)
    x = torch.randn(2, n_mels, 101)
    out = m(x)

    assert out.shape == (2, n_classes), f"unexpected output shape: {out.shape}"

    params = sum(p.numel() for p in m.parameters())
    assert params > 0

    # print some useful info for quick verification
    print(f"BCResNet params: {params}")
    print(f"Output shape: {out.shape}")
