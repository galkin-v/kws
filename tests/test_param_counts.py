import os
import sys
from pathlib import Path

import pytest

try:
    import hydra
    import omegaconf
except ImportError:  # pragma: no cover - dependency guard
    hydra = None
    omegaconf = None
import torch


def register_resolvers():
    if hydra is None or omegaconf is None:
        pytest.skip("hydra/omegaconf not available for param budget test", allow_module_level=True)
    omegaconf.OmegaConf.register_new_resolver("len", lambda x: len(x))
    omegaconf.OmegaConf.register_new_resolver("getindex", lambda lst, idx: lst[idx])
    omegaconf.OmegaConf.register_new_resolver(
        "function", lambda x: hydra.utils.get_method(x)
    )


CONFIGS = [
    "bcresnet_slim.yaml",
    "bcresnet_ref.yaml",
    "conv1d_depthwise.yaml",
    "crnn.yaml",
    "dsresnet_lite.yaml",
    "dsresnet10.yaml",
    "tcn.yaml",
    "conformer_tiny.yaml",
    "mobilenetv3_small.yaml",
]


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_param_budgets_under_10k():
    register_resolvers()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    cwd = str(repo_root)
    for cfg_name in CONFIGS:
        conf_path = os.path.join(cwd, "conf", cfg_name)
        conf = omegaconf.OmegaConf.load(conf_path)
        model = hydra.utils.instantiate(conf.model)
        params = count_params(model)
        assert (
            params < 10_000
        ), f"{cfg_name} has {params} parameters (expected < 10k)"
