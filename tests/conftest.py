import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from halugen.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tok(cfg):
    from halugen.chat import load_tokenizer
    try:
        return load_tokenizer(cfg)
    except Exception as e:  # gated repo / no HF token
        pytest.skip(f"cannot load Mistral tokenizer (HF login + accepted license needed): {e}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not _cuda(), reason="needs a CUDA GPU (run on Colab)")


@pytest.fixture(scope="session")
def model(cfg):
    if not _cuda():
        pytest.skip("needs a CUDA GPU")
    from halugen import hf_backend as hf
    hf.setup_determinism(cfg["seed"])
    return hf.load_model(cfg)
