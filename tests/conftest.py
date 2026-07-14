import numpy as np
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require a CUDA device")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def det():
    """Factory for a single-detection array [x1, y1, x2, y2, score]."""
    def _det(x, y=100.0, w=50.0, h=120.0, score=0.9):
        return np.array([[x, y, x + w, y + h, score]], dtype=np.float32)
    return _det
