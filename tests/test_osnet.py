import numpy as np
import pytest
import torch

from bytetrack.osnet import build_osnet, osnet_x0_25


class TestBuildOsnet:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_osnet('resnet50', pretrained=False, device='cpu')

    def test_forward_shape(self):
        model = build_osnet('osnet_x0_25', pretrained=False, device='cpu')
        assert not model.training
        with torch.no_grad():
            out = model(torch.randn(2, 3, 256, 128))
        assert out.shape == (2, 512)
        assert model.feature_dim == 512

    def test_local_checkpoint_roundtrip(self, tmp_path):
        model = osnet_x0_25(pretrained=False)
        ckpt = tmp_path / "osnet_x0_25.pth"
        torch.save(model.state_dict(), ckpt)

        loaded = build_osnet('osnet_x0_25', device='cpu', checkpoint=str(ckpt))
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_wrong_architecture_checkpoint_raises(self, tmp_path):
        ckpt = tmp_path / "bogus.pth"
        torch.save({'fc.weight': torch.zeros(3, 3)}, ckpt)
        with pytest.raises(RuntimeError, match="No layers"):
            build_osnet('osnet_x0_25', device='cpu', checkpoint=str(ckpt))
