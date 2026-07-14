# Credits and Acknowledgments

This project builds upon and extends two excellent open-source projects. We are grateful to the original authors for their pioneering work.

## Original Projects

### 1. ByteTrack - Multi-Object Tracking

**Repository**: [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)

**Paper**: Zhang, Y., Sun, P., Jiang, Y., et al. (2021). "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." arXiv preprint arXiv:2110.06864.

**License**: MIT License

**What We Use**:
- Core tracking algorithm and data association logic
- CPU-based implementation (`byte_tracker.py`)
- Kalman filter implementation (`kalman_filter.py`)
- Matching functions (`matching.py`)
- BaseTrack and STrack classes
- Track state management

**Our Modifications**:
- Added GPU-accelerated batch processing (`batch_gpu_tracker.py`)
- Implemented GPU Kalman filter with vectorized operations
- Added numerical stability safeguards (covariance regularization, safe aspect ratio, NaN isolation)
- Extended for multi-stream simultaneous tracking
- Integrated appearance (ReID) fusion into the association step
- Cleaned up imports and package structure, added a pytest suite

### 2. Deep Person ReID - Re-Identification Models

**Repository**: [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)

**Paper**: Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2019). "Omni-Scale Feature Learning for Person Re-Identification." IEEE International Conference on Computer Vision (ICCV).

**License**: MIT License

**What We Use**:
- OSNet model architecture (`osnet.py`)
- Model variants (x1.0, x0.75, x0.5, x0.25, ibn_x1.0)
- Pretrained ImageNet weights (re-hosted on this repository's GitHub releases
  with sha256 verification for reliable direct downloads; original source is
  the deep-person-reid Google Drive)
- Feature extraction for appearance-based matching

**Our Modifications**:
- Added `build_osnet()` wrapper function for easier model loading
  (including local-checkpoint support for offline setups)
- Safe weight loading (`torch.load(..., weights_only=True)`, sha256 checks)
- Integrated with GPU tracker for real-time feature extraction
- Optimized for batch processing of multiple streams
- Added performance benchmarks for different model sizes

## File-by-File Attribution

### From ByteTrack (FoundationVision/ByteTrack)

| File | Original Path | Modifications |
|------|---------------|---------------|
| `bytetrack/byte_tracker.py` | `yolox/tracker/byte_tracker.py` | Fixed imports, minor cleanups |
| `bytetrack/kalman_filter.py` | `yolox/tracker/kalman_filter.py` | Fixed imports |
| `bytetrack/matching.py` | `yolox/tracker/matching.py` | Replaced cython_bbox with pure Python IoU |
| `bytetrack/basetrack.py` | `yolox/tracker/basetrack.py` | No changes |

### From Deep Person ReID (KaiyangZhou/deep-person-reid)

| File | Original Path | Modifications |
|------|---------------|---------------|
| `bytetrack/osnet.py` | `torchreid/models/osnet.py` | Added `build_osnet()` wrapper |

### Our Original Contributions

| File | Description |
|------|-------------|
| `bytetrack/batch_gpu_tracker.py` | GPU-accelerated multi-stream tracker |
| `bytetrack/gpu_kalman_filter.py` | GPU Kalman filter with batched operations |
| `bytetrack/gpu_matching.py` | GPU matching with vectorized distance computation |
| `bytetrack/batch_track_state.py` | Track state management for GPU tracker |
| `bytetrack/__init__.py` | Clean package exports |
| `tools/*` | Demo and benchmark scripts |
| `examples/*` | Usage examples |
| `tests/*` | Pytest suite |
| `docs/*` | Comprehensive documentation |
| `scripts/*` | Helper scripts |

## Other Dependencies

- **LAP**: [gatagat/lap](https://github.com/gatagat/lap) (BSD-2) — Jonker-Volgenant
  linear assignment solver used for Hungarian matching.

## Test Video

`test_videos/6387-191695740_medium.mp4` — pedestrians filmed from above,
a free stock video from [Pixabay](https://pixabay.com/) (2016 upload,
distributed under the then-current CC0/Pixabay free license; no attribution
required). Used only as a small sample input for examples and benchmarks.

## Key Improvements Over Original

### Performance
- Batched Kalman filtering and IoU computation on GPU across all streams
- Parallel Hungarian assignment on a CPU thread pool
- Measured speedups per stream count: see [docs/benchmarks.md](docs/benchmarks.md)

### Stability
- Covariance regularization (prevents overflow in multi-stream scenarios)
- Safe aspect ratio computation (prevents division by zero)
- NaN/inf detection and isolation of corrupted tracks

### Usability
- Clean, flat package structure: `from bytetrack import BatchGPUTracker`
- Comprehensive documentation and examples
- Pytest suite (`pytest tests/`)

## Citation

If you use this project in your research, please cite both original papers:

```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2110.06864},
  year={2021}
}

@article{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## License

This project maintains the MIT License from both original projects. See [LICENSE](LICENSE) file for details.

## Contact

- **Original ByteTrack**: See [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)
- **Original OSNet**: See [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- **This Project**: Open issues at [BatchGpuByteTrack](https://github.com/sweetlhare/BatchGpuByteTrack/issues)

## Thank You

We extend our sincere gratitude to:
- **Yifu Zhang** and the ByteTrack team for the excellent tracking algorithm
- **Kaiyang Zhou** and the deep-person-reid team for the OSNet models
- The open-source community for their continued support and contributions

---

*Last Updated: February 2026*
