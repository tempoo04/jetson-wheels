# jetson-wheels

Prebuilt Python wheels for NVIDIA Jetson (linux_aarch64) that aren't available
on PyPI.

## Available wheels

| Package | Version | Python | JetPack | CUDA | cuDNN | TensorRT |
|---|---|---|---|---|---|---|
| onnxruntime-gpu | 1.20.1 | 3.11 | 6.x | 12.6 | 9.3 | 10.3 |

## Install

Pick the release matching your setup and install directly from the URL:

```bash
pip install https://github.com/tempoo04/jetson-wheels/releases/download/v1.20.1/onnxruntime_gpu-1.20.1-cp311-cp311-linux_aarch64.whl
```

Or pin it in `pyproject.toml`:

```toml
[project]
dependencies = [
    "onnxruntime-gpu @ https://github.com/tempoo04/jetson-wheels/releases/download/v1.20.1/onnxruntime_gpu-1.20.1-cp311-cp311-linux_aarch64.whl ; sys_platform == 'linux' and platform_machine == 'aarch64'",
]
```

## Why this repo exists

PyPI's `onnxruntime-gpu` package ships only `manylinux_x86_64` and Windows
wheels — no `linux_aarch64`. The official NVIDIA Jetson AI Lab index covers
the default JetPack Python (3.10), but not other Python versions. This repo
fills those gaps with wheels built directly on Jetson hardware.

## Compatibility

Each wheel is built for a specific Python version (`cpXYZ` ABI tag). Match
the Python version exactly — a cp311 wheel will not install on Python 3.10
or 3.12. See each release's notes for full build environment details.

## Verify after install

```bash
pip show onnxruntime-gpu
```

Expected: name `onnxruntime-gpu`, version `1.20.1`, location inside your active environment's `site-packages`.

```python
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
# Expect: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Quick GPU sanity check — runs a tiny model on CUDA and confirms it doesn't silently fall back to CPU:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "your_model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("Using:", sess.get_providers()[0])  # should print CUDAExecutionProvider
```
