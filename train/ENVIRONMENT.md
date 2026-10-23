# Training environment (M4+)

`train/.venv` reuses the host's NVIDIA PyTorch build (CUDA, arm64) through `--system-site-packages`
and adds only PEFT, installed **without dependencies** so pip can't replace the CUDA torch:

```bash
uv venv --system-site-packages --python /usr/bin/python3 train/.venv
uv pip install --python train/.venv/bin/python --no-deps peft==0.21.2
```

| Package | Version | Source |
| --- | --- | --- |
| torch | 2.7.0a0+ecf3bae40a.nv25.02 (CUDA 12.8) | host (NVIDIA container) |
| transformers | 5.14.1 | host |
| accelerate | 1.15.0 | host |
| peft | 0.21.2 | train/.venv, --no-deps |

A plain `uv pip install peft` tried to resolve torch from PyPI (a multi-GB download that would
shadow the CUDA build); it was stopped after 10 minutes.
