# Basic pytorch3D exercises

_Python Version_: **3.11.8**

## Conda

```bash
conda create -n torch3d python=3.11
conda activate torch3d
conda install pytorch=2.5.1 pytorch-cuda=12.4 torchvision torchaudio cuda-toolkit=12.4 -c pytorch -c nvidia
conda install black jupyter matplotlib scipy
python -m pip install open3d

# This is tricky to install on Windows. Need to fulfil compilation environment precisely.
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# PyTorch3D precompiled binaries:
python -m pip install pytorch3d==0.7.8+pt2.5.1cu124 --extra-index-url https://d-k-ivanov.github.io/packages-py

# Confirmation
python -c "import torch; import pytorch3d; print('PyTorch3D version:', pytorch3d.__version__)"
```

## PIP

TBA

```bash
python -m venv venv
. venv/Scripts/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
