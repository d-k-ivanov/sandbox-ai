# Basic pytorch3D exercises

_Python Version_: **3.11.8**

## Conda

```bash
conda create -n torch3d python=3.10
conda activate torch3d
conda install pytorch=2.4.1 pytorch-cuda=11.8 torchvision torchaudio cuda-toolkit=11.8 -c pytorch -c nvidia
conda install black
conda install jupyter
conda install scipy
python -m pip install open3d

# This is tricky to install on Windows. Need to fulfil compilation environment precisely.
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Third-Party precompiled binaries:
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu118
```

## PIP

TBA

```bash
python -m venv venv
. venv/Scripts/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Fixes

```txt
# ['ninja', '-v'] -> ['ninja', '--version']
C:\tools\miniconda3\envs\torch3d\lib\site-packages\torch\utils\cpp_extension.py
```
