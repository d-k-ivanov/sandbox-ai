# Basic pytorch3D exercises - WIP

_Python Version_: **3.11.8**

## Conda

```bash
conda create -n torch3d python=3.10
conda activate torch3d
conda install pytorch pytorch-cuda=12.4 torchvision torchaudio cuda-toolkit -c pytorch -c nvidia
conda install black
conda install jupyter
python -m pip install open3d
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## PIP

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
