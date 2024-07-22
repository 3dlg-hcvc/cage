# CAGE
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white"></a>

### CAGE: Controllable Articulation GEneration

[Jiayi Liu](https://sevenljy.github.io/), [Hou In Ivan Tam](https://iv-t.github.io/), [Ali Mahdavi-Amiri](https://www.sfu.ca/~amahdavi/), [Manolis Savva](https://msavva.github.io/)

CVPR 2024

<img src="docs/static/images/teaser.webp" alt="drawing" style="width:100%"/>

[Page](https://3dlg-hcvc.github.io/cage/) | [Paper](https://arxiv.org/abs/2312.09570) | [Data](https://aspis.cmpt.sfu.ca/projects/cage/data.zip) (alternative link for data: [OneDrive](https://1sfu-my.sharepoint.com/:u:/g/personal/jla861_sfu_ca/EVFoahRzu4hMpQiGi4OsDbYBU170oPAjvWa02iohyj5sTg?e=qnBra3))

## Setup
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies. The environment was tested on Ubuntu 20.04.4 LTS.
```
# Create a conda environment
conda create -n cage python=3.10
conda activate cage

# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyGraphviz
conda install --channel conda-forge pygraphviz

# Install other packages
pip install -r requirements.txt

# Install PyTorch3D (not required for training):
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```
## Data
We share the training data ([here](https://aspis.cmpt.sfu.ca/projects/cage/data.zip)~101MB) preprocessed from [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset. Once downloaded, extract the `data` and put it directly in the project folder. The data root can be configured with `system.datamodule.root=<path/to/your/data/directory>` in `configs/cage.yaml` file. If you find it slow to download the data from our server, please try this alternative link on [OneDrive](https://1sfu-my.sharepoint.com/:u:/g/personal/jla861_sfu_ca/EVFoahRzu4hMpQiGi4OsDbYBU170oPAjvWa02iohyj5sTg?e=qnBra3).

## Training
Run `python main.py --config configs/cage.yaml --log_dir <folder/for/logs>` to train the model from the scratch. The experiment files will be recorded at `./<log_dir>/cage/<version>`. The original model was trained on two NVIDIA A40 GPUs.


## Citation
Please cite our work if you find it helpful:
```
@inproceedings{liu2024cage,
    title={CAGE: Controllable Articulation GEneration},
    author={Liu, Jiayi and Tam, Hou In Ivan and Mahdavi-Amiri, Ali and Savva, Manolis},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={17880--17889},
    year={2024}
}
```
## Acknowledgements
This implementation is partially powered by 🤗[Diffusers](https://github.com/huggingface/diffusers).
