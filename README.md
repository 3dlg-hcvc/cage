# CAGE
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white"></a>

### CAGE: Controllable Articulation GEneration

[Jiayi Liu](https://sevenljy.github.io/), [Hou In Ivan Tam](), [Ali Mahdavi-Amiri](https://www.sfu.ca/~amahdavi/), [Manolis Savva](https://msavva.github.io/)

<img src="docs/static/images/teaser.webp" alt="drawing" style="width:100%"/>

[Page](https://3dlg-hcvc.github.io/cage/) | [Paper]() | [Data](https://aspis.cmpt.sfu.ca/projects/cage/data.zip)

## Setup
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies. The environment was tested on Ubuntu 20.04.4 LTS with a single NVIDIA A40.
```
# Create an environment from the `environment.yml` file.

conda env create -f environment.yml
conda activate cage

# Install PyTorch3D:

conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```
## Data
We share the [training data](https://aspis.cmpt.sfu.ca/projects/cage/data.zip) preprocessed from [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset. Once downloaded, extract the `data` folder and the data root can be configured as `system.datamodule.root=<path/to/your/data/directory>` in `configs/cage.yaml` file.

## Training
Run `python main.py --config configs/cage.yaml` to train the model from the scratch. The experiment files will be recorded at `./tb_logs/cage/<version>`.

### Pretrained Model
To be released. 

## Citation
Please cite our work if you find it helpful:
```

```
## Acknowledgements
This implementation is partially powered by 🤗[Diffusers](https://github.com/huggingface/diffusers).