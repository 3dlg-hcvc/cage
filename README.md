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
# Create an environment from the `environment.yml` file:
conda env create -f environment.yml
conda activate cage

# Install PyTorch3D (not required for training):
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```
## Data
We share the training data ([here](https://aspis.cmpt.sfu.ca/projects/cage/data.zip)~101MB) preprocessed from [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset. Once downloaded, extract the `data` and put directly in the project folder. The data root can be configured with `system.datamodule.root=<path/to/your/data/directory>` in `configs/cage.yaml` file.

## Training
Run `python main.py --config configs/cage.yaml --log_dir <folder/for/logs>` to train the model from the scratch. The experiment files will be recorded at `./<log_dir>/cage/<version>`.


## Citation
Please cite our work if you find it helpful:
```
@article{liu2023cage,
    author  = {Liu, Jiayi and Tam, Hou In Ivan and Mahdavi-Amiri, Ali and Savva, Manolis},
    title   = {{CAGE: Controllable Articulation GEneration}},
    year    = {2023},
    journal = {arXiv preprint arXiv:xxx}
}
```
## Acknowledgements
This implementation is partially powered by 🤗[Diffusers](https://github.com/huggingface/diffusers).