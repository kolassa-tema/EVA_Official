# Expressive Gaussian Human Avatars from Monocular RGB Video (NeurIPS 2024)
Code repository for the paper:
**Expressive Gaussian Human Avatars from Monocular RGB Video**

[Hezhen Hu](https://alexhu.top/), [Zhiwen Fan](https://zhiwenfan.github.io/), [Tianhao Wu](https://chikayan.github.io/), [Yihan Xi](), [Seoyoung Lee](https://seoyoung1215.github.io/), [Georgios Pavlakos](https://geopavlakos.github.io/), [Zhangyang Wang](https://vita-group.github.io/group.html)

[Project Page](https://evahuman.github.io/) | [Paper](https://arxiv.org/abs/2407.03204)

![teaser](assets/teaser.jpg)

## Installation and Setup
<!-- --- -->
First, please clone the repo and download the necessary required files via [OneDrive](https://utexas-my.sharepoint.com/:u:/g/personal/hezhen_hu_austin_utexas_edu/ESGMO8k169ZDtkuh44rRUxABE1p5g_4tKpHHqsSU6tjXbA?e=JaPdVv).
Remember to place the checkpoint file in the root path.

Then, follow the commands below to set up necessary environments.
Our configuration is based on NVIDIA-driver 535.183.01.

```bash
# Based on CUDA 12.1
conda create -n eva -y python=3.10
conda activate eva
bash scripts/env_install.sh
bash scripts/bug_fix_eva.sh
conda deactivate

# For SMPLer-X
conda create -n smpler_x python=3.8 -y
conda activate smpler_x
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
pip install -r preprocess/SMPLer-X/requirements.txt
cd preprocess/SMPLer-X/main/transformer_utils
pip install -v -e .
cd ../../../../
pip install setuptools==69.5.1 yapf==0.40.1 numpy==1.23.5
bash scripts/bug_fix.sh
```


## Pipeline for a real-world video
First, process the data as follows (here is a sample video [sample](https://utexas-my.sharepoint.com/:u:/g/personal/hezhen_hu_austin_utexas_edu/ERi1KuGI2H9DlCHjFDiQPbcBrgbb85pLY6GG2eCR78bjWw?e=Fh8iRL).)
```
010 (video_name)/  
└── images/  
```

Please run the following command to do the data preprocessing and avatar modeling.
```bash
ROOT_PATH={PATH_TO_VIDEO_FOLDER} bash Full_running_command.sh
```

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HAMER](https://github.com/geopavlakos/hamer)
- [SMPLer-X](https://github.com/caizhongang/SMPLer-X)
- [DWPose](https://github.com/IDEA-Research/DWPose)
- [SGHM](https://github.com/cxgincsu/SemanticGuidedHumanMatting)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [GauHuman](https://github.com/skhu101/GauHuman)

## Citation
<!-- --- -->

We would appreciate it if you could cite the following work.

```bibtex
@inproceedings{hu2024expressive,
  title={Expressive Gaussian Human Avatars from Monocular RGB Video},
  author={Hu, Hezhen and Fan, Zhiwen and Wu, Tianhao and Xi, Yihan and Lee, Seoyoung and Pavlakos, Georgios and Wang, Zhangyang},
  booktitle={NeurIPS},
  year={2024}
}
```
