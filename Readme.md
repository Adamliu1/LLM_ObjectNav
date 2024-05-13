# Quick Start:

## Create Conda env

NOTE: use python 3.9 otherwise conda version of habitat sim not working
OR build from pip, source

setup pytorch (I use cuda 11.8) (note: seems like habitat-baseline requires pytorch and will configure it)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
cuda 11.7
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```


Install Habitat Sim
```
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```
Or build from pip
https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md
```
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install . -v
```

Install Habitat baseline and Habitat Lab
```
pip install -e habitat-lab
pip install -e habitat-baselines
```

## Set OpenAI keys
To use the API locally please provide an API key and organization ID in a .env file in the root directory of the repository.
```
OPENAI_API_KEY =  sk-***
OPENAI_ORG =  org-***
```

## Install dependencies

List of dependencies
```
pip install transformers scikit-image scikit-fmm algo wandb openai==0.28 python-dotenv retry
```


Install ``detectron2``
Update see: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Download Segmentation Model to ``RedNet/model`` path
```
https://drive.usercontent.google.com/download?id=1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv&export=download&authuser=0
```
linux cmd (not sure if it works yet)
```
mkdir -p RedNet/model
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv' -O RedNet/model/rednet_semmap_mp3d_40.pth
```

## Download HM3D dataset
Full instruction here: https://github.com/facebookresearch/habitat-sim/blob/089f6a41474f5470ca10222197c23693eef3a001/datasets/HM3D.md
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d --data-path data/
```
Download Habitat-Lab HM3D ObjectNav task dataset
https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets
```
cd data/ # datapath

wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip

unzip objectnav_hm3d_v2.zip
mkdir -p datasets/objectnav/hm3d
mv objectnav_hm3d_v2 datasets/objectnav/hm3d/v2
rm objectnav_hm3d_v2.zip
```

rename hm3d to hm3d_v0.2
```
data/scene_datasets/hm3d -> data/scene_datasets/hm3d_v0.2
```

