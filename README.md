<div align="center">

<h1>Towards Robust Multimodal Domain Generalization via Modality-Domain Joint Adversarial Training</h1>

</div>

The official implementation of the paper "Towards Robust Multimodal Domain Generalization via Modality-Domain Joint Adversarial Training".

## Code

### Environments

The code was developed with:

```text
Python 3.10.4
torch 1.11.0+cu113
mmcv-full 1.2.7
mmaction2 0.13.0
NVIDIA GeForce RTX 4090
```

The implementation additionally depends on `soundfile`, `scipy`, `imageio`, and `tqdm`.

## EPIC-Kitchens Dataset

### Prepare

#### Download pretrained weights

1. Download the [SlowOnly flow model](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) into `EPIC-rgb-flow-audio/pretrained_models`.
2. Download the [SlowFast video model](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) into `EPIC-rgb-flow-audio/pretrained_models`.
3. Download the [VGGSound audio model](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it to `vggsound_avgpool.pth.tar`, and place it in the same directory.

#### Download EPIC-Kitchens

```bash
bash download_script.sh
```

Download the audio files from [EPIC-KITCHENS-audio.zip](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/blob/main/EPIC-KITCHENS-audio.zip). Arrange the data as follows:

```text
EPIC-KITCHENS
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
│   ├── train
│   │   ├── D1
│   │   ├── D2
│   │   └── D3
│   └── test
│       ├── D1
│       ├── D2
│       └── D3
├── flow
│   ├── train
│   └── test
└── audio
    ├── train
    └── test
```

### Video and Flow and Audio

<details>
<summary>Click for commands</summary>

```bash
cd EPIC-rgb-flow-audio

python train_EPIC.py --use_video --use_flow --use_audio -s D2 D3 -t D1 --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/EPIC-KITCHENS/ --seed 0 --use_dsu
python train_EPIC.py --use_video --use_flow --use_audio -s D1 D3 -t D2 --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/EPIC-KITCHENS/ --seed 0 --use_dsu
python train_EPIC.py --use_video --use_flow --use_audio -s D1 D2 -t D3 --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/EPIC-KITCHENS/ --seed 0 --use_dsu
```

</details>

To run every multi-source two-modal and three-modal configuration with one seed:

```bash
cd EPIC-rgb-flow-audio
bash multi_domain_DG.sh
```

## HAC Dataset

### Prepare

Download HAC from the [Human-Animal-Cartoon dataset page](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main). Download the same three pretrained backbones described above into `HAC-rgb-flow-audio/pretrained_models`.

Arrange the data as follows:

```text
HAC
├── human
│   ├── videos
│   ├── flow
│   └── audio
├── animal
│   ├── videos
│   ├── flow
│   └── audio
└── cartoon
    ├── videos
    ├── flow
    └── audio
```

### Video and Flow and Audio

<details>
<summary>Click for commands</summary>

```bash
cd HAC-rgb-flow-audio

python train_HAC.py --use_video --use_flow --use_audio -s animal cartoon -t human --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/HAC/ --seed 0 --use_dsu
python train_HAC.py --use_video --use_flow --use_audio -s human cartoon -t animal --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/HAC/ --seed 0 --use_dsu
python train_HAC.py --use_video --use_flow --use_audio -s human animal -t cartoon --lr 1e-4 --bsz 16 --nepochs 20 --datapath /path/to/HAC/ --seed 0 --use_dsu
```

</details>

To run every multi-source two-modal and three-modal configuration with one seed:

```bash
cd HAC-rgb-flow-audio
bash multi_domain_DG.sh
```

Checkpoints and logs are written under `models/JAT` and `results/JAT`.
