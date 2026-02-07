### Prerequisites

The code is built with following libraries:

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1
pip mim install mmcv-full==1.5.0
pip install mmdet==2.20.0
Please set up the remaining parts according to the environment deployment instructions for BEVfusion and md4all_main.
After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

You can then create a symbolic link `data` to the `/dataset` directory in the docker.

### Data Preparation

### Model

download:https://pan.baidu.com/s/1HQnuKoCoPeYDSkT4kW9c4w?pwd=c67t
according to md4all_main README file, download md4allDDa_nuscenes.ckpt and put it into checkpoints

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```

### Evaluation

We also provide instructions for evaluating our pretrained models. Please download the checkpoints using the following script: 

```bash
./tools/download_pretrained.sh
```

Then, you will be able to run:

```bash
torchpack dist-run -np [number of gpus] python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval [evaluation type]
```

For example, if you want to evaluate the detection variant of BEVFusion-camera-only+DGIFM, you can try:

```bash
torchpack dist-run -np 8 python tools/test.py configs\BEVfusion_DPTHead.yaml runs-max-DPT-varible0.35-0.55\run-1b009e1e\epoch_27.pth --eval bbox
```

While for the BEVFusion-camera-only+SAMBA:

```bash
torchpack dist-run -np 8 python tools/test.py configs\BEVfusion_sappm.yaml runs-sappm\run-86eb1edc\epoch_25.pth --eval map
```
It is worth noting that due to the incompatibility between DGIFM and SAMBA, extra attention is needed for mmdet3d\models\fusion_models\bevfusion.py and mmdet3d\models\vtransforms\lss.py.
For example: After running DGIFM, you need to comment out the relevant code in mmdet3d\models\fusion_models\bevfusion.py and mmdet3d\models\vtransforms\lss.py before you can run SAMBA.
### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train BEVFusion-camera-only+DGIFM, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs\BEVfusion_DPTHead.yaml
```

For BEVFusion-camera-only+SAMBA model, please run:

```bash
torchpack dist-run -np 8 python tools/train.py configs\BEVfusion_sappm.yaml
```

