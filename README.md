# adversarial_training_imagenet
This repository contains the code for ARES-Bench ([A Comprehensive Study on Robustness of Image Classification Models: Benchmarking and Rethinking](https://arxiv.org/abs/2302.14301)), 
a Python library for adversarial machine learning research focusing on benchmarking adversarial 
robustness on image classification correctly and comprehensively.

### Installation

- Install `CUDA 11.3` with `cuDNN 8` following the official installation guide of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).

- Setup conda environment:
```bash
# Create environment
conda create -n RobustBench python=3.9 -y
conda activate RobustBench

# Install requirements
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch -y

# Clone RobustBench
git clone https://github.com/thu-ml/adversarial_training_imagenet.git

# Install other requirements
pip install -r requirements.txt
```
The requirements.txt includes its dependencies.
## Files in the folder
- `adv/`: PyTorch implementation of Attack methods.
- `ares_attack_torch/`: PyTorch implementation of Attack methods from ARES.
- `model/`: PyTorch implementation of models and a model zoo.
- `src_ckpt/`: Storage for checkpoints downloaded.
- `src_data/`: Storage for datasets.
- `test_out/`: Output directory for model training and testing.
- `train_configs/`: Training configs for adversarial training.

## Example to run the codes

ARES-Bench provides command line interface to run benchmarks. For example, you can train a robust model of ResNet50 with corresponding configuration:

    python -m torch.distributed.launch --nproc_per_node=<num-of-gpus-to-use> adversarial_training.py --configs=./train_configs/resnet50.yaml

There are 5 eval_***.py files in the folder that evaluate the adversarial robustness benchmarks on ImageNet and its variants. For example, if you want to evaluate the robustness of ResNet50 on ImageNet-C dataset, you can run the following command line:

    python -m torch.distributed.launch --nproc_per_node=<num-of-gpus-to-use> eval_ood.py --model_name=resnet50_normal --imagenet_c_path=<imagenet-c-path>

## Citing ARES-Bench

```
@article{liu2023comprehensive,
  title={A Comprehensive Study on Robustness of Image Classification Models: Benchmarking and Rethinking},
  author={Liu, Chang and Dong, Yinpeng and Xiang, Wenzhao and Yang, Xiao and Su, Hang and Zhu, Jun and Chen, Yuefeng and He, Yuan and Xue, Hui and Zheng, Shibao},
  journal={arXiv preprint arXiv:2302.14301},
  year={2023}
}
```


