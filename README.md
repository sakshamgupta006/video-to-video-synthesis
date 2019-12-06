# vid2vid
### [Project](https://tcwang0509.github.io/vid2vid/) | [YouTube(short)](https://youtu.be/5zlcXTCpQqM) | [YouTube(full)](https://youtu.be/GrP_aOSXt5U) | [arXiv](https://arxiv.org/abs/1808.06601) | [Paper(full)](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf)

Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation. It can be used for turning semantic label maps into photo-realistic videos, synthesizing people talking from edge maps, or generating human motions from poses. The core of video-to-video translation is image-to-image translation. Some of our work in that space can be found in [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [SPADE](https://github.com/NVlabs/SPADE). <br><br>
[Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/)  
 [Ting-Chun Wang](https://tcwang0509.github.io/)<sup>1</sup>, [Ming-Yu Liu](http://mingyuliu.net/)<sup>1</sup>, [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/)<sup>2</sup>, [Guilin Liu](https://liuguilin1225.github.io/)<sup>1</sup>, Andrew Tao<sup>1</sup>, [Jan Kautz](http://jankautz.com/)<sup>1</sup>, [Bryan Catanzaro](http://catanzaro.name/)<sup>1</sup>  
 <sup>1</sup>NVIDIA Corporation, <sup>2</sup>MIT CSAIL  
 In Neural Information Processing Systems (**NeurIPS**) 2018  

## Prerequisites
- Linux 
- Python 3
- NVIDIA GPU + CUDA(v 9.0) and cuDNN(v 7.0)
- PyTorch 1.0


## Getting Started
### Installing required libraries and software
- Download and install [Ananconda](https://www.anaconda.com/distribution/).
- Create environment
  ```bash
  conda create --name env_name
  ```
- Install Pytorch 
  ```bash
  conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
  ```
- Install python libraries [dominate](https://github.com/Knio/dominate), requests and dlib.
  ```bash
  pip install dominate requests dlib
  ```
- Installing Tensorboard 
  ```bash
  pip install tensorboard
  ```
### Dataset
- Cityscapes 
  - To download the Cityscapes dataset. Use the following link [Cityscapes download](https://www.cityscapes-dataset.com/)
  - To run the model, copy the downloaded images to the 'datasets' folder. 
 
- Face
  - To download the Face dataset. Use the following link [FaceForensics](http://niessnerlab.org/projects/roessler2018faceforensics.html)

### Downloading Datasets Using Scripts
- To download the dummy dataset, run the script `python scripts/download_datasets.py`.
- To download the [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch), run the script `python scripts/download_flownet2.py`.
- To download our pre-trained model for CityScapes datasets, download from the following [link]().
  - To test the model 
    ```bash
    python test.py --name label2city_2048 --label_nc 35 --loadSize 1024 --n_scales_spatial 3 --use_instance --fg --use_single_G
    ```
    The results will be saved in: `./results/label2city_2048/test_latest/`.

### Training Configuration
- We use the following platform and hardware to train our model and evaluate results
    - Ubuntu 18.04 
    - Cuda 9 with cudnn 7 
    - Intel Core i7 8700k (8 CPUS)
    - HyperX 16Gb Ram
    - Nvidia RTX2080 Graphic Card 
- Training Time : ~50 hours.

### Training 
- First, download the FlowNet2 checkpoint file by running `python scripts/download_models_flownet2.py`.
  - We trained our models using Single RTX2080 GPU. For convenience, we provide some sample training scripts for GPU users. Performance is not guaranteed using these scripts.
  - For example, to train a 256 x 128 video with a single GPU 
  ```bash
  python train.py --name label2city_256_g1 --label_nc 35 --loadSize 256 --use_instance --fg --n_downsample_G 2 --num_D 1 --max_frames_per_gpu 6 --n_frames_total 6
  ```
- To run tensorboard, 
  ```bash
  tensorboard --logdir=runs
  ```
## Citation
```
@inproceedings{wang2018vid2vid,
   author    = {Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Guilin Liu
                and Andrew Tao and Jan Kautz and Bryan Catanzaro},
   title     = {Video-to-Video Synthesis},
   booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},   
   year      = {2018},
}
```

