# vid2vid
### [Website](http://vid-2-vid.herokuapp.com/index.html) | [arXiv](https://arxiv.org/abs/1808.06601) | [Project Github Link](https://github.com/sakshamgupta006/video-to-video-synthesis)

Our project aims to implement NVIDIA's Video-to-Video Synthesis research paper, the goal of the research is to learn a mapping function from an input source video (sequences of semantic segmentation masks) to an output photo-realistic video that precisely depicts the content of the source video.

## Contributors 
- Saksham Gupta (saksham@vt.edu)
- Pronnoy Goswami (pronnoygoswami@vt.edu)
- Parth Vora (parthv@vt.edu)
- Mohak Bheda (mohakbheda@vt.edu)

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
    - Intel Core i7 8700k (8 Cores)
    - HyperX 16GB Ram
    - Nvidia RTX 2080 Graphic Card 
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

