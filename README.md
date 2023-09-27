# nanoflownet-cnns
This repository contains the code used to create and train NanoFlowNet, and includes pre-trained networks.

Please refer to the [main repository](https://github.com/tudelft/nanoflownet) for the AI-deck implementation and the Crazyflie obstacle avoidance application

## Install

The easiest way to set up is to install all requirements into a docker environment:

`docker run -v <path to FlyingChairs2>:/workspace/FlyingChairs2 -v <path to flow datasets dir>:/workspace/flowData -v <path to this repo>:/workspace/nanoflownet --gpus all -it tensorflow/tensorflow:2.8.0-gpu`

Inside the created docker container:

`pip install opencv-python-headless==4.5.5.64`

`pip install tensorflow_model_optimization==0.7.2`

`pip install tqdm==4.64.0`

`pip install tensorflow_addons==0.16.1`

`pip install wandb==0.12.14`

`pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110==1.12.0`

in another terminal find the docker container id with `docker ps -l`

and commit the changes

`docker commit [container_id] nanoflownet`

This concludes the set-up. The correct container can be now opened (without re-installing the pip requirements) by replacing `tensorflow/tensorflow:2.8.0-gpu` with `nanoflownet`:

`docker run -v <path to FlyingChairs2>:/workspace/FlyingChairs2 -v <path to flow datasets dir>:/workspace/flowData -v <path to this repo>:/workspace/nanoflownet --gpus all -it nanoflownet`

## Publication:
[arXiv preprint](https://arxiv.org/abs/2209.06918)

[IEEE-ICRA 2023 paper](https://ieeexplore.ieee.org/document/10161258)

Please cite us as follows:
```
@inproceedings{bouwmeester2023nanoflownet,
  title={Nanoflownet: Real-time dense optical flow on a nano quadcopter},
  author={Bouwmeester, Rik J and Paredes-Vall{\'e}s, Federico and De Croon, Guido CHE},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1996--2003},
  year={2023},
  organization={IEEE}
}
```
