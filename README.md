## Installation

The easiest way to set up is to install all requirements into a docker environment:

`docker run -v `<i>< path to FlyingChairs2 ></i>`:/workspace/FlyingChairs2 -v `<i>< path to flow datasets dir ></i>`:/workspace/flowData -v `<i>< path to this repo ></i>`:/workspace/nanoflownet --gpus all -it tensorflow/tensorflow:2.8.0-gpu`

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

`docker run -v `<i>< path to FlyingChairs2 ></i>`:/workspace/FlyingChairs2 -v `<i>< path to flow datasets dir ></i>`:/workspace/flowData -v `<i>< path to this repo ></i>`:/workspace/nanoflownet --gpus all -it nanoflownet`

