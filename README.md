# Distributed-deep-learning

## TensorFlow

### Versions

Package | Version 
------------ | -------------  
Python | 3.8.8
TensorFlow | 2.4.1
TensorBoard | 2.4.1
TensorBoard Profiler | 2.4.0
TensorFlow Datasets | 4.2.0
CUDA Toolkit | 10.2 or 11.0 
cuDNN | 7.6.5 or 8.0 

### Setup

On Linux, if using an Anaconda environment, `conda install tensorflow-gpu`, `pip install tensorflow-datasets`, and `pip install -U tensorboard-plugin-profile`

On Windows, TF2.4 is not yet available on Anaconda. `pip install tensorflow-gpu`, `pip install tensorflow-datasets`. Then `conda install cudatoolkit`, it should package CUDA Toolkit 10.2 and cuDNN 7.6.5. Finally, `pip install -U tensorboard-plugin-profile`

NOTE: If TF throws errors looking for CUDA 11, remove cuDNN and CUDA Toolkit, then use `conda install -c conda-forge cudnn==8.0.5.39`. This should install CUDA Toolkit 11.0 and cuDNN 8.0. I had a lot of issues with this, reach out if you have problems.

NOTE: TF Datasets versioning is unclear. Pip is 4.2, anaconda is 1.2, they appear to be the same.


### Usage

To use the TensorFlow framework:

2. Modify `./TF.py` to point to the desired dataset. Provide that dataset's `element_spec` to `TF.py` so it can load the data properly (annoying requirement of the experimental TF.Datasets save/load functions).
3. Run `TF.py` to train, modifying parameters at will! Use `TF_submit_job.sh` as a template batch script for deepthought2 usage.

#### I. Preparing the Data

Use `./datasets/TF_dataset_loader.py` to prepare the MNIST and/or ImageNet for training. This script is used to download online tensorflow_datasets datasets into the local </datasets> directory. This must be done off of HPC, and then files manually transferred to HPC. HPC denies remote access to download datasets.

NOTE: FOLLOW THESE INSTRUCTIONS. ImageNet does not play nice, and requires special instructions.

For MNIST:

1. Define script variable `dataset_name` as 'MNIST'

2. Define script variable `path` as the directory you'd like it stored to, <MNIST_dir>

3. Run the script. It will download MNIST into `path` in the correct format, with `train` and `test` subdirectories

4. Move <MNIST_dir> to HPC, in desired location, and pass that location to `TF.py` when you run it.

For ImageNet:
1. Manually download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from image-net.org. You'll need ~150 gb memory.

2. Store them in `<imagenet_dir>/downloads/manual`, where <imagenet_dir> is a directory of your choice

3. Define script variable `dataset_name` as 'ImageNet'

4. Define script variable `path` as your <imagenet_dir> from step 2.

5. Run the script. It will unpack the tarballs, split them, shard them, and save them to `<imagenet_dir>/imagenet2012/5.1.0`. NOTE: This requires an additional ~150 gb of memory.

6. Move <imagenet_dir>/imagenet2012/5.1.0 and its contents to lustre on HPC. Maintain the file structure. New path should be `<hpc_imagenet_dir>/imagenet2012/5.1.0>`. Note that location, and pass <hpc_imagenet_dir> to `TF.py` when you run it.

#### II. Training the Network

1. Write a slurm batch script appropriate for the job you wish to run. The `./batch_templates` directory contains some example scripts when using Anaconda.

2. Within that script, run `TF.py` with the appropriate flags for your desired job:

User inputs:
    --model         : 'ResNet50', 'ResNet152', 'DenseNet121', 'DenseNet201'
    --dataset       : 'MNIST', 'ImageNet'
    --data_path     : string, point to desired dataset
    --num_epochs    : int, number of epochs to train
    --batch_size    : int, batch size; recommend 128 for MNIST and 16 for ImageNet
    --learning_rate : float, optimizer learning rate; default is fine for most use cases
    --ignore_gpu    : 'True' or do not provide
    --mGPU          : 'True' or do not provide
    --mCPU          : 'True' or do not provide
    --mWorker       : Not supported
    --mWorkerGPU    : Not supported


### Logging and Profiling

The `TF.py` script will automatically save a TensorBoard log file for every run into the same directory as `TF.py`. This saves log information on the training performance of the network, as well as profiling information. This can be opened by running tensorboard and pointing it to the log directory: `tensorboard --logir <insert/path/to/tensorboard/log>`. 

This will run a local host instance that can be opened via web browser, and all profiling and training data can be viewed.