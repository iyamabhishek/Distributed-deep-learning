# Distributed-deep-learning

## TensorFlow

### Versions

Package | Version 
------------ | -------------  
Python | 3.8.8
TensorFlow | 2.4.1
TensorBoard | 2.4.1
TensorFlow Datasets | 4.2.0 (1.2?)
CUDA Toolkit | 10.2 or 11.0 
cuDNN | 7.6.5 or 8.1 

### Setup

On Linux, simply `conda install tensorflow-gpu`.

NOTE: As of 04/05/2021, if using Anaconda on Windows, TF 2.4 is not available on Anaconda. Use pip to install Tensorflow, TF Datasets, and TF Models.
NOTE: TF Datasets versioning is unclear. Pip is 4.2, anaconda is 1.2, they appear to be the same.

WARNING: TF may look for CUDA Toolkit 11.0 and cuDNN 8.1. In this case, remove earlier versions of CUDA Toolkit and cuDNN, then install cudnn 8.1 from conda-forge, and reinstall CUDA Toolkit.

### Usage

To use the TensorFlow framework:

1. Run `./datasets/TF_dataset_loader.py`to download your desired dataset to the datasets directory. use `<dataset>.element_spec` to get TensorSpec info of the dataset.
2. Modify `./TF.py` to point to the desired dataset. Provide that dataset's `element_spec` to `.TF.py` so it can load the data properly (annoying requirement of the experimental TF.Datasets save/load functions).
3. Run TF.py to train, modifying parameters at will! Use `TF_submit_job.sh` as a template batch script for deepthought2 usage.