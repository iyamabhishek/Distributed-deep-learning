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
cuDNN | 7.6.5 or 8.0 

### Setup

On Linux, simply `conda install tensorflow-gpu` and `conda install tensorflow-datasets`.

On Windows, TF2.4 is not yet available on Anaconda. `pip install tensorflow-gpu`, `pip install tensorflow-datasets`. Then 'conda install cudatoolkit`, it should package CUDA Toolkit 10.2 and cuDNN 7.6.5.  

NOTE: If TF throws errors looking for CUDA 11, remove cuDNN and CUDA Toolkit, then use `conda install -c conda-forge cudnn==8.0.5.39`. This should install CUDA Toolkit 11.0 and cuDNN 8.0. I had a lot of issues with this, reach out if you have problems.

NOTE: TF Datasets versioning is unclear. Pip is 4.2, anaconda is 1.2, they appear to be the same.


### Usage

To use the TensorFlow framework:

1. Run `./datasets/TF_dataset_loader.py`to download your desired dataset to the datasets directory. use `<dataset>.element_spec` to get TensorSpec info of the dataset.
2. Modify `./TF.py` to point to the desired dataset. Provide that dataset's `element_spec` to `TF.py` so it can load the data properly (annoying requirement of the experimental TF.Datasets save/load functions).
3. Run `TF.py` to train, modifying parameters at will! Use `TF_submit_job.sh` as a template batch script for deepthought2 usage.