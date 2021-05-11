# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:26:59 2021

@author: Matthew Ziemann

This script is used to download online tensorflow_datasets datasets into the local </datasets> directory.
This must be done off of HPC, and then files manually transferred to HPC. HPC denies remote access to download datasets.

ImageNet_subset is a 10% subset of ImageNet that balances the number of class samples (it's 10% the size of ImageNet), 
with the same size validation set. Use it on DT2, because the GPUs are old and the full ImageNet takes too long to train.

NOTE: FOLLOW THESE INSTRUCTIONS. ImageNet does not play nice, and requires special instructions.

For MNIST:
    1. Define `dataset_name` as 'MNIST'
    2. Define `path` as the directory you'd like it stored to, <MNIST_dir>
    3. Run this script. It will download MNIST into `path` in the correct format, with `train` and `test` subdirectories
    4. Move <MNIST_dir> to HPC, in desired location, and pass that location to `TF.py` when you run it.

For ImageNet or ImageNet_subset:
    1. Manually download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from image-net.org. You'll need ~150 gb memory.
    2. Store them in `<imagenet_dir>/downloads/manual`, where <imagenet_dir> is a directory of your choice
    3. Define `dataset_name` as 'ImageNet' or 'ImageNet_subset'
    4. Define `path` as your <imagenet_dir> from step 2.
    5. Run this script. It will unpack the tarballs, split them, shard them, and save them to `<imagenet_dir>/imagenet2012/5.1.0`
       or `<imagenet_dir>/imagenet2012_subset/10pct/5.1.0`. NOTE: This requires an additional ~150 gb of memory for the full 
       ImageNet set.
    6. Move <imagenet_dir> and its contents to lustre on HPC (you do not need to move the ImageNet tarballs). Maintain the file 
       structure. New path should be <hpc_imagenet_dir>` containing 'imagenet2012' and/or 'imagenet2012_subset/10pct'. 
       Note that location, and pass <hpc_imagenet_dir> to `TF.py` when you run it.

"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Define these according to the instructions from the header.
dataset_name = 'ImageNet_subset'
path = 'E:/TF_ImageNet'


########################################################################################################################
# Load selected dataset into a tf.data.Dataset (separate train and test Dataset)

if dataset_name == 'MNIST':
    ds_train, ds_test = tfds.load('mnist',                  # Define the dataset
                                  split=['train','test'],   # Define how to split the dataset
                                  as_supervised=True,       # State whether dataset is supervised; (features, label)
                                  shuffle_files=True)       # State whether to shuffle input files
    
    
    # Save the datasets into the local <./datasets> directory
    path_train = str(path+'/train')
    path_test = str(path+'/test')
    tf.data.experimental.save(ds_train, path_train)
    tf.data.experimental.save(ds_test, path_test)


elif dataset_name == 'ImageNet':
    # Note: You must download the ImageNet train and validation date separately, place in <imagenet_dir>/downloads/manual, and 
    # direct tfds.load() to <imagenet_dir>.
    
    imagenet_dir = path  # <imagenet_dir> from above
    # Extract, split, and shard the dataset. Will save to <imagenet_dir>/imagenet2012/5.1.0
    ds_train, ds_test = tfds.load('imagenet2012',                 # Define the dataset
                                  split=['train','validation'],   # Define how to split the dataset
                                  data_dir = imagenet_dir,        # Point to downloaded files
                                  as_supervised=True,             # State whether dataset is supervised; (features, label)
                                  shuffle_files=False)            # State whether to shuffle input files


elif dataset_name == 'ImageNet_subset':
    # Note: You must download the ImageNet train and validation date separately, place in <imagenet_dir>/downloads/manual, and 
    # direct tfds.load() to <imagenet_dir>.
    
    imagenet_dir = path  # <imagenet_dir> from above
    # Extract, split, and shard the dataset. Will save to <imagenet_dir>/imagenet2012/5.1.0
    ds_train, ds_test = tfds.load('imagenet2012_subset/10pct',    # Define the dataset
                                  split=['train','validation'],   # Define how to split the dataset
                                  data_dir = imagenet_dir,        # Point to downloaded files
                                  as_supervised=True,             # State whether dataset is supervised; (features, label)
                                  shuffle_files=False)            # State whether to shuffle input files
