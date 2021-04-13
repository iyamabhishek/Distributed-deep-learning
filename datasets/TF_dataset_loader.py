# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:26:59 2021

@author: Matthew Ziemann

This script is used to download online tensorflow_datasets datasets into the local </datasets> directory. 
Necessary for working on HPC.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

########################################################################################################################
# Load selected dataset into a tf.data.Dataset (separate train and test Dataset)
ds_train, ds_test = tfds.load('mnist',                  # Define the dataset
                                split=['train','test'],   # Define how to split the dataset
                                as_supervised=True,       # State whether dataset is supervised; (features, label)
                                shuffle_files=True)       # State whether to shuffle input files


# Save the datasets into the local <./datasets> directory
path_train = './TF_MNIST/train'
path_test = './TF_MNIST/test'
tf.data.experimental.save(ds_train, path_train)
tf.data.experimental.save(ds_test, path_test)
