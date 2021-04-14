# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:42:14 2021

@author: Matthew Ziemann
"""


#- Uncomment the below two lines to run CPU only, and ignore GPU resources.
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time
import tensorflow as tf
#import tensorflow_datasets as tfds
#import numpy as np
#import matplotlib.pyplot as plt

############################################################################### 
#-- Debugging options.
from tensorflow.python.client import device_lib
print(str(device_lib.list_local_devices()))
print('TF Found Devices: '+str(tf.config.list_physical_devices()))

###############################################################################
#-- Run options
num_epochs = 10
batch_size = 128
learning_rate = 0.001

###############################################################################
#-- Dataset pipeline
# Start a global timer
start_time = time.perf_counter()
# Load the dataset from local </datasets> directory.
path_train = './datasets/TF_MNIST/train'
path_test = './datasets/TF_MNIST/test'
# Define dataset structure, from Dataset.element_spec.
## MNIST = (tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int64))
structure = (tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int64))
ds_train = tf.data.experimental.load(path_train, structure)
ds_test = tf.data.experimental.load(path_test, structure)

#-- Training data pipeline
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def pad_img(image, label):
    """Pads image from (28,28,1) to (32,32,1)"""
    paddings = [[2,2],[2,2],[0,0]]
    return tf.pad(image, paddings, mode="CONSTANT", constant_values=0.0), label

# Normalize the image data (for MNIST) from uint8 to float32
ds_train = ds_train.map(normalize_img, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Zero pad the data (for MNIST) from (28,28,1) to (32,32,1)
ds_train = ds_train.map(pad_img,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

# If dataset fits in memory, cache it for better performance. NOTE: Perform random transformations after caching.
ds_train = ds_train.cache()
# Shuffle the data; set shuffle buffer >= dataset size.
ds_train = ds_train.shuffle(100000)
# Batch after shuffling to get unique batches at each epoch. Padded batch for MNIST to pad sizes to (32,32,1)
ds_train = ds_train.batch(batch_size)
# Prefetch for performance boost
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

#-- Evaluation data pipeline. Similar to training data pipeline.
ds_test = ds_test.map(normalize_img, 
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
# Zero pad the data (for MNIST) from (28,28,1) to (32,32,1)
ds_test = ds_test.map(pad_img,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
# Cache after batching since we don't need shuffling.
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


###############################################################################

#-- Load a premade model.
model = tf.keras.models.Sequential()

model.add(tf.keras.applications.ResNet50(include_top=False,       # Whether to include the fully-connected layer at the top of the network.
                                         weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                         input_shape=(32,32,1),
                                         pooling='avg'))   # Input shape. Minimum (32,32,1)

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss='sparse_categorical_crossentropy',  # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=['accuracy']  #tf.keras.metrics.SparseCategoricalAccuracy()
)

start_train_time = time.perf_counter()

model.fit(
    ds_train,
    epochs=num_epochs,
    validation_data=ds_test
)

end_time = time.perf_counter()

print(f"Total train time: {end_time-start_train_time:0.4f} seconds")
print(f"Total runtime: {end_time-start_time:0.4f} seconds")