# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:42:14 2021

@author: Matthew Ziemann

This script will load user-defined dataset, load user-defined model, and train it with profiling.

Distributed training is supported.

User inputs:
    --model         : 'ResNet50', 'ResNet152', 'DenseNet121', 'DenseNet201'
    --dataset       : 'MNIST', 'ImageNet', 'ImageNet_subset'
    --data_path     : str
    --num_epochs    : int
    --batch_size    : int
    --learning_rate : float
    --ignore_gpu    : 'True' or do not provide
    --mGPU          : 'True' or do not provide
    --mCPU          : 'True' or do not provide
    --mWorker       : Not yet supported
    --mWorkerGPU    : Not yet supported

"""

import os
import time
from argparse import ArgumentParser

###############################################################################
#-- Run options. These must be done before importing tensorflow.
# Choose which model to use: ResNet50, ResNet152, DenseNet121, or DenseNet201
model_name = 'ResNet50'
dataset_name = 'MNIST'  # Not currently used. Implement when you add datasets.
data_path = ''
num_epochs = 5
batch_size = 128  # Sets the number PER GPU
learning_rate = 0.001
IGNORE_GPU = False  # Blocks GPUs from TensorFlow if True.


#-- Select distribution strategy. Can only pick mGPU or mCPU below, or neither of them for standard training.
# Using multi-GPU per node? This enables simple MirroredStrategy for single-node GPU distribution.
MULTI_GPU = False  # WARNING: THIS MAY NOT WORK WITH MULTI_WORKER BELOW. NEED TO VERIFY.
# Using multi-CPU per node? This will assign intra-ops to additional CPUs.
MULTI_CPU = False

# Distributing across nodes? Enables MultiWorkerMirroredStrategy || WARNING: CURRENTLY BROKEN. NEED TO FIX.
MULTI_WORKER = False
if MULTI_WORKER:
    # Using GPUs on multiple nodes? Multi_worker will default to CPU training. If you want to use multi-worker GPU, make this true.
    MW_GPU = False
    
#- Overwrite defaults with user input, if provided. WARNING: Boolean type turns "true" for ANY input. Don't use False inputs.
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model_name", type=str,
                    help="Select the premade model to use")
parser.add_argument("-d", "--dataset", dest="dataset_name", type=str,
                    help="Select the dataset to use")
parser.add_argument("-p", "--data_path", dest="data_path", type=str,
                    help="Provide the path to your dataset")
parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int,
                    help="Number of epochs to train")
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int,
                    help="Size of batches to train with")
parser.add_argument("-l", "--learning_rate", dest="learning_rate", type=float,
                    help="Select the optimizer learning rate to use")
parser.add_argument("-i", "--ignore_gpu", dest="IGNORE_GPU", type=bool,
                    help="Set to True to hide the GPU from TF. Defaults to False.")
parser.add_argument("-g", "--mGPU", dest="MULTI_GPU", type=bool,
                    help="Set to True to use MirroredStrategy for single node GPU. Defaults to False.")
parser.add_argument("-c", "--mCPU", dest="MULTI_CPU", type=bool,
                    help="Set to True to use multi CPUs per node. Defaults to False.")
parser.add_argument("-w", "--mWorker", dest="MULTI_WORKER", type=bool,
                    help="Set to True to use MultiWorker across nodes. BROKEN. Defaults to False.")
parser.add_argument("-wg", "--mWorkerGPU", dest="MW_GPU", type=bool,
                    help="Set to True to if using MultiWorker and want to use GPUs. Defaults to False.")
args = parser.parse_args()
if args.model_name:
    model_name = args.model_name
if args.dataset_name:
    dataset_name = args.dataset_name
if args.data_path:
    data_path = args.data_path
if args.num_epochs:
    num_epochs = args.num_epochs
if args.batch_size:
    batch_size = args.batch_size
if args.learning_rate:
    learning_rate = args.learning_rate
if args.IGNORE_GPU:
    IGNORE_GPU = args.IGNORE_GPU
if args.MULTI_GPU:
    MULTI_GPU = args.MULTI_GPU
if args.MULTI_CPU:
    MULTI_CPU = args.MULTI_CPU
if args.MULTI_WORKER:
    MULTI_WORKER = args.MULTI_WORKER
if args.MW_GPU:
    MW_GPU = args.MW_GPU

# Print run conditions for record keeping
print("\nRun Parameters: \n")
print(f'model_name = {model_name}')
print(f'dataset_name = {dataset_name}')
print(f'data_path = {data_path}')
print(f'num_epochs = {num_epochs}')
print(f'batch_size = {batch_size}')
print(f'learning_rate = {learning_rate}')
print(f'IGNORE_GPU = {IGNORE_GPU}')
print(f'MULTI_GPU = {MULTI_GPU}')
print(f'MULTI_CPU = {MULTI_CPU}')
print(f'MULTI_WORKER = {MULTI_WORKER}')
if MULTI_WORKER:
    print(f'MW_GPU = {MW_GPU}')
print('')

###############################################################################
#-- Set up environment
if IGNORE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


if MULTI_CPU:
    # Get number of threads from Slurm
    numThreads = int(os.getenv('SLURM_CPUS_PER_TASK',1))

    # Set number of threads for inter-operator parallelism,
    # start with a single thread
    numInterOpThreads = 1
    
    # The total number of threads must be an integer multiple
    # of numInterOpThreads to make sure that all cores are used
    assert numThreads % numInterOpThreads == 0
    
    # Compute the number of intra-operator threads; the number
    # of OpenMP threads for low-level libraries must be set to
    # the same value for optimal performance
    numIntraOpThreads = numThreads // numInterOpThreads
    os.environ['OMP_NUM_THREADS'] = str(numIntraOpThreads)


import tensorflow as tf
import tensorflow_datasets as tfds

# Set the random seed for reproducability
tf.random.set_seed(43)

#-- Print debugging data.
from tensorflow.python.client import device_lib
print(str(device_lib.list_local_devices()))
print('TF Found Devices: '+str(tf.config.list_physical_devices()))

# If distributing, set up the distribution environment
if MULTI_CPU:
    # Configure TensorFlow to use multiple CPUs on a node.
    tf.config.threading.set_inter_op_parallelism_threads(numInterOpThreads)
    tf.config.threading.set_intra_op_parallelism_threads(numIntraOpThreads)
elif MULTI_GPU:
    # Then define distribution strategy
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    batch_size *= num_gpus
    print(f'Number of GPUs: {num_gpus}')
    print('')
    print('--------------------------------------------------')
    print('')
elif MULTI_WORKER:
    # Reset the TF_CONFIG env variable
    os.environ.pop('TF_CONFIG', None)

    # Define communication scheme based on GPU or CPU; Defaults to CPU
    if MW_GPU:
        communication_options = tf.distribute.experimental.CommunicationOptions(
                                    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    else:
        communication_options = tf.distribute.experimental.CommunicationOptions(
                                    implementation=tf.distribute.experimental.CommunicationImplementation.RING)
    # Then define distribution strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communications_options=communication_options)
    # num_workers = len(tf_config['cluster']['worker'])
    # batch_size *= num_workers
    # print(f'Number of Workers: {num_workers}')
    print('')
    print('--------------------------------------------------')
    print('')
    
# Start a global timer to record total runtime (dataset loading and training)
start_time = time.perf_counter()

###############################################################################
#-- MNIST dataset pipeline
if dataset_name == 'MNIST':
    # Load the dataset from local </datasets> directory.
    path_train = str(data_path+'/train')
    path_test = str(data_path+'/test')
    # Define dataset structure, from Dataset.element_spec.
    ## MNIST = (tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int64))
    structure = (tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8), tf.TensorSpec(shape=(), dtype=tf.int64))
    ds_train = tf.data.experimental.load(path_train, structure)
    ds_test = tf.data.experimental.load(path_test, structure)
    
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    def pad_img(image, label):
        """Pads image from (28,28,1) to (32,32,1)"""
        paddings = [[2,2],[2,2],[0,0]]
        return tf.pad(image, paddings, mode="CONSTANT", constant_values=0.0), label
    
    #-- Training data pipeline
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
    # Batch after shuffling to get unique batches at each epoch.
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
    
elif dataset_name == 'ImageNet' or dataset_name == 'ImageNet_subset':
    # Load the dataset from <imagenet_dir>/imagenet2012/5.1.0
    imagenet_dir = data_path  # <imagenet_dir> from above path
    if dataset_name == 'ImageNet':
        data_name = 'imagenet2012'
    else:
        data_name = 'imagenet2012_subset/10pct'
    # imagenet_dir = '/lustre/cmsc714-1ves/datasets/TF_ImageNet'  # Example for HPC
    ds_train, ds_test = tfds.load(data_name,                      # Define the dataset
                                  split=['train','validation'],   # Define how to split the dataset
                                  data_dir = imagenet_dir,        # Point to downloaded files
                                  as_supervised=True,             # State whether dataset is supervised; (features, label)
                                  shuffle_files=True,             # State whether to shuffle input files
                                  download = False)               # This must be from an already downloaded file.
    
    def imagenet_preprocess(image, label):
        """
        Dataset map function for preprocessing imagenet. Resizes to (224, 224, 3) and applies preprocessing.
        """
        i = image
        i = tf.cast(i, tf.float32)
        i = tf.image.resize_with_crop_or_pad(i, 224, 224)
        if model_name == 'ResNet50' or model_name == 'ResNet152':
            i = tf.keras.applications.resnet.preprocess_input(i)
        else:
            i = tf.keras.applications.densenet.preprocess_input(i)
        return (i, label)
    
    #-- Training data pipeline. We don't cache because ImageNet is huge.
    ds_train = ds_train.map(imagenet_preprocess)
    # Batch data. 
    ds_train = ds_train.batch(batch_size)
    # Prefetch for performance boost
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    
    #-- Evaluation data pipeline. Similar to training data pipeline.
    ds_test = ds_test.map(imagenet_preprocess)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

###############################################################################
#-- Load and compile a premade model. Select from one of the four premade models, given by user.
if dataset_name == 'MNIST':
    input_shape = (32,32,1)
    include_top = False
elif dataset_name == 'ImageNet' or dataset_name == 'ImageNet_subset':
    input_shape = (224,224,3)
    include_top = True
    
if MULTI_GPU or MULTI_WORKER:
    with strategy.scope():
        model = tf.keras.models.Sequential()
        
        if model_name == 'ResNet50':
            model.add(tf.keras.applications.ResNet50(include_top=include_top, # Whether to include the fully-connected layer at the top of the network.
                                                     weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                     input_shape=input_shape, # Input shape. Minimum (32,32,1)
                                                     pooling='avg'))          # Apply pooling to the output of the last convolutional block.
        elif model_name == 'ResNet152':
            model.add(tf.keras.applications.ResNet152(include_top=include_top, # Whether to include the fully-connected layer at the top of the network.
                                                     weights=None,             # Pretrained weight set to use. `None` cues random initialization.
                                                     input_shape=input_shape,  # Input shape. Minimum (32,32,1) 
                                                     pooling='avg'))           # Apply pooling to the output of the last convolutional block.
        elif model_name == 'DenseNet121':
            model.add(tf.keras.applications.DenseNet121(include_top=include_top, # Whether to include the fully-connected layer at the top of the network.
                                                        weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                        input_shape=input_shape, # Input shape. Minimum (32,32,1)
                                                        pooling='avg'))          # Apply pooling to the output of the last convolutional block.
        
        elif model_name == 'DenseNet201':
            model.add(tf.keras.applications.DenseNet201(include_top=include_top, # Whether to include the fully-connected layer at the top of the network.
                                                        weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                        input_shape=input_shape, # Input shape. Minimum (32,32,1)
                                                        pooling='avg'))          # Apply pooling to the output of the last convolutional block.
        
        # MNIST can't use the stock fully connected layer, since it only has 10 classifications. Manually input it here.
        if dataset_name == 'MNIST':  
            model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',  # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics=['accuracy']  #tf.keras.metrics.SparseCategoricalAccuracy()
        )
else:
    model = tf.keras.models.Sequential()
    
    if model_name == 'ResNet50':
        model.add(tf.keras.applications.ResNet50(include_top=include_top,       # Whether to include the fully-connected layer at the top of the network.
                                                 weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                 input_shape=input_shape,   # Input shape. Minimum (32,32,1)
                                                 pooling='avg'))          # Apply pooling to the output of the last convolutional block.
    elif model_name == 'ResNet152':
        model.add(tf.keras.applications.ResNet152(include_top=include_top,       # Whether to include the fully-connected layer at the top of the network.
                                                 weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                 input_shape=input_shape,   # Input shape. Minimum (32,32,1) 
                                                 pooling='avg'))          # Apply pooling to the output of the last convolutional block.
    elif model_name == 'DenseNet121':
        model.add(tf.keras.applications.DenseNet121(include_top=include_top,       # Whether to include the fully-connected layer at the top of the network.
                                                    weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                    input_shape=input_shape,   # Input shape. Minimum (32,32,1)
                                                    pooling='avg'))          # Apply pooling to the output of the last convolutional block.
    
    elif model_name == 'DenseNet201':
        model.add(tf.keras.applications.DenseNet201(include_top=include_top,       # Whether to include the fully-connected layer at the top of the network.
                                                    weights=None,            # Pretrained weight set to use. `None` cues random initialization.
                                                    input_shape=input_shape,   # Input shape. Minimum (32,32,1)
                                                    pooling='avg'))          # Apply pooling to the output of the last convolutional block.
    
    # MNIST can't use the stock fully connected layer, since it only has 10 classifications. Manually input it here.
    if dataset_name == 'MNIST':    
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',  # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics=['accuracy']  #tf.keras.metrics.SparseCategoricalAccuracy()
    )

###############################################################################
#-- Set tensorboard callback for profiling
if MULTI_CPU:
    log_dir_name = str('./'+dataset_name+'_'+model_name+'_'+str(numThreads)+'CPUs'+'_log')
elif MULTI_GPU:
    log_dir_name = str('./'+dataset_name+'_'+model_name+'_'+str(num_gpus)+'GPUs'+'_log')
elif MULTI_WORKER:
    log_dir_name = str('./YOU_NEED_TO_MAKE_A_NAME_WHEN_THIS_IS_SUPPORTED')
else:
    log_dir_name = str('./'+dataset_name+'_'+model_name+'_log')
    
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir_name,
                                                      histogram_freq=1,
                                                      update_freq=1)  # How often to log histogram visualizations

###############################################################################

start_train_time = time.perf_counter()

model.fit(ds_train,
          epochs=num_epochs,
          validation_data=ds_test,
          callbacks=tensorboard_callback)

end_time = time.perf_counter()

print(f"{model_name} - Total train time: {end_time-start_train_time:0.4f} seconds")
print(f"{model_name} - Total runtime: {end_time-start_time:0.4f} seconds")
print(f"Data logged to: {log_dir_name}")