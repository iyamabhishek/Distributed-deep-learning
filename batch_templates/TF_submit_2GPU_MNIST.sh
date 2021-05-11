#!/bin/bash

#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH -t 10:0:0
#SBATCH --constraint=rhel8

module load cuda
module load cudnn

source ~/anaconda3/bin/activate
conda activate DDL_GPU

python TF.py -m ResNet50 -d MNIST --mGPU True
python TF.py -m ResNet152 -d MNIST --mGPU True
python TF.py -m DenseNet121 -d MNIST --mGPU True
python TF.py -m DenseNet201 -d MNIST --mGPU True