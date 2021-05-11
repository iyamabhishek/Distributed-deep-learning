#!/bin/bash

#SBATCH -p standard
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH -t 30:0:0
#SBATCH --constraint=rhel8

module load cuda
module load cudnn

source ~/anaconda3/bin/activate
conda activate DDL_GPU

python TF.py -m ResNet50 -d ImageNet_subset -p /lustre/cmsc714-1ves/datasets/TF_ImageNet --mGPU True -b 16