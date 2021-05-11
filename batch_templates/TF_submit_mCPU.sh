#!/bin/bash

#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 10:0:0
#SBATCH --constraint=rhel8

# Allow threads to transition quickly
export KMP_BLOCKTIME=0
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,0,0

source ~/anaconda3/bin/activate
conda activate DDL

srun python TF.py -m ResNet50 -d MNIST -p ./datasets/TF_MNIST --mCPU True
srun python TF.py -m ResNet152 -d MNIST -p ./datasets/TF_MNIST --mCPU True
srun python TF.py -m DenseNet121 -d MNIST -p ./datasets/TF_MNIST --mCPU True
srun python TF.py -m DenseNet201 -d MNIST -p ./datasets/TF_MNIST --mCPU True