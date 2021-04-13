#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 10
#SBATCH --constraint=rhel8

source ~/anaconda3/bin/activate
conda activate DDL

python TF.py