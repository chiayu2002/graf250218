#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o output0220_mod_ssp.txt				
#PBS -e error0220_mod_ssp.txt				
#PBS -N nerf
cd ~/graf250218												

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf250218/configs/default.yaml