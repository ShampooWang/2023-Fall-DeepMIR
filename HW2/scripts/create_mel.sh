#!/bin/bash
#SBATCH --job-name=create_mel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/create_mel.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2

python /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/utils/create_mel_data.py