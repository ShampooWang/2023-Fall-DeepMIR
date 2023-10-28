#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/logs/inference.log

lscpu | egrep 'CPU\(s\)'

python /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/inference.py