#!/bin/bash
#SBATCH --job-name=train_sawsinsub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/sawsinsub.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/ddsp-singing-vocoders

# SawSing as an example
python main.py --config ./configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub \