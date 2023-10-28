#!/bin/bash
#SBATCH --job-name=wavlm_large_weighted_sum
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/logs/val/wavlm_large_weighted_sum.log

lscpu | egrep 'CPU\(s\)'
cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification

ckpt="/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/exp/wavlm_large_weighted_sum/epoch=23-step=27815-val_top1_acc=0.6452.ckpt"

python3 run_task.py \
    "TrainSingerClassifier" \
    --gpus 1 \
    --njobs 8 \
    --seed 322 \
    --eval \
    --resume_ckpt ${ckpt}