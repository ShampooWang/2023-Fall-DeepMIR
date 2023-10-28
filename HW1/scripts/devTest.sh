#!/bin/bash
#SBATCH --job-name=devTest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/logs/devTest2.log

lscpu | egrep 'CPU\(s\)'
cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification

EXP_ROOT="/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/exp/devTest"
CFG_FILE="/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/code_base/config/devTest.yaml"
mkdir -p $EXP_ROOT

python3 run_task.py \
    "TrainSingerClassifier" \
    --config ${CFG_FILE} \
    --gpus 1 \
    --njobs 8 \
    --seed 322 \
    --train \
    --save_path ${EXP_ROOT}