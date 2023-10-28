#!/bin/bash
#SBATCH --job-name=train_bigvsan_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/bigvsan_base.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/bigvsan

python train.py \
    --config /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/bigvsan/configs/bigvsan_base_22khz_80band.json \
    --input_wavs_dir /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset \
    --input_training_file /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/train_wav_path.txt \
    --input_validation_file /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/dataset/valid_wav_path.txt \
    --checkpoint_path /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/bigvsan_base \
    --eval_subsample 1 \