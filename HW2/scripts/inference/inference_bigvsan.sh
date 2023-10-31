#!/bin/bash
#SBATCH --job-name=small_dev_bigvsan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/inference/small_dev_bigvsan.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/bigvsan

python inference.py \
    --checkpoint_file "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/bigvsan_spec_norm/g_ep40" \
    --output_dir "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/inference_result/small_dev_bigvsan"
