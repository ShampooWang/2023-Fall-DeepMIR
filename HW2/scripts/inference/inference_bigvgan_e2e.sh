#!/bin/bash
#SBATCH --job-name=devTest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/inference/devTest.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/bigvgan

python inference_e2e.py \
    --checkpoint_file "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/bigvgan_pretrained_spec_norm/g_ep0" \
    --output_dir "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/inference_result/devTest"
