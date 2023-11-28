#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DeepMIR/HW3/logs/gpt2_greedy.log


cd /work/jgtf0322/Homework/2023-Fall-DeepMIR/HW3

python code_base/util/eval_metrics.py \
    --output_file_path "/work/jgtf0322/Homework/2023-Fall-DeepMIR/HW3/generated_samples/llama/topp095_temp095/midi"