#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp1d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DeepMIR/HW3/logs/generate.log

lscpu | egrep 'CPU\(s\)'
cd /work/jgtf0322/Homework/2023-Fall-DeepMIR/HW3


OUT_DIR="generated_samples/llama/topp095_temp095"
mkdir -p $OUT_DIR

python3 generate.py \
    --output_dir ${OUT_DIR} \
    --do_sample true \
    --temperature 0.95 \
    --top_p 0.95 \
