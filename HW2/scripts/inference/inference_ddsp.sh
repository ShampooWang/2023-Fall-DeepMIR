#!/bin/bash
#SBATCH --job-name=Full_debuzz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/inference/Full_debuzz.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/ddsp-singing-vocoders

rm -rf "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/ddsp/Full/runtime_gen"

# SawSing as an example
python main.py --config ./configs/full.yaml \
               --stage  validation \
               --model_ckpt "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/ddsp/Full/ckpts/vocoder_best_params.pt" \
               --output_dir "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/exp/ddsp/Full/runtime_gen" \
               --model Full \
               --is_part true \