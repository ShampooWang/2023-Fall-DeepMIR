#!/bin/bash
#SBATCH --job-name=train_sawsinsub_pretrained
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account="MST110260"
#SBATCH --partition=gp4d
#SBATCH --output=/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/logs/sawsinsub_pretrained.log

cd /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/ddsp-singing-vocoders

# SawSing as an example
python main.py --config /work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/ddsp-singing-vocoders/configs/sawsinsub.yaml \
               --stage  training \
               --model SawSinSub \
               --model_ckpt "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW2/ddsp-singing-vocoders/exp/f1-full/sawsinsub-256/ckpts/vocoder_best_params.pt" \