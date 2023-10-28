#!/bin/bash
python /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/inference.py\
    --dataset_root /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/artist20/preprocess_devTest\
    --ckpt  "/work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/exp/large_last_fix_acc/epoch=9-step=11589-val_top1_acc=0.9076.ckpt"\
    --output_csv /work/jgtf0322/Homework/2023-Fall-DLMAG/HW1_singer_classification/inference.csv\
