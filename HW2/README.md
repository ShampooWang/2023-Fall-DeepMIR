# 2023 Fall Deep Learning for Music Analysis and Generation HW2: GAN-based Mel-Vocoder

## 0. Install packages

- Python 3.8
- Pytorch 2.0.1

After install above packages, install `requirements.txt` by

```bash
cd bigvgan
pip isntall -r requirements.txt
```

## 1. Inference by the provided bigvgan model

```bash
cd bigvgan

python inference_e2e.py \
	--test_dir /path/to/your/input/dir \
    --output_dir /path/to/your/target/dir
```

## 2. Train the bigvgan model

```bash
cd bigvgan

python train.py \
    --config ./configs/bigvgan_22khz_80band.json \
    --checkpoint_path /path/to/your/saving/dir \
    --input_wavs_dir /path/to/your/dataset/root \
    --input_training_file /path/to/your/train_wav_path.txt \
    --input_validation_file /path/to/your/valid_wav_path.txt \
    --eval_subsample 1 \
```