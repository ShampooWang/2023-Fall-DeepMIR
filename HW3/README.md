# 2023 Fall DeepMIR HW3: Transformer-based music generation

## 0. Install packages

- Python 3.9
- Pytorch 1.13.1

After install above packages, install `requirements.txt` by

```bash
pip isntall -r requirements.txt
```

## 1. Inference by the provided music language model

```bash
python generate.py \
    --output_dir /path/to/your/target/dir
```

## 2. Train the music language model

```bash
bash scripts/train_gpt2.sh or scripts/train_llama.sh
```