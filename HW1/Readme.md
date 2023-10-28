# 2023 Fall Deep Learning for Music Analysis and Generation HW1: Singer classification

## Install requirements

### Main packages

- python ≥ 3.8
- pytorch: 1.10.1+cu111
- pytorch_lightning: 1.5.10
- unsilence
- demucs

```python
pip install -r requirements.txt
```

### Since we do not include`demucs`in `requirements.txt`, you need to install it by

```python
python3 -m pip install -U demucs
```

## Inference

### 0. Preprocess data

Remove instrumental sounds and silence by `demucs` and `unsilence`, respectively. After that segment songs into 5 seconds of snippets.

```python
python preprocess.py\
    --target_dir /path/to/your/target_dir\
    --output_dir /path/to/your/output_dir\
```

Make sure that the target directory structure is like:

```python
-target_dir
	|- 0001.wav
	|- 0002.wav
		...
```

Then, the output directory will be:

```python
-output_dir
	|-0001-
	|	|-seg0.wav
	|	|-seg1.wav
	|	...
	|-0002-
	|	|-seg0.wav
	|	...
	|
	| inference.json
...
```

We will use `inference.json` for the next step’s prediction

### 1. Create predicting csv file

```python
python inference.py\
    --dataset_root /path/to/the/preprocessed/dataset\
    --ckpt  /path/to/your/model/checkpoints\
    --output_csv /path/of/the/ouputting/csv\
```

Note that `dataset_root` should containing `inference.json` created in the previous step.