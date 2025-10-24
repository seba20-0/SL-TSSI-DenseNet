# SL-TSSI-DenseNet

This repo trains a DenseNet121 model on TSSI (Temporal-Spatial Skeleton Image) representations built from 2D landmarks. It expects datasets that provide sequences of skeleton coordinates (not raw RGB frames).

Quick start with a prepared dataset (e.g., `wlasl100_tssi`) and custom dataset instructions are below.

## Quick Start (prepared TFDS dataset)

1) Download a prepared dataset into `./datasets`:

- Windows/Powershell: manually download and unzip `wlasl100_tssi.zip` from the URL in `datasets/download_datasets.sh`.

2) Train:

```
python train.py --dataset wlasl100_tssi --use_wandb false --batch_size 32 --num_epochs 5
```

## Use Your Own Dataset (videos -> NPZ -> training)

If you only have videos (e.g., `MLR511-ArabicSignLanguage-Dataset-MP4`), you must first extract 2D landmarks (pose + hands) and convert them into a simple NPZ format the trainer can read.

1) Install requirements (Python):

```
pip install tensorflow mediapipe opencv-python numpy
```

2) Prepare NPZ dataset from your videos:

```
python tools/prepare_mlr511_to_npz.py \
  --src datasets/MLR511-ArabicSignLanguage-Dataset-MP4 \
  --dst datasets/mlr511_npz \
  --val_users user11,user12
```

This creates:
- `datasets/mlr511_npz/samples/*.npz` with arrays: `pose` (frames x joints x 2), `label` (int)
- `datasets/mlr511_npz/splits/train.txt` and `validation.txt`
- `datasets/mlr511_npz/label_map.json`

3) Train on your NPZ dataset:

```
python train.py --dataset mlr511_npz --use_wandb false --batch_size 32 --num_epochs 10 --augmentation true --pipeline default
```

Notes:
- `--use_wandb false` disables Weights & Biases logging.
- You can change the input temporal size via `config.MLR511_INPUT_HEIGHT` (default 64). The dataloader pads/resize frames to this height.
- The default pipeline builds the blue channel as an angle from x/y and applies speed/scale/flip augmentations.

## Evaluate Per Signer

To report accuracy and inference time for specific users (e.g., 1, 8, 11):

```
python test.py --dataset mlr511_npz \
  --weights_dir <path_to_saved_weights_or_wandb_export> \
  --use_wandb false \
  --batch_size 32 \
  --pipeline default \
  --test_users 1,8,11
```

Output is printed and saved to `artifacts/reports/test_per_user_report.json`.

## Datasets Directory

- Prepared TFDS-style datasets (e.g., `wlasl100_tssi`) can be placed directly under `./datasets` (unzipped).
- Custom NPZ datasets should follow `./datasets/mlr511_npz` as created by the tool.

## Troubleshooting

- If you see: `Dataset <name> not found`, ensure you used `--dataset wlasl100_tssi` for prepared sets or `--dataset mlr511_npz` for the NPZ path above.
- If Mediapipe fails on some videos during preparation, the script will skip them and continue.
