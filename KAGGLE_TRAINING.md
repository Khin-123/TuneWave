# Kaggle CUDA training

This project trains the stereo L/R vocal-mask U-Net from compressed MUSDB18
`.stem.mp4` files.  The training script decodes chunks on demand through the
`musdb`/`stempeg` stack; it does not convert the dataset to WAV or cache all
spectrograms.  Only `train/` is used for training and validation.  The official
`test/` directory remains untouched for final evaluation.

## Kaggle setup

Upload or clone TuneWave into the notebook, and attach the compressed MUSDB18
dataset.  The dataset path passed to the script must contain `train/` and
`test/`, for example `/kaggle/input/musdb18`.

```bash
!apt-get update -qq
!apt-get install -y -qq ffmpeg
!pip install -q musdb==0.4.3 stempeg==0.2.6 torch numpy scipy librosa soundfile
!ffmpeg -version
```

```python
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Use a writable output directory.  Checkpoints and records are written below
`/kaggle/working`; the dataset under `/kaggle/input` is read-only.

```bash
%cd /kaggle/working/TuneWave/backend
```

## Three-epoch cloud smoke test

This bounded run uses eight train tracks, two validation tracks, three epochs,
and AMP when CUDA is available:

```bash
python train_full_vocal_unet.py --dataset-root /kaggle/input/musdb18 --checkpoint-dir /kaggle/working/tunewave_checkpoints/cloud_smoke --cloud-smoke --run --batch-size 2 --num-workers 2 --device cuda --mixed-precision
```

## Full training commands

Maximum 20 total epochs with early stopping after eight stale validation
epochs:

```bash
python train_full_vocal_unet.py --dataset-root /kaggle/input/musdb18 --checkpoint-dir /kaggle/working/tunewave_checkpoints/full --batch-size 2 --num-workers 2 --optimizer adam --learning-rate 1e-3 --loss l1 --epochs 20 --early-stopping-patience 8 --checkpoint-interval 1 --device cuda --mixed-precision --run
```

Exactly 20 total epochs (patience 20 prevents an earlier stop in normal
training; use this when every epoch must run):

```bash
python train_full_vocal_unet.py --dataset-root /kaggle/input/musdb18 --checkpoint-dir /kaggle/working/tunewave_checkpoints/full --batch-size 2 --num-workers 2 --optimizer adam --learning-rate 1e-3 --loss l1 --epochs 20 --early-stopping-patience 20 --checkpoint-interval 1 --device cuda --mixed-precision --run
```

`--epochs 20` is the final total epoch.  A resumed run starts at the saved
epoch plus one and never adds 20 epochs to the checkpoint epoch:

```bash
python train_full_vocal_unet.py --dataset-root /kaggle/input/musdb18 --checkpoint-dir /kaggle/working/tunewave_checkpoints/full --resume-from /kaggle/working/tunewave_checkpoints/full/last_model.pth --batch-size 2 --num-workers 2 --optimizer adam --learning-rate 1e-3 --loss l1 --epochs 20 --early-stopping-patience 8 --checkpoint-interval 1 --device cuda --mixed-precision --run
```

Each completed epoch writes `last_model.pth` and a periodic
`checkpoint_epoch_NNNN.pth`; `best_model.pth` is replaced only when validation
loss improves.  `training.log`, `training_history.json`, and
`training_config.json` are kept in the same checkpoint directory.

## Exporting results

Create an artifact that contains checkpoints, logs, history, and configuration:

```bash
cd /kaggle/working
zip -r tunewave_checkpoints_full.zip tunewave_checkpoints/full
```

Download `tunewave_checkpoints_full.zip` from the Kaggle output panel.  Copy
`best_model.pth` into the local TuneWave checkpoint directory, for example:

```powershell
Copy-Item .\best_model.pth `
  'D:\UIT\4th yr 2nd sem\PROJECT\TuneWave\backend\models\best_model.pth'
```

