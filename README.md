# DSRUL

Reference implementation of **DSRUL: A Novel Adaptive Dual-Stream Relative
Uncertainty Learning Method for Facial Expression Recognition**.

> **Status:** this public snapshot contains the corrected RAF-DB training path
> for AUEM, DSRUC, and CCAR. Focused tests and development-time mini-batch
> checks have been completed, but full-dataset training and the numerical
> results reported in the manuscript have not yet been reproduced from this
> repository snapshot.

## Overview

DSRUL targets facial expression recognition with noisy or ambiguous labels.
The current RAF-DB implementation combines three components:

- **AUEM (Attentional Uncertainty Estimation Module)** predicts a sample
  importance score from normalized pooled features. The score defines the two
  batch regions and contributes an importance-separation regularizer.
- **DSRUC (Dual-Stream Relative Uncertainty Comparison)** pairs differently
  labeled samples across those regions, mixes their embeddings using the
  model's independent uncertainty branch, and trains the mixed representation
  against both source labels.
- **CCAR (Cluster-Contrastive Adaptive Relabeling)** maintains class
  prototypes, measures intra/inter-class geometric inconsistency, and combines
  that signal with sample importance to produce a dynamic relabeling threshold.

CCAR corrections are batch-local targets for the CCAR classification loss.
The original dataset labels are not permanently overwritten.

## Supported Scope

| Area | Status |
| --- | --- |
| RAF-DB, 7-class training | Supported by the primary entrypoint |
| AUEM + DSRUC + CCAR | Connected in the primary training objective |
| CCAR unit tests | Included |
| FERPlus/AffectNet corrected paths | Not yet integrated |
| Standalone evaluation CLI | Not yet included |
| Full paper-result reproduction | Pending |

The release intentionally excludes incomplete historical experiments,
`MaskDUL1`, cached bytecode, machine-specific paths, and plotting scripts whose
values are not traceable to versioned run artifacts.

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- implementation-notes.md
`-- src/
    |-- train_rafdb.py       # Primary RAF-DB training entrypoint
    |-- resnet.py            # ResNet-18, uncertainty, and importance branches
    |-- iresnet.py           # Optional IResNet-18 backbone
    |-- ccar.py              # Prototype, threshold, relabeling, and CCAR loss
    |-- image_utils.py       # Image augmentation helpers
    `-- test_ccar.py         # Focused CCAR tests
```

See [implementation notes](docs/implementation-notes.md) for the exact active
loss definitions and manuscript-to-code decisions.

## Requirements

The training entrypoint currently uses CUDA and `DataParallel`, so a
CUDA-capable PyTorch environment is required for training. Install a PyTorch
and torchvision build compatible with your CUDA driver from the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then
install the remaining packages:

```bash
pip install -r requirements.txt
```

Exact Python, PyTorch, CUDA, and GPU versions used for paper-grade reproduction
will be added with the reproducibility release. The CCAR tests can run on CPU.

## RAF-DB Preparation

RAF-DB is not distributed with this repository. Obtain it from the dataset
owner and follow its license. The loader expects:

```text
raf-basic/
|-- EmoLabel/
|   |-- list_patition_label.txt  # Clean official train/test labels
|   `-- 0.1noise_train.txt       # Noisy labels for official training images
`-- Image/
    `-- aligned/
        |-- train_00001_aligned.jpg
        `-- test_0001_aligned.jpg
```

Each label file must contain whitespace-separated image names and labels in
`1..7`:

```text
train_00001.jpg 5
train_00002.jpg 2
```

The noisy and clean files must contain the same `train*` images. The clean
file must also contain the official `test*` images. The training script reports
the observed noisy-label ratio before training.

The official training partition is split into a fixed, class-stratified train
and clean-validation split. Defaults are `--val_ratio 0.1` and
`--split_seed 0`; the exact image lists are saved in `split_manifest.json`.

## Pretrained Weights

Pretrained weights are not included. Pass a compatible backbone checkpoint:

```text
--pretrained /path/to/ijba_res18_naive.pth.tar
```

Omit `--pretrained` to use random initialization. Random initialization is a
supported code path, but it is not established as the manuscript reproduction
setting. Record the source and checksum of any pretrained checkpoint used in
reported experiments.

## Training

Run from the repository root:

```bash
python src/train_rafdb.py \
  --raf_path /path/to/raf-basic \
  --train_label_file 0.1noise_train.txt \
  --clean_label_file list_patition_label.txt \
  --pretrained /path/to/ijba_res18_naive.pth.tar \
  --save_dir runs/rafdb-noise-01 \
  --batch_size 256 \
  --epochs 100 \
  --lr 0.0002 \
  --weight_decay 0.0001
```

Important method defaults:

| Argument | Default | Role |
| --- | ---: | --- |
| `--beta` | `0.2` | CCAR base threshold |
| `--gamma` | `0.6` | Importance-dependent threshold adjustment |
| `--alpha` | `0.4` | Geometry adjustment; an implementation choice not numerically specified in the PDF |
| `--prototype_momentum` | `0.9` | Class-prototype EMA coefficient |
| `--ccar_temperature` | `0.1` | Distance-score temperature |
| `--relabel_epoch` | `8` | First epoch with batch-local relabeling |
| `--margin_1` | `0.5` | AUEM regularization offset |

Use `python src/train_rafdb.py --help` for all data, optimizer, scheduler,
sampling, augmentation, EMA, and checkpoint options.

### Early stopping

The default `--target_val_accuracy 0.885` stops training once that validation
accuracy is reached. For a controlled full-epoch run, set the target above the
attainable range, for example `--target_val_accuracy 1.01`, and record that
choice with the experiment.

### Resume

```bash
python src/train_rafdb.py \
  --raf_path /path/to/raf-basic \
  --checkpoint runs/rafdb-noise-01/best.pth \
  --save_dir runs/rafdb-noise-01
```

The checkpoint restores the backbone, external classifier, CCAR state,
optimizer, scheduler, and epoch. RNG and sampler state are not currently saved,
so a resumed run is not bitwise identical to an uninterrupted run.

## Evaluation and Outputs

The best validation checkpoint is saved under `--save_dir`:

```text
<save_dir>/
|-- best.pth
`-- split_manifest.json
```

Unless `--skip_test_eval` is supplied, training reloads `best.pth` and performs
one final evaluation on the official RAF-DB test split. Use
`--skip_test_eval` during model selection to avoid repeated test-set access.

Current evaluation reports cross-entropy loss and accuracy. Macro-F1, balanced
accuracy, calibration, noise-detection quality, and relabeling precision/recall
are not yet implemented.

## Tests

Run the focused CCAR tests from the repository root:

```bash
python src/test_ccar.py
```

They cover prototype initialization and EMA updates, missing classes, dynamic
thresholds, relabel gating, scaled-logit loss behavior, and finite gradients.
These tests validate component behavior; they do not replace full-dataset
training or result reproduction.

## Reproducibility Boundary

- Data, noisy-label files, pretrained weights, trained checkpoints, and paper
  result logs are not included.
- The current complete method path supports RAF-DB only.
- The active objective is `L_CCAR + L_AUEM + L_DSRUC`, with coefficient 1 for
  each term.
- The current CCAR classification term scales each sample's logits by its
  importance before cross-entropy. It is not ordinary sample-weighted CE.
- Full RAF-DB training, repeated-seed statistics, and the manuscript tables
  remain to be reproduced and released.

No manuscript accuracy should be attributed to this snapshot without the
matching environment, dataset/noise protocol, checkpoint, seed, and run logs.

## Citation

Publication metadata is not final. A BibTeX entry and `CITATION.cff` will be
added when the author list and venue metadata are ready.

## License

No software license has been added yet. Until a license is selected, normal
copyright restrictions apply. Dataset and pretrained-model licenses are
separate and must be followed independently.
