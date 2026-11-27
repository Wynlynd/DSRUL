# DSRUL

Reference implementation of **DSRUL: A Novel Adaptive Dual-Stream Relative
Uncertainty Learning Method for Facial Expression Recognition**.

> **Version:** `2.0.0-experiment.c-preupdate-proposal` (Experiment C)
>
> **Branch:** `experiment-c-preupdate-proposal`
>
> **Experiment baseline:** [`version-2-paper-alignment@dd772ae`](https://github.com/Wynlynd/DSRUL/tree/dd772aef62e3904fb77e3a05b6648ad0290b85a2)
>
> **Changes:** [CHANGELOG.md](CHANGELOG.md)

> **Status:** This diagnostic branch changes only the CCAR operation order once
> relabeling is enabled: the current batch is judged with the prototypes from
> previous batches, then the prototypes are updated. It is intended to measure
> self-inclusion bias and is not presented as the manuscript's original
> update-before-proposal implementation. Full RAF-DB training remains pending.

## Experiment C

The V2 baseline updates a prototype with the current batch before using that
prototype to judge the same samples. Experiment C removes that self-inclusion:

```text
before --relabel_epoch: update prototypes; keep original batch targets
from   --relabel_epoch: propose targets with pre-update prototypes; then update
```

The prototype update still uses every sample's original observed training
label. Relabeling remains batch-local and affects only `L_CCAR`; dataset labels
are not overwritten. All AUEM, DSRUC, optimizer, scheduler, augmentation, and
evaluation behavior is unchanged from the experiment baseline.

At startup, the training log prints the implementation version and method
contract. A valid Experiment C run must report
`2.0.0-experiment.c-preupdate-proposal` and
`ccar_update_order: propose_then_update_when_relabeling`.

If relabeling is enabled before enough prototypes exist, CCAR's existing
validity mask keeps those samples unchanged and the batch then initializes or
updates the prototypes normally. "Pre-update" therefore means one batch behind,
not one epoch behind.

## Overview

DSRUL targets facial expression recognition with noisy or ambiguous labels.
The current RAF-DB implementation combines three components:

- **AUEM (Attentional Uncertainty Estimation Module)** predicts a sample
  importance score from the per-sample min-max-normalized `mu` embedding, as
  specified by Equation (1). The score defines the two batch regions and
  contributes an importance-separation regularizer.
- **DSRUC (Dual-Stream Relative Uncertainty Comparison)** pairs differently
  labeled samples across those regions, mixes their embeddings using the
  model's independent uncertainty vectors, and trains the mixed representation
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
| AUEM/objective/CCAR/checkpoint tests | Included |
| FERPlus/AffectNet corrected paths | Not yet integrated |
| Standalone evaluation CLI | Not yet included |
| Full paper-result reproduction | Pending |

The release intentionally excludes incomplete historical experiments,
`MaskDUL1`, cached bytecode, machine-specific paths, and plotting scripts whose
values are not traceable to versioned run artifacts.

## Repository Layout

```text
.
|-- VERSION
|-- CHANGELOG.md
|-- README.md
|-- requirements.txt
|-- docs/
|   `-- implementation-notes.md
`-- src/
    |-- train_rafdb.py       # Primary RAF-DB training entrypoint
    |-- resnet.py            # ResNet-18, uncertainty, and importance branches
    |-- ccar.py              # Prototype, threshold, relabeling, and CCAR loss
    |-- objectives.py        # Equation-aligned AUEM/DSRUC target helpers
    |-- image_utils.py       # Image augmentation helpers
    `-- test_*.py            # Formula, model, checkpoint, and CCAR tests
```

See [implementation notes](docs/implementation-notes.md) for the exact active
loss definitions and manuscript-to-code decisions.

## Requirements

The training entrypoint currently uses CUDA and `DataParallel`, so a
CUDA-capable PyTorch environment and Python 3.9 or newer are required for
training. Install a PyTorch
and torchvision build compatible with your CUDA driver from the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then
install the remaining packages:

```bash
pip install -r requirements.txt
```

Exact Python, PyTorch, CUDA, and GPU versions used for paper-grade reproduction
will be added with the reproducibility release. The test suite can run on CPU
and also exercises CUDA paths when CUDA is available.

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

Pretrained weights are not included. Pass a compatible model checkpoint:

```text
--pretrained /path/to/ijba_res18_naive.pth.tar
```

Omit `--pretrained` to use random initialization. Random initialization is a
supported code path, but it is not established as the manuscript reproduction
setting. Warm starts load exact name/shape matches for the trunk and embedding
path, while deliberately skipping the internal classifier, AUEM importance
head, and uncertainty head. Record the source and checksum of any pretrained
checkpoint used in reported experiments.

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

The training batch sampler retains all samples and merges a tail that is too
small for `BatchNorm1d` into the preceding batch. The configured batch size
must provide at least two samples per visible CUDA device.

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

V2 checkpoints restore the backbone, external classifier, CCAR state,
optimizer, scheduler, epoch, and EMA history when enabled. They carry a format
version and method contract (`normalized_mu`, embedding dimension,
`pairwise_normalized_vector_mean`, ResNet-18, and the CCAR update order). V1
checkpoints cannot be resumed because the importance and uncertainty head
shapes changed. V2 baseline full checkpoints are also rejected because they do
not declare the Experiment C update order. Use `--pretrained` for the restricted
warm start described above. Resume rejects
changes to behavior-defining data, method, optimizer, scheduler, augmentation,
and EMA settings. Resume-location/output controls, the historical warm-start
source field, early-stop target, and the final test-evaluation switch are
excluded from that equality check. The total `--epochs` plan remains fixed
because it defines the cosine scheduler's `T_max`; V2 resume is therefore for
continuing an interrupted run within its original epoch plan, not silently
extending that plan. RNG and sampler state are not currently saved, so a
resumed V2 run is still not bitwise identical to an uninterrupted run.

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

Run all paper-alignment and component tests from the repository root:

```bash
python -m unittest discover -s src -p "test_*.py" -v
```

They cover the `mu -> normalization -> importance` path, the documented V2
interpretations of Equations (2)-(6), vector-uncertainty normalization and
reduction, three-term gradient reach, prototype initialization and EMA, dynamic
thresholds, pre-update-prototype relabel ordering, scaled-logit loss behavior,
warm-start filtering, and checkpoint compatibility. These tests validate
component behavior; they do not replace full-dataset training or result
reproduction.

## Reproducibility Boundary

- Data, noisy-label files, pretrained weights, trained checkpoints, and paper
  result logs are not included.
- The current complete method path supports RAF-DB only.
- The active objective is `L_CCAR + L_AUEM + L_DSRUC`, with coefficient 1 for
  each term.
- The current CCAR classification term scales each sample's logits by its
  importance before cross-entropy. It is not ordinary sample-weighted CE.
- Experiment C changes only the relabel/update order. It does not filter noisy
  samples from prototype updates and must not be described as a clean-prototype
  method.
- Full RAF-DB training, repeated-seed statistics, and the manuscript tables
  remain to be reproduced and released.

No manuscript accuracy should be attributed to this snapshot without the
matching environment, dataset/noise protocol, checkpoint, seed, and run logs.

## Version History

| Version | Branch/baseline | Main change |
| --- | --- | --- |
| `2.0.0-experiment.c-preupdate-proposal` | `experiment-c-preupdate-proposal`, based on `dd772ae` | Uses pre-update prototypes to propose batch-local CCAR targets, then updates prototypes with the current batch. |
| `2.0.0-dev` | `version-2-paper-alignment` | Computes AUEM importance from normalized `mu`, restores vector uncertainty for DSRUC, adds regression coverage for the documented V2 interpretations, and adds versioned checkpoints. No paper accuracy is claimed. |
| V1 baseline | `main@0d5f3a2` | First public RAF-DB path with AUEM, DSRUC, CCAR, validation checkpointing, and explicit reproducibility limits. |

For a new major research line, create
`version-<major>-<short-description>` (for example,
`version-3-new-objective`). For later minor or patch releases, use
`version-<semver>-<short-description>` (for example,
`version-2.1.0-training-protocol`). Create every branch from the exact preceding
stable commit or release tag, record that baseline in the changelog, and never
force-push an older version branch. Every version branch must update `VERSION`,
this version table, and `CHANGELOG.md` with the exact changes, compatibility
boundary, validation commands, and remaining limitations. Branch names use
lowercase ASCII and hyphens. Formal releases also receive an annotated SemVer
tag; the current V1 state is a baseline commit, not a tagged release.

## Citation

Publication metadata is not final. A BibTeX entry and `CITATION.cff` will be
added when the author list and venue metadata are ready.

## License

No software license has been added yet. Until a license is selected, normal
copyright restrictions apply. Dataset and pretrained-model licenses are
separate and must be followed independently.
