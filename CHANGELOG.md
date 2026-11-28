# Changelog

All notable changes are documented here. Version branches are preserved rather
than overwritten so each research-code state remains auditable.

## [2.0.0-experiment.d-oracle-soft-diagnostics] - 2026-11-28

Branch: `experiment-d-oracle-soft-diagnostics`
Baseline: `experiment-c-preupdate-proposal@ec77f6f`

### Added

- Added a unified `--ccar_variant` switch for A baseline, C preupdate, and D
  oracle-soft experiments without maintaining divergent training loops.
- Added `--seed`, explicitly seeded samplers/workers, and separate recording of
  the fixed split seed and training seed.
- Added weighted observed-class prototype centers. D uses weight 1.0 for clean
  samples and `--oracle_noisy_weight` (default 0.5) for known noisy samples.
- Added JSONL relabel/prototype/loss diagnostics, validation/test macro-F1,
  balanced accuracy, per-class recall and confusion matrices, plus ordered
  final-test prediction export.
- Added a controlled A/C/D three-seed server matrix script and regression tests
  for update order, weighted centers, metric calculations, relabel quality,
  seed repeatability, and resume identity.
- Vectorized per-class prototype aggregation to avoid batch-level GPU
  synchronization, and aligned JSONL history to the checkpoint epoch on resume.

### Boundaries

- Clean labels are detached diagnostics for A/C. D additionally uses them only
  to choose prototype weights; clean labels never become loss targets.
- D is an oracle mechanism test, not a deployable method.
- Full checkpoints from A, C, and D are mutually incompatible when their method
  contracts differ; shape-compatible warm starts remain supported.

### Validation

- `python -m compileall -q src` passed.
- `python -m unittest discover -s src -p "test_*.py" -v` passed all 29 tests,
  including CUDA paths.
- `python src/train_rafdb.py --help` and `git diff --check` passed.
- A one-epoch CUDA RAF-DB `oracle_soft` smoke run completed and produced a
  checkpoint, JSONL metrics, run manifest, and split manifest.
- Full repeated-seed RAF-DB training remains pending.

## [2.0.0-experiment.c-preupdate-proposal] - 2026-11-27

Branch: `experiment-c-preupdate-proposal`
Baseline: `version-2-paper-alignment@dd772ae`

### Changed

- Changed the CCAR order after relabeling is enabled from
  `update -> propose` to `propose -> update`. The proposal therefore uses the
  prototypes saved after previous batches and the current sample cannot move
  its own prototype before being judged.
- Preserved the baseline behavior before `relabel_epoch`: prototypes update and
  the original observed labels remain the batch targets.
- Added the CCAR update order to the checkpoint method contract.

### Added

- Added ordering tests for both sides of `relabel_epoch` and a numerical
  regression case that distinguishes pre-update from post-update prototype
  decisions.
- Added the implementation version and method contract to the training startup
  log so uploaded server runs identify the active experiment unambiguously.
- Added a server-use guide with the controlled Experiment C command.

### Compatibility

- Checkpoint tensor shapes are unchanged, but V2 baseline full checkpoints are
  intentionally rejected for resume because their method contract lacks the
  Experiment C update order. Shape-compatible weights can still be loaded with
  `--pretrained`.

### Validation

- Verified with Python 3.13.11 and PyTorch 2.9.1+cu128 from the project's
  `pytorch_env` environment.
- `python -m compileall -q src` passed.
- `python -m unittest discover -s src -p "test_*.py" -v` passed all 19 tests.
- `python src/train_rafdb.py --help`, the implementation-version assertion, and
  `git diff --check` passed.
- Full RAF-DB training is pending.

### Known limitations

- This experiment does not remove noisy-label samples from prototype updates.
- The changed operation order is a diagnostic intervention rather than a claim
  about the manuscript's original Equation (7)-(12) execution order.
- Full training and repeated-seed comparisons against the V2 baseline remain
  pending.

## [2.0.0-dev]

Target version: `2.0.0`
Branch: `version-2-paper-alignment`
Baseline: `main@0d5f3a2`

### Changed

- Changed AUEM importance from the normalized 512-D pooled trunk feature to the
  per-sample min-max-normalized `mu` embedding required by Equation (1).
- Changed the importance head from `Linear(512, 1)` to
  `Linear(out_dimension, 1)`; the default input width is 64.
- Made `out_dimension` consistently configure the embedding, importance head,
  uncertainty vector, classifier, CCAR prototype width, and checkpoint method
  contract.
- Restored the Equation (4) uncertainty vectors and the Equation (5) operation
  order: pairwise component normalization followed by a feature-dimension mean
  to obtain scalar mixture weights.
- Narrowed the supported public training path to the manuscript's ResNet-18;
  removed the non-executable IResNet CLI path and source file.
- Extracted the active AUEM, DSRUC, and batch-local CCAR target operations into
  testable helpers while preserving the documented V2 numerical behavior for
  Equations (2)-(14).
- Applied the Equation (4) uncertainty denominator symmetrically to both DSRUC
  weights, with `clamp_min` used only as zero-division protection.
- Replaced permissive pretrained loading with exact name/shape warm-start
  filtering, explicit skipped-key reporting, and exclusion of legacy
  classifier, importance, and uncertainty heads.
- Made V2 resume reject behavior-changing configuration drift, require
  scheduler state, and continue saved EMA history. Output controls and the
  early-stop target may change; the scheduler-bound total epoch plan may not.
- Added a minimum-size batch sampler that merges unsafe tails instead of
  dropping samples or passing singleton shards through `BatchNorm1d`.

### Added

- Added equation-level tests for AUEM grouping/loss, DSRUC normalization,
  vector reduction, mixing and two-label reduction, and CCAR
  update-before-propose behavior.
- Added model tests for the actual normalized-`mu` input to the importance head
  and gradient reach through the three-term objective.
- Added V2 checkpoint format and method-contract validation.
- Added resume-contract, EMA-restoration, warm-start-boundary, and safe-tail
  sampler regression tests.
- Added `VERSION`, this changelog, and the repository versioning policy.

### Compatibility

- V1 full training checkpoints are not resume-compatible with V2 because
  `alpha.0.weight` changed from `[1, 512]` to `[1, out_dimension]`, the
  uncertainty head now emits `out_dimension` values instead of one scalar, and
  the optimizer state refers to the old parameters. V2 rejects them before
  partial loading. Shape-compatible trunk and embedding tensors can still be
  warm-started through `--pretrained`; classifier, importance, and uncertainty
  heads are always reported as skipped.

### Validation

- Verified on 2026-11-26 with Python 3.13.11, PyTorch 2.9.1+cu128,
  torchvision 0.24.1+cu128, and CUDA available.
- `python -m compileall -q src` passed.
- `python -m unittest discover -s src -p "test_*.py" -v` passed all 17
  tests; the three-term gradient-path test ran on both CPU and CUDA.
- `python src/train_rafdb.py --help` and the `VERSION == 2.0.0-dev` assertion
  passed.
- `git diff --check`, the credential/local-path scan, and the repository scope
  scan passed. No dataset, checkpoint, run output, or `MaskDUL1` content is in
  this version.
- Full RAF-DB training was not run because the dataset and paper checkpoint are
  intentionally not part of this repository snapshot.

### Known limitations

- Full RAF-DB training, repeated-seed statistics, and manuscript table
  reproduction remain pending.
- The complete corrected path remains RAF-DB only.
- The manuscript leaves the extra weighted-CE prose, Equation (13) target
  notation, `alpha`, and several relabeling details underspecified. V2 preserves
  the documented implementation choices instead of presenting them as paper
  facts.

## V1 baseline

Baseline: `main@0d5f3a2`

### Added

- Published the first focused RAF-DB implementation of AUEM, DSRUC, and CCAR.
- Added CCAR tests, clean validation splitting, checkpoint selection and resume,
  and explicit reproducibility boundaries.

### Known limitations

- V1 computed AUEM importance from normalized pooled trunk features rather than
  the manuscript's `mu` embedding.
- Datasets, weights, run logs, and reproduced manuscript results were not
  included.
