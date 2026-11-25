# Changelog

All notable changes are documented here. Version branches are preserved rather
than overwritten so each research-code state remains auditable.

## [Unreleased]

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
