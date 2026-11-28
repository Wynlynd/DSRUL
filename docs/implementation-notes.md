# DSRUL Experiment D implementation notes

This document records the active behavior of the public RAF-DB path. It
separates implementation choices from claims that are explicit in the
manuscript.

Experiment D is based on Experiment C commit `ec77f6f`. The underlying V2
implementation was aligned against the
14-page manuscript PDF with SHA-256
`DA235CB8344C22411137C4C84CFDBBB7238827122A72E6FB648C63D952D30184`.

This branch supports three explicit variants: A baseline (`update -> propose`),
C preupdate (`propose -> update`), and D oracle-soft (C order with weights 1.0
for clean-label matches and 0.5 by default for known noisy-label samples). D is
a diagnostic upper-bound test, not a manuscript claim or deployable method.

The training entrypoint prints the implementation version and complete method
contract before loading data. Server logs therefore retain the experiment
identity even when the uploaded directory name is changed.

## Active dataflow

The primary path is:

```text
image
  -> ResNet-18 trunk
  -> 64-D embedding (mu) -------------> external 7-class classifier
  -> 64-D uncertainty vector ----------> DSRUC mixture weights
  -> scalar sample importance --------> AUEM split/regularizer and CCAR
```

The backbone and external classifier are both included in the optimizer and in
the saved checkpoint.

## AUEM

The importance head is a `Linear(out_dimension, 1) + Sigmoid` module. Following
Equation (1), it receives the per-sample min-max-normalized `mu` embedding. The
default `out_dimension` is 64, and the same value configures the embedding,
importance head, external classifier, CCAR prototypes, and checkpoint contract.

Within each mini-batch, samples are sorted by importance. The lower half is
treated as the high-uncertainty region and the upper half as the low-uncertainty
region. For an odd batch size, the high-uncertainty region has `floor(B/2)`
samples and the low-uncertainty region has `ceil(B/2)`; pairing then uses the
smaller count. Because both learned branches contain `BatchNorm1d`, the training
batch sampler guarantees at least two samples per visible CUDA device. It
merges an unsafe final tail into the preceding batch instead of dropping data.

The active AUEM regularizer is:

```text
L_AUEM = log((mean(low-importance) + delta) / mean(high-importance))
```

with `delta = 0.5` by default. The denominator has a small numerical clamp.

The manuscript also mentions a weighted cross-entropy in prose but gives no
formula or fourth term in Equation (14). V2 therefore does not invent an extra
classification loss; the importance-dependent classification behavior remains
the scaled-logit CCAR loss in Equation (13).

## DSRUC

The two importance-defined groups are independently shuffled and truncated to
the same pair count. Same-label pairs are removed. If no different-label pair
remains, the batch has no DSRUC loss.

For a retained high/low pair, the independent uncertainty vectors are first
normalized component-wise as required by Equation (4). The normalized vectors
are then averaged across their feature dimension to produce the scalar mixture
weights in Equation (5):

```text
denominator[k] = max(u_high[k] + u_low[k], epsilon)
normalized_high[k] = u_high[k] / denominator[k]
normalized_low[k]  = u_low[k] / denominator[k]
w_high = mean(normalized_high)
w_low  = mean(normalized_low)
mixed  = w_high * mu_high + w_low * mu_low
```

The same mixed feature is classified against both source labels, and the two
cross-entropies are added with equal coefficient.

## CCAR

Each class prototype is initialized from the first observed batch center and is
then updated using an exponential moving average. A and C use an ordinary mean.
D uses `sum(w_i * f_i) / sum(w_i)` within each observed-label class. A noisy
sample is downweighted but never removed or reassigned to a clean/predicted
class. Classes absent from the batch, or with zero total weight, retain their
previous prototype.

For original label `y`, CCAR computes Euclidean distance to the `y` prototype
and to the closest initialized non-`y` prototype. The geometric score is:

```text
phi = sigmoid((d_intra - d_inter) / tau)
```

The threshold combines geometry and importance relative to the current batch
mean:

```text
threshold = beta / (1 + gamma * sign(i - mean(i))) * (1 - alpha * phi)
```

Default values are `beta=0.2`, `gamma=0.6`, `alpha=0.4`, prototype momentum
`0.9`, and `tau=0.1`. The PDF does not state a numeric `alpha`; `0.4` is an
explicit implementation choice.

The classifier softmax supplies `Pmax` and the probability of the original
label. When `Pmax > Pgt + threshold`, the predicted class becomes the target
for the current batch's CCAR loss. The dataset label is not permanently
modified, and prototype updates continue to use original labels.

Before `relabel_epoch`, every variant updates prototypes and retains observed
targets. Starting at `relabel_epoch`, A updates then proposes; C and D use:

```text
1. Compute the proposal with the prototypes saved after previous batches.
2. Update the prototypes with the current embeddings and original labels.
3. Use the proposal as the current batch's L_CCAR target where its mask is true.
```

Consequently, a sample does not move its own class prototype before its
geometric inconsistency is measured. If the required prototypes are not yet
initialized, the existing validity mask prevents relabeling and the update
still occurs. The prototypes are delayed by one batch, not by one epoch.

The active CCAR classification term is:

```text
L_CCAR = cross_entropy(importance * logits, batch_local_target)
```

Scaling logits changes each sample's softmax temperature/confidence. It is not
equivalent to multiplying an already-computed per-sample CE by importance.

## Total objective

The active training objective is:

```text
L_total = L_CCAR + L_AUEM + L_DSRUC
```

All three terms use coefficient 1. When no valid DSRUC pair exists,
`L_DSRUC = 0` for that batch.

## Validation and test use

The clean official RAF-DB training labels are used to create a fixed,
class-stratified validation split. Noisy labels are used for the remaining
training images. The official test split is evaluated only after the best
validation checkpoint is reloaded, unless `--skip_test_eval` is specified.

The split image names and split parameters are saved in `split_manifest.json`.
The independent training seed is saved in both manifests and the checkpoint.
The method contract records the selected variant, order, weighting policy, and
D's oracle weight, so incompatible full checkpoints are rejected on resume.
`--pretrained` remains available for the restricted shape-compatible warm start.
Resume requires scheduler state, restores EMA history when enabled, and rejects
changes to behavior-defining CLI values. RNG/sampler state is not saved, so a
resumed run is not bitwise equivalent to an uninterrupted run. Epoch metrics
are written to `metrics.jsonl`. On resume, records after the checkpoint epoch
are removed and duplicate epochs retain only the latest record before new
records are appended. Final class metrics and ordered predictions are written
after the selected checkpoint is evaluated on the official test split.

Clean training labels serve three isolated roles: validation split creation,
detached relabel diagnostics in all variants, and D's explicit oracle prototype
weights. They are never CCAR/DSRUC loss targets. The diagnostics report correct
noisy-label corrections, false changes to clean samples, noisy-to-wrong changes,
per-class counts, component losses, prototype drift/norm, update coverage,
weight sums, and per-epoch weight ESS. The ESS is an update-stream proxy because
batch centers are normalized before the prototype EMA; it is not the exact
effective sample count of the final EMA prototype.
