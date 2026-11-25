# DSRUL implementation notes

This document records the active behavior of the public RAF-DB path. It
separates implementation choices from claims that are explicit in the
manuscript.

## Active dataflow

The primary path is:

```text
image
  -> ResNet-18 trunk
  -> 64-D embedding (mu) -------------> external 7-class classifier
  -> scalar uncertainty --------------> DSRUC mixture weights
  -> scalar sample importance --------> AUEM split/regularizer and CCAR
```

The backbone and external classifier are both included in the optimizer and in
the saved checkpoint.

## AUEM

The importance head is a `Linear(512, 1) + Sigmoid` module. It receives the
per-sample min-max normalized global-pooled backbone feature, not the final
64-D `mu` embedding.

Within each mini-batch, samples are sorted by importance. The lower half is
treated as the high-uncertainty region and the upper half as the low-uncertainty
region. For odd batch sizes, one region contains one additional sample.

The active AUEM regularizer is:

```text
L_AUEM = log((mean(low-importance) + delta) / mean(high-importance))
```

with `delta = 0.5` by default. The denominator has a small numerical clamp.

An earlier ordinary sample-weighted cross-entropy implementation remains only
as disabled source comments. It is not part of the active objective.

## DSRUC

The two importance-defined groups are independently shuffled and truncated to
the same pair count. Same-label pairs are removed. If no different-label pair
remains, the batch has no DSRUC loss.

For a retained high/low pair, the independent uncertainty outputs define
complementary mixture weights:

```text
w_high = u_high / (u_high + u_low + epsilon)
w_low  = 1 - w_high
mixed  = w_high * mu_high + w_low * mu_low
```

The same mixed feature is classified against both source labels, and the two
cross-entropies are added with equal coefficient.

## CCAR

Each class prototype is initialized from the first observed batch mean and is
then updated using an exponential moving average. Classes absent from the
current batch retain their previous prototype. Relabeling is disabled until the
given class and at least one other class have initialized prototypes.

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
The repository does not currently save RNG/sampler state, a full resolved CLI
configuration, metric history, or prediction files.
