# Native-pixel observed flow development pilot: data protocol draft

Status: local prospective draft for review. The loader and fabricated-file
tests implement data preparation only. There is no trained runner, real-data
read, fitted model, submitted job or experimental result under this draft.

## Endpoint and existing development history

The endpoint is unconditional native CIFAR-10 RGB 32x32: every one of its 3,072
coordinates enters each model. The candidate uses no VAE. Comparators may
learn their own encoders and decoders from the same fitting arrays, with all
training charged. No resizing, crop, hidden teacher variable, externally
pretrained encoder, or class-conditioning input is supplied. An exact
logit/orthonormal-Haar chart may be provided equally to all methods with its
Jacobian retained. Generation must use learned coarse coordinates and all
declared stochastic decoder inputs, never coarse values from a real image.

Preserve the current adaptive partition, not the superseded 45k/5k split.
`adaptive_cifar_repair_split` first uses the class-stratified 20260823 split,
then repartitions the old fit set with 20260824, leaving 40,000 fitting,
5,000 recycled repair and 5,000 excluded discovery records. The repair images
have already informed previous experiments and earlier fitting. They provide
adaptive development evidence only; new architecture or seed choices cannot
make them untouched confirmation. Original discovery pixels remain excluded
from every returned array and model/evaluation statistic. The official test
batch remains unopened. This module provides no test-loading API.

Canonical record i is the zero-based row in ordered concatenation of
`data_batch_1,...,data_batch_5`: file 1+i//10000, row i%10000. The new loader
`qalt/src/qalt/observed_flow_data.py` reuses the exclusive five-file loader
and existing adaptive split function. Labels determine memberships and class
quotas only, and are not returned to models.

## Frozen input selection and verification

Before deserialization, verify each of the five training file SHA-256 hashes
against `qalt/data/observed_manifest_v1.json`; reject symlink batch paths.
Before selecting pixels, require the complete split hash
`814c2280cea33403d9370b02ea12c83df3a5f89bbbf593567fed6e960ef321e7`,
full fitting ID hash
`4f002c7cfe2e3d1ca54a7c8847982941d655060e6989a66f87094e4d632be457`,
and full repair ID hash
`ced1fb315bba0eeb81c03b6f2fb296afd3208cb02a7cff6dc513327708ea8939`.
The split hash uses canonical sorted compact JSON. ID hashes use raw
contiguous little-endian int64 bytes. Source file loading necessarily reads
all allowed training-file records before subsetting; no excluded discovery
pixel is returned, hashed as a selected pixel, or used in a fitting statistic.

Select 400/class from fitting and 100/class from repair (4,000/1,000 totals).
Rank eligible IDs lexicographically by SHA-256 digest of the ASCII string
`observed-flow-cifar-selection-v1|ROLE|IIIII`, where ROLE is `fit` or `repair`
and IIIII is the five-digit zero-padded original ID. Break any hash tie by
record ID. Take the fixed quota independently within each class, then sort
selected IDs globally. This selection uses no pixels or scores. Check both
selected sets are disjoint and contain no original discovery IDs.

`allow_noncanonical_fixture=True` bypasses canonical file/split hash equality
only for fabricated unit-test inputs. It preserves 50k shapes, the full split
algorithm and quotas; its ledger says `canonical_dataset_verified=False` and
names the data as a fabricated fixture. Any future empirical runner must
reject this metadata. Default real-data loading cannot bypass the checks.

## Shared dequantization and model inputs

For each selected role and original ID, take the SHA-256 of ASCII
`observed-flow-cifar-dequantization-v1|ROLE|IIIII`, interpret the complete
digest as a little-endian integer, and seed NumPy PCG64 explicitly. Draw
3x32x32 integers K uniformly from 0,...,2^32-1 in row-major order, then compute
`U=(K+0.5)/2^32` and `X=(pixel+U)/256` in float64. Midpoints ensure X is
strictly inside (0,1), even for pixels 0 and 255, without clipping. This is a
finite-precision uniform dequantization convention; the same realized arrays
are shared by every method and every training seed. It is not tuned after
seeing a model or evaluation result.

Return read-only float64 NCHW arrays, selected original IDs and an integrity
ledger; return no class labels. Compute only selected raw-pixel/dequantized
hashes, full membership hashes, source file hashes, explicit opened-file
allowlist, overlap counts, salts, schema/NumPy version and dataset status.
Full source arrays are released after per-record construction; no redundant
float64 copy of all 50,000 images is built. IDs and ledger permit the same
record-keyed dequantization to be checked independently.

Do not cast these unit-cube arrays directly to float32: a pixel close to one
can round to the boundary and invalidate a strict unit-interval transport.
For the planned GPU models, compute the shared logit coordinates
`Y=log(X)-log1p(-X)` in float64 first, then cast Y to float32 identically for
all arms. Preserve the per-image float64 log-Jacobian
`sum[-log(X)-log1p(-X)]` explicitly for observation-space density evaluation.
Use the spline model's real-coordinate mode (`unit_interval=False`) on these
already transformed inputs, avoiding a second logit. Generation returns
through the same sigmoid chart. Check and report numerical reconstruction
error; no silent pixel clipping or claimed exact float32 arithmetic is allowed.

For a future image likelihood report, distinguish continuous dequantized NLL
from exact discrete pixel likelihood. The usual 8-bit offset interpretation
requires the declared dequantization convention; do not silently label a
single finite-precision draw as exact integrated discrete likelihood.

## Next review required before execution

Freeze learned spline, full FM, hierarchical stochastic FM, and learned-analysis
latent FM with a globally dependent stochastic residual decoder,
shared prior dimension 3,072, minibatch schedule, training/time caps, evaluator
preprocessing, numerical limits and output preservation before real-data
loading or training. Include the exact stochastic-latent-copy equality
control, charging its shared fitted model's cost. No quality, efficiency,
representation or SOTA claim follows from this data loader. This draft does
not modify confirmation permissions or invent independent coverage for the
recycled repair set.

The tests use five tiny fabricated pickle files that expand to canonical
array shapes in memory, with constant synthetic pixels and fabricated labels.
They verify exact quotas, deterministic ranking/dequantization, native-pixel
recovery, five-file access, poisoned-test exclusion, strict-default rejection,
split-check ordering, immutable outputs and explicit fixture metadata. They
do not train any model or open real data.
