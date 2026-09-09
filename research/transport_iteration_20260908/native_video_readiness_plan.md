# Native-video readiness: prospective subset plan

Status: design only, after the current CIFAR development decision. No decoder
was implemented, no archive was opened in this task, and no video/frame/test
payload was decoded or extracted. This plan reuses the earlier same-session
PSC header audit and locally saved inventory/manifest. It makes no video
quality, latent-model superiority, or data-readiness pass claim.

## Located archive and already verified header metadata

Exact PSC path:
`/ocean/projects/mth250006p/ywang26/datasets/ucf101-subset/UCF101_subset.tar.gz`.
The file is an uncompressed POSIX tar despite its suffix; recorded size is
171,386,880 bytes. Pinned archive SHA-256:
`e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb`.
Pinned `sayakpaul/ucf101-subset` revision:
`b9984b8d2a95e4a1879e1b071e9433858d0bc24a`.

The earlier read-only header audit confirmed:

| Split | Clips | Distinct source groups | Payload eligible for prospective development? |
|---|---:|---:|---|
| train | 300 | 195 | yes, after loader/configuration freeze |
| val | 30 | 25 | yes, development only |
| test | 75 | 30 | no; retain sealed |

Every pair of split group sets has zero overlap. In particular there are
**25 validation groups, not 30**: clips are not independent groups.
Known literal header entries include:

- `UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c03.avi`
- `UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g04_c07.avi`
- `UCF101_subset/val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi`
- `UCF101_subset/val/ApplyEyeMakeup/v_ApplyEyeMakeup_g14_c05.avi`

The ten classes are ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling,
BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk and
BenchPress. This is a ten-class subset, not the complete 101-class UCF101
benchmark. The complete 405-member header index was not saved locally in the
inspected artifacts; a future metadata-only indexing step must materialize
and hash it before a decoder can receive an explicit allowlist. Current
availability is based on the earlier same-session directory/header check,
not a new PSC filesystem query.

Local evidence: `work/real-data-pilot-inventory.md`,
`qalt/data/observed_manifest_v1.json`, and
`qalt/src/qalt/data_integrity.py`. Existing `parse_ucf_member` checks path
safety, split identity and class/filename consistency. `inspect_ucf_archive`
checks duplicate group/clip IDs and cross-split group overlap without decoding
any payload. These helpers are usable; no observed-video decoder was found in
the prior repository inventory.

## Strict future loader boundary

1. Read headers into a manifest containing original member path, byte offset,
   member byte length and tar block bounds, split, class, source group and clip
   identifier. Require regular nonsymlink members, no duplicate identities,
   no absolute paths or traversal, and the recorded counts/group disjointness.
   Preserve group identity `(class_name,gNN)` and clip identity `cNN`.
2. Produce two explicit payload allowlists: original `train` and original
   `val`. Keep test metadata solely for the disjointness/count check. Do not
   expose a test phase, optional test path or permissive split string to the
   development decoder. Reject any request not in the exact hashed allowlists.
3. Stream only an allowed member through a bounded, seekable file-like reader
   into the video decoder. The reader must restrict reads/seeks to that
   member's offset/length. Do not extract files to disk or call `extractall`.
   A supported tar member file-like API may provide this bounded reader, but
   its bounds/seek behavior must be tested on fabricated archives first.
   Supplying the whole tar stream to a decoder would violate this design.
4. Emit an opened-member ledger, parent-group identity, selected timestamps,
   decoded dimensions, preprocessing/version hashes and deterministic failures.
   No rejected clip is silently replaced by another class/group or retried with
   more favorable preprocessing. All methods consume the same cached endpoint
   tensors and exclusions. Test payloads remain unopened.

Before any payload read, freeze how archive integrity is checked. The pinned
whole-file hash is already available as provenance; a future full raw-byte
checksum is a container-integrity operation, not decoding, and must be
explicitly recorded if performed. This task did not reread archive bytes.

## Proposed common endpoint: eight frames at 64, then 128

These are two explicit possible development resolutions, not a scale chosen
after comparing quality. Start with eight RGB 64x64 frames if the CIFAR decision
supports proceeding. A subsequent 128x128 extension requires its own frozen
resource/configuration record. Do not run both and report only the favorable
one. The historical observed protocol proposed 16x64x64; this eight-frame
proposal is a separately recorded endpoint change, not that experiment.

A concrete temporal rule to freeze prospectively is:

- Decode presentation-order frames with finite strictly increasing timestamps.
  Use the complete source clip's first/last timestamps to define its center.
- Select eight target times `center+(k-3.5)/8`, k=0,...,7, giving 8 Hz spacing
  across a 0.875-second window. Choose the closest source frame to each target,
  with ties going to the earlier presentation timestamp. Record both requested
  and realized timestamps and indices.
- Require the complete target window to lie inside the source time range and
  require eight distinct selected frames. Reject short/irregular/missing-PTS
  clips instead of duplicating/padding frames or inferring motion frames.
  Freeze rejection handling before decoding. Header metadata alone cannot
  tell how many clips will meet these requirements.
- A first streaming decode pass can obtain frame timestamps, and a second can
  retain only the eight chosen frames. Charge both passes and share their
  output across all methods. No large all-frame tensor is needed.

No temporal blending/interpolation is proposed: nearest observed timestamps
preserve actual frames, while recording irregular timing. Spatially, convert
to RGB using one pinned decoder/color-conversion stack, resize the shorter
side to the chosen resolution with one fixed antialiased bilinear rule, then
center-crop to square. For source H,W and target R, set new dimensions to
`floor(H*R/min(H,W)+0.5)` and `floor(W*R/min(H,W)+0.5)` (positive half-up
rounding), then start the crop at `floor((newH-R)/2), floor((newW-R)/2)`.
Pin the actual antialias implementation/library version before execution. Use no class-conditioned crop or quality-based
frame rejection. Frame metadata and pixel format/color-range interpretation
must be recorded; unusual unsupported conversion should fail consistently.

Spatial resampling/cropping and temporal selection are **lossy preprocessing
from the original source video**. After that declared endpoint is fixed, the
model retains every endpoint coordinate. Do not call the original source video
fully preserved or transfer a likelihood statement through these noninvertible
preprocessing operations. Upsampling, where needed, creates interpolated pixels
and information redundancy, not new native detail. All models get precisely
the same endpoint and preprocessing, so those operations create no method's
private information advantage.

To keep preprocessing and dequantization unambiguous, freeze the cached endpoint
as RGB uint8 after the deterministic resize/crop, with an explicit rounding
rule if the chosen stack outputs floats. Then use one shared record-and-frame
keyed finite-precision uniform dequantization `(pixel+U)/256` in float64,
using interior midpoint noise as in the image loader. Compute a shared
float64 logit and its Jacobian before any float32 cast. Never cast near-boundary
unit-cube values to float32 before logit or silently clip them. The precise
salts and key encoding must be added to the eventual configuration before
payload access; they are not implemented by this plan.

## Dimension, memory and architectural implications

| Declared RGB endpoint | Full prior/representation coordinates D | Compared with CIFAR32 D=3,072 |300 training clips uint8|300 training clips float64|
|---|---:|---:|---:|---:|
|8x64x64|98,304|32 times|28.125MiB|225MiB|
|8x128x128|393,216|128 times|112.5MiB|900MiB|

The 30 validation clips add one tenth of those array sizes. These are data-array
sizes only, excluding model activations, gradients, caches and optimizer state.
Doubling spatial resolution quadruples D and pixel-level activation work.
A full Gaussian source has exactly D coordinates, counting every stochastic
decoder noise coordinate. Any latent compression or source-video downsampling
must be described separately, not disguised by that count.

Packing eight RGB frames as 24 channels is a lossless reshape of the chosen
endpoint and could test input plumbing with the existing 2D implementation.
It does not create temporal weight sharing, time-translation equivariance,
3D context or a competitive video architecture. A realistic video candidate
and baseline need explicitly shared temporal modeling and globally dependent
stochastic decoding. The proof for frozen low-complexity contexts does not
establish such representation sufficiency for these clips.

## Group-level inference and limited scientific scope

Use original source groups as independent units. Average clip scores within
group before group-level summaries/uncertainty; do not count frames, pixels,
multiple clips from one group, or augmentation views as independent examples.
Training can draw groups uniformly then draw a clip within group, identically
for all methods, so groups with more source clips do not silently dominate.
Any independent discovery/head-fitting partition must also be group-disjoint,
further reducing the 195 available training units. Preserve the original
train/val/test split; internal splits do not unlock official test clips.

With only 25 validation source groups and a narrow ten-class subset, FVD or
frame-FID plug-in scores cannot justify a strong unconditional-video quality
claim. The immediate use is a bounded real-video input/conditional-density
or unconditional-sampling feasibility check with explicit small-data scope.
Do not compare its scores with papers using full UCF101, different clip lengths,
resolutions, pretrained information or evaluation sample sizes. A competitive
latent-video claim still requires adequate data, trained strong matched
baselines and replicated observations; this plan provides none of those.

The Pro5 development protocol says a CIFAR pass authorizes UCF development
**design only**, not sealed tests. Follow the root's decision after that study;
no implementation or allocation is authorized by this document.

## Decoder dependency status

Read-only local checks found no `ffmpeg` executable on PATH and no `av`,
`imageio_ffmpeg` or `cv2` module in `work/venv312`. This is local audit-runtime
metadata, not proof that PSC lacks them. The known PSC Python remains
`/ocean/projects/mth250006p/ywang26/pytorch/bin/python`, but its video-decoder
stack was not queried in this task. A future authorized environment check
must establish PyAV/linked FFmpeg availability/version without opening video
payloads, then test bounded file-like decoding using a fabricated clip. No
package installation or fallback transcoding was attempted here.
