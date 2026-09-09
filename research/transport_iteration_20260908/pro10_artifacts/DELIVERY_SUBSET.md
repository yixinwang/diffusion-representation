# Partial Git delivery: original text files only

This directory is **not the complete original 17-file inspection package**.
It publishes the sixteen original text files byte-for-byte unchanged and adds
this separate delivery note. The original notes, code, numerical results, raw
text timing records, and `SHA256SUMS` have not been edited.

## Binary file deliberately absent from Git

`local_focused_sources.npz` is **NOT included in this Git delivery**.
It contains the actual saved local dictionary/source banks, not regenerated
replacements. Its exact original identity is:

- Size: **4,629,183 bytes**.
- SHA-256: `89994a3aadb4fcd2eb2404feffbef2849fce3fa62cda2ee43b58d87f422ab718`.
- Location inside the unchanged downloadable archive:
  `pro10_review/local_focused_sources.npz`.

The omission is a delivery limitation, not an experimental change. The available
GitHub connector accepts inline text/base64 rather than a local-file upload;
the preceding full-package publication could not transfer this binary. The user
explicitly authorized this text-only subset. No source banks were recreated,
no models were fitted, and no numerical experiments or jobs were rerun for
this publication.

## Complete original downloadable archive

The complete unchanged archive remains a separate ChatGPT-conversation artifact:

- File: `pro10_review_package.zip`.
- Download: [original inspection package](sandbox:/mnt/data/pro10_review_package.zip).
- Original local artifact location: `/mnt/data/pro10_review_package.zip`.
- Size: **4,663,488 bytes**.
- SHA-256: `8865fc56bde75e6a79f1ba695a44294c7393beaa701b078450f8bac996eb850c`.

The `sandbox:` link is conversation-scoped, not a GitHub-hosted or generally
portable download URL. Obtain the original archive from the accompanying
ChatGPT delivery. No public hosting or permanent availability is claimed.

## Original manifest intentionally preserved

`SHA256SUMS` is the ORIGINAL 1,367-byte manifest, including the entry for the
absent `local_focused_sources.npz`. Its own SHA-256 is
`39852427a3a43b3a01c0eadf1f8ef78580b5872f681f5456042e08f3b4e9604f`.
The missing entry has NOT been removed or rewritten. This new delivery note
is not part of that original manifest.

Running `sha256sum -c SHA256SUMS` in a fresh checkout of this directory is
expected to fail for the missing NPZ. That failure accurately identifies the
incomplete Git delivery; it must not be described as a full-package hash pass.
To check the complete original package, verify the ZIP hash, extract its
`pro10_review/` directory, and run the original manifest check there. Any
original-note reference to an included source-bank archive refers to that
complete downloadable package, not this Git subset.

## Actual Git inventory

The sixteen unchanged original text files are:

```text
MATH_AUDIT.md
PROPOSED_PSC_PROTOCOL.md
PROVENANCE.json
README.md
RUN_NOTES.md
SHA256SUMS
audit_numbers.json
local_focused_live.jsonl
local_focused_timing.json
local_numerical.json
local_quality.json
local_timing_live.jsonl
pro10_kernels.py
quality_quadrature.py
run_focused_local.py
run_local.py
```

The sole added file is `DELIVERY_SUBSET.md` (this note). Thus Git receives
**17 text files: 16 original files plus 1 delivery note**, not the original
17-file package. It receives no NPZ or placeholder standing in for the NPZ.

Publication is isolated on `pro10-artifacts-20260909`, based on the reviewed
commit `e7ee7938a9b146aa5b167a1176e53f99f9ccd4bd`, and adds only
`research/transport_iteration_20260908/pro10_artifacts/`. The active branch
`agent/observation-transport-audit-20260908` is not a write target. This delivery
does not update the archived scientific claims with subsequent native results.
