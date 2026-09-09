# Pro17 original archive delivery

Transport-only publication of the unchanged 29,564-byte original TAR.XZ.
Archive SHA256: `80b9e676f913c0dfe2cfd99dee6eff44f6009632d3b98c2dde12ba16cccfe220`.

From this directory, run:

```sh
python verify_extract.py --parts . --destination /path/to/NEW_verified_pro17
```

The destination must not exist. The extractor checks all four part sizes and
SHA256 hashes, the reassembled archive, the exact 23-file inventory, safe regular
file paths, and every original file size/SHA256 before writing. It does not run
source code or regenerate fits/results. Full original manifests and historical
failure/read/execution receipts are also available in `original_receipts/`.

`PRE_TREE_VERIFICATION.json` records upload hash checks and exact read failures.
This publication does not convert original execution receipts into independently
verified passing-test or research-success claims.

`archive.part01.b64` is an earlier unverified fragment, retained unchanged for
provenance. It is NOT one of the four parts in `TRANSPORT_MANIFEST.json` and must
not be used to reassemble this archive.
