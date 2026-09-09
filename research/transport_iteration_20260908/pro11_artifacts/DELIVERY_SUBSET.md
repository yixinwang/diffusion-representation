# Delivery subset: Pro11 archival publication

This directory is a **partial delivery**, not the complete original package.
It contains 15 byte-for-byte original text files (117,659 bytes) and this new
`DELIVERY_SUBSET.md`. All six original NPZ fitted-state files are absent from
this Git publication. They remain intact in the original downloadable archive;
none was recreated, refitted, converted, trimmed, or substituted.

## Publication boundary

Repository: `yixinwang/diffusion-representation`.
New isolated branch: `pro11-artifacts-20260909`.
Base commit: `f024cc56651da7dc161d555117d9373a4ac8fa99`.
Only added path: `research/transport_iteration_20260908/pro11_artifacts/`.
No write targets `agent/observation-transport-audit-20260908` or any other
existing branch. The source, results, and scientific notes are archival copies,
not revisions. No fitting, generation, mathematical test, numerical audit,
training, PSC connection, or job was rerun during publication. Publication
checks are file inventories and byte/hash comparisons only.

The original README and provenance statements about no repository writes refer
to the original research run. This separate, subsequently authorized commit is
delivery only. It supplies no new native quality or representation evidence.

## Binary delivery limitation

The available GitHub writes in this session accept inline UTF-8/base64 strings,
not mounted-file upload parameters. No authenticated local GitHub CLI or Git
credential helper is configured. A usable byte-preserving file-backed transfer
route for these six multi-megabyte NPZ files was unavailable in this session.
This is not a claim that GitHub rejects NPZ files or their sizes, or that its
blob API lacks binary support. No binary upload is claimed to have succeeded.
The original archive contains no separate smaller fitted-state files: the
states and their verification banks are packaged together in the six NPZs.

## Complete unchanged original archive

Filename: `pro11_dependence_theory.zip`.
Exact archive size: **18,185,693 bytes**.
SHA256:

```
64c631099d465614bc7706f14b89770afde6c2b29b90cc5ac916296e963a83c4
```

Original download location in this ChatGPT conversation:
`sandbox:/mnt/data/pro11_dependence_theory.zip`.
Original attachment identity: `file_00000000903881f5ab8f8a9737fa79cd`.
That sandbox location is a conversation attachment, not a public GitHub URL;
it must be obtained from the original conversation's download link. No new
archive, externally hosted replacement, or regenerated dataset is substituted.

The ZIP contains the original `pro11_dependence_theory/` directory with
**21 files**, including all six NPZs. Its total uncompressed file bytes are
18,265,569. The complete archive and all 20 entries of its original manifest
were checked by SHA256 before this subset was assembled.

## Files absent from Git

All sizes and hashes below are from the actual unchanged bytes in the original
ZIP. The six omitted files total **18,147,910 bytes**.

| Original filename | Bytes | SHA256 |
|---|---:|---|
| `fit_positive_1109101.npz` | 3025874 | `425eb6c30310b98443fb6310d655d2bb8cea3bf95098c8c1e296309838dd9f0b` |
| `fit_positive_1109102.npz` | 3026156 | `f3e3dc0cfda9f28c48667097b366bba7cea590e2d566556016fad4491631f319` |
| `fit_positive_1109103.npz` | 3025702 | `693bfeaff04af7baeb9ae509a9ecda023c905df9f136631376255c9e24370830` |
| `fit_zero_mean_change_1109101.npz` | 3023355 | `2f5285b39632d19107be2d4898271ed7780134d7071b128b45f67c7aef3a2658` |
| `fit_zero_mean_change_1109102.npz` | 3023660 | `d825833cc9a700fd82ff38601359331347a396287284f40af9b9749e1bebe5be` |
| `fit_zero_mean_change_1109103.npz` | 3023163 | `0529e4499a793b3da795f15e3cd6bb4e1d5ab85d166fb67e85f8bd2282a28322` |

These files contain the three positive-law and three changed-law fitted states,
evaluator-only truth objects, and saved 64x3072 Gaussian/generated verification
banks. The published seed records and `saved_audit.json` retain their original
results and hashes, including the changed-law structural failures. Those text
records do not substitute for the absent fitted arrays. The complete 4000x3072
training banks were already excluded from the original package, as documented
in its unchanged README and provenance; this publication makes no new claim
to deliver them.

## Original text inventory published unchanged

| Filename | Bytes |
|---|---:|
| `PROTOCOL.md` | 13275 |
| `PROVENANCE.md` | 8525 |
| `README.md` | 5199 |
| `RESULTS.json` | 6337 |
| `SHA256SUMS` | 1732 |
| `THEORY.md` | 20155 |
| `audit_saved.py` | 4193 |
| `learning_seed_1109101.json` | 8459 |
| `learning_seed_1109102.json` | 8462 |
| `learning_seed_1109103.json` | 8459 |
| `math_checks.json` | 4899 |
| `model.py` | 11095 |
| `requirements-observed.txt` | 169 |
| `run_checks.py` | 13158 |
| `saved_audit.json` | 3542 |

The original executable modes of `audit_saved.py` and `run_checks.py` are
retained. All other original files have regular non-executable file mode.

## Manifest and preserved qualifications

`SHA256SUMS` is the **original unchanged** 1,732-byte manifest. Its SHA256 is:

```
1806ccef2ab1baaed43c46a8540278ed5690519c21382f0b252a6b799268abf6
```

Its six absent-NPZ entries are deliberately retained. Running an ordinary full
`sha256sum --check SHA256SUMS` on this Git subset alone should therefore report
14 present entries as matching and six missing files, and exit nonzero. Do not
remove those entries, silently use `--ignore-missing` as a full-package pass,
or treat a regenerated close numerical match as the original binary. The
manifest excludes itself, and this new delivery note is not retroactively added
to it. Restoring the exact six NPZ members from the verified original archive
is sufficient for the original 20-entry checksum verification; no fit is needed.

The originals retain the 32-search-step cost (approximately 46,080 conditional
CDF evaluations per complete array), the exact analytic-copy tie, the
continuous-density versus floating-point atomic-law distinction, the separate
source-generation and fitting times, and the lack of a matched global-FM speed
or quality result. None of those qualifications was shortened or changed for
publication. The estimator and theorem still require independent review before
PSC replication or native claims.
