# Pro16: same-real-map spline numerical audit

Frozen source: `168efc227e361a86a2a6aa6952786e5d0e13e30f`.
Publication base: `929c48767ba1fe1e64318c9ab165da28afcda795`.
Only this isolated artifact directory is added. No active source or PSC job is changed.

The complete package is stored in five byte-exact binary archive parts, not a delivery subset. It contains 26 files: the full report and derivation, proposed rerun protocol, unchanged original source, standalone candidate, test scripts, frozen test plans, all original synthetic failures and actual test outputs, provenance and full per-file manifest. `stable_spline.py` and `original/` are also exposed directly for inspection and match the archived files byte-for-byte.

## Extract and verify after fetching this branch

From this directory, with Python 3.10 or later:

```sh
python extract_package.py --out /tmp/pro16-audit
cd /tmp/pro16-audit/pro16_artifacts
```

The extractor first checks all part hashes, the complete archive hash, and every payload byte count/hash, then writes into an exclusive new output directory. It performs no network access and never executes archived code. Read the extracted `README.md`, `RERUN_PROTOCOL.md`, `MANIFEST.json` and `results/`.

Rerun the local scalar tests in new directories (Torch and mpmath required):

```sh
python run_audit.py --out results/independent_audit
python run_supplement.py --out results/independent_supplement
```

The original results remain untouched. The supplement replays the retained original synthetic witnesses, not a native failed model.

## Findings and limits

The frozen inverse algebra is correct in real arithmetic. Synthetic endpoint-neighbor tests reproduce out-of-bin roots, including theta=1.0000001192092896. The candidate uses reflected direct endpoint distances, scaled positive odds coordinates, positive-term logdet evaluation and unconditional binary64 scalar work, with caller-dtype outputs. No interior clipping, discriminant clamping, tolerance rescue or detached gradient is used.

44,800 tested scalar coordinates: the original dense kernel rejected eight inverse banks; the candidate passed all twenty banks. Independent 100-digit bisection on 96 bins gave maximum candidate inverse value error 1.449e-16 and logdet error 1.023e-13. First/second derivative checks and invalid-input rejection passed. Nine saved synthetic failure witnesses were independently replayed using the original stored knots. A reflection-only diagnostic also passed those nine: the odds implementation is not proved uniquely necessary.

Important retained limitations: an extreme-logit test had rounded inverse-forward residual 0.006, so scalar oracle agreement does not imply the native 0.001 roundtrip gate. A finite binary64 bin with derivatives 1e308 is correctly rejected when its scaled discriminant underflows. This is not an all-finite-input guarantee.

No actual failed PSC model/optimizer/RNG/batch was accessed or replayed here. The aggregate exception does not identify which predicate failed. No native quality, GPU qualification, runtime advantage or full-flow admission is established.

The proposed repaired study restarts all seven arms for all three original seeds from scratch, preserves original information/counts/budgets/gates, retains both larger RQS controls, charges implementation cost, and opens no quality until all 21 fits are frozen. Adequate baseline training remains to be demonstrated; baseline numerical failure or undertraining is never a candidate win. The protocol authorizes no jobs.

## Complete archive identity

TAR.XZ: 40,952 bytes; SHA256 `8c3b10d6ff93a39f090a15682b2c0de62feefaef8384d985b21c28be0d5df853`.
Payload MANIFEST SHA256: `8f21080a6781d8c4ceee3b8d77348181022efb5bb5c25426e55a66d113ede889`.
The chat-delivered ZIP contains the identical 26 files: 59,799 bytes; SHA256 `6fe0c323d147f42ee2037493c66f47ba7d2504693e99191f4eb162b68f1f40aa`.
See `DELIVERY.json` for every part size/hash. No payload is omitted or regenerated for publication.
