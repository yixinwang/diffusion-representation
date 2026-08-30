# Reproduction

The complete local scientific package is carried by Git head `abc36a1b99160126271b48e5e2ee408808cadce8` and tag `rmpf-r20-realistic-high-dependence-failure-20260830`.

Core commands, with one CPU thread fixed:

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
python -m pytest -q experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/tests/test_real_r19_stratum.py
python experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/source/run_source_gate.py --data-root data/rmpf32 --output experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/results/source_gate
python experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/source/run_quality_stratum.py --data-root data/rmpf32 --output experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/results/quality_smoke
python experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/source/run_transfer_diagnosis.py --data-root data/rmpf32 --output experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/results/diagnosis
python experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/source/verify_final.py --data-root data/rmpf32 --output experiments/real_development/rmpf_r20_realistic_high_dependence_stratum_20260830/artifacts/R20_FINAL_VERIFICATION.json
```

Immutable data hashes:

- CIFAR opened-development file: `0365c4be3460d833c3f308cd9e4433a246dcf07401610a1d2e1489d4fd98ba4a`.
- UCF opened-development file: `1fbdbcc20447c11c66b71c921a449d0c7cd2e46bc27eeb931a680e3c6febed2c`.
- Frozen image R19 parameters: `2515df6934a841b4cbb1220bffc44072dc5b1a2076b4af7ed9951a3581e55c74`.
- C4G parent archive: `28694f85d3683f2e43645747d007ba744ad2d9960bcf2b238e6507021ccd9f6f`.

Source gate took 6.50 seconds with 268,940 KiB peak RSS. The successful complete quality run took 34.56 seconds with 1,145,748 KiB peak RSS. Transfer diagnosis took 5.30 seconds with 443,708 KiB peak RSS. Two tests and 34 independent final-verifier checks passed. UCF quality, model-seed replication 9401–9404, and untouched confirmation were not opened.
