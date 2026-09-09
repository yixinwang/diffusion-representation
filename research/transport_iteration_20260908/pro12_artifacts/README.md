# Pro12 — certified restricted tilted-normal / Heun frontier

**Delivered locally, not published. No fits, PSC jobs, active-branch changes,
new image/video data, source-bank regeneration, or sampler timing runs.**

The new result is a constructive real-interval integration certificate for
all five fixed Heun grids, plus a sharper finite-learning correction. The
arithmetic backend is MPFR, not Arb. No complex analytic continuation is used.

## Results

All units are forward KL nats per complete 3072-dimensional array (192 common
root coordinates and 2880 conditionally independent residual coordinates).
The primary conditional intervals have widths at most 4.66e-10. The final
expected-training-risk intervals have widths at most 1.098e-8. Both meet the
requested 2e-8 width. Every interval includes a directed analytic Gaussian tail
bound, at most 7.144081089334284e-18 outside |z| <= 12.

For targets 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, the fastest quality-eligible
primary endpoint-Heun grids are **8, 16, 32, 64, 64, none**, respectively, both
conditional on correct catalog recovery and under the finite-learning expected
risk bound. Exact central/tail transport qualifies at every requested target.
This is an ideal-real-law quality certificate paired with observed binary64
implementation costs, not a KL certificate for a discrete rounded output law.

Read `FRONTIER.md` for tables; `MATH.md` for the proof; `PROTOCOL.md` for assurance
boundaries and evaluation rules; `INSPECTION.md` for immutable source provenance.
`FAILURES.md` preserves failed setup/compile attempts and superseded bounds.

## Reproduce

Requirements: C++17 compiler, GMP headers/library, MPFR 4.2.2 (the version used
here), and Python 3. No Python numerical packages are required. Prefer the
vendor `mpfr.h`; `mpfr_minimal.h` contains a small ABI fallback for the tested
Linux x86-64 runtime-only installation. No library binaries are redistributed.

From this directory:

```sh
python3 run.py --output /tmp/pro12-independent-check --timeout 240 --repeat-order8
```

The output directory must not exist. Each process has a hard bound; failures
retain logs and never produce a misleading completion marker. The runner builds
four programs, runs consistency/global-bound checks, certifies all five grids,
recomputes sharper risks, then constructs the frontier. It runs no fitting, scheduler or network commands; only the specified new local
output directory is written.

For a single smaller local check, add `--n 4`. For manual Linux compilation:

```sh
g++ -O2 -std=c++17 -fno-fast-math -ffp-contract=off certify.cpp -l:libmpfr.so.6 -lgmp -o /tmp/pro12-certify
/tmp/pro12-certify 4 6 48 4 5e-9 128
```

## Evidence files

`results/n*_first_attempt.json` are the unchanged primary integration receipts.
Their `learning_correction` and `unconditional_upper` fields contain the first,
valid but weaker, stochastic-order correction; **do not use those fields for
the final frontier**. The final sharper bounds are in `results/n*_risk.json`,
with both endpoints and outward width bounds. `results/frontier.json` consumes
only those final risks and keeps decimal endpoints as strings.

The five primary integrations took 143.917737513 seconds total locally. A
separate order-8, 192-bit, changed-mesh N4 certificate also passed; its agreement
is corroboration, not the basis for either certificate. The 256 consistency
checks passed, and all-domain constants were separately evaluated outwardly.
These are computational certificates under the documented arithmetic/software
assumptions, not proof-assistant formalizations.

The archive contains text/source/JSON/logs only. `SHA256SUMS` covers every
other delivered file. No precompiled executable, MPFR/GMP library, fitted model,
source-bank NPZ, or original PSC output-array binary is included.
