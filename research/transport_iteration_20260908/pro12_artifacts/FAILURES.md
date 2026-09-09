# Preserved attempts and limitations

## Backend acquisition did not succeed

`python-flint` was absent. A bounded installation attempt (12-second bound)
failed to reach the package index because DNS resolution was unavailable.
The retained original output is `results/setup_attempt.log`. Additional direct
network access/download probes did not establish a usable wheel. Connector
GitHub reads did work. No Arb integration was actually executed, so this is a
backend-availability failure, NOT evidence that Arb fails mathematically or
numerically on this problem. The delivered independent solution uses the
already-installed MPFR/GMP runtime and no complex arithmetic.

## Initial C++ compile failure

The first draft passed a `std::string` to an interval constructor accepting
`const char*`. Its exact source is preserved at
`attempts/certify_initial_compile_failed.cpp`. The correction was to use
`tol.c_str()`; it changes input conversion plumbing, not the mathematical rule.
The original compiler stderr was not saved to a file. The unchanged failing
source was later compiled in a bounded syntax-only check; that reproduced
stderr and status are explicitly labeled
`results/compile_failure_reproduction.{log,json}`. They are not presented as
original-time logs.

## Local execution interface

An interactive/streaming container invocation was unsupported. The longer
N32/N64 local computations were run with bounded local process wrappers; both
completed within this response. Their process records are retained. There was
no remote cluster/scheduler action and no promise of asynchronous work.

## First correction was valid but not sufficiently narrow

The initial integration receipts include the stochastic-order failure bound
M<=2.8519519316104041 and a learning correction <=7.0582438656742994e-7. It
improves Pro10's conservative bound but is not the final risk interval. The
receipts are unchanged, including those superseded fields.

`risk_bounds.cpp` implements the later proved positive-part/Pinsker argument.
Its final risk receipts meet width <=2e-8 at every N. The first correction is
still used as a safe root-wrong bound inside the final argument.

## Completed and partial numerical work

All five primary compact integrations completed on their first executed
numerical attempts. There were no hidden failed numerical grids or quadrature
timeouts. The certificate's subdivisions are normal adaptive error control,
not discarded fits or samples. No domain-positivity box was rejected in these
five successful runs. The changed-order N4 run also completed. A later smoke
run of the full reproduction wrapper was intentionally scoped to N4; it is
labeled partial-grid and does not replace the already completed five-grid data.

The checker is not proof-assistant verified. The original 129 PSC payloads and
102 binary outputs were not independently replayed in this iteration. Native
quality, floating-generator KL, exact ML under rounded arithmetic, GPU latency,
and memory superiority remain outside this certificate.
