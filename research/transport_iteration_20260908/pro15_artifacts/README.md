# Pro15: phase-invariant conditional graph recovery

**Result:** zero-mean conditional dependence is identifiable through a registered
context-feature margin, but the original 4,000-array budget has no recovery
certificate. A conservative Lane A-scale guarantee requires 69,507 graph arrays
plus 4,000 parameter arrays. The original 2,000 graph arrays also encounter an
information-theoretic exact-matching lower bound. A smooth off-basis falsifier
remains fatal to the harmonic method and is handled better by the strong
histogram-likelihood comparator. There is no universal latent/FM advantage.

Read THEOREM.md for definitions, full proofs, all approximation and graph-error
terms, sample/compute bounds and the precise missing native assumption.
PROTOCOL.md is the unchanged preregistered local protocol; RESULTS.md reports
actual execution. FAILED_ATTEMPTS.md preserves every scientific failure and
access limitation. REFERENCES.md records the inspected frozen source context
and prior weighted-conditional-covariance methodology.

## Complete implementation

- src/model.py: normalized full-D Gaussian-to-observed transport and inverse,
  full density, fitted roots, validated partial matching, constrained context
  coefficients, exact-copy decoder, serialization.
- src/discover.py: phase-invariant weighted Grams, old unconditional discovery,
  split conditional-likelihood baseline with exact mathematical screening,
  root and separate pooled parameter fitting; no evaluator import.
- src/bounds.py: explicit graph, information, root and parameter constants.
- src/fixture.py and src/evaluate.py: evaluator-owned fabricated truth and two
  independent population-integration formulas.
- src/run.py: original all-fit-before-score execution and complete array saving.
- src/audit_saved.py: added after execution for read-only saved-state checks;
  it does not refit or draw new sources.
- tests/test_pro15.py: 36 tests, including source contracts, endpoints,
  normalization, inverse/Jacobian, graph threshold and deliberate invalid inputs.

## Actual data and states

results/run_initial contains both seeds' original full Gaussian observation
sources and all observed arrays for three laws. Low-budget fits use explicit
prefix slices of those larger banks. The fitting interface never sees the
Gaussian sources. All 72 actual fitted states, graph/regression arrays, common
generation sources, generated outputs, inverses, log determinants, log densities,
copy outputs, population scores and timings are present. Coefficients and root
probabilities are serialized losslessly as JSON float values; replay is bitwise.
Fit-statistic -infinity values in likelihood arrays are deliberate sentinels for
screened-out/unexamined edges, not nonfinite density or model parameters.

results/run_initial/ALL_FITS_FROZEN.json sealed all model hashes before any
population evaluation. registration.json records the unchanged pre-run protocol
and six execution-source hashes. The original executed source versions are also
preserved in attempts/initial_sources. results/saved_audit.json independently
checks all frozen models and records original payload hashes. Tests' source,
output and numerical fixture arrays are in results/tests_initial.

The separate full_dimension_smoke directory contains a constructed, unfitted
3,072-dimensional model and its complete 64-source numerical outputs. Do not
mistake it for learned large-dimensional graph validation or native images.

## Reproduction

Dependencies: Python 3.10+ syntax, NumPy and SciPy. Actual versions are recorded
in results/run_initial/environment.json. No GPU, network, native data or PSC is
needed. From the package root:

```sh
./run.sh results/reproduction
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src PRO15_TEST_OUTPUT=results/tests_reproduction \
  python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python src/audit_saved.py \
  --run results/run_initial --output results/audit_reproduction.json
sha256sum -c SHA256SUMS
```

Existing run directories are never overwritten. New reproductions are separate
runs; they do not replace the original arrays or timing receipts. Source-to-
output replay can differ on a different numeric platform; the original bytes
remain authoritative.

## Delivery

The requested target is the new isolated pro15-artifacts-20260909 branch, only
under research/transport_iteration_20260908/pro15_artifacts/, based on
ce2b6ba6b1075e62fe048b42e84110632688fabb. This session exposed no GitHub write
action, so **no publication branch or commit is claimed**. The complete original
archive is delivered instead, with no missing binary arrays and no encoded
archive blobs in a repository. publication_request.json specifies the target.
MANIFEST.json inventories all payloads; SHA256SUMS also hashes MANIFEST.json.
The external archive-identity receipt supplies the archive and SHA256SUMS hashes,
avoiding self-referential manifests. No files were omitted or regenerated as
substitutes. No PSC job, active-branch edit or native-study change occurred.
