# Preserved failures and scope

## Unexpected fabricated execution failure

attempts/mechanism_initial.py and attempts/checks_initial.py are the initial
executed sources. attempts/run_initial.log is the original redirected output;
attempts/results_initial.json is its partial receipt with traceback. The float64
small and full-dimensional checks ran, then the float32 full-dimensional case
raised: expected m1 and m2 to have the same dtype, but got double != float.

The cause was independently written scalar-harness arithmetic x0=-bound+dx*ix,
with an integer index and global float64 default. The corrected harness uses
ix.to(dtype=x.dtype). This is not a bug finding in the frozen native source,
which already explicitly casts its index. The original files are unchanged.

attempts/mechanism_corrected.py, checks_corrected.py, run_corrected.log and
results_corrected.json retain the first successful correction. The final harness
adds source hashes, a hidden-overflow rejection and frame-gradient check, and
avoids an unnecessary graph-retention logging warning. attempts/run_final.log and
fabricated_results.json record the final run. No fit was run or regenerated.

## Deliberately invalid/failing mechanisms

The final receipt retains a deliberately omitted new logdet, which produces a
nonzero .44330536 discrepancy and is rejected. It also retains a fixed-base
correlated-follower target with .22314355 nats of unavoidable conditional TC for
that restricted diagonal-response family. This is not a lower bound for the
trainable full model. These are intentional negative controls, not represented
as accidental earlier discoveries.

## Access failures

A GitHub git-data request with abbreviated adcb87c returned404; a URL-encoded
branch-name fetch was rejected by the connector's path validation. The normal
commit endpoint subsequently resolved adcb87c to
adcb87c43d1720a167933d4d8bd83759865edb5f. Frozen files were successfully read using
the GitHub connector at1f6be303c8b93ba45ac63c43e7c5971fa38a0082.

A subsequent direct sandbox urllib request for the already inspected frozen
cached_global_innovation.py at raw.githubusercontent.com failed with
URLError: [Errno -3] Temporary failure in name resolution. This paragraph is a
summary of the observed tool exception, not a purported original redirected log.
No source bytes were downloaded by that attempt. The delivery therefore records
connector-inspected blob identities and does not claim local byte-identical
copies of the frozen repository files. This access failure did not require a
fallback to native data or a repository modification.
