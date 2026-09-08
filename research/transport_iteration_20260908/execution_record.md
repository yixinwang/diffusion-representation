# Execution record

The unchanged covariance experiment at source `4553e04f71c545e401409d18d82731589974760b` uses an isolated PSC worktree. The original dirty checkout remains untouched.

1. Submission with four CPUs and 8G was rejected before allocation because the partition permits 2,000 MiB per core. The replacement requested 8,000 MiB, below the registered 8 GiB ceiling.
2. Job `45550358` was accepted but exited at its required-environment check before loading any data. Its batch environment omitted `RESULT_ROOT`.
3. Job `45550810` explicitly exports `RESULT_ROOT` and `SOURCE_COMMIT`. Its log is retained separately. Source, seed, methods, thresholds, and data roles remain unchanged.

The result directory is `/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-covariance`. Its separate job logs preserve both attempts.

Local focused verification passed 14 transport/screen tests and 23 covariance tests. A broader local suite stopped during collection because system Python 3.9 could not evaluate a pre-existing Python 3.10 union annotation. No test body ran in that invocation. The broader suite then passed all 129 tests under Python 3.12 in 41.45 seconds.

The source-only numerical runner initially failed on JSON serialization of a NumPy Boolean. Native scalar conversion repaired this output-format defect, and its regression check passed. No result was overwritten. The successful local fixture is recorded in `local_fixture.json`.

ChatGPT launched three separate Pro conversations before displaying a temporary request limit. The saved conversation links identify the work; complete Pro reviews remain pending retrieval.

The GitHub research branch was published at `5b58f34`, including the seven earlier PSC commits. Job `45550810` was verified running on node `r277`.

Completed runs:

PSC job 45550810 completed the frozen covariance experiment in 182.68 seconds of runner time. First failed criterion: paired_nll. The full layer beats B4 but fails its block and Student requirements. All result files were retrieved through an SSH archive after the SFTP transfer service closed its connection. The archive checksums were independently verified; no result content changed. The large diagnostics array remains on PSC, with its hash in the published manifest.

PSC job 45551078 completed at db4429e with fourteen focused tests passing and numerical round-trip errors below 2.2e-12. Source-only fixture output is published verbatim in psc_fixture.json. Both jobs have finished; neither accessed the official CIFAR test batch.

The covariance source review identified one unused failure-reporting defect: an exception during later diagnostics could retain an earlier stage name. This completed run raised no such exception; its ordered summary correctly records paired_nll as the first failure. The frozen runner was not changed.
