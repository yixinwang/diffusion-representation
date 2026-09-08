# Complete-flow pilot failure preservation

The runner now preserves partial development records and checks committed source identity before creating any data. This change makes failed runs inspectable; it does not change the registered models or establish a quality, representation, or efficiency advantage.

## Source identity

The runner compares the bytes of every listed project source, test, specification, and launcher against its Git blob at the recorded HEAD commit. The list includes `run.py`, `PROTOCOL.md`, `run.slurm`, and the separately registered `run_cpu.slurm`. It also checks that imported project modules resolve to this checkout. The manifest records both Git blob identifiers and SHA-256 hashes. A dirty listed file causes failure before training or evaluation data are created. The final files must be committed before a production run; there is no production bypass.

This identifies checked files in a stable checkout. It is not a lock against another process changing files after checking, and it does not identify all third-party library binaries. Python and Torch versions remain recorded. Failures in the scheduler shell, CUDA launcher probe, or Python imports occur before the runner owns an output directory and are outside its exception handler.

## Preserved records

The shared training-array hash is written before device transfer. After each completed training arm, the runner saves its checkpoint, updates `training.json`, and updates the manifest's completed-arm list. A training exception attempts to save the current model state, completed update count, first and last completed objectives, exception text, and traceback. The outer handler saves a separate failure record for any exception after creating the new output directory. If a damaged device prevents checkpoint serialization, the failure record reports that preservation error; a checkpoint cannot be guaranteed in that situation.

Generated arrays are saved before finite-value checks, metric calculations, and numerical roundtrip validation. Completed metric records and available numerical checks are written incrementally. The exact-copy samples are preserved separately. Atomic replacement protects this run's metadata and sample archives from partial writes. Failure records use exclusive creation with numbered alternatives, so an existing failure record is retained. An existing output directory is rejected before the failure handler takes ownership; its contents are untouched.

All training arms still finish before the independent evaluation draw. There is no evaluation-driven fitting, arm selection, or restart. The model constructors, data generator, seeds, optimizer, learning rate, clipping, time and update caps, inference grids, metrics, and inversion tolerances are unchanged. Progress bookkeeping and preservation add small overhead; capped execution can consequently finish a different number of updates. Report the actual completed updates and measured times. A separately registered CPU run is a CPU development record and cannot supply GPU timing evidence.

## Verification

A temporary fabricated-model smoke check passed these cases:

- The data generator, metrics, distances, and device synchronization functions have identical abstract syntax to the previous runner.
- A second-arm exception after two completed optimizer updates preserves the failed state, training hash, completed first arm, and traceback; it creates no evaluation data.
- A deliberate numerical roundtrip failure retains all five generated arrays, the exact-copy samples, and all completed metric records.
- Reusing an existing output directory leaves every existing byte unchanged.
- Repeated failure preservation creates a second record without replacing the first.
- An isolated Git fixture passes with exact blobs and rejects a changed CPU launcher.

The smoke uses substituted dummy learners and fabricated arrays only. Its outcomes verify failure-preservation behavior. No PSC job, CUDA environment change, model-training experiment, or new quality selection was performed for this patch.
