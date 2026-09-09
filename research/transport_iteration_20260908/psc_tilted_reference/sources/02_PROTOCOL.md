# Frozen tilted-normal finite-family synthetic validation

This CPU validation uses the delivered `positive_class_reference.py` without edits or runtime replacement of its functions. Its SHA-256 is `38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b`. The runner checks its bytes against that digest and checks the entire declared source closure against the submitted full Git commit before executing precisely the verified reference bytes. There is no native-data path, network action, source override, or tuning flag.

## Three repeated full-dimensional reference runs

Use seeds 2026090901, 2026090902, and 2026090903. For each, call the original `run(seed, repeats=9)`. Its disclosed truth is root coefficient +0.5 and dictionary index 7. It generates 32 root-fitting arrays and 256 separate summary-fitting arrays, then fits sign and index using only the observed synthetic image arrays and the public dictionary. Every observation contains 3,072 coordinates, partitioned into 192 roots and 2,880 residuals. These synthetic Haar/sigmoid images do not establish realistic image quality.

Preserve every returned reference report, including all nine timing repetitions for batch sizes one and 64. Exact sampling and all five locked Gaussian-reference Heun comparators use 4, 8, 16, 32, and 64 actual velocity calls. The original three warmups per arm and randomized interleaving remain unchanged. No reported NFE or seed is selected or omitted according to results. The common fitted model and complete Gaussian source are shared across arms, with the same public dictionary and summary cache.

The runner separately reconstructs the original initial fitting streams, numerical-check source, timing source banks, and interleaving sequence. Their hashes and the dictionary hash are saved. This provenance calculation occurs outside the original timing function, whose implementation and calls remain unchanged. Its extra work is included only in the harness's total wall time. Fit observations and source arrays can be reconstructed from the saved seed and verified code; compact successful reports retain their hashes rather than all regenerated arrays.

The original numerical report must have full-source roundtrip error no larger than 1e-9, scalar tail roundtrip error at absolute source values through 50 no larger than 1e-10, orthogonal-dictionary error no larger than 1e-12, and a bitwise-equal same-information conditional copy. All errors must be finite. The complete raw report is saved before enforcing these gates. They are finite numerical checks, not global interval proofs. Incorrect learned sign/index selections are preserved as outcomes, not grounds for choosing replacement seeds.

## One recovery run for each dictionary condition

Evaluate signs -0.5 and +0.5, each with indices 0 through 15, in that order. Their fixed seeds are 2026091001 through 2026091032 in the same order: negative-sign index j uses 2026091001+j; positive-sign index j uses 2026091017+j. Each cell generates 32 root-fitting and 256 summary-fitting arrays with its own seed. The simulator's truth parameters are isolated from `fit(root_images, head_images, dictionary)`; fitting has no truth-parameter or source-innovation argument. Preserve the selected sign/index, all 16 observed likelihood scores, score gap, fitting time, simulation time, and input/source/dictionary hashes.

These are 32 different conditions with one declared training seed each. They are not 32 repeated training seeds for every condition, nor a direct empirical verification of the tiny theoretical misclassification probability. There is no generation-quality evaluation or likelihood oracle supplied to fitting.

## Preservation and resource limits

The output directory must be new. Source snapshots, machine versions, host and scheduler identity, seeds, dimensions, counts, thresholds, command, raw reports, and progress persist. Failures preserve their traceback, completed reports, and available partial original-run timing/fit values. Available arrays in the failed original reference frame are saved separately; a failed recovery fit preserves its observation arrays. A final `COMPLETE.json` records completion or failure and SHA-256 hashes for every other preserved payload file. Forced termination or a storage failure can still prevent complete preservation.

The launcher requests two CPUs, 4,000 MB, and ten minutes on RM-shared, with numerical libraries restricted to one thread. A warning 60 seconds before the time limit requests partial preservation. No empirical study is run before review and source commit. Software tests use fabricated reports or injected failures only. The long interval audit is separate and is not repeated by this runner. This study concerns the specified exact canonical velocity and locked Heun grid; it does not compare all flow-matching parameterizations, trained neural models, native image/video quality, or GPU costs.
