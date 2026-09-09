# Optimizer comparison execution record

The comparison source was frozen at a397981d13f1bbe2121b58d5dac9f13089c243a7 after review and tiny fabricated instrumentation checks. The complete QALT suite passed 223 tests in 19.22 seconds. All 36 original observation streams and both fixed 500-update solvers are prescribed before fitting. This study measures optimization only.

The first scheduler request was rejected before any job existed: RM-shared permits at most 2000 MB per allocated CPU, while the launcher requested one CPU and 4 GB. The next submission overrides memory to 2000 MB with the same one CPU and 60-minute cap. The scientific source, solver settings and streams remain at the frozen revision. This resource amendment precedes every comparison result. It does not alter the queued native-image experiment.

PSC accepted the amended request as job 45568902. Checkout: `/ocean/projects/mth250006p/ywang26/diffusion-optimizer-20260909`. Result directory: `/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-optimizer-comparison`. Submission succeeded; scientific results are still pending.
