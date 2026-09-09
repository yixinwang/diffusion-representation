# Local reproduction of the delivered tilted-normal reference

The delivered reference was reviewed and executed unchanged with one-thread numerical-library settings, seed 77181 and nine randomized interleaved timing repetitions. Its SHA256 is 38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b. All ten delivered SHA256SUMS entries match.

The observed-array fit recovered root sign +0.5 and summary index 7. Full-source round-trip maximum error was 3.11e-15; declared scalar tail checks through absolute source 50 gave 1.42e-14; exact-copy arrays matched bitwise. The experiment uses all 3,072 actual coordinates, but its Haar/sigmoid synthetic arrays are not natural images.

On this local CPU, batch-64 medians were 12.06 ms for the analytic generator and 15.28, 28.74, 55.68, 109.35 and 219.56 ms for the fixed Gaussian-reference population field at 4, 8, 16, 32 and 64 actual calls. These reproduce the direction of the delivered CPU result, not its exact hardware timings. All source coordinates, fitted root and summary, output transform, fitting data and fitting procedure are shared. No GPU, peak-memory, native-perceptual or general learned-FM superiority follows.

This local run uses the original single seed and original truth. A separately frozen PSC replication is being prepared. The interval lower-bound code is under independent audit; a dense numerical search alone does not establish a uniform error lower bound.
