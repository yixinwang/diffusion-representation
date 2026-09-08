# Independent covariance result verification

Reviewed the retrieved completed seed-2100 result files from source commit `4553e04f71c545e401409d18d82731589974760b`. All five payload SHA-256 digests match both `SHA256SUMS` and `COMPLETE.json`; the checksum-manifest digest also matches. Completion reports `all_checks_pass=false` and first failure `paired_nll`, consistent with the summary.

Independently recomputed paired contrasts from `scores.npz`, using `(log_score_full-log_score_comparator)/2304` per image, then balancing ten class means. There are 5,000 unique image IDs and exactly 500 images per class. The recomputation used image-level sample variances, Welch–Satterthwaite degrees of freedom, two-sided t intervals, one-sided margin tests, and a three-comparison Holm adjustment. Stored contrasts match bitwise; means, standard errors, intervals, and adjusted p-values reproduce the summary within 1e-14.

The registered comparisons, with descriptive 95% intervals, were:

- B4: gain 0.3184876558, interval [0.3111779721, 0.3257973396], required gain >0.01, passed.
- Block covariance: gain -0.0000028361, interval [-0.0000108078, 0.0000051356], required gain >0.01, failed.
- Frozen Student: gain -0.1834002741, interval [-0.1904972013, -0.1763033470], required gain >-0.01, failed.


Positive values favor full covariance. Full covariance improves the B4 conditional score, shows no practical gain over block covariance, and loses substantially to the frozen Student rival. Holm-adjusted p-values are numerically 0, 1, and 1; the first value is floating-point underflow for an extremely small tail probability. The gain cannot support a cross-band mechanism or a complete density-successor claim.

The completed run passes source/parent reproduction, positivity, reversible-density checks, coordinate means, angular checks, and the local CPU requirement. It fails aggregate and stratum-specific uncentered second-moment calibration, band-energy shares, band-energy correlations, and radial PIT. These later diagnostics remain diagnostic after the earlier primary NLL failure.

The holdout covariance optimum reports 0.3185854754 nat/detail maximum gain over B4 on this same array, just 0.0000978196 above the fitted full covariance. Its role is an upper bound computed on already evaluated development coordinates. It is neither a fitted competitor nor population evidence. Its near match to fitted full covariance offers no route to overcoming the 0.1834 nat/detail Student deficit within this global zero-mean Gaussian family on this array.

This is a completed failed adaptive-development run on the reused repair split. Its intervals carry no confirmation coverage. It warrants preserving the result and withholding promotion. It establishes no unconditional image-generation gain, VAE or latent-diffusion superiority, or video result. The summary records that official test data were not deserialized; this review accessed only the retrieved development result files and did not independently trace remote file access. No frozen code or data were changed.
