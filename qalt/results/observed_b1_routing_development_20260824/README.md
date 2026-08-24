# Observed B1 scalar-routing development result

This is a preserved failed registered iteration from source commit `a2b86ad`, Slurm array job `44306107`, and development seeds `1100..1104`. Every method used the same 45,000 fitting images, 5,000 validation images, record-keyed dequantization per seed, 250,000 fitting sites, fixed Haar chart, and conditional information declared in the protocol. The official CIFAR-10 test batch was not deserialized.

The marginal mechanism succeeded consistently. The four-component one-shot conditional improved validation NLL over the fixed Gaussian by `0.12595` nat per detail coefficient (seed range `0.12592..0.12600`). Coarse-energy conditioning added `0.004391` (`0.004383..0.004405`). Four and eight one-shot components were practically identical: `O4-O8` averaged `-1.20e-6` nat/detail and stayed between `-3.30e-6` and `2.11e-7` across seeds.

The route failed for a specific reason. Relative to the autoregressive comparator, the first RGB channel of each Haar band tied within roughly `1e-6`, while the second and third channels lost `0.0799/0.0799`, `0.0715/0.0709`, and `0.0233/0.0228` nat per full detail coefficient. All five simultaneous routing decisions therefore abstained to the depth-nine full route. The pattern isolates within-band cross-color dependence: increasing marginal mixture capacity does not fix it, and cross-band parents add almost nothing to the first channel of later bands.

The next registered mechanism is a one-shot joint RGB block conditional per Haar band, using a dense covariance shape and non-Gaussian radial scale mixture. It directly represents the dependence that scalar autoregression exploited while retaining block-parallel sampling.

This launch is not promoted as a complete B1 execution because the runner omitted the registered PIT, angular, residual-energy, and spatial diagnostics. Those omissions do not change the frozen likelihood gates or the failure, but the corrected block study must emit them. This result contains no active/coarse generator, full diffusion, VAE, FID, or endpoint evidence.
