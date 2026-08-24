# Frozen Multiscale QALT Stage-A confirmation

Freeze commit precedes every access to confirmation seeds `800..829`. The implementation, image/video shapes, train/validation/test sizes, 20 solver steps, parent features, EM iterations, exact/diagonal/coarse/Euler controls, timing repetitions, and live-array accounting remain those that passed corrected development.

Confirmation consists of 30 paired units for both images and videos. The inferential family contains 16 one-sided paired Student tests with Holm correction at familywise level 0.05:

- two TOST components per modality for exact-split minus QALT within `[-0.02, 0.02]` nat per dimension;
- QALT-minus-oracle NLL below `0.01` per modality;
- positive diagonal-minus-QALT, coarse-only-minus-QALT, and finite-Euler-minus-QALT NLL per modality;
- latency ratio below one and explicit peak live-array ratio below one per modality.

Token-accounting agreement within 10%, exact full-token equality, and Haar round-trip error below `1e-10` are hard deterministic checks. A failure is retained and stops Stage B. No component can be removed, no threshold can change, and no second Stage-A confirmation is permitted.

Passing supports only the declared procedural nonlinear/non-Gaussian image/video law. It does not establish observed-data quality or production accelerator speed. Stage B still requires separately frozen observed-data experiments against optimized latent, wavelet, and few-step baselines.
