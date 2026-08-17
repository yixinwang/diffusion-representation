# Densely rotated nonlinear identification

The nonlinear, non-Gaussian latent process is post-composed with a random dense orthogonal mixing,
so every observed coordinate contains stochastic variation. The learned chart is a
Cayley-parameterized orthogonal map followed by a shallow exactly invertible nonlinear shear.

Training uses only paired-future conditional variance and a log-determinant state-spread gauge.
Evaluation aligns the learned two-dimensional state to the true state with an affine map fitted on
training data. Ground-truth latents are never used to train the chart.

Baselines are the best two-dimensional linear pair-invariant subspace and reconstruction PCA. The
committed run uses five independent data, mixing, and training seeds.
