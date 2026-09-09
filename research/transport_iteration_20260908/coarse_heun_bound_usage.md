# Checkpoint-only coarse-Heun bound

New calculator: `qalt/experiments/observed_flow_pilot/check_coarse_heun_bound.py`.

Example command, after substituting the saved checkpoint and its recorded full source commit:

```sh
PYTHONPATH=qalt/src python qalt/experiments/observed_flow_pilot/check_coarse_heun_bound.py \
  --checkpoint /path/to/shared_latest.pt \
  --source-commit FULL_RECORDED_COMMIT \
  --output /path/to/new_coarse_bound.json
```

The default selects `coarse_prior.velocity`, grid side eight, and 16 Heun steps / 32 actual velocity calls. For an explicitly separate full-tensor FM checkpoint, use `--prefix velocity` and its actual spatial size and step count. The convolutional bound is independent of grid size, but the dimension-dependent determinant bounds are not.

The script compares the local implementation sources with the declared checkpoint revision, verifies the unconditional three-convolution / two-SiLU geometry and all selected tensor shapes, and checks finite weights and biases. It loads only a plain tensor state dictionary using `weights_only=True`. It hashes the checkpoint, sources, and calculator. The link between checkpoint and source revision must still be checked against the originating run manifest; state dictionaries alone do not establish that provenance. There is no observation loader or fitting routine.

For each convolution, the reported bound is the sum of channel-matrix Frobenius norms over all spatial offsets. The first convolution uses only its state-input columns; its fixed time channel and biases do not affect state derivatives. With $b=1+1/e$, the script computes $L=b^2K_1K_2K_3$ and $a=L/N+L^2/(2N^2)$. The mathematical sufficient condition is strict: $a<1$. Ordinary float64 scalar calculations do not supply outward-rounded interval bounds, so the report always sets `validated_interval_certificate` to false. A passing numerical comparison indicates that the conservative sufficient condition appears to hold. A failing comparison gives no conclusion about actual invertibility.

The output also reports a strict upper limit on scaling the last convolution's weights that would meet this bound, and a sufficient step-count threshold. These are mathematical diagnostics. They do not authorize modifying an existing checkpoint, changing its source law, or altering a frozen inference grid. No trained checkpoint has been evaluated as part of implementation.

Four fabricated-weight tests pass: zero-output identity and invariance to time-channel/bias changes; an explicit three-factor norm product and its scaling threshold; comparison against the exact dense finite-grid convolution norm; and rejection of conditional, nonfinite, or mismatched fields.

## A possible future constrained parameterization

One prospective construction is to parameterize every convolution as

$$
W_j=\frac{\alpha_j\widetilde W_j}{\epsilon+\sum_r\|\widetilde W_{j,r}\|_F},\qquad \epsilon>0.
$$

Then $K_j\le\alpha_j$ and a prescribed product $b^2\alpha_1\alpha_2\alpha_3<N(\sqrt3-1)$ enforces the real-arithmetic sufficient condition for the chosen number of steps. The first layer can normalize all input columns for simplicity; that also bounds its state-column restriction. Biases can remain unconstrained finite parameters. This construction changes the fitting model and requires a new comparison; it gives no approximation or generation-quality improvement guarantee.

A library's usual spectral normalization of a flattened convolution kernel does not directly bound the full spatial convolution operator by the same number. Spatial overlap matters. One must either bound the actual configured convolution operator or retain a valid offset-sum bound with the required spatial factors. Power iteration alone produces an estimate and needs an error bound before it can support a strict global claim. These considerations apply to the existing convolutional coarse field; they do not establish a global Lipschitz bound for an attention-based residual field.

The calculator also bounds each zero-padded convolution through a larger periodic convolution. For an n-by-n input and an odd k-by-k kernel, zero-pad the input into a torus of side n+k-1 and crop the output back to the original sites. These embedding and projection maps have operator norm one. The periodic convolution is diagonalized by the discrete Fourier transform; its norm is the largest channel-matrix singular value over all frequencies. This upper bound can improve the sum of offset Frobenius norms, so the calculator takes the smaller per-layer value. It uses float64 FFT/SVD, without directed rounding. Fabricated dense zero-padding matrices verify the inequality for small grids and both 1-by-1 and 3-by-3 kernels. No fitted coarse checkpoint has been tested yet.
