# Exact radial flow and failure conditions

For a color-whitened residual vector `z in R^3`, write `r=||z||`. A strictly increasing rational-quadratic log-radius spline `h_s` defines

`g_s(r)=exp(h_s(log r))` and `T_s(z)=g_s(r) z/r`.

The inverse is `T_s^{-1}(w)=g_s^{-1}(||w||) w/||w||`. The derivative has one radial eigenvalue `g_s'(r)` and two tangential eigenvalues `g_s(r)/r`, so

`log |det DT_s| = log h_s'(log r) + 3(log g_s(r)-log r)`.

Composing these cells with the unchanged exact FSMLF/C3 trunk and R7 endpoint yields

`log p(x|y) = -0.5 ||F(x,y)||^2 - D/2 log(2 pi) + log|det DF(x,y)| - D log 127.5`.

Every visible coordinate remains in the flow. Inactive cells are exactly identity. There is no VAE, encoder, decoder, ELBO, reconstruction loss, stochastic fiber, or deleted coordinate.

## Finite advantage

If `Z=R U` conditional on context `S=s`, `U` is uniform on the sphere and independent of `R`, and `R` has density `p_s`, a standard Gaussian shares the angular law and has chi-3 radius density `q_3`. An exact radial transport removes

`Delta_s = KL(p_s || q_3)`.

## No-headroom and first failure

There is no radial headroom when `R|S=s` is already chi-3. More generally,

`KL(P_{R,U|S} || Q_chi x Uniform) = KL(P_{R|S}||Q_chi) + E KL(P_{U|R,S}||Uniform)`.

The radial map can remove only the first term. It leaves angular parity, cross-site total correlation, semantic multimodality, and stage-aligned feedback unchanged. Applying independent radial bijections also preserves the copula across residual vectors.

## Transfer error

If the fitted and ideal radial maps differ by at most `epsilon_T` in transformed coordinates and `epsilon_J` in log-Jacobian on the registered annulus, then the per-vector Gaussian-base log-density error is bounded by

`(||T*(z)|| + epsilon_T/2) epsilon_T + epsilon_J`,

plus the identity-tail remainder. If the inverse remaining flow is `L`-Lipschitz and the RMS radial inverse error is `epsilon_R`, then `W2 <= L epsilon_R`, and Euclidean energy-distance discrepancy is at most `2 W2`.

## Systems boundary

The radial state is `O(levels * orientations * classes * bins)` and ideal fused inference is linear in visible dimension. The measured implementation nevertheless incurs repeated parent/color passes, sixty class-conditioned spline cells, quantile fits, gathers/scatters, and R7 refitting. The arithmetic proxy was small, but measured fit and batch latency failed. The unchanged parent-plus-color image fit alone already exceeded the frozen latent-flow fit allowance, ruling out another endpoint-only systems adjustment under the same contract.
