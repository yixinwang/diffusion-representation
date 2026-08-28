# Exact HSRF derivation

## Normalized all-coordinate map

Let `H` be block-diagonal orthonormal Haar, and let `S0,Sf` be positive diagonal affine normalizations. For one split `(A,B)`, HSRF uses

```text
Z_A = A
Z_B = {B - Phi(A) C_mu U_mu^T} exp{-s(A)}
s(A) = s_max tanh(Phi(A) C_s U_s^T / s_max).
```

The inverse is explicit:

```text
B = Z_B exp{s(A)} + Phi(A) C_mu U_mu^T.
```

The coupling Jacobian is block triangular and

```text
log |det J| = -sum_j s_j(A).
```

Haar and fixed permutations have determinant magnitude one. Thus, for `z=F(x)` and standard-normal base,

```text
log p_F(x) = -0.5 ||z||^2 - D/2 log(2 pi) + log |det D F(x)|.
```

No dimension is discarded.

## Finite global-dependence advantage

Suppose after the shared chart

```text
B = Phi(A) W_* + epsilon,
epsilon ~ N(0, sigma^2 I_q),
Cov(Phi(A)) = I_m.
```

With `n>m+1`, ordinary least squares has independent-test error

```text
E ||Phi(A)(W_hat-W_*)||^2 = q sigma^2 m/(n-m-1).
```

Therefore its expected per-visible-dimension NLL advantage over a zero-global model is

```text
Delta_full = ||W_*||_F^2/(2 D sigma^2)
             - q m/{2 D (n-m-1)}.
```

If `W_*=C_* U_*^T` has output rank `r`, reduced-rank regression changes the finite term to

```text
Delta_r = ||C_*||_F^2/(2 D sigma^2)
          - r m/{2 D (n-m-1)}
          - epsilon_subspace - epsilon_projection.
```

This gives a smooth signal-to-estimation crossover.

## Conditional-scale headroom

If the correct conditional Gaussian variance is `v_j(A)`, the oracle gain over the best constant variance is

```text
Delta_scale = (1/(2D)) sum_j [log E v_j(A) - E log v_j(A)].
```

This is nonnegative by Jensen and is zero exactly when the conditional variance is constant almost surely.

## Impossibility boundary

- If `E[B|A]=0`, an additive mean coupling has zero population advantage for dependence that exists only in covariance, tails, or multimodal support.
- If the conditional variance is constant, diagonal-affine HSRF has zero scale advantage.
- Dependence outside the hierarchical feature span leaves an irreducible `L2` projection error.
- Dependence outside the selected output rank leaves a singular-value tail error.
- A same-information baseline that copies the complete map ties exactly.

These conditions explain why a global diagnostic can improve while exact NLL, precision/recall, or feature geometry does not.