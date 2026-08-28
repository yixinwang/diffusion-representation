# Exact projection-spline derivation and registered falsifier

Let `F_BS` be the unchanged exact data-to-base map and let `b in R^q` be its calibrated detail residual. Let `U in R^{q x r}` have orthonormal columns and write `z=U^T b`, `b_perp=(I-UU^T)b`.

For each selected coordinate, a monotone rational-quadratic spline uses ordered input knots `x_k`, output knots `y_k`, and positive derivatives `d_k`. In bin `k`, let `w=x_{k+1}-x_k`, `h=y_{k+1}-y_k`, `delta=h/w`, and `theta=(x-x_k)/w`. The map is

```
R(x)=y_k+h [delta theta^2+d_k theta(1-theta)] /
              [delta+(d_{k+1}+d_k-2delta)theta(1-theta)].
```

Its derivative is

```
R'(x)=delta^2 [d_{k+1}theta^2+2delta theta(1-theta)+d_k(1-theta)^2] /
       [delta+(d_{k+1}+d_k-2delta)theta(1-theta)]^2 > 0.
```

The inverse is the unique root in `[0,1]` of

```
a theta^2+b theta+c=0,
a=(y-y_k)(d_k+d_{k+1}-2delta)+h(delta-d_k),
b=h d_k-(y-y_k)(d_k+d_{k+1}-2delta),
c=-delta(y-y_k).
```

Outside the frozen tail interval the map and derivative are exactly identity.

The endpoint layer is

```
T_U(b)=b+U(R(U^T b)-U^T b).
```

Completing `U` to an orthogonal basis gives coordinates `(z,w) -> (R(z),w)`. Hence

```
T_U^{-1}(v)=v+U(R^{-1}(U^T v)-U^T v),
log|det DT_U(b)|=sum_j log R'_j(u_j^T b).
```

Composing with `F_BS` gives the exact normalized law

```
log p_R6(x)=log phi(T_U(F_BS(x)))
            +log|det D F_BS(x)|
            +sum_j log R'_j(u_j^T b(x)).
```

No encoder, decoder, VAE, deleted coordinate, variational bound, or unnormalized energy appears.

## Finite non-Gaussian advantage

Assume

```
b=U z+U_perp g,
g~N(0,I),
z_j independent with densities p_j,
```

and each spline is the exact Gaussianizing transport `Phi^{-1} o F_j`. Then the orthogonal complement is already Gaussian and

```
KL(P_b || N(0,I))-KL(P_b || Q_R6)
  = sum_j KL(p_j || phi).
```

For approximate splines with transformed log-density errors at most `epsilon_j` outside exceptional masses `eta_j`, the gain is bounded below by

```
sum_j KL(p_j||phi)-sum_j epsilon_j-2 B_tail sum_j eta_j.
```

This predicts a finite advantage only when projected marginal negentropy exceeds approximation, tail, and selection error.

## No-headroom and first failure conditions

If every selected projected marginal is standard normal, the population-optimal spline is identity and rank zero is optimal after any positive byte charge.

Coordinatewise monotone splines preserve the copula. If

```
p(b) != phi(U_perp^T b) product_j p_j(u_j^T b),
```

then the irreducible error includes the selected-coordinate multi-information plus any non-Gaussian complement error. Global parity/sign laws can have identical one-dimensional marginals while retaining a nontrivial global copula, so independent one-dimensional splines cannot identify them. A learned PCA basis additionally fails when non-Gaussian directions are not isolated by a covariance eigengap.

## Frozen real-data falsifier

Proper energy, feature Frechet, pixel/temporal SWD, precision, recall, support, and variance were primary; exact NLL was secondary. Rank, bins, tails, support thresholds, fitting/search, bytes, training RSS/time, inference latency/peak, and FLOPs were frozen before opened development. Promotion required nonzero rank and a joint image/video quality and systems win over unchanged `B+S`, RCC, random/shuffled/coordinate spline controls, full flow, full VP diffusion, and positive-rate latent flow/diffusion. One domain failure closed confirmation.