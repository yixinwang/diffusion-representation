# RMPF-R7 exact derivation and independent audit

## 1. One coherent normalized flow

Let the unchanged exact R4/B+S map send an observation `x` to standardized coordinates `u=(a,b)`, where `a` is the retained coarse state and `b in R^q` contains every Haar detail coefficient. No coordinate is removed.

Let `U in R^{q x r}` contain orthonormal selected DCT-II columns and let `H in R^{r x r}` be a normalized Walsh-Hadamard butterfly. Define

```
p = H U^T b = (c,t),
```

with `c,t in R^{r/2}`. The balanced global state is

```
s(c) = 1{ product_i sign(c_i) > 0 }.
```

For target coordinate `j`, parameters are `(mu_{j,s}, sigma_{j,s})`, `sigma>0`. The data-to-base map is

```
y_c = c,
y_{t,j} = (t_j - mu_{j,s(c)}) / sigma_{j,s(c)},
b' = b + U H^T (y-p).
```

The full data-to-base flow is the composition of the unchanged B+S map and this endpoint map. The source is the same full-dimensional standard Gaussian used by every exact-flow control.

## 2. Exact inverse

Because the conditioner block is unchanged, `c=y_c` is available to the inverse. Therefore

```
t_j = y_{t,j} sigma_{j,s(y_c)} + mu_{j,s(y_c)},
b = b' + U H^T (p-y).
```

The DCT columns and Hadamard matrix are orthonormal, so `U^T U=I` and `H^T H=I`. The low-rank correction modifies exactly the selected subspace and leaves its orthogonal complement unchanged. Hence the displayed inverse is exact.

The numerical audit found maximum round-trip error `4.44e-16` for the selected-DCT implementation and maximum disagreement `2.22e-15` relative to a full orthonormal DCT implementation.

## 3. Exact Jacobian and likelihood

Away from the sign hyperplanes, `s(c)` is locally constant. In `(c,t)` coordinates,

```
D(c,y_t)/D(c,t) = [[I, 0], [*, diag(sigma_{j,s}^{-1})]].
```

Thus

```
log |det J_R7| = - sum_j log sigma_{j,s(c)}.
```

DCT, Hadamard, and the orthogonal-complement identity have determinant magnitude one. If `F_BS` is the unchanged exact data-to-base map and `ell_BS(x)` its exact log-Jacobian, then

```
log p_R7(x)
 = -0.5 ||F_R7(x)||^2 - D/2 log(2 pi)
   + ell_BS(x) - sum_j log sigma_{j,s(c)}.
```

The sign hyperplanes have Lebesgue measure zero and do not affect normalization. Finite-difference log-determinant error in the registered known-truth run was at most `3.36e-9`.

## 4. Finite global-copula advantage

Assume the selected target block satisfies

```
C ~ p_C,
T_j | S=s ~ N(mu_{j,s}, sigma_{j,s}^2),
T_j independent across j conditional on S=s(C),
```

and the orthogonal complement is standard normal and independent. R7 is then exact on the selected joint law.

Let the strongest coordinatewise baseline fit the exact marginal density `p_j(t_j)` but ignore `S`. The population log-score advantage of the exact coupled law is

```
E log p(T|S) - E log product_j p_j(T_j)
 = sum_j I(T_j;S),
```

because the targets are conditionally independent given the common state. This is strictly positive whenever any conditional law differs across states.

For a balanced binary state and well-separated conditional normals, `I(T_j;S)` approaches `log 2`. With four affected coordinates and `D=32`, the experiment-aligned ceiling is

```
4 log(2) / 32 = 0.08664 nat per visible dimension.
```

The executed C2 known truth obtained `0.09247 [0.09227,0.09266]` nat/dimension over the coordinatewise R6 spline; the small excess over the simple ceiling reflects additional marginal-scale mismatch removed by the coupled Gaussian fit.

With finite samples, each state estimates one mean and one variance. Writing `n_min` for the smaller state count, the expected plug-in penalty is `O(r/n_min)`. Therefore a sufficient finite noninferiority condition is

```
sum_j I(T_j;S) > C r / n_min + approximation_error + tail_error.
```

## 5. No-headroom and impossibility conditions

There is no headroom if, for every selected target coordinate,

```
T_j | S=0  =_d  T_j | S=1 =_d N(0,1).
```

The fitted layer then shrinks to identity.

The first projection-factorization failure occurs when any material dependence is:

1. outside `span(U)`;
2. in the unchanged orthogonal complement;
3. not measurable by the single parity state `S`;
4. conditionally non-Gaussian or multimodal after conditioning on `S`;
5. stage-aligned inside the iterative trunk rather than removable at the endpoint.

Even parity on more coordinates than the conditioner state can summarize is a lower-bound witness. A finite state cannot distinguish all global configurations; a copied full model or a richer global state is then required.

## 6. Compute and memory

The selected DCT implementation computes only `bU` and the low-rank correction; it does not materialize a full DCT. For batch size `N`,

```
C_A = O(N q r + N r log r),
workspace_A = O(qr + Nr),
learned_bytes_A = O(r),
```

where the deterministic DCT columns are generated from the stored integer indices. A dense full residual update costs at least `Omega(N q h)` per stage and is repeated across NFE.

The endpoint layer itself passed the intended latent-like oracle:

- image endpoint/full batch latency ratio: `0.0616`;
- video endpoint/full batch latency ratio: `0.0576`;
- image incremental FLOP ratio: `0.0293`;
- video incremental FLOP ratio: `0.0147`;
- learned stored-state ratios below `0.0002`.

However the total unchanged B+S trunk remained the bottleneck. Even after an algebraically exact collapsed affine realization, total candidate/full latency ratios were `5.238` for images and `3.531` for video. The video deterministic-basis workspace ratio was also `0.118`, above the frozen `0.10` limit.

## 7. Independent audit conclusions

- The inverse and determinant are exact almost everywhere.
- The known-truth advantage is a genuine copula-information term, not marginal leakage.
- The initial sequential construction failed because greedy ordering destroyed state balance; the block construction repaired exactly that premise and passed.
- The cost blocker is independent of the scientific endpoint: the endpoint itself is cheap, but the preserved B+S trunk is already several times slower than the strongest retained full flow.
- Opening CIFAR/UCF R7 outcomes after the failed cost gate would violate the preregistered `then and only then` rule. No such outcomes were opened.
- A further small endpoint modification cannot repair the total systems ratio. Reopening requires a new trunk realization or a different multiresolution family, not another copula endpoint.