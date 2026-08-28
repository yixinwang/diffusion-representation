# Reversible Multirate Pyramid Flow: exact finite model, bounds, and adversarial audit

Frozen known-truth design: see `KNOWN_TRUTH_PREREGISTRATION.md` and amendment S1. This note is written after the frozen confirmation only to assemble derivations already encoded and tested; it does not change any gate.

## 1. Model and exact target

Let `D=2d` and let `H in R^{D x D}` be an orthonormal Haar analysis matrix. For a standard normal source `(Z_A,Z_B) ~ N(0,I_d) x N(0,I_d)`, define a coordinatewise bijection

\[
  g_\tau(z)=\frac{\sinh(\tau z)}{\tau},\qquad
  g_\tau^{-1}(x)=\frac{\operatorname{asinh}(\tau x)}{\tau}.
\]

Write `A_0=g_tau(Z_A)` and `B_0=g_tau(Z_B)`. At stage `k=0,...,K-1`,

\[
 A_{k+1}=s_k\odot A_k+c_k,
\]

with all coordinates of `s_k` positive. The detail state is updated by an additive triangular transport

\[
 B_{k+1}=B_k+h\{\ell_k(A_{k+1})+m_k(A_{k+1})\}.
\]

The local term has four per-coordinate features,

\[
 \ell_{k,j}(a)=\beta_{k,j,0}+\beta_{k,j,1}\tanh(a_j)
 +\beta_{k,j,2}\tanh(a_{j+1})+\beta_{k,j,3}a_j,
\]

where indices wrap cyclically. The structured global term is present only at controller-selected stages. Its rank-`r` feature is a balanced binary tree. For each feature `q`, initialize

\[
 h^{(0)}_{q,j}=\tanh\{\gamma \sigma_{q,j}a_{\pi_q(j)}\},
\]

then recursively communicate

\[
 h^{(l+1)}_{q,j}=\tanh\{\eta h^{(l)}_{q,2j}h^{(l)}_{q,2j+1}\},
\]

until one scalar `phi_q(a)` remains. Thus

\[
 m_k(a)=W_k\phi(a),\qquad \phi(a)=(\phi_1(a),\ldots,\phi_r(a))^\top.
\]

Every visible coordinate is retained:

\[
 X=H^\top(A_K,B_K).
\]

There is no encoder, decoder, stochastic bottleneck, reconstruction loss, KL regularizer, discarded coordinate, route selector, or unnormalized endpoint law.

## 2. Exact invertibility

Because each `s_k` is positive,

\[
 A_k=(A_{k+1}-c_k)\oslash s_k.
\]

Given the recovered coarse trajectory, the detail inverse is

\[
 B_0=B_K-h\sum_{k=0}^{K-1}\{\ell_k(A_{k+1})+m_k(A_{k+1})\}.
\]

Finally apply `g_tau^{-1}` and `H`. Hence the inverse exists globally and is explicit. The implementation tests round-trip error below `1e-10`; the observed maximum is reported in the test log.

## 3. Exact Jacobian and density

Order transformed coordinates as `(A,B)`. One stage has Jacobian

\[
 J_k=\begin{bmatrix}
 \operatorname{diag}(s_k)&0\\
 hD\{\ell_k+m_k\}(A_{k+1})\operatorname{diag}(s_k)&I
 \end{bmatrix}.
\]

It is block triangular, so

\[
 \det J_k=\prod_{j=1}^d s_{k,j}.
\]

The additive local/global update contributes zero log-determinant. Haar is orthonormal and has absolute determinant one. The scalar texture map contributes

\[
 \log |g_\tau'(z)|=\log\cosh(\tau z).
\]

Therefore, for the recovered source `z=(z_A,z_B)`,

\[
 \log p_X(x)=-\frac12\|z\|^2-\frac D2\log(2\pi)
 -\sum_i\log\cosh(\tau z_i)
 -\sum_{k,j}\log s_{k,j}.
\]

This is the exact likelihood used by every fitted candidate. An autograd dense-Jacobian oracle independently checks the analytic log determinant.

## 4. Exact comparison target and attributable component

Let `B` be a full repeated global model. Let `S` be the same orthonormal pyramid, source, stages, local update, fitting data, and schedule, but no structured tree communication. Let `A` be only the learned tree global term and the frozen multirate controller. Thus:

- `B+S`: `no_global`;
- `B+S+A`: `rmpf`;
- zero `A`: no global weights;
- random `A`: random Fourier features with learned output;
- frozen `A`: random feature output frozen;
- shuffled `A`: correct tree feature paired with a shuffled response;
- generic matched `A`: parameter-matched random features;
- local-fiber comparator: `mcqf_local_fiber`;
- strongest same-information global controls: validation-selected dense attention and RFF;
- unrestricted equivalence control: exact RMPF copy.

The estimand is the paired confirmation difference in normalized negative log likelihood and proper energy score, with the systems ratios measured in the same process.

## 5. Finite experiment-aligned advantage

Suppose the target detail shift is

\[
 q(a)=q_0(a)+W\phi(a),
\]

where `q_0` lies in the local class and the columns of `W` are orthogonal. Under the exact unit-variance additive source, fitting a model that omits `W phi` gives conditional Gaussian KL

\[
 \operatorname{KL}\{N(q(a),I)\|N(q_0(a),I)\}=\frac12\|W\phi(a)\|^2.
\]

Averaging and normalizing by `D`,

\[
 \Delta_{\mathrm{NLL}}=\frac{h^2}{2D}\operatorname{tr}\{W^\top W\,\mathbb E[\phi\phi^\top]\}.
\]

For global strength `alpha`, `W=alpha W_0`, so the leading gain is exactly quadratic:

\[
 \Delta_{\mathrm{NLL}}(\alpha)=\alpha^2\Delta_{\mathrm{NLL}}(1).
\]

This predicted the observed smooth crossover: the registered sweep rose from approximately zero at strength zero to `0.000447, 0.00675, 0.02574, 0.05458, 0.13224` at strengths `1,4,8,12,20`. Strength 12 was frozen before confirmation because it crosses the unchanged practical margin `0.05`.

If the fitted mixer retains only rank `r<r_*`, and the omitted singular values of the target global map are `sigma_{r+1},...`, the irreducible conditional KL is

\[
 \Delta_r\ge \frac{h^2\lambda_{\min}(\Sigma_\phi)}{2D}
 \sum_{j>r}\sigma_j^2.
\]

This gives a quantitative rank transition and is the experiment-aligned lower bound tested in `controlled_rank_sweep.csv`.

## 6. Compute and memory bounds

Let the coarse/detail width be `d`, detail output width `b`, global rank `r`, total stages `K`, and global stages `G`. A local update costs `Theta(Kbd)` only through fixed-width pointwise features; in the implementation it is `8Kbd` scalar operations per batch item. A balanced tree feature costs `Theta(rd)` and its projection costs `Theta(rb)` per global stage. Hence

\[
 C_{\mathrm{RMPF}}=\Theta\{K b+G r(d+b)+D\log D\}.
\]

A dense attention global update costs `Theta(d^2+db)` per head per selected stage:

\[
 C_{\mathrm{full}}=\Theta\{G H(d^2+db)+Kb+D\log D\}.
\]

For fixed `G,H,r` and growing `d`, the global communication ratio is `O(r/d)`. The full model is recovered monotonically by increasing tree rank to a complete basis, selecting all stages, and replacing structured mixing by the unrestricted full global block.

Streaming generation stores only `(A,B)`, two local temporaries, and the largest global feature buffer:

\[
 M_{\mathrm{RMPF}}=O\{D+r\}\quad\text{per sample},
\]

whereas dense attention materializes `O(d^2)` scores. Reversible training can recompute stage activations, giving `O(D+r)` activation memory instead of `O(KD)` checkpoint storage, at the declared recomputation cost. Stored parameter bytes are

\[
 P_{\mathrm{RMPF}}=O(Kb+rGb),\qquad P_{\mathrm{attention}}=O(Kb+HGdb)
\]

for this finite implementation. All routing, Haar, mixer, schedule, and serialization bytes are charged.

The prespecified break-even condition is

\[
 G\{C_{\mathrm{full,global}}-C_{\mathrm{tree}}\}
 > C_{\mathrm{controller}}+C_{\mathrm{Haar}}+C_{\mathrm{fusion}}.
\]

Confirmation measured batch latency ratio `0.1076 [0.0939,0.1214]`, peak-memory ratio `0.2145`, and stored-byte ratio `0.3152 [0.3151,0.3154]` against the validation-selected dense-attention global control.

## 7. Transport and proper-score perturbation bound

Let `T_*` be the target endpoint and `T_hat` the fitted endpoint under the same source. Decompose per-stage detail velocity error into local, global-mixer, schedule/discretization, and optimization terms:

\[
 e_k(a)=e_{\ell,k}(a)+e_{m,k}(a)+e_{s,k}(a)+e_{o,k}(a).
\]

The coarse affine map is exact in the known-truth experiment. For a nonlinear realistic model with stage Lipschitz factors `L_k`, define the downstream amplification

\[
 \Gamma_k=\prod_{j>k}(1+hL_j).
\]

A common-source coupling gives

\[
 W_2(P_*,P_{\hat T})
 \le \epsilon_{\mathrm{chart}}+\epsilon_{\mathrm{coarse}}
 +h\sum_{k=0}^{K-1}\Gamma_k
 \left(\epsilon_{\ell,k}+\epsilon_{m,k}+\epsilon_{s,k}+\epsilon_{o,k}\right).
\]

For an energy score on a bounded domain of diameter `B`, the expected score regret is at most a constant multiple of this `W_1<=W_2` bound. For exact likelihood, triangular Gaussian detail shifts give the sharper identity

\[
 \mathbb E[-\log \hat p+\log p_*]
 =\frac{1}{2D}\sum_k h^2\mathbb E\|e_k(A_{k+1})\|^2
\]

when stage residual components are orthogonal; otherwise the squared norm of the accumulated shift replaces the sum.

The realistic transfer error is the sum of: non-affine coarse approximation; finite global tree rank; failure of the selected schedule to remain stable; numerical solver error; optimizer error; and hardware fusion overhead. The first condition that invalidates the known-truth mapping is a global dependence functional outside the span/communication graph of the structured mixer at the frozen rank and schedule.

## 8. Communication lower bound and hidden global copula

Let detail coordinates be vertices of a communication graph. After `G` global rounds, an output can depend only on coordinates in its `G`-hop receptive set unless a global summary crosses that cut. Construct two source laws with identical marginals on every such set but opposite parity across two disconnected components. Every local or insufficient-rank model produces the same conditional output law under both targets. Le Cam's two-point argument then gives minimax testing error at least `1/2` and nonzero endpoint risk.

For the even-parity distribution on `{-1,+1}^m` versus the independent product law, every proper-subset marginal is identical, including all pairwise covariances, while

\[
 \operatorname{TV}=1/2,\qquad \operatorname{KL}=\log 2,\qquad W_2=\sqrt2.
\]

Thus pairwise covariance and independent local fibers are information-theoretically insufficient. A rank-`r` mixer also fails whenever the target communication tensor has stable rank greater than `r`; the omitted-singular-value KL lower bound above quantifies the failure.

The balanced tree has logarithmic path length and explicitly computes parity-like high-order products. It therefore distinguishes the previous MCQF global-sign adversary. Confirmation shows MCQF bit-error exceeds RMPF by a positive paired interval, while pairwise correlations remain small.

## 9. Noninferiority condition

Let the full model endpoint error be `E_full` and its measured cost be `C_full`. Let the RMPF error decomposition above sum to `E_R`. For quality margin `delta` and efficiency ratio `rho<1`, sufficient conditions are

\[
 E_R\le E_{\mathrm{full}}+\delta,
 \qquad C_{\mathrm{RMPF}}\le \rho C_{\mathrm{full}},
\]

plus no worse registered precision/recall/diversity or catastrophic-failure rates. In the exact known-truth Gaussian-shift model, the full model and RMPF share the same coarse map; if the true mixer lies in the tree span and the local regressions are consistent, `E_R` tends to zero. The full model cannot be strictly beaten by an unrestricted equivalence control, which exactly copies RMPF and ties.

## 10. Confirmation result

Across untouched seeds `9100--9104`:

- no-global minus RMPF NLL: `0.05296 [0.05173,0.05419]` nat/dimension;
- no-global minus RMPF global-bit error: `0.32570 [0.30553,0.34587]`;
- full-attention minus RMPF proper energy score: `0.004286 [0.002411,0.006160]`;
- RMPF/full-attention batch latency: `0.10764 [0.09386,0.12143]`;
- peak-memory ratio: `0.21452`;
- stored-byte ratio: `0.31522 [0.31506,0.31538]`;
- equivalence-copy mismatch: exactly zero.

All frozen known-truth gates passed. This promotes the mechanism only to genuine-data development, not to real-data confirmation or the project-level promotion claim.
