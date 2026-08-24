# Certified adaptive transport-depth routing

Status: **Proposal; derivation independently adversarially audited.** This note states the exact guarantee used to design observed-data Stage B1. It is not a claim of FID/FVD parity, universal latent-method dominance, or VAE representation superiority.

## 1. Generative story and frozen objects

Let blocks `V={1,...,B}` follow a fixed topological order of a directed acyclic graph. Suppress an external condition `c`. The observed data law has proper conditional kernels

`P(dx)=prod_v p_v(dx_v | x_pa(v))`.

Training, calibration clusters, and final test clusters are disjoint. Training returns two frozen proper kernels for each block:

- `q_v^I`: the iterative conditional;
- `q_v^O`: the cheap one-shot conditional.

A frozen route `r in {I,O}^B` defines the joint law

`Q_r(dx)=prod_v q_v^{r_v}(dx_v | x_pa(v))`.

The full comparator is `F=(I,...,I)`. Assume common DAG/conditioning variables; `P << Q_r`; integrable log ratios; identical inclusion of chart Jacobians, dequantization constants, priors, and decoder likelihoods; and independent calibration clusters. Frames, clips, or patches from a common video source are one cluster unless source-level independence is justified.

## 2. Conditional KL decomposition, line by line

Start from the definition,

`KL(P||Q_r) = E_P log {dP(X)/dQ_r(X)}`.

Insert the common DAG factorizations,

`KL(P||Q_r) = E_P log {prod_v p_v(X_v|X_pa(v)) / prod_v q_v^{r_v}(X_v|X_pa(v))}`.

The logarithm turns the product ratio into a sum,

`KL(P||Q_r) = sum_v E_P log {p_v(X_v|X_pa(v)) / q_v^{r_v}(X_v|X_pa(v))}`.

Apply iterated expectation conditional on each parent tuple,

`KL(P||Q_r) = sum_v E_{P_pa(v)} KL(p_v(.|X_pa(v)) || q_v^{r_v}(.|X_pa(v)))`.

Every empirical conditional must therefore be scored at held-out **data parents**. Scoring at generated parents changes the estimand and can hide error where an upstream model avoids difficult parent values.

Subtract the same identity for the full route. The relative regret is

`Delta_r := E_P log {q_F(X)/q_r(X)} = KL(P||Q_r)-KL(P||Q_F)`.

Consequently, `Delta_r <= epsilon` certifies forward-KL regret of at most `epsilon` relative to the full comparator. It does not say that either law is close to truth. If a separate result establishes `KL(P||Q_F)<=eta`, then

`KL(P||Q_r) <= eta + epsilon`.

Exact quality parity requires `eta=epsilon=0`; strict improvement by `gamma` requires `Delta_r <= -gamma`.

## 3. Finite-family calibration certificate

For independent calibration clusters `Z_1,...,Z_n`, define the paired dimension-normalized loss

`D_{i,r} = {-log q_r(Z_i)+log q_F(Z_i)}/N_i`.

The normalization fixes the target: equal-source and equal-frame estimands are different. Let the frozen route family have size `M`. Suppose, conditional on training, `D_{i,r}` is independent over clusters and lies in an interval of known width `R_r`. Define

`hat Delta_r = n^{-1} sum_i D_{i,r}`,

`U_r = hat Delta_r + R_r sqrt{log(M/alpha)/(2n)}`.

For one route, Hoeffding gives

`Pr(Delta_r > U_r) <= exp{-2n(U_r-hat Delta_r)^2/R_r^2} = alpha/M`.

Applying the union bound to all `M` frozen routes yields

`Pr(for all r: Delta_r <= U_r) >= 1-alpha`.

The calibration-dependent selector

`hat r in argmin_{r: U_r<=epsilon} C(r)`

therefore satisfies, with probability at least `1-alpha`,

`KL(P||Q_hat_r) <= KL(P||Q_F)+epsilon`.

If the eligible set is empty, the algorithm abstains and uses `F`. Gaussian log-score differences are unbounded, so the Hoeffding form is not automatically available. The executable must instead freeze one valid choice: a clipped-score estimand, justified sub-Gaussian/sub-exponential bounds, a valid confidence sequence, or another assumption-backed cluster bound. Ordinary Student or bootstrap intervals are descriptive unless their assumptions prove the advertised coverage.

## 4. Blockwise certificate

Define a paired block loss and its expectation,

`d_{i,v}=N_i^{-1} log{q_v^I(X_iv|X_i,pa(v))/q_v^O(X_iv|X_i,pa(v))}`,

`delta_v=E d_{i,v}`.

If `O(r)` is the set of one-shot blocks, direct cancellation in the joint log ratio gives

`Delta_r=sum_{v in O(r)} delta_v`.

If simultaneous bounds `delta_v<=u_v` hold for every block, then every adaptively selected route satisfies

`Delta_r <= sum_{v in O(r)} u_v`.

This replaces a union bound over routes by simultaneous block bounds but can be conservative because uncertainty radii are summed. Marginal 95% intervals cannot simply be added.

## 5. Compute result

Under an explicitly additive work model,

`C(r)=C_chart+C_router+C_shared+C_decode + sum_{r_v=I} K_v c_v^I + sum_{r_v=O} c_v^O`.

Subtracting the routed cost from the full route cancels every shared term,

`C(F)-C(r)=sum_{r_v=O}(K_v c_v^I-c_v^O)`.

Thus routing is strictly cheaper if every routed block has `c_v^O<=K_v c_v^I` and at least one inequality is strict. If a matched latent comparator belongs to the candidate family, passes the same quality certificate, and costs are known, minimum-cost selection gives `C(hat r)<=C(r_latent)`. For measured latency the analogous evidence needs simultaneous cost bounds, such as `U_C(hat r)<=L_C(r_latent)`; critical-path latency need not be additive.

## 6. Algorithm

1. Fit every iterative and one-shot conditional on training clusters only.
2. Freeze the DAG, models, route family, score normalization, tolerance, cluster definition, compute accounting, and stopping rules.
3. Evaluate every conditional on calibration data at data parents.
4. Form paired joint or blockwise log-score differences against the full route.
5. Construct a valid simultaneous one-sided upper bound.
6. Mark a route eligible only when its bound is at most the frozen tolerance.
7. Abstain to the full route if no cheap candidate is eligible.
8. Otherwise select the minimum-cost eligible route using a frozen tie rule.
9. Freeze the chosen route and implementation.
10. Generate in topological order with the selected conditional for each block.
11. Open the test set once and report proper scores, endpoint metrics, latency, memory, and all failures.

Any extra adaptive round needs a new independent calibration split or a formally valid reusable-holdout/sequential method.

## 7. Consequence and exact boundary

If `KL(P||Q_F)<=eta` and `Delta_hat_r<=epsilon`, Pinsker's inequality gives

`TV(P,Q_hat_r) <= sqrt{(eta+epsilon)/2}`.

For a bounded score `s in [m,M]`,

`|E_P s-E_Q s| <= (M-m) sqrt{(eta+epsilon)/2}`.

FID, FVD, and unbounded feature moments do not follow without separate moment and continuity assumptions. A same-information baseline with the same chart, kernels, noise, schedules, decoder, and accounting is the same measurable map and must tie. Any strict claim must name the restricted decoder, solver, route, tokenizer, or compute budget that creates the separation.

## 8. Adversarial checks that constrain implementation

- Duplicate frames do not create independent calibration units; use images or source-video groups.
- Generated-parent scoring can conceal an arbitrarily bad downstream conditional.
- Creating candidates after viewing calibration data invalidates the finite-family bound.
- Near-zero Gaussian variance makes unbounded NLL tails invalidate Hoeffding.
- A weak full comparator can be far from truth even when relative regret is nonpositive.
- Cyclic separately normalized conditionals need not define a coherent joint.
- Diffusion losses, ELBOs, and block-normalized scores are not automatically joint NLLs.
- Excluding the chart, router, decoder, data movement, or variable solver evaluations can reverse a compute comparison.
