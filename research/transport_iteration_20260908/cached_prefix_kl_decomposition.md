# Cached-prefix factorization and the full-density KL budget

Fix a fitted invertible analysis map $A$ and fitted decoder functions. The decomposition below separates four population errors: root modeling, conditional dependence omitted within blocks, information discarded by summaries, and scalar conditional-density fitting. A tractable normalized decoder alone does not make any of those errors small.

## Model and measure assumptions

Let $Y$ be a fresh observation and write

$$
(C,R_1,\ldots,R_B)=A(Y).
$$

Each residual block $R_b=(R_{b1},\ldots,R_{bm_b})$ is a vector. Its complete preceding history is

$$
H_b=(C,R_1,\ldots,R_{b-1}).
$$

Let $S_{bj}=s_{bj}(H_b)$ be a deterministic summary of that complete history. The fitted density in analysis coordinates is

$$
q(c,r_1,\ldots,r_B)=q_C(c)
\prod_{b=1}^B\prod_{j=1}^{m_b}
q_{bj}(r_{bj}\mid s_{bj}(h_b)).
$$

The decoder is conditionally independent **within each block** given its preceding history and uses only the corresponding summaries. At generation the history consists of generated values. Observed preceding values can be used to evaluate training or population likelihood, but must not replace generated history during sampling.

The map $A$ is a measurable bijection with measurable inverse; a differentiable flow with nonzero Jacobian is a sufficient density-level example. Spaces are standard Borel so regular conditional laws exist. State the equality first when all relevant KL quantities are finite and conditional densities exist relative to appropriate product reference measures. The same chain-rule statement extends through conditional relative entropy to extended nonnegative values. In particular, a singular true conditional against a decoder density can produce infinite error; formal density-log manipulations must not assume it away.

Denote the pushforward true law by $P$. Conditional mutual information and total correlation below are always under this true law in the fixed analysis coordinates. Define

$$
\operatorname{TC}(R_b\mid H_b)
:=E_{H_b}\operatorname{KL}\left(
P_{R_b\mid H_b}\,\middle\|\,
\bigotimes_jP_{R_{bj}\mid H_b}\right).
$$

This convention already averages over the history. If conditional total correlation is instead written as a function of a fixed history, the outer expectation must be included.

## Exact decomposition

With the preceding conventions,

$$
\boxed{\begin{aligned}
\operatorname{KL}(P_Y\|Q_Y)
={}&\operatorname{KL}(P_C\|q_C)\\
&+\sum_b\operatorname{TC}(R_b\mid H_b)\\
&+\sum_{b,j}I(R_{bj};H_b\mid S_{bj})\\
&+\sum_{b,j}E_{S_{bj}}\operatorname{KL}
\left(P_{R_{bj}\mid S_{bj}}\|q_{bj}(\cdot\mid S_{bj})\right).
\end{aligned}}
$$

All terms are nonnegative. No independence of pixels or independent sites within a training array is assumed for this identity. Conditional independence is a restriction of the fitted decoder, and its mismatch appears explicitly as total correlation.

### Proof

Invariance of KL under $A$ and the blockwise chain rule give

$$
\operatorname{KL}(P_Y\|Q_Y)
=\operatorname{KL}(P_C\|q_C)
+\sum_bE_{H_b}\operatorname{KL}
\left(P_{R_b\mid H_b}\,\middle\|\,
\bigotimes_jq_{bj}(\cdot\mid S_{bj})\right).
$$

Insert the product of true full-history scalar conditionals inside each block's log density ratio. Integration separates the joint dependence term and scalar terms:

$$
E_{H_b}\operatorname{KL}
\left(P_{R_b\mid H_b}\,\middle\|\,
\bigotimes_jq_{bj}(\cdot\mid S_{bj})\right)
=\operatorname{TC}(R_b\mid H_b)
+\sum_jE_{H_b}\operatorname{KL}
\left(P_{R_{bj}\mid H_b}\|q_{bj}(\cdot\mid S_{bj})\right).
$$

For one scalar coordinate, insert $P_{R_{bj}\mid S_{bj}}$. Because $S_{bj}$ is a deterministic function of $H_b$, conditioning on both is the same as conditioning on $H_b$. The first resulting term is

$$
E\log\frac{p(R_{bj}\mid H_b)}{p(R_{bj}\mid S_{bj})}
=I(R_{bj};H_b\mid S_{bj}).
$$

The second term depends only on $(R_{bj},S_{bj})$, so averaging histories gives

$$
E\log\frac{p(R_{bj}\mid S_{bj})}{q_{bj}(R_{bj}\mid S_{bj})}
=E_{S_{bj}}\operatorname{KL}
\left(P_{R_{bj}\mid S_{bj}}\|q_{bj}(\cdot\mid S_{bj})\right).
$$

Summing proves the formula. These conditional relative-entropy chain rules also establish the nonnegative extended-value version, without subtracting infinite entropies.

## What the history and cache must contain

The displayed $H_b$ is the entire preceding coordinate prefix. A cache that is merely a deterministic function of that prefix is another summary, not automatically the complete history. One may still define each $S_{bj}$ as a composition of the cache and scalar-summary functions; the formula remains valid with $H_b$ as the full prefix. Information omitted by both cache and summary then appears in $I(R_{bj};H_b\mid S_{bj})$.

If one relabels a lossy cache as $H_b$ and applies the chain rule as though it were the complete prefix, an additional omitted-history error can disappear incorrectly. Its disappearance requires a proved conditional-sufficiency assumption. Also, summaries cannot use observed coordinates from the current block while retaining a within-block product decoder. If within-block preceding coordinates are used sequentially, split the block into scalar steps or change the factorization; the relevant conditional total-correlation term changes with that model.

Even summaries that retain all history cannot eliminate a nonzero block total-correlation term. Conversely, singleton blocks have zero within-block total correlation, but can still discard predictive history through their summaries.

## Conditioning on fitted training information

Let $\mathcal T$ include the training arrays, all optimization randomness, model selection, fitted $A$, and the fitted summaries and densities. Assume a fresh observation $Y$ independent of $\mathcal T$ from the target law. Conditional on $\mathcal T=t$, the map and fitted functions are fixed. Apply the identity to the pushforward law $P^{(t)}=A_t\#P_Y$ and the fitted $Q_t$.

Taking expectations over training gives the corresponding expected-risk equality, with every population term evaluated under its training-dependent pushforward law. A high-probability upper bound for the full KL requires bounds on the sum of all four terms, under the same fitted model. Separate component events can be combined by a union bound; an in-sample loss estimate is not such an event. The expectation identity does not permit replacing a random learned analysis by a fixed-analysis theorem without proving the needed uniformity, conditioning, or sample splitting.

An independent evaluation sample permits evaluation conditional on the fitted training information. Repeatedly selecting models with that evaluation sample changes the conditioning and requires adaptive-inference control or an untouched sample for confirmation. Training-site pooling does not multiply the number of independent full-array sampling units.

## Stochastic-summary nuance

If a summary is random, let a fixed kernel $K(ds\mid h)$ generate it from the history. Suppose its randomness is independent of the current response conditional on the history. Augment the true distribution with that same kernel. The marginal fitted conditional decoder is

$$
\overline q(r\mid h)=\int q(r\mid s)K(ds\mid h).
$$

Comparing **augmented** joint laws with the same summary kernel gives

$$
E_{H,S}\operatorname{KL}(P_{R\mid H}\|q(\cdot\mid S))
=I(R;H\mid S)+E_S\operatorname{KL}(P_{R\mid S}\|q(\cdot\mid S)).
$$

Here the equality uses $R\perp S\mid H$ under the augmented true law. Marginalizing the summary contracts KL, so the output-density error obeys

$$
E_H\operatorname{KL}(P_{R\mid H}\|\overline q(\cdot\mid H))
\le E_{H,S}\operatorname{KL}(P_{R\mid H}\|q(\cdot\mid S)).
$$

Thus the augmented identity generally supplies an **upper bound**, rather than the deterministic equality, for the marginal decoder. Independent fresh summary draws per scalar coordinate preserve a product of mixed scalar conditionals. A shared random summary can itself couple coordinates, so the original product-factorization identity no longer describes that marginal block density without modification. If the summary uses information from the current response unavailable at generation, even the Markov condition fails.

Stochastic summary noise is an additional source of generation randomness unless it is included in, or deterministically derived from, the declared source budget. A conditioning argument does not make those coordinates free.

## Why good conditional fit does not establish sufficient summaries

The scalar fitting term is minimized by $q_{bj}=P_{R_{bj}\mid S_{bj}}$. It can equal zero while the discarded-history information is positive. For example, a constant summary can fit the true marginal perfectly even when the residual depends strongly on the preceding history. Likewise, perfect scalar conditionals cannot repair within-block dependence omitted by a product decoder.

A correctly evaluated independent full joint log likelihood does include these penalties implicitly, but its absolute value does not identify KL without the true-law entropy or a justified comparison. Good agreement on a finite evaluation sample alone does not upper-bound the unobserved population terms, especially when log densities are unbounded or rare dependence patterns are absent from that sample.

There is one useful comparison identity with explicit limits. For any reference full-history conditional density $g(r\mid h)$ and any summary density $q(r\mid s)$, the population loss difference is

$$
E[-\log q(R\mid S)]-E[-\log g(R\mid H)]
=I(R;H\mid S)
+E_S\operatorname{KL}(P_{R\mid S}\|q)
-E_H\operatorname{KL}(P_{R\mid H}\|g).
$$

If an upper bound $\varepsilon_g$ on the reference's population KL is available, then discarded-history information is bounded above by the population loss difference plus $\varepsilon_g$. Without that reference-error bound, a small loss difference does not show sufficient summaries: both predictors may miss the same information. A confidence bound for the population loss difference is also needed when replacing it by an empirical estimate. It requires an appropriate tail or boundedness condition and independent full-array evaluation units.

The general bounds $I(R;H\mid S)\le I(R;H)$ and, when the relevant information is finite, $I(R;H\mid S)=I(R;H)-I(R;S)$ follow because $S$ is deterministic from $H$. They do not make the information loss small. Continuous variables have no universal finite mutual-information bound merely from coordinate count or a bounded observation cube; differential entropy can be arbitrarily negative and mutual information can diverge. A finite summary alphabet limits its retained information from above but supplies no general upper bound on information discarded by that summary.

These qualifications leave the decomposition useful as a target for a learning theorem. They prevent normalized densities, low scalar optimization gaps, or good reused evaluation fits from being promoted into unproved representation-sufficiency or full-model quality claims.
