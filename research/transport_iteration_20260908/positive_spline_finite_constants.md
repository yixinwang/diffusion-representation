# Finite constants do not establish a practical separation

The finite learning theorem has a favorable asymptotic exponent, but its constants do not establish the requested practical improvement. This calculation substitutes the actual constants of the smooth cosine-pair example. It uses no observations and selects no experimental hyperparameter.

Take correlation parameter $\rho=0.65$, one context coordinate, two shared groups, failure probability $0.05$, and assume zero empirical optimization error. The theorem has $m=0.35$, $M=1.65$, $H=4\pi^2\rho=25.66097$, $\ell=0.175$, $U=3.3$, approximation constant $A_1=135.36162$, and concentration constant $K=5697.30110$. The restricted independent-coordinate comparator has a lower bound of $0.01600379$ nats per coordinate.

An integer search over all permitted bin counts through 20,000 gives these smallest values of the theorem's displayed upper bound:

- 256 independent arrays: 6,051.42 nats per coordinate, at three bins.
- 4,096 arrays: 948.4433, at five bins.
- 40,000 arrays: 203.6873, at seven bins.
- One million arrays: 23.8612, at eleven bins.
- One billion arrays: 0.26245, at 33 bins.
- One trillion arrays: 0.0029564, at 102 bins.

The density-ratio upper bound $\log(M/\ell)\approx2.244$ improves the largest numbers but still does not fall below the comparator's lower bound. These values are sufficient upper bounds. They do not estimate the sample size an actual fitted model requires. They demonstrate that this proof cannot establish a separation at the proposed development sizes. They do not prove that such a separation is empirically impossible or that tighter analysis cannot establish it.

The companion JSON records every constant and numerical term. The calculation assumes an exact empirical minimizer and fixed correct contexts, so optimization error and context learning cannot rescue its practical certificate. The independent-coordinate comparator is also weaker than a capable latent model. Even a much sharper bound below that floor would leave the main latent-diffusion comparison unresolved.

## Sharper bound checked after the fifth Pro review

The fifth Pro review supplies a substantially smaller excess-risk coefficient through the convex log-loss central condition. With $B=\log(U/\ell)$, use $V_B=B^2/(B-1+e^{-B})$ and coefficient $2V_B+4B/3$. The proof and its assumptions are preserved in `research/transport_iteration5_20260908/LOGLOSS_SHARPENING.md` and were checked independently. This improvement retains complete-array sampling and does not assume a correctly specified conditional model.

For the same cosine example and constants used above, the coefficient falls from 5,697.3011 to 12.58487. Applying the improved bound separately to the two groups with simultaneous coverage and their maximum coefficient count gives 2.858685 nats per coordinate at40,000 arrays, minimized at18 bins within the recorded integer search. The density-ratio bound remains smaller, but neither bound establishes the required0.01600379 separation. Even at100 million arrays the displayed learning bound is0.0188652. These are sufficient-bound calculations with zero optimization error, not required sample-size estimates. They use no data or fitting output.

The exact numbers are in `positive_spline_sharper_constants.json`. Pro5's separate cubic-pair example uses a different correlation, smoothness constant, and joint-per-pair comparison; its numerical threshold is not substituted for this experiment's per-coordinate threshold. The practical gap remains after correcting the original unnecessarily large concentration constant.
