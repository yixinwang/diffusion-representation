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
