# Diffusion representation


## 12/26/2925

suppose x = f(z) for linear f and low dimensional z, what would the diffusion score, or diffusion denoising function or diffusion mean prediction or noise prediction at different time step look like, what about differences across score trajectories, how would the low rank structure look like? what are would diffusion map and LTSA , and Levina Bickel estimator and how would they work in this? how about for quadratic f (nonlinear)? what if it is a low dimensional manifold but may not have global coordinates, how do I get representation?


What if we represent manifold not using global coordinates, but only constraints?

Also use log space smoothing to emphasize support learning?

The causal changes are small and sparse, so the generation of transformation from one image to another maybe described by low dimensional latents., or can conditional on known transfomration of previous frame.


- Use score-induced riemannian metric for kNN, 
- Levina-Bickel intrinsic dim responds to the score-induced metric
- Not sure why LTSA doesn’t work; no low-dimensional structure at all
- Diffusion map has a low-dimensional structure no matter whether we use score or not

Among score statistics

- E[x0 | xt] has the low-dimensional structure instead.

If (x_0=f(z)) (i.e., the data live exactly on the manifold parameterized by (z)), then the posterior mean is “manifold-projected”:
$$\mathbb E[x_0\mid x_t]=\mathbb E[f(z)\mid x_t]=\int f(z),p(z\mid x_t),dz.$$
In DDPM Gaussian corruption,$$p(x_t\mid z)=\mathcal N!\big(x_t;\ \sqrt{\bar\alpha_t},f(z),\ (1-\bar\alpha_t)I\big),]so by Bayes,[p(z\mid x_t)\ \propto\ p(z),\exp!\left(-\frac{1}{2(1-\bar\alpha_t)}\big|x_t-\sqrt{\bar\alpha_t},f(z)\big|^2\right).$$
What “shows up” in (\mathbb E[x_0\mid x_t])
* It is always in the range/convex hull (in expectation) of (f): it’s an average of points on the manifold ( {f(z)}) under the posterior over (z).
* In the small-noise regime ((1-\bar\alpha_t) small), (p(z\mid x_t)) concentrates near[z^*(x_t)=\arg\min_z |x_t-\sqrt{\bar\alpha_t},f(z)|^2 -2(1-\bar\alpha_t)\log p(z),]and then[\mathbb E[x_0\mid x_t]\approx f(z^*(x_t))](plus small curvature/uncertainty corrections). So it behaves like a projection/denoiser onto the manifold.
* If the mapping is non-injective (multiple (z) give similar (f(z))), then (\mathbb E[x_0\mid x_t]) becomes a mixture average across those modes—potentially landing “between” different manifold points (the classic MMSE vs MAP difference).
If you tell me whether you’re thinking discrete (t) DDPM or continuous VP/VE SDE, I can write the exact same story in that notation (it’s identical conceptually; only the noise parameter changes).



- Shall read the REPA paper, prism hypothesis, and the apple alignment paper
- https://arxiv.org/abs/2410.06940
- https://arxiv.org/abs/2512.19693
- https://www.arxiv.org/abs/2512.07829


https://arxiv.org/pdf/2512.20963
https://arxiv.org/pdf/2510.02305


How to use causality to make video diffusion more efficient? How in general can use causality to improve efficiency in learning?