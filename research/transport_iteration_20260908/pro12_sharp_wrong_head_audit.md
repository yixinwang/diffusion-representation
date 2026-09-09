# Sharper wrong-head correction using the reviewed Arb enclosures

This is a new analytical correction applied to the already reviewed Pro10 Arb receipts. It does not verify, import or rely on any pending Pro12 MPFR receipt. The original conservative correction, frontier outputs and underlying enclosures remain unchanged. No integration, fit, model evaluation, PSC job or runtime measurement was performed.

## Scalar change of measure

For e,f in[.4,.6], write p_e(y)=phi(y)[1+e psi(y)], psi(y)=2Phi(y)-1 in[-1,1]. The density ratio obeys p_e/p_f<=3/2: the largest negative-feature endpoint ratio is(.6)/(.4)=3/2, while the positive-feature endpoint ratio is1.6/1.4<3/2. Fractional-linear monotonicity in psi and the endpoint choices of e,f establish the uniform bound.

Also, by KL<=chi-square and E_phi psi^2=1/3,

KL(p_e||p_f) <=(e-f)^2 E_phi[psi^2/(1+f psi)]
 <=(.2)^2/[3(1-.6)]=1/30.

Let q_f be the output density of the fixed finite-Heun sampler targeting p_f, and let k_f=KL(p_f||q_f). It is positive and normalized by the previously established global Heun bijection/density result. Put L=log(p_f/q_f). On the set p_f<q_f, log(q_f/p_f)<=q_f/p_f-1, hence

integral p_f L_- <=integral_{p_f<q_f}(q_f-p_f)=TV(p_f,q_f).

The convention is TV=sup_A|P(A)-Q(A)|=one-half integral|p-q|. It follows that

integral p_f L_+ =k_f+integral p_f L_- <=k_f+TV(p_f,q_f)
 <=k_f+sqrt(k_f/2).

Dropping the negative part under p_e and using p_e<=1.5p_f therefore gives

KL(p_e||q_f)=KL(p_e||p_f)+E_{p_e}L
 <=1/30+1.5[k_f+sqrt(k_f/2)].

The signs and direction are forward KL throughout. This argument is valid without a triangle inequality for KL, which would not be available.

## Correct-root, wrong-index event

Condition on all training randomness and on correct root-sign estimation. A wrong learned head is now one fixed public orthonormal dictionary row. On a fresh root array, every such row still gives a standard Gaussian projection and hence a uniform transformed feature. Its fitted scalar tilt f(C) is uniform on[.4,.6]. Selecting the row using independent training arrays does not change this fresh-array marginal. True e(C) and fitted f(C) need not be treated as independent for the bound.

Let m=2880 and b be an upper bound on the reviewed oracle joint residual risk m E_f k_f for this call count. Sum the scalar inequality and apply Jensen to the square root:

W_N(b)=m/30+1.5b+1.5 sqrt(mb/2)
 =96+1.5b+1.5 sqrt(1440b).

This is a uniform bound on expected wrong-head risk conditional on any correct-root/wrong-index training outcome. The public-row uniformity is essential: if the learned head can produce an arbitrary f(C) distribution, an average oracle risk b alone does not give this correction without additional control. The wrong-root event does not have this uniformity, so retain the older valid17408 bound rather than extending Jensen incorrectly.

With prior finite-learning bounds p_g and p_j (the latter conditional on a correct root), and reviewed oracle lower/upper bounds a,b, the full expected learned-Heun risk is enclosed by

(1-p_g-p_j)a <=E_training KL(P||Q_hat)
 <=b+p_j W_N(b)+17408p_g.

The lower bound uses only the correct-root/correct-index event and nonnegativity elsewhere. The upper bound conservatively pays b even without multiplying by the success probability. These statements concern fresh-array population KL conditioned on each fitted model, then averaged over training.

## Reviewed receipt arithmetic

The checker `pro12_sharp_wrong_head_check.py` verifies all18 entries of the preserved Pro10 package's payload manifest before arithmetic. It verifies each uniform-e report's scope, call count, source SHA and imaginary enclosure, uses the compact lower endpoint plus the already tail-corrected upper endpoint, and performs all new arithmetic at128-bit Arb precision with exact rational constants. It records its own source hash, package manifest hash and all five input report hashes in `pro12_sharp_wrong_head_check.json`. No old receipt is overwritten.

The probability enclosures are p_j approximately8.593339505e-11 and p_g approximately8.713732247e-38. The old correction1.4849290665e-6 remains a valid conservative bound; it is simply unnecessary for the correct-root wrong-head event once the change-of-measure argument is used. The old exact-sampler expected upper bound8.2496059248e-9 is unchanged.

| Actual field calls | Sharpened expected upper, approximate | Expected enclosure width, approximate |
|---|---:|---:|
|4|0.223724714750647|1.20149e-8|
|8|0.032062916069977|1.03808e-8|
|16|0.002146511341633|9.74182e-9|
|32|0.000137757349955|9.57597e-9|
|64|0.000008718960908|9.53291e-9|

The full machine balls preserve outward radii; this rounded table is explanatory. The script directly checks that *each expected interval*, not merely each oracle interval, has width less than exact2e-8. For64 calls it verifies expected upper<=exact1e-5. Its expected lower remains above1e-6, so64 is excluded at that tighter target. The32-call expected lower remains above1e-4. All target decisions are included, including adverse exclusions.

Thus the previous expected-risk ambiguity at1e-5 for64 calls is resolved by a proved sharper bad-event bound, without changing any fit, sampler, source, oracle enclosure or numerical budget. The trusted-computation scope of the original Arb certificate still applies; this is not proof-assistant verification, verification of a separate MPFR package, a claim about arbitrary FM learners/solvers, or evidence for native image/video performance or computational superiority.
