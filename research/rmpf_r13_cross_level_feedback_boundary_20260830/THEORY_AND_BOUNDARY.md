# Exact construction and finite boundary

Let the frozen parent map an observation to coarse/global state `a` and details `b`. A fixed rank-eight projection of level-2 details is appended to the already inverted coarse state, producing conditioner `c`. A fixed sparse rank-eight projection of level-1 details is `p = r u`. For train-only state `S(c)`, R13 applies

\[
M_\alpha(u)=\frac{(1-\|\alpha\|^2)u+2(1+\alpha^\top u)\alpha}
{1+\|\alpha\|^2+2\alpha^\top u},\qquad \|\alpha\|<1,
\]

and replaces `p` by `r M_{alpha_S(c)}(u)`. Radius and the orthogonal complement are unchanged. The conditioner is unchanged, so the exact inverse uses `M_{-alpha_S(c)}`. For rank eight,

\[
\log|\det DT|=7\left[\log(1-\|\alpha\|^2)-\log\{1+\|\alpha\|^2+2\alpha^\top u\}\right].
\]

All-zero parameters exactly recover the parent flow. Composition with the unchanged C4G/FSMLF and R7 maps remains one normalized, all-coordinate, standard-Gaussian-base flow.

For an aligned teacher whose conditional direction is the inverse Mobius image of a uniform sphere, exact recovery removes conditional KL. Strict propriety gives a positive energy-distance separation from a genuinely different pooled law. Observation-space transfer error is controlled by parent error, projection error, state-misclassification mass, within-state map error and unchanged R7 error.

The first failure condition is state nonidentification, within-state multimodality or zero first directional moment, dependence outside the selected cross-site span, or cyclic fine-to-coarse feedback.

## Executed boundary

C6 passed exactness, matched known truth and systems. Its real proper-energy gains were negligible. C7 showed that linear cross-moment states can be unidentified under symmetry. C8 recovered state alignment and large NLL/dependence gains but failed one seed's lower proper-energy margin. C9 supplied exact states; one of five seeds still failed the all-seed lower bound. A separate projective implementation worsened real C4G energy, an outcome-informed scalar audit found no passing real route, and an independent rank-32 harmonic oracle's upper interval was below its practical margin.

Therefore another state threshold, projection rank, scalar strength, endpoint selector or likelihood-only correction repeats a rejected premise. The next scientifically distinct mechanism would need joint within-state multimodal cross-plane/site structure and a finite theorem directly lower-bounding the registered proper score.