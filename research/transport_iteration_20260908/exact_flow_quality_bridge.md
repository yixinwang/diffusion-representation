# Full-model KL, total variation, and population energy score

This note uses whole observations as probability-distribution units. Pixels need not be independent. Every statement below concerns population distributions in exact arithmetic; it supplies neither a finite-sample confidence statement nor superiority over a fitted flow-matching comparator.

## Conventions and the observation metric

On the pixel cube $[0,1]^D$, use

$$
d(x,y)=\|x-y\|_2/\sqrt D\le1.
$$

For probability measures $P,Q$, total variation means

$$
\operatorname{TV}(P,Q)=\sup_B|P(B)-Q(B)|=\tfrac12\|P-Q\|_{\rm variation}.
$$

The population energy score of prediction $Q$ under observations $P$ is

$$
S(Q;P)=E_{X\sim Q,Y\sim P}d(X,Y)
-\tfrac12E_{X,X'\stackrel{\rm iid}{\sim}Q}d(X,X').
$$

Every expectation uses independent draws unless explicitly shared. Define

$$
\operatorname{ED}(P,Q)=2E_{P,Q}d-E_{P,P}d-E_{Q,Q}d.
$$

Then direct subtraction gives

$$
\Delta_S(P,Q):=S(Q;P)-S(P;P)=\tfrac12\operatorname{ED}(P,Q).
$$

For completeness, positivity can be seen without a density assumption. For scalar observations the identity $|x-y|=\int |1_{x>t}-1_{y>t}|\,dt$ gives score excess $\int(F_P(t)-F_Q(t))^2\,dt\ge0$. Euclidean distance is a positive constant times the spherical average of absolute one-dimensional projections: rotation invariance gives $E_\theta|\theta^Tv|=c_D\|v\|_2$ with $c_D>0$. Averaging the scalar identity proves $\Delta_S\ge0$ in every dimension. Bounded observations justify all integrals. The upper-bound argument below uses signed measures and the diameter and requires no pixel independence or densities.

## A squared total-variation bound

Let $\delta=\operatorname{TV}(P,Q)$. If $\delta=0$, the measures agree and the score excess is zero. Otherwise, the Jordan decomposition of $\mu=P-Q$ has positive and negative parts of equal mass $\delta$. Set

$$
P_+=\mu^+/\delta,\qquad P_-=\mu^-/\delta.
$$

These are probability measures, and $\mu=\delta(P_+-P_-)$. Bilinearity of integration against finite signed measures gives

$$
\Delta_S(P,Q)=-\tfrac12\iint d(x,y)\,\mu(dx)\mu(dy)
=\delta^2\left\{E_{P_+,P_-}d-\tfrac12E_{P_+,P_+}d-\tfrac12E_{P_-,P_-}d\right\}.
$$

The cross term is bounded by one because the diameter is one. Both subtracted terms are nonnegative. Hence

$$
0\le\Delta_S(P,Q)\le\operatorname{TV}(P,Q)^2\le1.
$$

Pinsker's inequality, with natural-log KL and the total-variation convention above, states $\operatorname{TV}(P,Q)^2\le\operatorname{KL}(P\|Q)/2$. One direct proof takes the positive Hahn event $B$, with $p=P(B),q=Q(B)$ and $p-q=\operatorname{TV}(P,Q)$. Grouping the likelihood ratio by $B$ and its complement and applying Jensen gives KL at least the Bernoulli KL. As a function of $p$ with $q$ fixed, Bernoulli KL has value and first derivative zero at $p=q$ and second derivative $1/[p(1-p)]\ge4$, so it is at least $2(p-q)^2$; endpoints follow by limits. With an infinite KL the right-hand KL bound is interpreted as vacuous. Combining the bounds gives

$$
\boxed{\quad 0\le\Delta_S(P,Q)\le\operatorname{TV}(P,Q)^2
\le\min\{1,\operatorname{KL}(P\|Q)/2\}.\quad}
$$

If a different observation metric has diameter $M$, the score bound is $M\operatorname{TV}(P,Q)^2$, with the corresponding factor $M$ throughout. The pixel normalization is essential to the dimension-free constant one.

### Sharp discrete examples and convention checks

Take the two opposite pixel-cube corners $u=(0,\ldots,0)$ and $v=(1,\ldots,1)$, so $d(u,v)=1$. For

$$
P=p\delta_u+(1-p)\delta_v,\qquad
Q=q\delta_u+(1-q)\delta_v,
$$

one has $\operatorname{TV}(P,Q)=|p-q|$ and $\Delta_S(P,Q)=(p-q)^2$. Thus the coefficient one in the squared-TV bound is exact, even for discrete distributions. In particular, opposite point masses give excess one and KL infinity. For $q=1/2$ and $p=1/2+\eta$, KL is $2\eta^2+O(\eta^4)$ and excess is $\eta^2$, so the KL factor $1/2$ is asymptotically sharp too.

If “TV” instead denotes the full variation norm $\|P-Q\|_{\rm variation}=2|p-q|$, the score bound has coefficient $1/4$, not one. Energy distance as defined here is twice the excess; omitting that factor would produce the wrong constant on these examples. Discrete examples are valid for the energy/TV inequalities even when an invertible continuous-flow density construction is being discussed elsewhere.

## Invertible analysis and the KL decomposition

Let $A$ be a measurable bijection with measurable inverse, and write $(C,R)=A(Y)$. Denote the two pushforward laws by $P_{CR}$ and $Q_{CR}$. KL is invariant under such a transformation:

$$
\operatorname{KL}(P_Y\|Q_Y)=\operatorname{KL}(P_{CR}\|Q_{CR}).
$$

For standard Borel spaces, regular conditional distributions exist and the chain rule gives

$$
\operatorname{KL}(P_Y\|Q_Y)
=\operatorname{KL}(P_C\|Q_C)
+E_{C\sim P_C}\operatorname{KL}(P_{R\mid C}\|Q_{R\mid C}).
$$

This equality applies as an extended-real identity under the usual absolute-continuity formulation. A finite right-hand side requires $P_C\ll Q_C$ and conditional absolute continuity for $P_C$-almost every context. A learned decoder specified at every context gives a particular conditional kernel. No differentiability or Jacobian calculation is needed for the measure-theoretic invariance; smooth density formulas are a sufficient special case.

The resulting quality bridge on the observation cube is

$$
\Delta_S(P_Y,Q_Y)\le\min\left\{1,\frac{\varepsilon_C+\varepsilon_R}{2}\right\},
\qquad
\varepsilon_C=\operatorname{KL}(P_C\|Q_C),\quad
\varepsilon_R=E_{P_C}\operatorname{KL}(P_{R\mid C}\|Q_{R\mid C}).
$$

Residual error is averaged under the true coarse distribution, as required by the KL chain rule. Replacing that average by one under generated coarse contexts requires another argument. Bounds on empirical conditional negative log likelihood or an optimization gap alone do not establish either population KL term.

## Coarse TV plus conditional residual KL

Suppose instead that only $\operatorname{TV}(P_C,Q_C)\le e_C$ is available, and that the model conditional kernel $Q_{R\mid c}$ is specified for every context needed under $P_C$. Form the intermediate joint law

$$
M(dc,dr)=P_C(dc)Q_{R\mid c}(dr).
$$

For measurable conditional probability kernels on the spaces above,

$$
\operatorname{TV}(P_{CR},M)
\le E_{P_C}\operatorname{TV}(P_{R\mid C},Q_{R\mid C})
\le E_{P_C}\sqrt{\tfrac12\operatorname{KL}(P_{R\mid C}\|Q_{R\mid C})}
\le\sqrt{\varepsilon_R/2}.
$$

The first inequality follows by conditioning the difference on each measurable event; conditional Pinsker gives the second and Jensen gives the third. Also,

$$
\operatorname{TV}(M,Q_{CR})\le\operatorname{TV}(P_C,Q_C),
$$

because integrating the same kernel is a probability transition and contracts TV. Projection onto the coarse coordinate supplies the reverse inequality, so equality actually holds for this joint construction. The triangle inequality gives

$$
\operatorname{TV}(P_Y,Q_Y)=\operatorname{TV}(P_{CR},Q_{CR})
\le\min\{1,e_C+\sqrt{\varepsilon_R/2}\}.
$$

Consequently,

$$
\boxed{\quad\Delta_S(P_Y,Q_Y)
\le\min\{1,(e_C+\sqrt{\varepsilon_R/2})^2\}.\quad}
$$

The square root encloses $\varepsilon_R/2$. Writing $\sqrt{\varepsilon_R}/2$ would be too small by a factor $\sqrt2$: the two-point small-perturbation example with a constant coarse variable already contradicts that alternative asymptotically.

The TV-only coarse version remains meaningful for singular or discrete coarse laws. An invertible synthesis maps their resulting joint law to a normalized observation law, although that law need not have a Lebesgue density. If $A$ is only a measurable synthesis rather than a bijection, pushforward contracts TV, so the TV-based upper bound still transfers from a specified latent joint law to its output law; the displayed KL equality must then be replaced by data processing for the appropriate pushforward laws.

## Interpretation

These are conditional guarantees: controlling the stated population errors controls population energy-score excess on a bounded observation space. They do not show that the proposed learner achieves those errors, dominates latent diffusion or flow matching, improves perceptual metrics, or has a better quality–compute tradeoff. A reused repair-set score estimate and an empirical optimization gap remain different quantities. Applying the inequalities to a random fitted model can be done conditionally on its training information, but any high-probability or expectation guarantee for its population errors must be supplied separately.
