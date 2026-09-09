# Finite-Heun maps, normalized laws, and densities

This note concerns the real-arithmetic map implemented by the explicit Heun update. The exact ODE flow is a separate map. A deterministic measurable sampler applied to a probability distribution always defines a normalized probability law. An ordinary Lebesgue density requires an additional argument. Neither normalization nor the existence of a density establishes generation quality.

## A sufficient condition for one update

Fix a dimension $d$, a positive step $h$, and globally Lipschitz vector fields $f_t,f_{t+h}:\mathbb R^d\to\mathbb R^d$ with constants $L_t,L_{t+h}$. Define

$$
H(x)=x+g(x),\qquad
g(x)=\frac h2\{f_t(x)+f_{t+h}(x+h f_t(x))\},
$$

and

$$
a=\frac h2\{L_t+L_{t+h}(1+hL_t)\}.
$$

If $a<1$, then $H$ is a global bi-Lipschitz bijection and

$$
(1-a)\|x-y\|\le\|H(x)-H(y)\|\le(1+a)\|x-y\|.
$$

**Proof.** The predictor $x\mapsto x+hf_t(x)$ has Lipschitz constant bounded by $1+hL_t$. Composition and addition imply $\operatorname{Lip}(g)\le a$. The displayed inequalities follow by the triangle and reverse-triangle inequalities. For each proposed output $y$, the equation $H(x)=y$ is equivalent to $x=y-g(x)$. The right-hand side is a contraction of the complete metric space $\mathbb R^d$ into itself. The contraction theorem supplies a unique solution for every $y$. This proves surjectivity and injectivity and gives $\operatorname{Lip}(H^{-1})\le(1-a)^{-1}$. The assumptions permit unbounded support and linearly growing velocity.

If both fields are $C^1$, then $H$ is a global $C^1$ diffeomorphism. $\|Dg(x)\|_2\le a$, so every singular value of $DH(x)=I+Dg(x)$ belongs to $[1-a,1+a]$. The local inverse theorem, combined with the global bijection, gives a global $C^1$ inverse. The path $I+sDg(x)$ is nonsingular for all $s\in[0,1]$; its determinant starts positive and cannot change sign. It follows that

$$
0<(1-a)^d\le\det DH(x)\le(1+a)^d.
$$

A Gaussian source with density $\varphi_d$ consequently has output density

$$
p_H(y)=\frac{\varphi_d(H^{-1}(y))}{\det DH(H^{-1}(y))}.
$$

This is an existence and change-of-variables statement. The Heun sampler does not evaluate this determinant or compute its inverse merely by running forward.

## Composition of all steps

For $N$ successive updates $H_i$ with constants $a_i<1$, let

$$
T=H_{N-1}\circ\cdots\circ H_0,\qquad
m_N=\prod_i(1-a_i),\quad M_N=\prod_i(1+a_i).
$$

Then $T$ is globally bi-Lipschitz with

$$
m_N\|x-y\|\le\|T(x)-T(y)\|\le M_N\|x-y\|,
\qquad m_N^d\le\det DT(x)\le M_N^d.
$$

The determinant is evaluated along the intermediate states and its logarithm is the sum of the individual step log determinants. This gives

$$
d\sum_i\log(1-a_i)\le\log\det DT(x)
\le d\sum_i\log(1+a_i).
$$

All claims follow by composing the one-step inequalities and applying the chain rule. The resulting density is $\varphi_d(T^{-1}(y))/\det DT(T^{-1}(y))$.

For a uniform global bound $L$ and integration interval $[0,1]$ with $h=1/N$, one can take

$$
a=L/N+L^2/(2N^2).
$$

The sufficient condition becomes $L/N<\sqrt3-1$. In the current coarse sampler, 32 velocity evaluations mean $N=16$ Heun steps, so a proven uniform bound $L<16(\sqrt3-1)\approx11.7128$ would suffice. This threshold is conditional. No trained field was measured. For an illustrative hypothetical bound $L=1$, $a=33/512\approx0.0644531$; even then the determinant envelope in dimension 192 spans the powers $(1-a)^{3072}$ and $(1+a)^{3072}$. This shows why a valid global density bound can be extremely loose for quantitative likelihood bounds.

Failure of this sufficient test does not show noninvertibility. For example, $f(x)=\lambda x$ with positive scalar $\lambda$ has an invertible Heun multiplier $1+h\lambda+(h\lambda)^2/2$ for every positive step, even when the contraction test fails.

## A global bound for the implemented coarse network

The inspected `qalt/src/qalt/flow_matching.py` defines `FullTensorFlowMatching.velocity` as `ConvolutionalVelocity`: a zero-padded $3\times3$ convolution, SiLU, a second zero-padded $3\times3$ convolution, SiLU, and a final $1\times1$ convolution. Time is broadcast as an additional input channel. The coarse field is unconditional and has no attention, dropout, batch statistics, or clipping. The final convolution starts at zero, but training changes its weights.

For finite weights and biases, this field is smooth in state and time and globally Lipschitz in state. One elementary global bound for SiLU $s(u)=u\sigma(u)$ is

$$
|s'(u)|=|\sigma(u)+u\sigma(u)(1-\sigma(u))|
\le1+1/e=:b.
$$

To verify this, write $r=|u|$ and use $\sigma(u)(1-\sigma(u))\le e^{-r}$ and $re^{-r}\le1/e$. A tighter derivative supremum can improve the bound, but is unnecessary for its validity.

Let $K_1$ be the operator norm of the first convolution restricted to the state-input channels, with time held fixed. Let $K_2,K_3$ bound the remaining convolution operator norms on the configured finite grid. Then the uniform state Lipschitz bound

$$
L\le b^2K_3K_2K_1
$$

is valid for every time. Biases and the time-channel contribution affect the value $f_t(0)$ but not this state Lipschitz bound. For an explicit conservative bound without constructing a dense convolution matrix, write each convolution as a sum of spatial shifts followed by channel matrices $W_r$. Zero-padding shifts have Euclidean operator norm bounded by one, so $K_j\le\sum_r\|W_{j,r}\|_2$. Use only state-channel columns for $K_1$. Exact singular values of the finite-grid convolution operators can give a tighter bound. Ordinary power iteration supplies an estimate. An upper bound requires a separate error bound.

The product of layer bounds can be much larger than the actual state Lipschitz constant. Nothing inspected enforces spectral constraints during fitting. Consequently, one cannot infer that the learned coarse map satisfies the threshold above from the layer definitions, small losses, bounded sample displacements, or a successful eight-source roundtrip check. No trained weights or observations were used for this note.

Finite weights also imply uniform linear growth on the compact time interval: $\|f_t(x)\|\le L\|x\|+B$ for finite $B=\sup_{t\in[0,1]}\|f_t(0)\|$. This supports global existence and uniqueness of the exact ODE solution and a smooth exact-time flow. It does not transfer exact-flow invertibility to a fixed explicit numerical solver. The global residual FM uses a different attention network; the convolution argument above does not bound that field.

## Finite-step density can fail

Smooth globally Lipschitz velocities can produce a singular finite-Heun map. In two dimensions take an autonomous linear field $f(x)=Ax$ with

$$
hA=\begin{pmatrix}-1&-1\\1&-1\end{pmatrix}=:B.
$$

Direct multiplication gives $I+B+B^2/2=0$. One Heun step maps every input to zero, although the exact ODE flow $e^{hA}$ is invertible. The Lipschitz constant here is $\sqrt2/h$ and the sufficient contraction bound correctly does not apply. This elementary counterexample is independent of the fitted study models.

The finite-Heun coarse output always defines a normalized probability law in ideal arithmetic, but a global density is not established by smoothness or FM training alone. A verified one-step bound for every coarse update would establish one. Other weaker sufficient conditions could also establish absolute continuity without global injectivity; the condition in this note is not necessary.

A continuous conditional residual decoder supplies a normalized kernel $Q(dr\mid c)$ for every coarse code. For any normalized coarse law $\mu(dc)$, their joint $\mu(dc)Q(dr\mid c)$ and its measurable synthesis pushforward are normalized. If $\mu$ has a density and the full synthesis is a $C^1$ diffeomorphism, the standard joint-density and change-of-variables formulas apply. If the coarse law is concentrated on a Lebesgue-null set, adding continuous residual coordinates in the remaining dimensions does not necessarily restore full-dimensional absolute continuity: the joint remains concentrated on that null set times the residual space, and a smooth diffeomorphism preserves this local null-set property. No conditional spline likelihood should be relabeled as the entire model's likelihood.

Finally, actual finite-precision sampling is supported on finitely many representable outputs. Continuous density statements conventionally refer to the real-arithmetic model. Floating-point finite checks and roundtrips test implementation behavior on sampled inputs; they do not prove a global real-arithmetic Lipschitz bound or a literal floating-point Lebesgue density.

## The collapse example belongs to the implemented convolutional family

The identity SiLU(u)-SiLU(-u)=u follows from sigmoid(u)+sigmoid(-u)=1. With width at least twice the input channel count, use only the central spatial weights and set biases and time weights to zero. The first convolution outputs the channel pairs (x,-x). After its SiLU, the second convolution takes their differences to output (x,-x) again. After the second SiLU, the last convolution takes A times the paired differences. The resulting field is exactly x mapped to Ax independently at every spatial location, in real arithmetic. Unused hidden channels can have zero weights.

For RGB coarse inputs the implemented width32 exceeds the six channels required by this construction. Embed the displayed two-dimensional matrix A in the first two channels and set its action on the third channel to zero. The finite Heun update collapses two channels at every coarse-grid location and preserves the third. The exact ODE remains invertible. This demonstrates a limitation within the implemented unconstrained family; it does not identify the trained study weights with this example or establish a failure of their density.
