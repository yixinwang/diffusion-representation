# Exact innovation-response decomposition and limits

This note concerns the implemented **precomposition** $B_H\circ F_H$, not the distinct projected-rank proposal. All fitted functions and the training sample are held fixed throughout each identity. No empirical result or finite-sample learning guarantee is asserted.

## Conditional density and full-model KL

Let $H=(C,R_{<b})$ be the observed/generated prefix. The old invertible block is $B_H$. Pull the true block through this fitted inverse and write $E=B_H^{-1}(R)=(U,V)$, with 16 anchors $U$ and 704 followers $V$. The response has

$$F_H(u,w)=(u,m(h(H),u)+\exp(s(h(H),u))\odot w),$$
$$m=U_f a(h,u),\qquad s=\log2\,\tanh(U_f b(h,u)).$$

Here $U_f$ denotes the follower rows of the **fixed fitted orthonormal frame**, not the anchor random vector. The frame is not a function of the current history in the implemented module. The summary $h(H)\in\mathbb R^{16}$ is deterministic. The exact pulled model density is

$$q(u,v\mid H)=\phi_{16}(u)\prod_j\mathcal N(v_j;m_j(h(H),u),\tau_j^2(h(H),u)),\qquad \tau_j=e^{s_j}.$$

Indeed its inverse leaves anchors unchanged and subtracts/divides every follower; its log determinant is $\sum_j s_j$. The old block inverse is applied before this response inverse. All original Gaussian coordinates remain present. Root normalization and invertibility of the fixed full-dimensional analysis imply normalization of the complete composition.

Assume regular conditional densities, absolute continuity relative to these positive model densities, finite conditional second moments, and finite entropies/KL sufficient to avoid undefined infinity-minus-infinity decompositions. Let $P$ below denote the true law after the fixed fitted pullback. Chain rule and conditional change of variables give the block decomposition

$$\begin{aligned}
E_H\mathrm{KL}(P_{U,V\mid H}\|q_{U,V\mid H})
={}&E_H\mathrm{KL}(P_{U\mid H}\|\phi_{16})\\
&+I(V;H\mid h(H),U)\\
&+E_{h,U}\mathrm{TC}(P_{V\mid h,U})\\
&+\sum_j E_{h,U}\mathrm{KL}(P_{V_j\mid h,U}\|\mathcal N(\mu_j,\sigma_j^2))\\
&+\frac12\sum_jE_{h,U}\left[\log\frac{\tau_j^2}{\sigma_j^2}
 +\frac{\sigma_j^2+(\mu_j-m_j)^2}{\tau_j^2}-1\right],
\end{aligned}\tag{1}$$

where $\mu_j=E[V_j\mid h,U]$ and $0<\sigma_j^2=\operatorname{Var}(V_j\mid h,U)<\infty$. Conditional total correlation is the KL to the product of conditional follower marginals. Formula (1) separates, in order: anchor error, omitted history, follower dependence after retaining the permitted information, scalar nonnormality, and exact Gaussian mean/variance mismatch.

Proof: first split the anchor marginal. The identity

$$E\mathrm{KL}(P_{V\mid H,U}\|q_{V\mid h,U})
=I(V;H\mid h,U)+E\mathrm{KL}(P_{V\mid h,U}\|q_{V\mid h,U})$$

uses deterministic $h$. Split the latter KL into conditional total correlation and scalar KL terms. Subtracting log densities of two Gaussians leaves a quadratic, whose expectation uses exactly $\mu_j,\sigma_j^2$; this proves the last two lines. If conditions for the refined split fail, retain the unsplit nonnegative conditional KL chain rule in extended-real form. A stochastic summary requires explicitly specifying and accounting for its randomization; the displayed mutual information identity must not be transferred silently.

For fixed fitted invertible analysis $A$, the full joint KL is

$$\mathrm{KL}(P_X\|Q_X)=\mathrm{KL}(P_C\|Q_C)+\sum_bE_{H_b}\mathrm{KL}(P_{R_b\mid H_b}\|Q_{R_b\mid H_b}),$$

and each block term equals (1) after its old-block pullback. The expectation over training can be taken afterward. Relearning analysis or $B_H$ changes the pulled true distribution itself; (1) does not prove that optimization decreases any particular term. In particular, omitted-history information concerns **pulled innovations**, not raw pixels, since $B_H$ already uses more than the global 16 summary scalars. Neither held-out likelihood nor a small sample covariance certifies that this omitted-information term vanishes.

## Retained SiLU variant: bounded scales, not uniformly bounded means

The prospective native choice retains the independently implemented endpoint-inclusive integer anchors and **SiLU** hidden activation in both I and P. It does not adopt the delivered reference's stride anchors or hidden tanh. The head is $o=W_2\operatorname{SiLU}(W_1x+b_1)+b_2$, where $x=[h,\tanh u,\tanh^2u]$ (or the matched prefix features). For fixed finite weights and finite inputs it has finite exact-real outputs. Since $|\operatorname{SiLU}(t)|\le|t|$,

$$\|o\|\le\|W_2\|\big(\|W_1\|\sqrt{\|h\|^2+2r}+\|b_1\|\big)+\|b_2\|.$$

Thus coefficients and follower means are bounded over anchors at each fixed finite history, and grow at most linearly with $\|h\|$ for fixed weights. They are **not uniformly bounded over arbitrary unbounded histories**. The retained frame has $\|U_f\|_{\rm op}\le1$, so it does not increase this mean norm bound. A finite second moment of $h$ is sufficient for finite response-mean second moments, but is an additional distributional premise; finite observed inputs alone do not establish the population premise in (1).

The final outer tanh still gives the uniform exact-real log-scale bound $|s_j|<\log2$, hence $1/2<\tau_j<2$, independent of history size. Finite floating arithmetic may round to a scale endpoint or overflow an earlier expression; numerical validity checks are separate from these exact-real bounds. Normalization of each conditional Gaussian kernel requires finite means and positive finite scales, not a uniform bound on means across histories. Neither a global Lipschitz bound for the full generator nor a finite-learning-risk bound follows from bounded log scales alone.

## What linear-ICA covariance does and does not restrict

For the old block at fixed history, scalar innovations are independent with diagonal variance $\Lambda$. Write $M=I+UKU^T$, $K=\operatorname{diag}(e^\alpha-1)$, and $D=\operatorname{diag}(e^\ell)$. Its conditional covariance is

$$\Sigma=D M\Lambda M^T D.$$

Subtracting the diagonal matrix $D\Lambda D$ leaves

$$D[UKU^T\Lambda+\Lambda UKU^T+UK(U^T\Lambda U)KU^T]D.$$

Its column space lies in $\operatorname{span}(DU,D\Lambda U)$ and therefore its rank is at most $2r$. This is **diagonal plus a symmetric rank-at-most-$2r$ correction**, not necessarily diagonal plus a positive semidefinite low-rank matrix. With identical scalar variances it reduces to a rank-at-most-$r$ correction. These are conditional statements: mixing different histories can produce a higher-rank marginal covariance.

The new precomposition adds nonlinear dependence, but does not promise a richer covariance matrix at every point. Before $B_H$, its follower covariance is $E\operatorname{diag}(e^{2s})+U_f\operatorname{Cov}(a)U_f^T$, with anchor/follower cross terms; the full covariance is again diagonal plus a correction of rank at most $2r$. After the old nonlinear scalar maps, this simple covariance argument need not survive. Therefore covariance alone is an inadequate criterion for its nonlinear benefit.

More explicitly, all expectations below are over the standard Gaussian anchors at fixed $H$:

$$E[V\mid H]=U_fE[a],\qquad\operatorname{Cov}(U,V\mid H)=\operatorname{Cov}(U,a)U_f^T,$$
$$E[\|V\|^2\mid H]=E\|U_fa\|^2+\sum_j E e^{2s_j}.$$

These follow by conditioning on anchors and averaging the independent zero-mean follower noise. They provide direct source-space mean and second-moment checks. They do not predict pixel gradient energy after a nonlinear $B_H$ and learned analysis; that diagnostic must be computed on actual decoded pixels.

## A normalized nonlinear example with zero off-diagonal covariance

Take two independent standard normals $Z_1,Z_2$, an anchor $U=Z_1$, and

$$V=Z_2+\gamma\{\tanh^2(U)-c\},\qquad c=E\tanh^2(Z_1),\quad\gamma\ne0.$$

Its positive normalized joint density is

$$p(u,v)=\phi(u)\phi\big(v-\gamma[\tanh^2(u)-c]\big),$$

with determinant one. Both means and the cross covariance are zero by symmetry, while $E[V\mid U]=\gamma(\tanh^2(U)-c)$ is nonlinear and nonconstant. Thus it is not jointly Gaussian and is not independent. This is representable by the response: take zero scales and one nonzero follower mean direction. In the default frame family, the leading-coordinate identity frame allows one such follower coordinate while retaining the configured anchors; all other coordinates can remain independent normals. The response features include $\tanh^2(U)$, and the two-hidden-unit identity $\operatorname{SiLU}(x)-\operatorname{SiLU}(-x)=x$ realizes a linear function of this feature exactly, with the centering constant in the output bias. This is a capacity example, not a claim that training discovers these weights.

It also lies outside a two-dimensional invertible linear mixing of independent scalar variables. For $f(u)=\gamma(\tanh^2u-c)$, the log density is $-(v-f(u))^2/2-u^2/2$. If independent linear coordinates existed, two independent constant directions would have zero mixed directional second derivative everywhere. The coefficient of $v f''(u)$ forces the product of their $u$ components to vanish. The remaining constant and nonconstant $f'(u)$ coefficients then force their $v$-component product and cross sum to vanish, contradicting independence of the directions. This is a small exact representational separation; it is not a comparison against general coupling flows or FM.

## Restrictions left unchanged

With $B_H$ fixed, the response cannot change its standard-normal, history-independent anchor marginal. A true pulled anchor $N(\delta,1)$ alone therefore gives block KL at least $\delta^2/2$, whatever the response head learns.

Followers remain conditionally independent Gaussians given $h,U$. For a nonlinear non-Gaussian failure world, let $a(U)=\exp(\eta\tanh U_1)$ with $0<\eta<\log2$ and set two true followers to

$$V_1=a(U)Z_1,\qquad V_2=a(U)(\rho Z_1+\sqrt{1-\rho^2}Z_2),\qquad0<|\rho|<1.$$

The joint law is non-Gaussian due to the nonlinear scale mixture, but its conditional total correlation is exactly $-\frac12\log(1-\rho^2)$. Equation (1) gives this lower bound for every response in the fixed-$B_H$ family. Permitting arbitrary response means or marginal scales cannot remove it. Jointly changing $B_H$, the frame or analysis might remove this obstruction; it is not a lower bound across those larger learned families. A stochastic exact-copy latent decoder can reproduce the entire composed map and must tie.

## Mean, energy and bounded observable diagnostics

In pulled coordinates, the follower negative log likelihood, up to a constant, is $\sum_j[s_j+(v_j-m_j)^2e^{-2s_j}/2]$. With $z_j=(v_j-m_j)e^{-s_j}$, its exact scores are

$$\frac{\partial L}{\partial m_j}=-z_je^{-s_j},\qquad \frac{\partial L}{\partial s_j}=1-z_j^2.$$

The rank coefficient gradients are $U_f^T(-z\odot e^{-s})$ and $U_f^T[(1-z^2)\odot\log2\{1-\tanh^2(U_fb)\}]$, followed by the head chain rule. Thus nonlinear anchor features can reveal energy misfit even if ordinary anchor/follower covariance is zero. Under a correct conditional law, $E[z\mid h,U]=0$ and $E[z^2-1\mid h,U]=0$. Consequently the proposed average outer products of $[\tanh U,\tanh^2U]$ with $zU_f$ and $(z^2-1)U_f$ vanish. They are projected moment diagnostics; they omit factors present in the exact scores, cannot certify conditional independence, and can miss misfit outside the frame/feature directions. Compare them as model-adequacy checks when fitted coordinate systems differ.

Use $\mathrm{TV}(P,Q)=\sup_A|P(A)-Q(A)|$. For any observable $f\in[a,b]$,

$$|E_Pf-E_Qf|\le(b-a)\mathrm{TV}(P,Q)\le(b-a)\sqrt{\mathrm{KL}(P\|Q)/2}.$$

For pixels in $[0,1]^D$, coordinate means satisfy this with range one. The edge-gradient energy $g(x)=|\mathcal E|^{-1}\sum_{(i,j)\in\mathcal E}(x_i-x_j)^2$ also lies in $[0,1]$, so its discrepancy has the same bound. Conversely an observed **population** discrepancy $\Delta$ gives KL at least $2\Delta^2$; a finite noisy estimate needs its own uncertainty allowance. Agreement of these moments is only necessary, not sufficient, for distributional agreement. Relative gradient-energy percentages become unstable if the true energy is small and are not covered by an absolute bound without its denominator.

For $f,g\in[0,1]$, $|\operatorname{Cov}_P(f,g)-\operatorname{Cov}_Q(f,g)|\le3\mathrm{TV}(P,Q)$ by splitting the product moment and two means. If the comparator is exactly the product of the two true marginals, their means coincide, giving $|\operatorname{Cov}_P(f,g)|\le\mathrm{TV}(P,P_fP_g)$ and hence $I(f;g)\ge2\operatorname{Cov}_P(f,g)^2$. **That constant cannot be reused for features in $[-1,1]$**: the product has range two, and the corresponding simple bound is $I\ge\operatorname{Cov}^2/2$. Small covariances do not rule out the nonlinear example above.

With normalized Euclidean distance $d(x,y)=\|x-y\|_2/\sqrt D$ on the pixel cube, the population energy-score excess satisfies

$$\mathrm{ES}(Q;P)-\mathrm{ES}(P;P)=\tfrac12\mathrm{ED}(P,Q)\le\mathrm{TV}(P,Q)^2\le\min(1,\mathrm{KL}(P\|Q)/2).$$

For the sharper first inequality, write the signed measure $P-Q=\delta(P_+-P_-)$ with $\delta=\mathrm{TV}$ and probability measures $P_+,P_-$. Energy distance is quadratic in this signed measure. The cross distance is at most one and within-law distances are nonnegative, hence half the residual energy distance is at most one. Without bounded diameter (for example unbounded logits), this particular bound requires additional assumptions. Finite first moments suffice for defining energy score but not for transferring the cube bound to logits.

Population KID is a kernel MMD squared. If its fixed feature/kernel pipeline has a proven uniform bound $|k(x,y)|\le K$, then $\mathrm{MMD}_k^2\le4K\mathrm{TV}^2\le2K\mathrm{KL}$. A polynomial kernel on unbounded features has no automatic such bound. Even on bounded input pixels an explicit network-dependent bound may be uselessly large. Empirical unbiased KID can be negative and is not itself a certified population divergence. Reused repair-set KID, gradient, covariance and energy gates remain engineering diagnostics unless separately designed uncertainty and evaluation independence justify stronger claims. None of these inequalities proves a native-image/video, universal FM, or VAE-representation advantage.
