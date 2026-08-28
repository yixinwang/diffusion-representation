# Exact RMPF-R10 construction and audited failure boundary

## Exact fused trunk

For each 2x2 polyphase block, let `e=x00` and let `t_o` denote the three remaining orientation/channel targets. For class `y`, orientation `o`, and channel `c`, R10 applies

\[
r_o=t_o-m_{o,c,y}-\alpha_{o,c}e,\qquad z_o=r_o/s_{o,c},\qquad c=e+\tfrac14(r_h+r_v+r_d).
\]

A fixed signed orthogonal one-stage butterfly mixes the standardized details. The same map is composed at two resolutions. The final coarse block is standardized class-conditionally and the unchanged exact R7 coupled global-copula endpoint is applied once.

The inverse first undoes R7 and the butterfly, then reconstructs

\[
r_o=s_{o,c}z_o,\qquad e=c-\tfrac14(r_h+r_v+r_d),\qquad t_o=r_o+m_{o,c,y}+\alpha_{o,c}e.
\]

Thus every visible coordinate is retained. There is no VAE, encoder, decoder, stochastic bottleneck, reconstruction objective, ELBO, discarded coordinate, or post-hoc selector.

With orthonormal butterfly and Haar/polyphase reindexing, the trunk log-Jacobian is

\[
\log|\det DF_{10}|=-\sum_j\log s^a_j-\sum_{\ell,o,c}N_{\ell,o,c}\log s_{\ell,o,c}+\log|\det DT_7|.
\]

For input scaling by 127.5, the exact normalized density is

\[
\log p(x\mid y)=-\tfrac12\|F_{10}(x,y)\|^2-\tfrac D2\log(2\pi)+\log|\det DF_{10}(x,y)|-D\log(127.5).
\]

The identity/ordinary-flow limit is obtained by zero predictors, unit scales, identity butterflies, and inactive R7 targets. Unrestricted conditioners, output rank, and mixing recover the corresponding full affine flow family.

## Compute and memory passes

Two lifting levels require a fixed number of streaming reads/writes. Local prediction, update, standardization, and the one-stage butterfly are fused, giving

\[
C_{10}(D,r)=\Theta(D)+\Theta(Dr+r\log r),\qquad M_{10}=\Theta(D+Dr),
\]

rather than materializing recurrent dense detail states. Measured systems gates pass against both full flow and positive-rate latent flow on the frozen image and video workloads.

## Finite hidden-copula advantage

The R7 endpoint observes selected orthogonal detail modes, applies a normalized Hadamard transform, freezes half as a conditioner block, computes a shared parity state, and conditionally standardizes the target half. Under the registered state-conditional Gaussian teacher, R7 is exact. A coordinatewise Gaussian control can match every target marginal but cannot use the shared parity state. Its population log-score deficit is

\[
\sum_j I(T_j;S)>0
\]

whenever at least one target law changes with state `S`. Five seeds 9600-9604 give coordinatewise-control minus R10 NLL 0.0107923 [0.0107017,0.0108829], with exact copy-control equality and candidate parity gap below 0.05.

## Real-development failure

Five CIFAR/UCF development seeds legally opened after the systems gate. R10 did not improve the registered downstream estimand:

- image local-minus-R10 energy: -0.00000613 [-0.00001957,0.00000730];
- image local-minus-R10 dependence error: 0.002318 [-0.004703,0.009338];
- video attribution: exactly 0 [0,0];
- image R10-minus-full energy: 1.04022 [0.85077,1.22967];
- image R10-minus-positive-rate-latent energy: 1.08710 [0.90383,1.27036];
- video R10-minus-positive-rate-latent energy: 1.48477 [1.36940,1.60013].

The R7 endpoint was inactive in every video seed and only intermittently active in images. Random, shuffled, and generic matched endpoints tied the candidate on proper quality and dependence.

## C3 one-layer falsifier

C3 changes only diagonal RGB residual scaling to one shared 3x3 Cholesky factor per level/orientation:

\[
z=L^{-1}r,\quad r=Lz,\quad \log|\det J|=-\log\det L.
\]

For Gaussian residual covariance `Sigma`, the exact per-site NLL advantage over diagonal whitening is

\[
\tfrac12\{\log\det\operatorname{diag}(\Sigma)-\log\det\Sigma\}.
\]

Known truth passes, and real NLL improves by 0.6870 nat/dimension on images and 0.6639 on video. Proper energy worsens, R7 remains unattributed, and fit budgets fail. Hence second-order color covariance is real but not the missing sample-quality mechanism.

## First unsupported dependence class

The sparse trunk cannot represent conditional multimodality, semantic or cross-site dependence outside its local linear and sparse butterfly spans, global dependence in the orthogonal complement of the selected R7 modes, or stage-aligned feedback into a conditioner already frozen by triangular ordering. Increasing rank or covariance toward the full model erases the systems gain.

The frozen active C4 child changes only the color-whitened residual radius through an exact monotone shared-scale map. Its radial Jacobian is

\[
\log g'(r)+(d-1)\{\log g(r)-\log r\}.
\]

There is no headroom when the radius is already chi-distributed and independent of direction. Failure would localize the next layer to angular/cross-site multimodality or stage-aligned dynamics. Confirmation remains sealed.
