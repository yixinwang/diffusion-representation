# Exact R8 lifting law and executed cost oracle

Let the two-level orthonormal Haar transform write

\[
Wx=(a_2,d_{2,1:3},d_{1,1:3}).
\]

For class or action label `y`, normalize the retained coarse state by

\[
z_a=(a_2-\mu_y^a)\oslash s^a,
\qquad s_j^a>0.
\]

At level `l`, orientation/channel group `g`, and site `s`, the data-to-base detail map is

\[
r_{lgs}=\frac{d_{lgs}-m_{lgy}-\alpha_{lg}p_{ls}-\beta_{lg}n_{ls}-[G_lC_l\phi(z_a)]_{lgs}}{\sigma_{lg}},
\qquad \sigma_{lg}>0.
\]

Level 2 is transformed first. Its original detail is recovered before constructing the level-1 approximation parent, so the map is triangular in `(a2,d2,d1)`. The inverse is explicit:

\[
d_{lgs}=\sigma_{lg}r_{lgs}+m_{lgy}+\alpha_{lg}p_{ls}+\beta_{lg}n_{ls}+[G_lC_l\phi(z_a)]_{lgs}.
\]

A signed long-range Hadamard lifting pair has determinant magnitude one. The unchanged exact R7 coupled endpoint is composed last. Hence

\[
\log|\det DF_8|=-\sum_j\log s_j^a-\sum_{l,g}N_{lg}\log\sigma_{lg}+\log|\det DT_7|.
\]

Identity is recovered by unit scales and zero coefficients. Unrestricted conditioners, full output rank, and dense mixing recover the associated full block-triangular affine-flow family and eliminate the cost separation.

The operation proxy is

\[
C_8=\Theta(D)+\Theta\{r(a+q)\}+\Theta(q)+C_{R7},
\]

with no NFE factor. The first unsupported dependence class includes conditional multimodality, covariance or copula components outside the fixed sparse/R7 span, stage-aligned feedback into an already frozen conditioner, and global interactions in the orthogonal complement of the selected R7 modes.

## Executed cost sequence

| Round | Domain | Fit s | Peak fit MB | Batch s | Single s |
|---|---|---:|---:|---:|---:|
| C0 | image | 4.529 | 364.0 | 0.1068 | 0.00260 |
| C0 | video | 4.124 | 508.9 | 0.0747 | 0.00794 |
| C1 | image | 1.864 | 174.7 | 0.0890 | 0.00332 |
| C1 | video | 1.859 | 255.5 | 0.0636 | 0.00626 |
| C2 | image | 1.915 | 174.7 | 0.0653 | 0.00256 |
| C2 | video | 1.764 | 255.5 | 0.0439 | 0.00368 |
| C3 | image | 1.642 | 104.1 | 0.0685 | 0.00356 |
| C3 | video | 2.080 | 170.7 | 0.0314 | 0.00419 |
| C3 replay | image | 1.695 | 104.1 | 0.0636 | 0.00336 |
| C3 replay | video | 1.777 | 171.3 | 0.0428 | 0.00571 |

C3 files were 18.7 KB and 65.4 KB; operation proxies were 87.9M and 50.5M. It passed every full-flow gate. The frozen image fit-time limit relative to positive-rate latent flow was 1.44778 seconds, which both C3 runs missed. Therefore no R8 CIFAR/UCF quality result was opened.