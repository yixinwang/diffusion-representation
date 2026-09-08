# Learning conditional transports from independent image clusters

Date: 2026-09-08. Author: theory_audit agent. Status: finite-sample derivations reviewed independently by the root and implementation-review agents. This is a mathematical construction and comparison of specified estimators. It supplies no image/video performance result or superiority theorem over unrestricted latent models.

## 1. Observation model and structural assumptions

An independent image or video clip is one statistical cluster. Suppose n independent identically distributed clusters are observed. Each cluster is an entire vector X in a D-dimensional observation space. A fixed C1 diffeomorphism T maps X to Y=(Y_1,...,Y_D) in (0,1)^D. The chart is specified before these n clusters are used, or trained on a separate independent sample. It may be nonlinear. Every method in the comparison receives the same observed clusters, chart, and specified context graph.

Fix a directed acyclic graph in coordinate order. For coordinate j, C_j consists of k_j selected preceding coordinates, with 0<=k_j<=k. The target factorization is

p(y)=product_{j=1}^D p_j(y_j|c_j).

This is a substantive conditional-independence assumption. It excludes dependence on preceding coordinates outside the selected context once C_j is known. The theorem later states the error when it fails. The contexts consist of observed chart coordinates. They require no ground-truth latent variables.

For 0<beta<=1, assume each conditional density obeys

a <= p_j(t|c) <= A,
|p_j(t|c)-p_j(t'|c')| <= L ||(t,c)-(t',c')||_2^beta,

for every t,t' in [0,1], every context c,c' in [0,1]^(k_j), and fixed constants 0<a<=1<=A<infinity and L<infinity. Boundary values can be interpreted by continuous extension. These assumptions include non-Gaussian and nonlinear conditionals, but exclude atoms, arbitrarily thin density troughs, and unbounded conditional irregularity. Constants and context dimension must remain controlled as D grows for the dimension comparison to retain its meaning.

Write KL(P||Q) for the Kullback-Leibler divergence of the two full joint laws. The same divergence holds before and after the fixed chart by change of variables. No independence among coordinates or image sites is assumed.

## 2. A fully specified conditional histogram estimator

Choose an integer b>=2 and divide each context coordinate and the response coordinate into b equal bins of width h=1/b. For factor j, there are K_j=b^(k_j) context cells. For context cell c and response bin r, count the n independent clusters whose jth context and response lie in those bins. Write N_c for the context count and N_cr for the joint count.

The add-one conditional response-bin probability is

q_hat(r|c)=(N_cr+1)/(N_c+b).

Within response bin r, the conditional density is b q_hat(r|c). An empty context cell produces the uniform response density. The joint fitted density is the product of these normalized conditional densities in graph order. Its normalization follows by successively integrating the final coordinate and proceeding backward. It uses observations alone.

All n clusters contribute one observation to each factor. Dependence between factors inside the same cluster does not change the argument: for a fixed j the sequence across clusters is independent, and expectations of the factor risks are added without assuming independence between those risks.

## 3. Finite conditional risk bound

For factor j, let p_star(t|c) be constant on every context/response cell, obtained by averaging the true conditional response-bin probabilities over the target distribution of C_j within that context cell and spreading each bin probability uniformly across the response bin. For an empty-probability context cell its definition is arbitrary, since that cell has zero target weight.

### Approximation error

Points in the same context/response cell have Euclidean distance bounded by sqrt(k_j+1)/b. Averaging the Holder inequality yields

|p_j(t|c)-p_star(t|c)| <= L (k_j+1)^(beta/2) b^(-beta).

Since p_star>=a, the inequality KL(p||q)<=integral (p-q)^2/q gives

E_{C_j} KL(p_j(.|C_j)||p_star(.|C_j))
<= [L^2 (k_j+1)^beta/a] b^(-2 beta).

### Estimation error with random context counts

Conditional on N_c=s, the response-bin counts are multinomial with s draws and true probabilities pi_1,...,pi_b. For a bin with pi_r>0, the binomial identity

E[1/(N_cr+1)|N_c=s]
= [1-(1-pi_r)^(s+1)]/[(s+1)pi_r]
<= 1/[(s+1)pi_r]

and Jensen's inequality for log imply

E log[pi_r/q_hat(r|c)] <= log[(s+b)/(s+1)].

Weighting by pi_r, summing r, and using log(1+x)<=x gives

E[KL(pi||q_hat(.|c))|N_c=s] <= (b-1)/(s+1).

Let w_c=P(C_j in cell c). The same binomial reciprocal identity, now for N_c with n draws and success probability w_c, gives

w_c E[1/(N_c+1)] = [1-(1-w_c)^(n+1)]/(n+1) <= 1/(n+1).

Summing context cells proves an integrated estimation bound K_j(b-1)/(n+1). No lower bound on context-cell probability is required for this step.

### Combining the two errors

The log ratio p_star/q_hat is constant within each context/response cell. Integrating it against the true distribution yields exactly the discrete conditional-bin KL term. This proves the decomposition and bound

E_training E_{C_j} KL(p_j(.|C_j)||q_hat_j(.|C_j))
<= [L^2(k_j+1)^beta/a] b^(-2 beta)
   + b^(k_j)(b-1)/(n+1).

The full joint KL chain rule now gives the explicit learning guarantee

E_training KL(P||Q_hat)
<= sum_j { [L^2(k_j+1)^beta/a] b^(-2 beta)
            + b^(k_j)(b-1)/(n+1) }.

This bound derives both approximation and fitting errors from stated regularity, the estimator, and n independent clusters. It does not assume that a learned model already has a small density error.

For q=k+1 and b=ceil(n^(1/(2 beta+q))), the expected KL per coordinate satisfies

E_training KL(P||Q_hat)/D
<= [L^2 q^beta/a + 2^q] n^(-2 beta/(2 beta+q)).

The bound applies for n>1; b can be set to at least two without changing the asymptotic rate. The floor constants can be adjusted for n=1. The numerator D is not discarded: to make total joint KL bounded by eta, the sufficient sample size scales with (D/eta)^((2 beta+q)/(2 beta)), up to the displayed constants.

Markov's inequality supplies the finite probability statement

Pr_training(KL(P||Q_hat)/D > R_n/delta) <= delta,

where R_n is the displayed upper bound per coordinate and 0<delta<1. This probability bound is conservative but requires no unproved concentration claim about correlated sites.

## 4. Gaussian-prior transport, normalization, and cost

Draw independent standard Gaussian coordinates Z_1,...,Z_D and set U_j=Phi(Z_j), where Phi is the standard Gaussian CDF. In graph order, set

Y_j=F_hat_j^(-1)(U_j|C_j),

where F_hat_j is the fitted conditional CDF. Every estimated response-bin mass is positive. Its CDF is continuous and strictly increasing in its response, so its inverse is uniquely defined away from endpoint conventions. Within a selected bin, inversion is linear. Finally output X=T^(-1)(Y).

The transformation is triangular and invertible outside boundaries of probability zero: the inverse recovers U_j=F_hat_j(Y_j|C_j) and Z_j=Phi^(-1)(U_j). The resulting joint density is exactly the normalized fitted model. This is one full-dimensional Gaussian-prior transport with no VAE, discarded coordinates, or added random inputs.

The unmodified context histogram has jumps when a context crosses a bin boundary. It is an almost-everywhere invertible transport, but generally not a continuous differentiable flow. Section 5 gives a continuous alternative with a proved rate. This distinction must remain explicit.

For fixed k, storing probabilities and cumulative probabilities costs O(D b^(k+1)). Counting the observed data costs O(n D (k+1)), plus O(D b^(k+1)) for table construction. Conditional CDF inversion by binary search costs O(log b) per coordinate after context lookup. Sampling or encoding costs O(D(k+log b)), plus the fixed chart cost. These are operation counts. No wall time or accelerator memory was measured.

Sampling depth equals the context graph depth. A chain has depth D; a fixed branching tree can have depth O(log D). Independent nodes in the same depth level can be generated in parallel. A low statistical context dimension alone does not prove low sequential sampling depth.

## 5. Continuous interpolation with a second explicit risk bound

A continuous transport can be obtained without assuming learned errors are small. Retain the same fitted response histograms at the centers of context cells. For each context c, use multilinear interpolation weights w_l(c) on its neighboring cell centers, clamping to the nearest center at the boundary. The weights are nonnegative, continuous, sum to one, and involve no more than 2^(k_j) centers. Define

q_tilde_j(t|c)=sum_l w_l(c) q_hat_j(t|cell l).

This density is normalized and positive, continuous in c, and piecewise constant in t. Its conditional CDF and inverse are continuous in both arguments. The triangular Gaussian-prior construction is consequently continuous and differentiable except at finitely many families of knots. The change-of-variables density holds almost everywhere. This construction does not claim a globally C1 Jacobian.

Assume the marginal density of C_j is bounded below by a_c,j>0. Under the structural model and the lower conditional bound a, the choice a_c,j=a^(k_j) is valid: each selected coordinate's density conditional on earlier selected coordinates is an average of a full preceding-coordinate conditional bounded below by a; multiplying these selected-coordinate conditional densities gives the bound.

If center l participates in interpolation at c, every point of its original cell is within coordinate distance 3/(2b) of c. The conditional-bin population density p_star_l obeys

KL(p_j(.|c)||p_star_l) <= [L^2(1+9k_j/4)^beta/a] b^(-2 beta).

For fixed c and l, the categorywise Jensen bound in Section 3 on E log[pi_l,r/q_hat_l,r] is valid even when weighted by the response-bin probabilities at c instead of pi_l,r: its upper bound log[(s+b)/(s+1)] does not depend on r. This gives

E_training KL(p_j(.|c)||q_hat_j(.|cell l))
<= [L^2(1+9k_j/4)^beta/a] b^(-2 beta)
   + (b-1) E[1/(N_l+1)].

Each context cell has probability at least a_c,j b^(-k_j), so the second term is bounded by b^(k_j)(b-1)/[(n+1)a_c,j]. Convexity of KL in its second density, followed by summing interpolation weights, proves

E_training E_{C_j} KL(p_j(.|C_j)||q_tilde_j(.|C_j))
<= [L^2(1+9k_j/4)^beta/a] b^(-2 beta)
   + b^(k_j)(b-1)/[(n+1)a_c,j].

Summing these bounds yields the full joint guarantee. The exponent remains 2 beta/(2 beta+k+1), with a context-density constant. In particular the construction is learned, normalized, continuous, nonlinear, and valid for non-Gaussian targets under explicit structural assumptions.

CDF evaluation combines no more than 2^k stored table entries. Binary search for inversion queries the interpolated CDF at each candidate bin boundary. Sampling costs O(D 2^k(k+log b)), plus chart cost, with the same graph-depth qualification. Learning rates and memory deteriorate with large k or small a; those costs cannot be omitted.

## 6. Learning a nonlinear non-Gaussian tree model

Take a rooted tree on the D coordinates, with bounded branching factor. Let the root be uniform on [0,1], and for every other coordinate use its single parent c as context, with

p_j(t|c)=1+theta cos(2 pi c) cos(2 pi t), for |theta|<1.

This density integrates to one, lies in [1-|theta|,1+|theta|], and is Lipschitz in (t,c), with a constant bounded by 2 pi |theta| sqrt(2). Its conditional CDF is

F_j(t|c)=t+[theta cos(2 pi c)/(2 pi)]sin(2 pi t).

Its derivative in t is strictly positive. Inverting this CDF gives a nonlinear conditional transport. The full joint distribution is non-Gaussian, and variables can have long-range dependence through ancestors. With k=1 and beta=1, the construction in Section 5 has expected KL per coordinate O(n^(-1/2)) and can be sampled in O(log D) graph depth for a balanced tree. Fixed nonlinear synthesis T^(-1) preserves the density-learning guarantee.

A coarse-to-fine image or video model can use such a graph as an explicit inductive assumption. The theorem does not establish that natural-image or video conditional laws follow that graph, meet the lower density bound, or have controlled Lipschitz constants. Verifying those assumptions requires separate modeling and empirical work. This tree model demonstrates an end-to-end nonlinear non-Gaussian learning result with image clusters as independent units; it does not establish state-of-the-art generative quality.

## 7. Error from omitted context

For an arbitrary target law, let V_j be all coordinates preceding j. Let p_j_full(.|V_j) be its true full conditional and p_j_context(.|C_j) the conditional averaged over omitted predecessors. For any fitted normalized conditional q_j(.|C_j), exact addition and subtraction of log p_j_context gives

KL(P||Q)=sum_j I_P(Y_j; V_j | C_j)
         + sum_j E_{C_j} KL(p_j_context(.|C_j)||q_j(.|C_j)).

Here I_P is conditional mutual information; repeated coordinates already included in C_j add no information in this notation. The first term is a population approximation floor. The learning theorem applies to the second term if the context-conditional densities meet its regularity assumptions.

One quantitative context approximation condition is also possible. Suppose the full conditional density varies with the full predecessor vector v according to

|p_j_full(t|v)-p_j_full(t|v')| <= L_v ||v-v'||^beta,

and is bounded below by a. Let r(C_j) be a fixed reconstruction of those predecessors, within a domain where the conditional density is defined and satisfies these inequalities. Since the true context-conditional is the KL-minimizing density among functions of context,

I_P(Y_j;V_j|C_j)
<= E KL(p_j_full(.|V_j)||p_j_full(.|r(C_j)))
<= (L_v^2/a) E||V_j-r(C_j)||^(2 beta).

This provides an explicit approximation bound from a specified context reconstruction. It offers no benefit if distant informative predecessors cannot be reconstructed accurately from the specified context.

## 8. A smooth distant-dependence failure world

Assume D>k+1, and each context contains only the preceding k coordinates. On [0,1]^D define

p_theta(y)=1+theta cos(2 pi y_1) cos(2 pi y_D), for 0<|theta|<1.

The density is positive, smooth, normalized, and non-Gaussian. Each selected local conditional is uniform: integrating out the distant member of the pair removes its cosine term, and the final context omits y_1. The population optimum within the specified local conditional model is the uniform joint law. No amount of data can remove this misspecification.

For z=theta cos(2 pi y_1) cos(2 pi y_D), the function (1+z)log(1+z)-z has second derivative 1/(1+z)>=1/(1+|theta|). Taylor's integral formula gives a lower bound z^2/[2(1+|theta|)]. The integral of z is zero and its squared integral is theta^2/4. Consequently

KL(P_theta||Uniform) >= theta^2/[8(1+|theta|)] > 0.

The target itself has a nonlinear full-dimensional Gaussian-prior transport: generate its first D-1 coordinates uniformly from Gaussian CDFs, then invert the final conditional CDF

F_D(t|y_1)=t+[theta cos(2 pi y_1)/(2 pi)]sin(2 pi t).

Omitted distant conditioning causes the failure. Applying a fixed invertible nonlinear synthesis chart preserves the KL failure floor.

## 9. Comparison with a specified full-dimensional histogram learner

Consider a baseline that estimates one joint histogram with b bins per coordinate, so K=b^D, using the same add-one probabilities. For a beta-Holder joint density bounded below, the corresponding approximation and fitting analysis gives

E KL(P||Q_hat_full) <= (L_full^2 D^beta/a_full)b^(-2 beta)
                      +(b^D-1)/(n+1).

Balancing these terms yields the exponent 2 beta/(2 beta+D). Comparing this upper bound with the conditional upper bound alone would not establish a strict rate advantage. The following finite lower bound supplies a concrete comparison for this specified estimator.

Take p(y)=1+theta(2y_1-1), for 0<theta<1/2, with all remaining coordinates independent uniform. This target belongs to the conditional model with k=0 and beta=1. Its full histogram population projection p_star is the mean in each cell. A second-order expansion of t log t, whose second derivative is at least 1/(1+theta) on the density range, gives

KL(p||p_star) >= theta^2/[6(1+theta)b^2].

To verify the constant, the squared error from replacing the linear slope 2theta by its bin mean integrates to theta^2/(3b^2); multiplying by 1/[2(1+theta)] gives the displayed bound.

For the stochastic histogram error, every cell probability pi lies between (1-theta)/K and (1+theta)/K. Suppose K>=2(1+theta) and n>=2K/(1-theta), ensuring pi<=1/2 and n pi(1-pi)>=1. The add-one estimate is q=(N+1)/(n+K), with N binomial(n,pi).

Here is an explicit conservative variance lower bound. Set Z=N-n pi and v=n pi(1-pi)>=1. Its fourth moment is bounded by 4v^2. For an independent copy Z', the difference has variance 2v and fourth moment bounded by 14v^2. Paley-Zygmund applied to (Z-Z')^2 gives P(|Z-Z'|>=sqrt(v))>=1/14. It follows that E|Z-Z'|>=sqrt(v)/14. The triangle inequality implies, for every real shift c,

E|Z+c| >= (1/2)E|Z-Z'| >= sqrt(v)/28.

Using c=1-K pi proves E|q-pi|>=sqrt(v)/[28(n+K)]. The Hellinger lower bound and Cauchy-Schwarz give

E KL(pi_vector||q_vector)
>= (1/2)sum_cells [E|pi-q|]^2/[pi+E q].

Since pi+E q<=3pi and n+K<=2n, each cell contributes at least 1/(37632 n). The total fitting error is at least K/(37632 n).

The exact histogram projection decomposition combines the bounds:

E KL(P||Q_hat_full)
>= theta^2/[6(1+theta)b^2] + b^D/(37632 n),

in the stated regime. For the specified full histogram schedule b approximately n^(1/(D+2)), its risk has order n^(-2/(D+2)), including a lower bound. The conditional estimator with k=0 has upper order D n^(-2/3). For fixed D>1, their risk ratio tends to zero in favor of the conditional estimator on this target. This is a strict asymptotic comparison against this particular full-dimensional histogram procedure.

The baseline receives the same data and permitted structural information. It fails to use the available conditional structure. A full-dimensional estimator that exploits the same graph can implement the conditional model and tie it. This result is not a lower bound for all full flows, diffusion, or image-density learners, and should not be presented as one.

## 10. Strong stochastic latent-copy equality

A competing latent generator can sample the same selected root or coarse coordinates and use the remaining Gaussian coordinates in the identical fitted conditional CDF decoder. With the same tables, context graph, and inputs, this stochastic decoder returns exactly the same sample distribution. Its normalized density, sample cost, memory, and learned risk coincide with the conditional transport construction.

Both constructions use the same D-dimensional standard Gaussian prior when all decoder noise is counted. Calling some coordinates latent and the remainder decoder noise does not change information, capacity, or computation. Consequently the theorem cannot establish strict improvement over an unrestricted stochastic latent flow/diffusion system that can use this decoder. A claimed separation requires a specified decoder restriction or a distinct proved computational limitation.

The learnability theorem does establish a concrete alternative to an assumption that conditional errors are small: a normalized observation-only estimator, a full sampling procedure, a finite rate, dependence on context regularity and dimension, costs, and a failure world. Its practical relevance requires testing those structural assumptions against strong matched stochastic decoders and against real observation-space metrics.

## Verification and limits

No training, validation, or test data were inspected for this note. The finite risk proof uses independent full clusters and random-count identities; it does not treat spatial sites as independent samples. The continuous interpolation extension has its own density-lower-bound constant and does not inherit the sharper raw-histogram constant without proof. The full histogram lower bound applies only under its stated count regime and to that estimator. Application review must verify the context-averaging decomposition, interpolation risk, Hellinger constants, chart freezing, and the expected-risk to probability conversion before publishing these claims.

## Prior work and scientific role

The construction is a proposal for a transparent learning guarantee, with no publication-novelty claim. Conditional histogram estimation and its convergence theory are established topics. Mathieu Sart's [Estimating the conditional density by histogram type estimators and model selection](https://arxiv.org/pdf/1512.07052), inspected here in its October 2016 PDF version, studies adaptive piecewise-constant conditional estimators under a target-weighted Hellinger loss. The present note instead spells out an add-one KL calculation and its composition across an observed context graph. This distinction does not establish novelty.

Bilodeau et al.'s [Minimax Rates for Conditional Density Estimation via Empirical Entropy](https://arxiv.org/abs/2109.10461), Annals of Statistics 2023, provides conditional-density KL risk theory under more general assumptions. The finite constants here should not be presented as the optimal known conditional-density theory. Their scientific role is to make the learning, observation model, cluster unit, sampling map, cost, and omitted-context failure jointly checkable for this project.
