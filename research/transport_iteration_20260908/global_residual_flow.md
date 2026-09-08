# Shallow global conditional residual transport

This component is a conditional normalizing flow. Its mathematical density is normalized on all residual coordinates; it does not supply a learning guarantee for native images. The companion shared-analysis experiment compares specified implementations and cannot prove superiority over the stochastic latent wrapper that copies it.

Let a fixed invertible learned analysis map logit pixels to a coarse vector c and all residual coordinates r. Lossless packing places every residual scale on the coarse spatial grid. A coupling layer fixes a subset A and transforms its complement B:

    v_B = exp(s(r_A,c)) * r_B + t(r_A,c),
    r'_B = RQS(v_B; h(r_A,c)),       r'_A = r_A.

The conditioner sees the masked residual tensor and coarse tensor through global attention. Its output provides shift t, bounded log-scale s=2*tanh(raw), and the positive normalized spline parameters. Linear identity tails extend each spline to a bijection of the real line. Since the fixed coordinates do not change, the same conditioner output is available for inversion. Each transformed-coordinate derivative is exp(s) times the positive spline derivative. Reordering A before B makes the layer Jacobian triangular. The determinant is the product of these diagonal derivatives, even when the conditioner depends globally and nonlinearly on A and c.

Compose four alternating masks. Their inverse reverses the layer order. For any fixed c, writing e=T_c^{-1}(r), the conditional density is

    q_R(r|c) = phi(e) * |det D_r T_c^{-1}(r)|.

Change of variables gives integral one. Every Gaussian residual coordinate is retained. Coupling masks also permit dependence across distant positions and different packed scales. This establishes capacity for global dependence, not sufficient approximation or successful estimation of image conditionals.

For any normalized coarse distribution Q_C, drawing C from Q_C and independent full residual Gaussian noise, applying T_C, and inverting the analysis gives a normalized full-dimensional generative probability law. A tractable joint density additionally requires a tractable coarse density. The finite-step FM implementation is evaluated through its actual samples; neither its training MSE nor the likelihood of the separate analysis is reported as joint generative NLL.

The conditional exactness describes real arithmetic. The implementation checks numerical inverses and log determinants separately. Tests cover tails, a dense derivative-based Jacobian, an independent central-difference Jacobian, distant cross-scale sensitivity, mask invariance, source preservation and conditional likelihood gradients. Numerical agreement does not prove image quality.

The cost includes the full analysis, coarse generation, four attention conditioners and expanded spline-parameter outputs. Packing reduces attention's token count but preserves every scalar coordinate. Global attention has quadratic token cost, and producing many spline parameters can be expensive. A smaller number of network calls does not establish lower measured latency, memory, total training work, or asymptotic complexity than latent FM. The comparator receives the identical packing and coarse context and sufficient parameters to match the candidate decoder's count.

The analysis-only Gaussian generator is a required control: the shared analysis was already trained to map all observations to a Gaussian. Fitting additional coarse or residual models might worsen its final generated distribution. All retained analysis parameters and their fitting cost belong to each full pipeline, including a latent copy. No equal-dimensional semantic representation improvement over a VAE follows from invertibility alone.
