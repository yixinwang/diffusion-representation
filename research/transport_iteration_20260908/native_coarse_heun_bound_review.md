# Native coarse Heun sufficient-bound check

The saved native shared checkpoint hash matches the completed run's terminal manifest: 6067ee07182283ac0ce2ed43fe8105ae5715d18fc29a30c77b1ea0bb6bc656a5. The read-only calculator validates the three-convolution unconditional field and source at 341dbaf. No observations were opened and no weights were changed.

At 16 Heun steps, the sufficient condition fails: the computed uniform Lipschitz upper estimate is 529.1473 and the residual-map estimate is 579.9407, whereas the sufficient residual threshold is one. This is inconclusive about actual invertibility. It neither proves collapse nor gives a complete-density certificate. These ordinary floating-point matrix bounds are not an interval certificate.

The new exact coarse-coupling component removes this particular unverified sampler assumption by construction in real arithmetic; its training quality and numerical behavior at scale still require experiments.
