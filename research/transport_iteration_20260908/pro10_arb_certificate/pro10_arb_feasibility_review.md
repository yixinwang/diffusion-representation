# Arb compact-domain certificate feasibility

The uniform-tilt compact-domain calculation succeeded for all five declared actual-call counts. These are interval enclosures from an independently written scalar program, subject to the trusted Arb implementation, correct callback contract and the previously proved analytic tail bound. They are not proof-assistant certificates or an audited production claim. No fitting, native arrays, PSC work, or generative runtime benchmark was performed.

## Dependency and reproducibility

`work/pro10_arb_compact.py` is the portable runner. It uses python-flint0.8.0, installed into the isolated `work/pro10-arb-deps`; the shared virtual environment was not modified. The runner searches this sibling directory first, and otherwise can use a normal python-flint installation. Runtime has no network calls. Invoke, for example:

```
python pro10_arb_compact.py --calls 64 --average --seconds 45 --output NEW.json
```

The original command used the workspace Python3.12 on macOSarm64. The downloaded wheel is `python_flint-0.8.0-cp312-cp312-macosx_11_0_arm64.whl`, 7.4MB approximately, SHA256884a75da741e4ebbfdf5c638629d9e8a34f5bbbbb315b815b08a7d664f7720b2. `work/pro10_arb_dependencies.json` records exact size, native-library hashes and environment. Other platforms require their own compatible wheel and recorded hash; the macOS wheel is not a portable Linux binary. Successful five-case reports include the exact runner SHA256. Original fixed-tilt and first uniform64 feasibility reports are retained separately; their earlier runner version predates source-hash reporting and must not be silently relabeled as produced by the final version.

Official [python-flint Arb documentation](https://python-flint.readthedocs.io/en/latest/arb.html) describes enclosing real balls and endpoint bounds. The [complex integration documentation](https://python-flint.readthedocs.io/en/latest/acb.html) specifies the analytic-callback obligation, including explicit branch-cut handling. Installed0.8.0 method docstrings were also inspected; the online documentation currently describes0.9.0. No code depends on an unverified newer API.

## Enclosure construction and review obligations

The code uses96-bit ball arithmetic, exact rational tilts/endpoints/time steps, and Arb pi/sqrt/erf/erfc/exp/log. For the actual discrete Heun map it propagates both H and its chain-rule derivative J. The integrand is

phi(z) [exp(ell) ell-expm1(ell)],

ell=-(H-z)(H+z)/2+log(1+e erf(H/sqrt2))+log J.

The field denominator uses1+e erf(k y/sqrt2), the exact tilted-normal expression. Complex erf and exp are entire; divisions are meromorphic and become nonfinite when their pole cannot be excluded. All square roots are of fixed positive *real time-dependent constants*, not of the complex integration variable. Both remaining complex logarithms receive the integration callback's `analytic` flag explicitly, so an enclosing domain that touches their branch cuts is rejected rather than passed as analytic.

For fixed e, integrate z over twenty unit segments covering[-10,10]. For the uniform case, an outer complex-ball integral covers exact e in[2/5,3/5] and multiplies byfive. When the outer callback requests analyticity in its entire complex e ball, every inner integrand evaluation forces `analytic=True` for both logarithms, even if the inner routine itself is performing a nonanalytic enclosure check. Thus a returned finite inner enclosure is for the parameterized integral over the full requested e ball; it is not just an enclosure at its midpoint. The inner evaluator must bound the correct function for every member of that ball. Forwarding only the inner flag would not establish the outer analyticity obligation.

Independent review should check this nested-parameter argument carefully: enclosure of values and holomorphy are distinct obligations. Along the real integration path, the previous global Heun derivative proof gives J>0 and the tilt factor ispositive. Logarithms agree there with the desired real branches. The callback's complex-domain checks support analytic continuation needed by the integrator. Neither a real-grid positivity test nor a principal logarithm evaluated without `analytic=True` would suffice.

Arb integration tolerances and evaluation limits are targets, not assertions of success: the report checks the actual returned enclosure. A finite wide enclosure remains an enclosure even if a requested tolerance was not reached. The script records nonfinite results/failures rather than converting them into a point estimate. Each bounded attempt has a45-second callback deadline, preserves exceptions and cost, refuses existing output files, and records integrand-call counts. No endpoint/midpoint conversion to ordinary float occurs inside the enclosure computation.

## Results and honest scope

Every uniform run returned a finite real enclosure; imaginary components were exactlyzero in these reports. The central joint value (multiply by2,880) agrees with the independent ordinary quadrature. Add the analytically proved nonnegative Gaussian-tail upper bound outside |z|<=10, evaluated with Arb; its joint value is at most approximately9.13949e-10. The resulting upper-bound balls are:

| Actual calls | Full joint upper-bound ball center, rounded for display | Local certificate calculation seconds |
|---|---:|---:|
|4|0.223724704158586|1.09|
|8|0.032062906940376|1.86|
|16|0.002146502865130|3.74|
|32|0.000137749042923|7.28|
|64|0.000008710696864368|14.99|

The machine JSON files `pro10_arb_average4/8/16/32/64.json` retain the enclosing radii; this rounded table is not itself an outward decimal certificate. The 64-call script comparison to exact1/50000 (=0.00002) returnedtrue; the analogous tests for all four smaller call counts returnedfalse. That threshold was only a feasibility output and is not a preregistered downstream quality requirement. Fixed e=1/2 had a different upper bound, approximately8.48886e-6; it is not interchangeable with the uniform result.

This supplies the missing type of compact-domain certificate rather than inferring it from quadrature convergence. It does not yet justify a manuscript-level certified number without independent code/proof review, pinned dependency provenance, and a documented trusted-computation scope. The previously established wrong-learning correction can be added symbolically, but no new eligibility or superiority claim is made here. The run durations above measure certification work, not generation, fitting, or inference performance.
