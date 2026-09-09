# Pro14 review package

Start with REVIEW.md. THEORY.md gives the complete mathematical proposal and
its approximation/observable qualifications. PROTOCOL.md specifies one proposed
3-by-2 native factorial plus the scalar reference. It is not an authorization
to launch or an already integrated native runner.

Run the fabricated-only check with:

    python checks.py

Dependencies used in the receipt: Python3.13.5, torch2.10.0+cpu,
NumPy2.3.5, SciPy1.17.0. There is no dependency installation or network/data access
in the executable files. No training or qalt import occurs.

Re-running checks.py writes a new fabricated_results.json and replaces that file's
receipt; copy the package first to retain the original result. The original
SHA256SUMS then intentionally detects any changed files. Existing attempts are
never rewritten by checks.py.

mechanism.py contains the proposed ResponseHead and InnovationResponse modules
plus clearly labeled fabricated surrounding scalar/SPD/root/analysis algebra.
The latter is independently written and does not validate native qalt behavior.
All normalized-density and inverse claims are exact-real mathematical statements;
finite tests and quadrature are not interval certificates.

SOURCE_INSPECTION.json records exactly which frozen paths were read and the
returned Git blob identities. FAILED_ATTEMPTS.md distinguishes a genuine
unexpected harness failure, deliberately rejected invalid variants, and access
failures. attempts/ preserves original source/log/receipt versions. The final
receipt hashes the exact final executable sources.

No native data, learned states, binary banks, model weights, or original native
payload are included or replaced. The SHA256SUMS here is new and applies only to
this review package; it is not any predecessor's manifest.
