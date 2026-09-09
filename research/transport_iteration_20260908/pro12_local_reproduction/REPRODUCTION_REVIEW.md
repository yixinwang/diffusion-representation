# Independent local reproduction of Pro12

The unchanged authenticated `pro12_artifacts/run.py` completed every requested check locally: five primary 128-bit/order-6 integrations, their 192-bit expected-risk postprocessing, and the changed-grid/order-8/192-bit N4 integration. No model fitting, sampler timing, native data, PSC job, or floating-generator-law certification was performed.

Before execution, all 47 delivered files matched the original publication commit `a8954f35cf3dc09c3337d41a45187a7b7e462b11` and local cherry-pick `90ed6da`; all 46 manifest entries matched. The SHA manifest file itself was authenticated by both Git blobs. Compiler preprocessing confirmed selection of the installed vendor `/opt/homebrew/include/mpfr.h`, avoiding the fallback ABI declaration. Compiler executable, vendor MPFR/GMP headers and linked library binary hashes, symlink resolutions, compiler version and library linkage were recorded in `work/pro12-local-authentication.json` and `work/pro12-local-vendor-header-probe.txt`.

Execution used `/usr/bin/clang++ -I/opt/homebrew/include`, `-L/opt/homebrew/lib -lmpfr`, the wrapper's existing `-lgmp`, and 240-second per-process limits. Runtime MPFR is **4.1.0**, on macOS arm64; the historical package reported MPFR 4.2.2 on Linux x86_64. Original files were not edited. The fresh output directory is `work/pro12-local-reproduction`.

All four builds, all 256 algebraic/derivative consistency checks, the independent global-constant checks, all integrations and all risk postprocessors succeeded. There were no failure or timeout receipts. The COMPLETE manifest authenticates 58 output payloads, independently rechecked by `work/check_pro12_local_reproduction.py`. The complete audit, process records and linkage are summarized in `work/pro12-local-independent-check.json`.

| Stages | Reproduced conditional KL | Reproduced expected-risk upper |
|---:|---:|---:|
| 4 | [0.22372470287378568, 0.22372470323412066] | 0.22372471382618164 |
| 8 | [0.032062905626935562, 0.032062906091946647] | 0.032062915221547965 |
| 16 | [0.0021465015491993599, 0.0021465020022553211] | 0.0021465104787589165 |
| 32 | [0.00013774772474983807, 0.00013774817896396868] | 0.00013775648599630906 |
| 64 | [0.0000087093770403469173, 0.0000087098339525953205] | 0.0000087180979953948377 |

All five conditional endpoint pairs match the historical printed decimals exactly. All five expected-risk upper endpoints match. The N8 and N64 expected-risk lower endpoints differ by approximately one binary64 ULP; the new values are respectively `0.03206290562418028` and `0.0000087093770395984892`. These small lower-endpoint differences are retained in raw receipts, not replaced with historical values. Every expected-risk width remains below `2e-8`. This review did not isolate whether the small lower-endpoint differences come from historical postprocessor argument serialization or cross-platform/library behavior; neither explanation is assumed.

The independent changed-order N4 certificate overlaps the primary interval. This agreement is a cross-check; the derivative remainder and directed arithmetic remain the basis for certification. N64's expected-risk upper bound is below `1e-5`; all supplied quality-target classifications are unchanged. Exact-copy ties, ideal-real/exact-selection restrictions, and the distinction between expected fitted-model KL and mixture KL remain in force.

Primary integration cost was **148.129108292 seconds** summed over the five integrator-reported clocks, plus separately preserved build/test/postprocessing/alternate-run costs. It is not the full invocation wall time and is not sampler latency. No historical PSC cost samples were replayed. The source-only audit `pro12_mpfr_independent_audit.md` remains unchanged as a record of the preceding review; this note adds independent execution evidence.
