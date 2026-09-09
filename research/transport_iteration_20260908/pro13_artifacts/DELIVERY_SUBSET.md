# Partial Pro13 delivery: 60 of 73 originals verified

The second recovery adds eight original zero-mean state files (367,323 bytes): six attached to verified artifact-root tree `627883eb9a151cc3a70e5ae39d94766081d32185`, plus two separately verified unattached Git blobs. Every new byte sequence matches its original SHA256SUMS entry and Git blob identity. The original SHA256SUMS and INVENTORY are unchanged. All 59 currently present entries of the 72-entry SHA256SUMS pass; including SHA256SUMS itself, 60 original files are present.

The proposed complete tree `59289225ee9e5d27c6aa5a662b2c6af89659a34f` was checked once and returned HTTP 422. It is not remotely authenticated. Thirteen original files remain absent, totaling 1329422 bytes. The current missing-file inventory follows; DELIVERY2_RECEIPT.json records every new source identity. DELIVERY_AUDIT.json, RECOVERED_BLOBS.json and RECOVERED_TREE.json retain the historical first recovery without alteration; DELIVERY1_SUBSET_HISTORICAL.md preserves its accompanying text.

No fits, missing fixtures or states were regenerated, and no new code or state replay was executed. The eight available zero-mean states do not establish complete nine-state delivery; the original matching fixture observations remain absent. Numerical state roundtrips could be separately checked on newly declared sources, but that would not reproduce the missing observed-fixture checks or archived fitting history. Prior unchanged-check failure and separately modified quadrature validation remain recorded in the adjacent pro13_local_checks package.

| Missing original path | Bytes | Expected SHA256 |
|---|---:|---|
| `standalone_results/centered_zero_mean_2_packed_state.json` | 46062 | `fac2693224872ce97bfdd6f2706292152b905a24383db6126d5a59fe282cb099` |
| `standalone_results/fixture_0.json` | 103739 | `f74ef6ea07aef81f4fdd8880a635db5cca5236027b374129f39cd8c2660422d2` |
| `standalone_results/fixture_1.json` | 103770 | `608d1bc20ef942c3c1ede101fd63ba67c277e70e474b7acc1b6def8415ce072f` |
| `standalone_results/fixture_2.json` | 103728 | `a5cbf4542dfb85a8e402a7e2864c59e319aaa1499b17a8ec5c9853866d650be7` |
| `standalone_results/positive_0_compiled_state.json` | 108055 | `63dda6009f4d4a0d17919a2c3d87048b9b8aff866279ae95d4827047bf995231` |
| `standalone_results/positive_0_dense_state.json` | 107901 | `25a2b677855ca8df4296a0cac99259c017a15567ebbf0588954b9ec997002212` |
| `standalone_results/positive_0_packed_state.json` | 108203 | `3a270895bd62ec7d30ca2c2f71e48c9ce4fbf9ed6bddd471a142b7ec254b3ba0` |
| `standalone_results/positive_1_compiled_state.json` | 108002 | `ad3b1c3896559511f679e0a573d2b163ef84edde1409a397826a814a344f2c82` |
| `standalone_results/positive_1_dense_state.json` | 107847 | `98d2c16a82a1bfc951965d4a5d7728978477d2d6e881ffc26d90470f6b676a5a` |
| `standalone_results/positive_1_packed_state.json` | 108152 | `5c140622790f5d29dacb4e35a197d6b3f8154a56ec7ab8ed8d70e2d0d0666611` |
| `standalone_results/positive_2_compiled_state.json` | 107991 | `abd0dbdf662c7576c4990e32027bec89a939bcd2eb3810ef16d096b29649c3a9` |
| `standalone_results/positive_2_dense_state.json` | 107834 | `4bc0732822ad6c9ee2ddaa9e8f010f6ffa38de3b3ce536b2bf6166f3ff475613` |
| `standalone_results/positive_2_packed_state.json` | 108138 | `ae081189123702328de198ab81a440209c494edcef6231209ea25924488cdafc` |
