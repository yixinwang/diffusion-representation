# Pro15 delivery subset and recovery record

Publication-only follow-up, September 9, 2026. No original scientific file,
model, result, log, test receipt, or manifest has been changed or regenerated.
No original research source was executed during this recovery.

## Corrected remote inventory

The formerly reported tree `35a6c09e283b0edb13b33dc75951777ec3d5aa85`
is NOT readable in `yixinwang/diffusion-representation`: a direct GitHub
connector read returned 404. The previous claim of 210 verified staged files
was incorrect. The actually readable earlier tree is
`22e87285c2902370ed06122e295c6bca3ca21b77`.

That earlier tree has 173 blobs: 163 match original ZIP bytes and 10 do not.
The ten mismatched files are excluded, not reused as originals. This clean
recovery tree reuses the 163 matching original blobs (344,792 bytes) without
retransmitting their contents. Thus 103 text originals, not 56, required
transfer. Exact mismatch identities and access failures are preserved in the
new delivery_transfer/FAILED_OPERATIONS.json; historical failure files remain
unchanged.

## What is present

The 163 matching text originals occupy their original paths. The other 103
text originals are in lossless binary transport parts, NOT yet expanded at
their original Git paths. These are new transport wrappers, not replacement
results or copies of the omitted original binary arrays:

* priority_manifests.tar.xz.part00 through part02: 2 unchanged original files,
  MANIFEST.json and SHA256SUMS (150,749 original bytes).
* remaining_texts.tar.xz.part00 through part06: the other 101 original text
  files (508,239 original bytes), including THEOREM.md, remaining fitted JSON
  states/receipts, the full-dimensional constructed JSON model, and audit.

After verified extraction, all 266 original text files (1,003,780 bytes) are
available, including all 72 actual fitted JSON states and original logs and
failures. The original full package had 451 files (103,035,126 bytes).
Exactly 185 original binary files (102,031,346 bytes) are omitted below.
No additional training, model-state regeneration, or scientific evaluation
is required to recover and hash the text files.

## Recovery without executing research source

From this package directory run the NEW stdlib-only delivery utility:

```sh
python3 delivery_transfer/restore_original_texts.py
```

It checks part hashes, concatenated archive hashes, safe regular-file paths,
the original manifests, and every recovered original text hash. It refuses to
overwrite any different existing file and reports omitted binary files.
Alternatively concatenate each bundle's numerically ordered parts, verify
its checksum in delivery_transfer/RECOVERY_INDEX.json, and use a standard
XZ/tar reader in a fresh directory. Nothing imports or executes src/ or tests/.
An ordinary `sha256sum -c SHA256SUMS` after text extraction will correctly
report missing binary originals; that is not a text-transfer hash failure.

Original ZIP: pro15_artifacts_complete.zip, 97,735,768 bytes; SHA256
`c4e7f4bf8ea105588bc949049f09f5706007bd20775d3a02ac9b0708a98d8c68`.
The ZIP itself is unchanged and is not republished here.

Original MANIFEST.json: 90,816 bytes; SHA256
`a9833ccfc16d68be73bac882a158413e2fecc195796ce985d01a7ccdcd20f76c`.
Original SHA256SUMS: 59,933 bytes; SHA256
`a03bc35e7d1c6d18b1e66fd0d1db1d686accc60bf830443567ba0657516ad8f7`.

Original README/publication_request/provenance may say publication was
unavailable. Those statements are preserved historical text, not a statement
about the status of this new delivery. The newer separate Bernstein/Fano note
and native spline developments have not been folded into the original results.
This delivery establishes no native quality result.

## All 185 omitted original binary files

For each section the original relative path is its directory heading plus
`/` plus the filename. All sizes below are original byte counts and all hashes
are original SHA256 values. No original binary file is inside either text bundle.

### results/run_initial/full_dimension_smoke

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| numerical.npz | 4483341 | 44754b7b3541f2c1c622476e2ec519e942b849b73a1802ae2d9519215248ffb9 |
| source.npy | 1572992 | 86c1111b5a11a2d23424205ffc18cbb31f886e6ca7529abc9d2fa59f8d042985 |

### results/run_initial/seed_150901

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed_source.npy | 11141248 | 17ca8c424dd65ab0396e01cb40d9d527d953a0041ea26a106bb73ea50801b438 |

### results/run_initial/seed_150901/harmonic

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | 9d6be13dc925d5727e746cee84f7c8a855d3fd60ce58a0e9129751a21bb5964b |

### results/run_initial/seed_150901/harmonic/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 22014 | c5b37297f238b33ff7c02090216318508a851b19d02278c368c788364fb11011 |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 70161 | caaf8de319a01c626bae900d32c32f10dfa3dbb21d55936610f73adb464d3960 |
| histogram_lr_numerical.npz | 63067 | 322c18bde549c63de318bf02e8f5aee677d4662aea87707369253f73cd2057d2 |
| oracle_graph_harmonic_fit_arrays.npz | 63590 | 7c15c26e5c5df069711f16f6686b78f27164729b003fa4088f4cc804c1420f2c |
| oracle_graph_harmonic_numerical.npz | 63055 | db1125c7aaa18bcad6fd717cb975f5fd640b1193944ebc88745e21625ab320e5 |
| oracle_graph_histogram_fit_arrays.npz | 63590 | 7c15c26e5c5df069711f16f6686b78f27164729b003fa4088f4cc804c1420f2c |
| oracle_graph_histogram_numerical.npz | 63068 | f649213c11fe3ec04a4e50f8feeaaa0e19d7aef64e351b7dc97e0664b09fee05 |
| product_fit_arrays.npz | 493 | 0b4e2990626f5907adb6d0c31a8306a0257e17591cf076e9356d2980dac51858 |
| product_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| spectral_fit_arrays.npz | 66714 | 31e8e34b82866c54d3faa137970997af8c8c2a62e61e55e823d0b8d22081e4db |
| spectral_numerical.npz | 63057 | bd426a4342a51715bbab8378148f8be99bdbf41efed3c23fd944a31d59b8a382 |
| unconditional_fit_arrays.npz | 1636 | 81bcf2395c32af489a9f9d605277d1553a2cdaaf396467f092065ad8753e9fee |
| unconditional_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |

### results/run_initial/seed_150901/harmonic/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21896 | fa633bb30e8394d23349e12d45d6dbac97906f00bb68da3a370da0e25842b43c |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 6975 | f27f546c7f6040b95b2fce3a2d8002d158ccadf3e871b6adc2b852071eb63410 |
| histogram_lr_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| oracle_graph_harmonic_fit_arrays.npz | 31531 | a4503b49a338e379d02a6ec3c9211d07a626dc1e25efb46c6a917c6a38be64ff |
| oracle_graph_harmonic_numerical.npz | 63060 | 63f3007f6ad7e65228b76235b3445e85102a15fc433d87f7f96eceb9141af9c5 |
| oracle_graph_histogram_fit_arrays.npz | 31531 | a4503b49a338e379d02a6ec3c9211d07a626dc1e25efb46c6a917c6a38be64ff |
| oracle_graph_histogram_numerical.npz | 63075 | 734e2614eed921be6a1a776704db2a6b3fe994f46292be91adadc0f559b7002a |
| product_fit_arrays.npz | 494 | afa14aaf7b604500e5d0b7a43430be0e83717313fc57a3140a8df5cf73c271bc |
| product_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| spectral_fit_arrays.npz | 3626 | 573f61f59c14484c6bf51e1ceeefcba0a17ded0c01408b9b6faf9ff4251d9008 |
| spectral_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| unconditional_fit_arrays.npz | 1644 | 3863c28c0f1774ed512194d39e6d49f39eebe24acc0ae46d85eaaa502678938f |
| unconditional_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |

### results/run_initial/seed_150901/null

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | 94459f897bb694ec4026df90555acb2d514afd6d63f851a34468084e539c1a38 |

### results/run_initial/seed_150901/null/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21882 | eb7858365fedda46b9856f0f9120330eddb4b864970330afc42fc6a365a5d982 |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 6967 | 64948c3c2c92c361b25d61103ed40f86fa682c546ec919aea66ce1b11a6a4768 |
| histogram_lr_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| oracle_graph_harmonic_fit_arrays.npz | 63587 | e4ca47765fa36fbed5b919fbe3081cd56db409cc9675e75dd45793cd7da01869 |
| oracle_graph_harmonic_numerical.npz | 63068 | 5cd2e0985c693317e31d753f1324abd78e669c5c0a8175449972e5d13e400378 |
| oracle_graph_histogram_fit_arrays.npz | 63587 | e4ca47765fa36fbed5b919fbe3081cd56db409cc9675e75dd45793cd7da01869 |
| oracle_graph_histogram_numerical.npz | 63051 | 0602aa9e6df6b1448a0fe1f266e17f387029f9f57eff42d9c08fa95f56e3db5c |
| product_fit_arrays.npz | 493 | 0b4e2990626f5907adb6d0c31a8306a0257e17591cf076e9356d2980dac51858 |
| product_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| spectral_fit_arrays.npz | 3624 | 56ff068587807877dcd4690a9f8a24088cfd09389a882669cdf6e9907dfa2ff1 |
| spectral_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| unconditional_fit_arrays.npz | 1638 | 42aab9836300e11c9fede9cf11ac252c989457baa93164b79ea5486ed3456a73 |
| unconditional_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |

### results/run_initial/seed_150901/null/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21896 | fa633bb30e8394d23349e12d45d6dbac97906f00bb68da3a370da0e25842b43c |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 6981 | 04aad91e76a7e73ef7fb97b26d1c7eb043a77db6b1619c3309ea4add974e0c8e |
| histogram_lr_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| oracle_graph_harmonic_fit_arrays.npz | 31531 | c1fc994689d30b56e64bf7e6ac44f87b81db9beb105e1c860f0783a6736aa1df |
| oracle_graph_harmonic_numerical.npz | 63048 | fc74d02a333b3f54647f0359aab84f5db5d70a0e61a9d806920907d0656c1c25 |
| oracle_graph_histogram_fit_arrays.npz | 31531 | c1fc994689d30b56e64bf7e6ac44f87b81db9beb105e1c860f0783a6736aa1df |
| oracle_graph_histogram_numerical.npz | 63063 | 962a1b2afd5d2a75a355831f5533c86a660de93024f39c9ed8793c23fc747692 |
| product_fit_arrays.npz | 494 | afa14aaf7b604500e5d0b7a43430be0e83717313fc57a3140a8df5cf73c271bc |
| product_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| spectral_fit_arrays.npz | 3624 | 5dd9a9a0252771e907121290c1654226006bd0d5fc6665e35fe9c98885d2c0e3 |
| spectral_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| unconditional_fit_arrays.npz | 1640 | 87d92b8fd9b78739c790cf1f456dfe8b541c612ae24e92a00ce1a5bccace06bb |
| unconditional_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |

### results/run_initial/seed_150901/off_basis

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | 47fa314a3ba93ff90eae195e187172110960aa8df627e79fd85878bf2f9c3cd0 |

### results/run_initial/seed_150901/off_basis/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21882 | eb7858365fedda46b9856f0f9120330eddb4b864970330afc42fc6a365a5d982 |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 70162 | 2d8e366d57a3f8f112933264823f54f723bb792e8db99bff9882250051915244 |
| histogram_lr_numerical.npz | 63061 | df0171c7c7510be6d5a68dfcd2f92148061a4862ccbc407065cf044d5882bcef |
| oracle_graph_harmonic_fit_arrays.npz | 63585 | f7221a5141370c39827c9e8a80261f07f87c38ddc235f5bcf52b40cc624e078f |
| oracle_graph_harmonic_numerical.npz | 63061 | 95aceb9ba4bcebf901d358dc0d976edd68946bdb4e60d0fb20fd761d6158c270 |
| oracle_graph_histogram_fit_arrays.npz | 63585 | f7221a5141370c39827c9e8a80261f07f87c38ddc235f5bcf52b40cc624e078f |
| oracle_graph_histogram_numerical.npz | 63054 | 7a1a6150e7666cc0588e0be442f1f03cbfedb07fa76a1a93f53b09db69ddb348 |
| product_fit_arrays.npz | 493 | 0b4e2990626f5907adb6d0c31a8306a0257e17591cf076e9356d2980dac51858 |
| product_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| spectral_fit_arrays.npz | 3623 | 39b16ee7e28ebbad2f6d9039d0d949a75d1896c46b978995891e91fbeb9cb05f |
| spectral_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |
| unconditional_fit_arrays.npz | 1638 | 1b39c57c098d0330eb27bb5d7efcad7e58d7b7b623955eaed737f0de834f8ed8 |
| unconditional_numerical.npz | 62927 | cbd7c264d4f8d3f6c9c384fa7d01824b53758bfcbfe5f2f484c0c6732f491bed |

### results/run_initial/seed_150901/off_basis/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21896 | fa633bb30e8394d23349e12d45d6dbac97906f00bb68da3a370da0e25842b43c |
| generation_source.npy | 20608 | 294f43604ba67901b4e5556d744f770875858609ab2fa4a31c330593e4ddc1dd |
| histogram_lr_fit_arrays.npz | 6975 | ebeacd818bb032910bdc404f0c51fe2b3af4b1e1bb673a973662fb109c867d4e |
| histogram_lr_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| oracle_graph_harmonic_fit_arrays.npz | 31505 | dad4c74aaabaa6ed79920361573d9fffbdeba5b772ae48bffd2a90f385be6ed0 |
| oracle_graph_harmonic_numerical.npz | 63059 | 30ac38e5f832f6c2f0b52f819b24d26e8c8225c31b7c3d9ae7573456898868b7 |
| oracle_graph_histogram_fit_arrays.npz | 31505 | dad4c74aaabaa6ed79920361573d9fffbdeba5b772ae48bffd2a90f385be6ed0 |
| oracle_graph_histogram_numerical.npz | 63074 | e4070d2d19674af5bdb12c63ab560d887f2b6c295c7e9225f8e8dbbd357af934 |
| product_fit_arrays.npz | 494 | afa14aaf7b604500e5d0b7a43430be0e83717313fc57a3140a8df5cf73c271bc |
| product_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| spectral_fit_arrays.npz | 3629 | 841a78a8d30ece700836514ec8decf3a3ec5e74722895d747aaedaccf44e8da9 |
| spectral_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |
| unconditional_fit_arrays.npz | 1641 | d16fdba72f55c9b12de7a4b5f72878b3dd2432e38a22cc387c2230e539505696 |
| unconditional_numerical.npz | 62938 | 10cf12d08fc95792521972c423836e5ded6302582629919c0fd029c9c7b04519 |

### results/run_initial/seed_150902

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed_source.npy | 11141248 | a1459fbed3792a94d33a43891c86abd3032ad74ca00002bf0fd37f6e56b7dc54 |

### results/run_initial/seed_150902/harmonic

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | 5acd565ad40e0f148fcfbdc2d7a3e01d0662cb1ac5656b75243549dc7894c5f9 |

### results/run_initial/seed_150902/harmonic/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 22008 | 3e674f6ae252d412adfde452677b276c1bde531f7076fa4b5a520600747e8a02 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 70105 | d20fbb0ce1634e3d19de820bcbff7bb665bbe6ab57ca6b9185931fd6e3bdf05e |
| histogram_lr_numerical.npz | 63004 | 1888d35e2b6cac93a5b14247368aa7dd453771ef42d8ca0a098f605caf6d5bbb |
| oracle_graph_harmonic_fit_arrays.npz | 63526 | a688ccac4cea83bdc826d7585c664d167d6f16e8f45637ccfd3d6e88e0cea12c |
| oracle_graph_harmonic_numerical.npz | 63014 | 51ed93dc35d26ed67fd0c2007384d17206c6aea8f9fc73e1b6beb1f71a16b594 |
| oracle_graph_histogram_fit_arrays.npz | 63526 | a688ccac4cea83bdc826d7585c664d167d6f16e8f45637ccfd3d6e88e0cea12c |
| oracle_graph_histogram_numerical.npz | 62999 | 22e99acd9cf6dfa88aac5a3f721df239984ab2b0852ebbdbeee90f7e8d8d3858 |
| product_fit_arrays.npz | 494 | b885e7eee54be579df729f435d063329032c188f5f467ed10511f6908d4ceca1 |
| product_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| spectral_fit_arrays.npz | 66655 | de4d6fab69f8576c6039cc2de08f19119263101fc09ed906be41f295d17ef569 |
| spectral_numerical.npz | 63013 | 5509338019a4b372561ead24268d7754d3817d5f3edd9e5667c213d7c4137e5a |
| unconditional_fit_arrays.npz | 1640 | dc5dbe37f53400d67d2f0a9d6fe0f62d1138e3d2ff73b772f512e510684bcb6e |
| unconditional_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |

### results/run_initial/seed_150902/harmonic/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21942 | 172a7ade2b1ce349765ac82165c0317605d14bc764f2f418e3804bcf57619608 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 6981 | 0a4cec89472d47f80ca477a804918a7e663a89520e5da8f44128b260405f3072 |
| histogram_lr_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| oracle_graph_harmonic_fit_arrays.npz | 31560 | 5a728f36a26f9cedb54fd27b031ac4327eb360293dbd653f33ab6d3af043404a |
| oracle_graph_harmonic_numerical.npz | 63000 | 2b847190421f00e4de189f4ba5f277196e6543ec9b400ebbd14f4282dff3b0ab |
| oracle_graph_histogram_fit_arrays.npz | 31560 | 5a728f36a26f9cedb54fd27b031ac4327eb360293dbd653f33ab6d3af043404a |
| oracle_graph_histogram_numerical.npz | 63019 | 1d29e1e0472d5a189d1d7b641df7ef3a2f73cbea0faf1e9cebc55f0565d893fd |
| product_fit_arrays.npz | 493 | 2eae3006381165891ab73f4a512dfab103427e57c42431f8e2115d2c1cd5966d |
| product_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| spectral_fit_arrays.npz | 3616 | cbd7e1eb8a4c1542695fdece8d324981b4f621399601bef6168e546a4680ea4f |
| spectral_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| unconditional_fit_arrays.npz | 1643 | 8e185d45647f5232650cece06ba404a66d784f499fa8383d80785af47c638ecb |
| unconditional_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |

### results/run_initial/seed_150902/null

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | 2b15fac864b45b33a54b865040dc8ebfd4b6bad9e25bcac5ca63c67f23aca01c |

### results/run_initial/seed_150902/null/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21924 | fef5a336b5494da7ad4f410f77a45769ac168c373845b44be868202bca083216 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 6969 | 89bf6e2fb54f70f503b8afacd3ecf30f4777a89b508f68ec2c534bb92835cb3f |
| histogram_lr_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| oracle_graph_harmonic_fit_arrays.npz | 63573 | bb82e28095356a272870ea01a8c68a82c47e7bcb1a31810587f6637d2a880f49 |
| oracle_graph_harmonic_numerical.npz | 62989 | c8bc0b21b4050cc7bac577518bbbbad94eb896be705359b3da86227c1c1d11fd |
| oracle_graph_histogram_fit_arrays.npz | 63573 | bb82e28095356a272870ea01a8c68a82c47e7bcb1a31810587f6637d2a880f49 |
| oracle_graph_histogram_numerical.npz | 63006 | d5d5febac7de1f445401fbae5b39160cb0260f2f07a1befb40a0bc0e483c5f93 |
| product_fit_arrays.npz | 494 | b885e7eee54be579df729f435d063329032c188f5f467ed10511f6908d4ceca1 |
| product_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| spectral_fit_arrays.npz | 3629 | c1521f9572baa9669ae56ecc5fbd85029aae505d75699090857332a77f383884 |
| spectral_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| unconditional_fit_arrays.npz | 1640 | 69f97c49e3351e718844dce69034023cbc43174092d899e47b99be44cb32cf5a |
| unconditional_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |

### results/run_initial/seed_150902/null/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21942 | 172a7ade2b1ce349765ac82165c0317605d14bc764f2f418e3804bcf57619608 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 6968 | 8d260213ac26da6a95b909189e6e1dbb8fd99c797620e90d4aa237d5a19427b0 |
| histogram_lr_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| oracle_graph_harmonic_fit_arrays.npz | 31552 | 073c22cce8a5732ac330d096cd9ee3e280b7b7488311bccf1534400fde7ab486 |
| oracle_graph_harmonic_numerical.npz | 63002 | 84972ec0d2a2c21e9dc5f1aeaa2c5a14647c2bf37d93b096d635970ede0bef16 |
| oracle_graph_histogram_fit_arrays.npz | 31552 | 073c22cce8a5732ac330d096cd9ee3e280b7b7488311bccf1534400fde7ab486 |
| oracle_graph_histogram_numerical.npz | 63029 | 79f62e63ab0f69333992d8b3bcb2d21ef76302389ac4e2ab63f00c63411dc885 |
| product_fit_arrays.npz | 493 | 2eae3006381165891ab73f4a512dfab103427e57c42431f8e2115d2c1cd5966d |
| product_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| spectral_fit_arrays.npz | 3643 | d2f6c29c6a203c26cf61c1a56220351b418fa7c90ca745845179aaedb218fa6f |
| spectral_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| unconditional_fit_arrays.npz | 1643 | a8803299914b5dec4e7826a98aefff9c91e4d3ad4d79d0798c7e830f96c05282 |
| unconditional_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |

### results/run_initial/seed_150902/off_basis

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| observed.npy | 11141248 | d790c8e8ddcf2a29a1d007b93e7059352ef20a26e4f7701f96c93d028f4ef7a4 |

### results/run_initial/seed_150902/off_basis/large

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21924 | fef5a336b5494da7ad4f410f77a45769ac168c373845b44be868202bca083216 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 70136 | 760890514b084e5a71586721565b3b7ccc609797082d96c965d7927d30c40252 |
| histogram_lr_numerical.npz | 63017 | a27743de75a661d79ce9c0dc1a211e52d6753607393049a90ab6b01d2bfa48e4 |
| oracle_graph_harmonic_fit_arrays.npz | 63561 | 28325f05733ae68b49b2f8e8648df26eaec7652b495c95b87474c3419e2ae7f4 |
| oracle_graph_harmonic_numerical.npz | 63001 | d3dc861c3162008a80f088f6173e44726b507f2a938e210bbd54f28c8b7e533d |
| oracle_graph_histogram_fit_arrays.npz | 63561 | 28325f05733ae68b49b2f8e8648df26eaec7652b495c95b87474c3419e2ae7f4 |
| oracle_graph_histogram_numerical.npz | 63017 | adc8cde3d3836c84bf5a642f8c16551abfe123853a285c9a3dc8e74ffbfee1bd |
| product_fit_arrays.npz | 494 | b885e7eee54be579df729f435d063329032c188f5f467ed10511f6908d4ceca1 |
| product_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| spectral_fit_arrays.npz | 3625 | 6ce0cea76029b7f00ea9ab443dacc02a8898fa945c45925bc9c0549659a6abbd |
| spectral_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |
| unconditional_fit_arrays.npz | 1637 | 86c1d7a7c9fa7b28adbe667f837318101b5318018846f8fd60838432374b5b56 |
| unconditional_numerical.npz | 62921 | 731fe568ca2d90c7b789f7df1c8a52786170597c9a57108f27259c909181940a |

### results/run_initial/seed_150902/off_basis/low

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| exact_copy.npz | 21942 | 172a7ade2b1ce349765ac82165c0317605d14bc764f2f418e3804bcf57619608 |
| generation_source.npy | 20608 | 4a22c70d2730267215c50171daf98c779ed100b783e8de05a5fd34304e88c0e0 |
| histogram_lr_fit_arrays.npz | 6993 | 83dbbad61a31dac899ddc6cddf9e915bc5b3f07033ef58b47df74cd91b5b3a82 |
| histogram_lr_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| oracle_graph_harmonic_fit_arrays.npz | 31542 | b1311b7a759eef49d988611bdcf7ac5822165481c2ffc917e633b658ded85362 |
| oracle_graph_harmonic_numerical.npz | 63000 | ca2494e300b070179bf668cacfb7968313c770061c79e13bb0b27d2594f36526 |
| oracle_graph_histogram_fit_arrays.npz | 31542 | b1311b7a759eef49d988611bdcf7ac5822165481c2ffc917e633b658ded85362 |
| oracle_graph_histogram_numerical.npz | 62993 | e4e03d5ef6df93484003d9aa48c736f53b2a1e777376c2fe48c09c8c2968a791 |
| product_fit_arrays.npz | 493 | 2eae3006381165891ab73f4a512dfab103427e57c42431f8e2115d2c1cd5966d |
| product_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| spectral_fit_arrays.npz | 3641 | 9cbdcd2278aaa6ba85349e7fbd9c8e9397547c4cc63a6508bbd00efc486c3712 |
| spectral_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |
| unconditional_fit_arrays.npz | 1643 | 44365d8663da8cdcfb1e68d7a5ea9d680e929ba289885d346008e152cda9ffeb |
| unconditional_numerical.npz | 62940 | 3c8c84e84f89d94d14509e573047eb54a562fc5d2161a57ab17280972da3e6b2 |

### results/tests_initial

| Original filename | Bytes | SHA256 |
| --- | ---: | --- |
| arrays.npz | 98064 | a90714cdb9041b02bcda44d84359c99bd4fa087d0be69eaf6fe01bc44f3d5398 |
| endpoints.npz | 2272 | 184e3356f9fdd59952e9f873ceb1acb3d894513a40a0f76c9f69fdd7d55348f9 |
| histogram_outputs.npz | 15372 | ccee511b0dfa8700bd488c9b8ed438238b8adafa3bd8fa00823b31da58460b60 |
| inverse.npz | 7700 | a863b2285a5e3011724d2eff12a6fa287cae28d3e5b2f5cf2a9a49ac268245d7 |
| jacobian.npz | 1396 | 9e1e5894df34377018529f45336a12d9ab935c1f63228828490ba076f2e21516 |
| quadrature.npz | 1472 | 1c9220e1cc93948a065bf4995aa7e274f2e964d68e2c9d1fcdb826e2983c5007 |
| small_fit_arrays.npz | 1204 | dd5879f9e5b1cc5c5f820fc24fbbbb3a67fbd5ada612d33942fe52b100b95cac |

