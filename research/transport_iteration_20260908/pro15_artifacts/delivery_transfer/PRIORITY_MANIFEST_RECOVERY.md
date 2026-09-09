# Original Pro15 manifests: lossless transfer

This transport contains exactly two unchanged original files from
pro15_artifacts_complete.zip. It contains no regenerated scientific output.
Concatenate the three binary parts in numerical order to recover a 19,332-byte
tar.xz archive. Its SHA256 must be:

8fceccbd44bd35cb7f273e8f52540f79271d100a1da424a1057c82493738c29b

The archive members, in order, are:

| Original path | Original bytes | Original SHA256 |
| --- | ---: | --- |
| MANIFEST.json | 90816 | a9833ccfc16d68be73bac882a158413e2fecc195796ce985d01a7ccdcd20f76c |
| SHA256SUMS | 59933 | a03bc35e7d1c6d18b1e66fd0d1db1d686accc60bf830443567ba0657516ad8f7 |

From a fresh recovery directory, after obtaining the three binary parts:

```sh
cat priority_manifests.tar.xz.part00 priority_manifests.tar.xz.part01 priority_manifests.tar.xz.part02 > priority_manifests.tar.xz
printf '%s  %s\n' 8fceccbd44bd35cb7f273e8f52540f79271d100a1da424a1057c82493738c29b priority_manifests.tar.xz | sha256sum -c -
tar -xJf priority_manifests.tar.xz
printf '%s  %s\n' a9833ccfc16d68be73bac882a158413e2fecc195796ce985d01a7ccdcd20f76c MANIFEST.json a03bc35e7d1c6d18b1e66fd0d1db1d686accc60bf830443567ba0657516ad8f7 SHA256SUMS | sha256sum -c -
```

This only extracts and hashes files. No original model, fixture, fitting,
evaluation, or test source needs to execute. These are transport parts, not
files named MANIFEST.json or SHA256SUMS containing compressed data.

Original full ZIP identity: 97,735,768 bytes, SHA256
c4e7f4bf8ea105588bc949049f09f5706007bd20775d3a02ac9b0708a98d8c68.
The ZIP itself has not been replaced or modified. All 185 original binary
array files are outside this text-only delivery.
