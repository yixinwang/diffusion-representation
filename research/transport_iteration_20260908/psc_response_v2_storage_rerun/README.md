# Fresh storage-corrected study 45632257

One submission was accepted after full authentication of the quota-failed job 45619353. All 21 fits start from scratch at source `3b8c0f7bef604a90ff329fd9608639527b67a972`, unchanged seeds, 90-minute allocation, and unchanged scientific/numerical gates. No partial checkpoint is reused.

The old project's actual block quota was exceeded. The new output directory under authorized project mth260022p passed exclusive write, fsync, readback, and removal of a tiny probe. Lustre project 559736 reported approximately 985 GiB of block headroom with ample inode headroom. Both study output and scheduler stdout now reside there. This is stronger evidence than the earlier rounded allocation report, but remains a timestamped check rather than a reservation of future space.

The launcher verified all 94 source hashes against the same frozen Git revision, pinned evaluator bytes, and unchanged original user checkout HEAD/dirty-state hash. Compute still uses the authorized cis260243p GPU account and the same typed V100 pool.

At 19:30:52 UTC on September 9, job 45632257 was RUNNING on v023 at elapsed 1:05, with source setup underway. No completed preflight result or quality result was available at this snapshot. All-fits freeze and numerical admissions must precede quality. The source is unchanged; this package is a submission/storage receipt, not a successful experiment.

The first supplemental quota-discovery calls used an older login Python and failed because its subprocess API lacked an argument. Those errors are preserved alongside the subsequent successful actual Lustre checks; the write/read probe itself passed.
