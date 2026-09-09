# CPU failure replay audit

Job45612327 completed0:0 in1:50 onr366, maximum sampled batch RSS598480K. Scientific diagnostic source2b9ccd26f0d36634e0b88f4af02954fb250fb6a5. The failed source remains168efc227e361a86a2a6aa6952786e5d0e13e30f.

All18 output hashes,4 diagnostic source Git blobs and88 archived reference source hashes pass independently. Full229,573,563-byte result and Slurm log are retained in `cpu-full`, with the original transport archive and receipt. The exact failed batch identities match the earlier independent RNG/order-hash reconstruction. Actual32logits, coarse/residual/analysis determinants, all four conditioner arrays and kernel local tensors are preserved.

CPU encode returned valid. Independent reconstruction of good_bins, good_disc, good_den, good_theta and good_terms from saved tensors exactly matches each recorded mask; no invalid interior coordinate exists in any layer. Minimum distance below theta=1 is4.172325134277344e-7 at encode layer3, coordinate[13,1,3,7], an active coupling coordinate. This is merely numerical proximity; it does not attribute the original failure.

The CPU uses the same PSC Torch2.10.0+cu128 build and original64-row convolution chunk shapes, but CPU floating arithmetic differs fromGPU. Additionally the frozen CPU diagnostic ran decoder under no_grad, whereas the original failed forward had gradients enabled. Thus the original GPU exception was not reproduced and no fix or root cause is established. A separately reviewed GPU diagnostic must preserve original decoder modes/requires_grad and enable_grad. No optimizer or fit was rerun, no repair quality evaluated, and the original dirty checkout HEAD/status was verified unchanged during worktree creation.
