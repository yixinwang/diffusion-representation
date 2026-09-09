# Same-clip timestamp diagnostic

This diagnostic follows failed job45578905; it does not change or rerun the original endpoint-selection protocol. The original runner remains unchanged. `diagnostic_metadata.py` retains the same fixed training member, metadata-manifest hash, source/dependency guards and bounded reader. Before decoding it recomputes the bounded member hash and requires exact equality with the original attempt's `699175c50544283f3b8537387403ff5b82958e4b18e590611129013131d1601a`.

It records every decoded frame's index, PTS, DTS, exact time base and exposed format/color metadata. It writes and flushes each offending row before flagging nonincreasing timestamps or nulls and continuing. It does not create RGB arrays, select frames, sort timestamps, impute times, change codecs or substitute clips. A fixed10000-frame cap remains in force. Complete means diagnostic coverage, not endpoint feasibility or a repaired timestamp policy.

The separate external `launch_video_diagnostic_guarded.py` targets this diagnostic source with the same full frozen-HEAD and all264 dependency-file guards. `video_diagnostic_first_clip.slurm` points to a separate proposed `video-diagnostic-launch` staging directory. A new immutable checkout, output directory and root-provided frozen commit are required. No original launcher or first-run result is changed. No remote execution is authorized by this document itself.

The fabricated diagnostic test preserves all six timestamps1,2,4,3,null,5 with flagged indices3 and4 and no array output. Original failed source, raw retrieval archive, all retrieved artifacts, preparation/monitoring records and audit are preserved in the adjacent `psc_video_first_clip_failure` directory with an explicit complete-file SHA256/size manifest.
