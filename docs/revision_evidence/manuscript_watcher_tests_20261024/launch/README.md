# Audited manuscript updater launch

The updater started on 24 October 2026 at 12:06:43 UTC in tmux `material-paper-refresh-20261024`. The live process was verified on CPU20 with a 2-GiB address-space limit. Its Python postprocessing lock is shared with production figure exports; child builders and compilers inherit the lock. It creates fresh isolated TeX/PDF candidates, binds their evidence and compiler outputs, and requires final scientific review before adoption.

At this snapshot, topology alone was complete; the primary main stage and the second dry starting-field control were running. The updater correctly reported `waiting_scientific_exports`, with no error. Its completion flag for the overall revision remains false. This launch record does not claim completion of the simulations, manuscript replacement or stability study.

Live status: `examples/data/material_production_20261024/candidate_manuscript/watcher_state.json`. Candidate pointers appear in `latest_candidate.json` under that directory when audited scientific exports arrive. The captured JSON files are timestamped launch evidence, not live status.
