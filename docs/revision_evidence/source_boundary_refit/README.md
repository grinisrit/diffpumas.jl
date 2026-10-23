# Completed no-air boundary-convention refit

The single control completed with exit 0 at 2026-10-23 09:12:36 UTC after 60m09.9s on 1 CPU (actual affinity 20), under a 2 GiB address-space cap. Source, runtime and affinity audits passed. It matched the recorded 600-cell robustness baseline numerical configuration, pinned physics dump, seed, 32 CSDA nodes, 24 calibration bins × 8,192 MC samples and 3 passes, except the explicit primary-altitude 0 control, output directory and resource/runtime wrapper.

This removes the air propagation leg for valid upward rays; it is a convention control, NOT altitude-corrected physical truth. Original absolute paths and identities are retained in these snapshots; the archive is not a replacement run directory. The external launcher source is preserved byte-for-byte.

Both baseline and control reached the 3-pass ceiling WITHOUT convergence. The no-air final relative field change is .03821, versus .04041 for baseline, above the .02 tolerance. The control versus baseline field relativeL2 difference is 3.465%, RMS parameter difference .003966 and maximum cell difference .02960. The supported-cell arithmetic mean mass-mixture parameter changes .133323→.128992 (−3.248%); this is not a groundwater inventory. Both supports contain 147 cells, with 117 above .001.

The fit mask changes 687→688 bins (added bin 1010). On the 687 common bins, each fit retaining its own fitted normalization, relative residual changes .370832→.371284 and log spread .758113→.756855. The normalization changes +0.175%. The separate alternate-seed baseline gives a 3.891% field-relativeL2 difference; one alternate seed does not calibrate uncertainty or establish that the boundary effect is negligible. Calibration uncertainty is not propagated, and changing air transport can change random-number consumption even under the same seed.

`descriptive_comparison_pending_outer.json` is explicitly a PROVISIONAL comparison of completed, output-hash-verified individual cases. The complete ten-scenario robustness outer audit was still pending at curation. This archive does not claim completion of that suite or substitute for the stricter final boundary-fit reporter. It does preserve the no-air case's own completed execution/source/runtime audit.

The original run log contains nonfatal Kaleido initialization warnings. The fit-only calculation nevertheless returned exit 0 with complete fit artifacts; raw log bytes, including whitespace, are preserved so recorded hashes remain valid.
