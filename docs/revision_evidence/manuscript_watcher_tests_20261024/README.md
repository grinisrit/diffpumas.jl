# Manuscript candidate watcher operational checks

The bounded CPU20/2-GiB watcher tests passed after correcting a fixture-only symlink-path assertion. The final run passed all six tests, including early-rejection log preservation, explicit interrupted-controller recovery, unchanged-signature deduplication, and recovery after a helper change. Child affinity/address-space inheritance is checked against the operating system. The compile test generates a genuine one-page PDF from an explicitly synthetic minimal document and verifies separate compiler/PDF/log provenance.

The paper builder and stage-parser assertion suites also passed independently under the shared Python postprocessing lock. These checks establish operational/template behavior, not numerical correctness of future production results. The watcher never edits the authoritative manuscript, marks every candidate for review, distinguishes incomplete/assembled/rejected states, and never labels the overall scientific revision complete.

Exact tested source hashes, resource limits and test coverage are in `summary.json`. Actual new-stage claims remain gated by their independent numerical source/runtime/output audits.
