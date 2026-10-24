# Full production launch and first-stage checkpoint

The authorized fresh production started 2026-10-24 at 10:56:08 UTC in tmux `material-production-20261024`, using the final frozen material correction and common absolute prior coefficient 3.954930197858131. Exact commands, source/input/runtime bindings and resource allocations are retained. The order is topology, main, detection, contrast, then the long reconstruction sweep. The requested sweep realization option 40 retains four actual realizations under the recorded implementation.

Topology completed with a clean source/runtime/output audit. All raw output digests and all candidate-export digests were independently checked. The exported grid contains 68,121 nodes (261 by 261); its exact-data parser and reviewed static figure exporter both succeeded. No transport simulation is performed by this topology stage.

Main started at 10:56:18 UTC and remains running at this checkpoint. The snapshot explicitly marks overall execution incomplete. Main, detection, contrast, reconstruction results and final-field stationarity are not asserted here. The durable orchestrator records each later stage separately and automatically exports audited candidate assets, while keeping execution completion distinct from scientific acceptance.
