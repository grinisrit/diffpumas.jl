# S7 analysis scripts

Post-run analysis of the S7 solver comparison (branch `s7-protocol`). They read the run output
directory and never write into it. Paths are not hard-coded:

- `S7_OUT`: run output directory (default `/mnt/big_data/agent-builds/diffpumas-s7-out`)
- `S7_WORK`: directory for intermediate files and outputs (default: current directory)

Julia readers are single-thread; run them as `nice -n 19 julia -t 1 <script>`. Python scripts print to stdout.

| script | inputs (in `S7_OUT`) | output |
|---|---|---|
| `s7_results_agg.py [OUT [FACTS]]` | `test_summary.tsv`, `selection.toml`, `cuts.toml`, `dry_calibration.toml`, `{dev,test,diag}/rows_*.tsv`, `log_*.txt`, `resources.log` | markdown facts file (`S7_FACTS`, default `s7_results_facts.md`) |
| `common.py`, `load_vec.py` | helpers: row loader, selection, t-interval; loads `S7_WORK/dump.tsv` and `gate/cells.tsv` | none |
| `mklist.py` | `test/rows_shard*.tsv` | `S7_WORK/need.txt`, the list of saved reconstructions needed |
| `dump2.jl` | `S7_WORK/need.txt`, `test/recs/*.jls` | `S7_WORK/dump.tsv`, reconstruction vectors |
| `tl2.jl` | `templates/template_test_{X,M}_*.jls`, `template_dev_X_*.jls` | `S7_WORK/tmpl.tsv`, templates and MC errors |
| `dump.jl`, `tl.jl` | one `test/recs/*.jls`, one template | key listing on stdout (exploration) |
| `chk.py` | `dump.tsv`, `test/rows_shard*.tsv` | check that saved vectors reproduce the logged `rmse_s` |
| `item1.py` | `test/rows_shard*.tsv`, `test_summary.tsv` | post hoc paired contrasts (GN-0 vs GN-lin, MLEM->GN vs GN-0, GN-lin vs MLEM) |
| `item2.py` | `dump.tsv`, test rows | GN-0 against MLEM->GN at equal c: iterate differences, data fit |
| `item3.py` | `tmpl.tsv` | template-noise diagnostic: counts per bin and counting error against the MC template error |
| `item4.py` | `test/rows_shard*.tsv`, `dev/rows_phase*.tsv` | solves, model evaluations and loaded wall times per solver |
| `item8.py` | `dump.tsv`, test rows, `gate/cells.tsv` | spread of secondary metrics and RMSE by height band; `S7_WORK/item8_bands.tsv` |
| `item9.py` | `dev/grids.toml`, `selection.toml` | grids actually run after edge extensions; SART q bound |
| `item13.py`, `item13b.py` | `dev/rows_phase*_shard*.tsv`, `selection.toml` | dev-stage margins of the selected setting to the runner-up (`item13.pkl`, then markdown) |
| `make_s7_figure.py [OUT [FIGDIR [FACTS]]]` | `test_summary.tsv`, `test/rows_shard*.tsv`, `gate/cells.tsv`, `test/recs/*.jls` | `fig_s7_sweep.{pdf,png}`, `fig_s7_field.{pdf,png}` in `FIGDIR` |

Order: `mklist.py`, `dump2.jl`, `tl2.jl`, then the `item*.py` scripts and `make_s7_figure.py`.
