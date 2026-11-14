#!/usr/bin/env python
"""Figure A of the S7 paper: RMSE_s against exposure (top) and paired change
against MLEM (bottom), arms X, Mc (lines) and M (open markers).
Reads only the run output directory (read-only): test_summary.tsv, test/rows_shard*.tsv, gate/cells.tsv and the
saved reconstructions test/recs/*.jls (read with Julia, single thread, nice 19).
usage: make_s7_figure.py [OUT_DIR [FIG_DIR [FACTS_MD]]]
  OUT_DIR  run output directory (env S7_OUT; default /mnt/big_data/agent-builds/diffpumas-s7-out)
  FIG_DIR  where fig_s7_sweep.* and fig_s7_field.* and the cached reconstruction text files go (env S7_FIGDIR; default .)
  FACTS_MD optional s7_results_facts.md (made by s7_results_agg.py) for the table cross-check (env S7_FACTS; skipped if absent)"""
import csv, glob, os, re, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

O = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("S7_OUT", "/mnt/big_data/agent-builds/diffpumas-s7-out")).rstrip("/") + "/"
HERE = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("S7_FIGDIR", "."))
os.makedirs(HERE, exist_ok=True)
FACTS = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("S7_FACTS", os.path.join(HERE, "s7_results_facts.md"))
EXPO = {1: 1e3, 3: 1e4, 4: 3.16e4, 5: 1e5, 6: 1e6, 7: 1e8}
EMPTY = 0.3291
T975_7 = 2.365

# style of make_figures.py (font 12, white face, grid alpha 0.3); sizes raised for 7 in -> 5.4 in scaling
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
})

# Okabe-Ito, one colour + one marker per solver
STY = {
    "MLEM":     dict(c="#0072B2", m="o", ls="-",  lw=1.5, ms=5.5),
    "GN-0":     dict(c="#D55E00", m="s", ls="-",  lw=1.5, ms=5.0),
    "MLEM->GN": dict(c="#009E73", m="^", ls="-",  lw=1.5, ms=6.0),
    "GN-lin":   dict(c="#E69F00", m="D", ls="--", lw=1.3, ms=4.5),
    "SART":     dict(c="#7F7F7F", m="v", ls=":",  lw=0.9, ms=3.5),
    "GD-0":     dict(c="#4D4D4D", m="X", ls="-.", lw=0.9, ms=3.8),
}
LAB = {"MLEM": "MLEM", "GN-0": "GN-0", "MLEM->GN": r"MLEM$\to$GN", "GN-lin": "GN-lin",
       "SART": "SART", "GD-0": "GD-0"}
PRIMARY = ["MLEM", "GN-0", "MLEM->GN"]
TOP = {"X": ["SART", "GD-0", "GN-lin", "MLEM", "GN-0", "MLEM->GN"],
       "Mc": ["SART", "GD-0", "GN-lin", "MLEM", "GN-0", "MLEM->GN"],
       "M": PRIMARY}
SHIFT = 1.12


def load_rows(pat):
    out = []
    for f in sorted(glob.glob(O + pat)):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["stage"] != "stage":
                out.append(r)
    return out


summ = list(csv.DictReader(open(O + "test_summary.tsv"), delimiter="\t"))
cell = {}
contrast = {}
for r in summ:
    k = r["kind"]
    if k == "cell":
        mean = float(r["mean"])
        sd = float(r["sd"]) if r["sd"] else float("nan")
        lo = float(r["lo"]) if r["lo"] else mean - T975_7 * sd / np.sqrt(8)
        hi = float(r["hi"]) if r["hi"] else mean + T975_7 * sd / np.sqrt(8)
        cell[(r["arm"], r["solver"], int(r["exp_index"]))] = (mean, lo, hi)
    elif k in ("C1", "C2"):
        contrast[(k, r["arm"], int(r["exp_index"]))] = (float(r["mean"]), float(r["lo"]), float(r["hi"]))
n_c1 = sum(1 for k in contrast if k[0] == "C1")
n_c2 = sum(1 for k in contrast if k[0] == "C2")
assert (n_c1, n_c2) == (14, 14), (n_c1, n_c2)

# single realisations (selected = 1, observation weights)
T = load_rows("test/rows_shard*.tsv")
single = collections.defaultdict(dict)
for r in T:
    if r["selected"] == "1" and r["weights"] == "obs":
        single[(r["arm"], r["solver"], int(r["exp_index"]))][int(r["rep"])] = float(r["rmse_s"])

# --- checks: means vs rows, paired contrasts vs rows
for k, (m, lo, hi) in cell.items():
    v = list(single[k].values())
    assert len(v) == 8 and abs(np.mean(v) - m) < 1e-9, k
for (kind, arm, e), (m, lo, hi) in contrast.items():
    s = "MLEM->GN" if kind == "C1" else "GN-0"
    d = [single[(arm, s, e)][i] - single[(arm, "MLEM", e)][i] for i in range(1, 9)]
    assert abs(np.mean(d) - m) < 1e-9, (kind, arm, e)


def check_against_facts():
    """plotted means equal the facts section-3 tables to 4 decimals"""
    if not os.path.exists(FACTS):
        print("facts file not found; skipping table check"); return 0
    txt = open(FACTS).read().split("## 3.")[1].split("Secondary metrics")[0]
    n = 0
    arm = None
    cols = None
    for line in txt.splitlines():
        m = re.match(r"^Arm (\w+):", line)
        if m:
            arm = m.group(1); continue
        if line.startswith("| solver"):
            cols = [int(c.split(":")[0]) for c in line.split("|")[2:-1]]; continue
        if arm and cols and line.startswith("| ") and not line.startswith("| ---"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            sol = parts[0]
            if sol not in STY:
                continue
            for e, p in zip(cols, parts[1:]):
                mm = re.match(r"\**([0-9.]+) ±", p)
                if mm:
                    assert abs(float(mm.group(1)) - cell[(arm, sol, e)][0]) < 5.1e-5, (arm, sol, e, p)
                    n += 1
    return n


n_checked = check_against_facts()


def jitter(x, n, rng):
    return x * np.exp(rng.uniform(-0.03, 0.03, n))   # +-3 % in tau


def draw_top(ax, arm, shift=1.0, overlay=False, rng=None):
    for s in TOP[arm]:
        st = STY[s]
        exps = [e for e in EXPO if (arm, s, e) in cell]
        if not exps:
            continue
        x = np.array([EXPO[e] for e in exps]) * shift
        y = np.array([cell[(arm, s, e)][0] for e in exps])
        lo = np.array([cell[(arm, s, e)][1] for e in exps])
        hi = np.array([cell[(arm, s, e)][2] for e in exps])
        primary = s in PRIMARY
        z = 4 if primary else 3
        if s in PRIMARY:   # 8 single realisations per cell
            for e in exps:
                v = np.array([single[(arm, s, e)][i] for i in sorted(single[(arm, s, e)])])
                ax.scatter(jitter(EXPO[e] * shift, len(v), rng), v, s=5, color=st["c"], alpha=0.4,
                           linewidths=0, zorder=2)
        kw = dict(color=st["c"], marker=st["m"], ms=st["ms"], lw=st["lw"], ls=st["ls"], zorder=z,
                  elinewidth=0.9 if primary else 0.6, capsize=2 if primary else 1.5,
                  mew=0.9 if overlay else 0.6)
        if overlay:
            kw.update(ls="none", mfc="white", elinewidth=0.9, mew=1.1)
        elif not primary and s in ("SART", "GD-0"):
            kw.update(alpha=0.9)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], **kw)


def draw_bottom(ax, arm, shift=1.0, overlay=False):
    for s, kind in (("GN-0", "C2"), ("MLEM->GN", "C1")):
        st = STY[s]
        exps = [e for e in EXPO if (kind, arm, e) in contrast]
        x = np.array([EXPO[e] for e in exps]) * shift
        y = np.array([contrast[(kind, arm, e)][0] for e in exps])
        lo = np.array([contrast[(kind, arm, e)][1] for e in exps])
        hi = np.array([contrast[(kind, arm, e)][2] for e in exps])
        kw = dict(color=st["c"], marker=st["m"], ms=st["ms"], lw=st["lw"], ls=st["ls"], zorder=4,
                  elinewidth=0.9, capsize=2, mew=0.6)
        if overlay:
            kw.update(ls="none", mfc="white", mew=1.1)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], **kw)


fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex="col", sharey="row",
                        gridspec_kw=dict(height_ratios=[2, 1], hspace=0.17, wspace=0.07,
                                         left=0.115, right=0.985, top=0.955, bottom=0.105))
(a, b), (c, d) = axs
rng = np.random.default_rng(7)
draw_top(a, "X", rng=rng)
draw_top(b, "Mc", rng=rng)
draw_top(b, "M", shift=SHIFT, overlay=True, rng=rng)
draw_bottom(c, "X")
draw_bottom(d, "Mc")
draw_bottom(d, "M", shift=SHIFT, overlay=True)

for ax in (a, b):
    ax.axhline(EMPTY, color="k", lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.set_ylim(0, 0.35)
for ax in (a, b):
    ax.text(1.6e8, EMPTY - 0.006, "empty field", ha="right", va="top", fontsize=9)
for ax in (c, d):
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    ax.set_ylim(-0.17, 0.06)
for ax in axs.ravel():
    ax.set_xscale("log")
    ax.set_xlim(5e2, 2e8)
    ax.set_xticks([10.0 ** k for k in range(3, 9)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
a.set_title("(a) X", loc="left", fontsize=11, pad=3)
b.set_title("(b) Mc (filled), M (open)", loc="left", fontsize=11, pad=3)
c.set_title("(c) X", loc="left", fontsize=11, pad=3)
d.set_title("(d) Mc (filled), M (open)", loc="left", fontsize=11, pad=3)
a.set_ylabel(r"$\mathrm{RMSE}_s$")
c.set_ylabel(r"$\Delta=\mathrm{RMSE}_s-\mathrm{RMSE}_s$(MLEM)", fontsize=9.5)
for ax in (c, d):
    ax.set_xlabel(r"per-bin exposure $\tau$ [m$^2$ sr s]")
for ax in (b, d):
    ax.tick_params(labelleft=False)
# tau = 7.25e6: counting error of the median bin equals the 1.99 % MC-template error
for ax in (b, d):
    ax.axvline(7.25e6, color="k", lw=0.8, ls=(0, (1, 2)), zorder=1)

# single legend, panel (a), lower left (no data there)
hs = []
for s in ["MLEM", "GN-0", "MLEM->GN", "GN-lin", "SART", "GD-0"]:
    st = STY[s]
    hs.append(Line2D([], [], color=st["c"], marker=st["m"], ms=st["ms"], lw=st["lw"], ls=st["ls"],
                     label=LAB[s]))
hs.append(Line2D([], [], color="0.4", marker="o", ms=2.5, lw=0, alpha=0.5,
                 label="realisations"))
a.legend(handles=hs, loc="lower left", ncol=1, frameon=True, framealpha=0.9, borderpad=0.4,
         labelspacing=0.3, handlelength=1.8)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_s7_sweep." + ext), dpi=200)
plt.close(fig)

# spot values
for arm in ("X", "Mc", "M"):
    print("MLEM->GN minus MLEM at 1e8, arm", arm, "= %.4f" % contrast[("C1", arm, 7)][0],
          "[%.4f, %.4f]" % contrast[("C1", arm, 7)][1:])
print("C1 points:", n_c1, "C2 points:", n_c2, "; means checked vs facts tables:", n_checked)


# =====================================================================
# Figure B: truth, MLEM, MLEM->GN and difference, arm Mc, tau = 1e8, realisation 1
# =====================================================================
import subprocess
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

RECS = O + "test/recs/"
RECFILES = {"MLEM": ("test_obs_test_Mc_e7_r1_MLEM_1p6.jls", "n=50", "rec_mlem.txt"),
            "MLEM->GN": ("test_obs_cap20_test_Mc_e7_r1_MLEMtoGN_0p01.jls", "K=2", "rec_mlemgn.txt")}


def read_rec(name):
    f, key, out = RECFILES[name]
    p = os.path.join(HERE, out)
    if not os.path.exists(p):
        jl = ('using Serialization\nx=deserialize("%s"); v=x["%s"]\n'
              'open("%s","w") do io; for a in v; println(io,repr(a)); end; end\n' % (RECS + f, key, p))
        subprocess.run(["nice", "-n", "19", "julia", "-t", "1", "-e", jl], check=True)
    return np.loadtxt(p)


if all(os.path.exists(RECS + v[0]) for v in RECFILES.values()):
    cells = list(csv.DictReader(open(O + "gate/cells.tsv"), delimiter="\t"))
    cx = np.array([float(r["x"]) for r in cells]); cy = np.array([float(r["y"]) for r in cells])
    cz = np.array([float(r["z"]) for r in cells])
    crossed = np.array([r["crossed"] == "1" for r in cells])
    evalm = np.array([r["evaluated"] == "1" for r in cells])
    wt = np.array([float(r["w_test"]) for r in cells])
    wm = read_rec("MLEM"); wg = read_rec("MLEM->GN")
    assert len(wm) == len(wg) == len(cells) == 600 and crossed.sum() == 76
    D_s = evalm & crossed
    rm = lambda w: float(np.sqrt(np.mean((w[D_s] - wt[D_s]) ** 2)))
    # the vectors must reproduce the RMSE_s of the rows of the results file
    row = {r["solver"]: float(r["rmse_s"]) for r in T if (r["arm"], r["exp_index"], r["rep"], r["selected"], r["weights"])
           == ("Mc", "7", "1", "1", "obs") and r["solver"] in ("MLEM", "MLEM->GN")}
    assert abs(rm(wm) - row["MLEM"]) < 1e-6 and abs(rm(wg) - row["MLEM->GN"]) < 1e-6, (rm(wm), rm(wg), row)
    print("fig B: RMSE_s MLEM %.4f, MLEM->GN %.4f (match rows); crossed %d, nonzero truth %d (%d uncrossed)"
          % (rm(wm), rm(wg), crossed.sum(), (wt > 0).sum(), ((wt > 0) & ~crossed).sum()))

    VMAX = 0.9
    err = wg - wt
    ecap = float(np.max(np.abs(err[crossed])))
    ecap = np.ceil(ecap * 20) / 20           # symmetric limit, rounded up to 0.05
    cmap_seq, cmap_div = "YlGnBu", "RdBu_r"
    GREY = "#cfcfcf"

    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(2, 2, left=0.0, right=0.985, top=0.955, bottom=0.14, hspace=0.06, wspace=0.0)
    axes3 = [fig.add_subplot(gs[0, 0], projection="3d"), fig.add_subplot(gs[0, 1], projection="3d"),
             fig.add_subplot(gs[1, 0], projection="3d"), fig.add_subplot(gs[1, 1], projection="3d")]
    cax1 = fig.add_axes([0.07, 0.075, 0.38, 0.022]); cax2 = fig.add_axes([0.57, 0.075, 0.38, 0.022])

    def panel(ax, w, title, cmap, vmin, vmax, norm=None, show_uncrossed=True, outline_unc=False):
        m = crossed
        size = 14.0 + 130.0 * (np.abs(w[m]) / vmax)
        sc = ax.scatter(cx[m], cy[m], cz[m], c=w[m], cmap=cmap, vmin=None if norm else vmin,
                        vmax=None if norm else vmax, norm=norm, s=size, depthshade=False,
                        edgecolors="0.35", linewidths=0.3, zorder=3)
        if show_uncrossed:
            u = ~crossed
            ax.scatter(cx[u], cy[u], cz[u], color=GREY, s=7, alpha=0.55, linewidths=0, depthshade=False, zorder=1)
        if outline_unc:   # truth: shown everywhere; nonzero uncrossed cells outlined
            u = (~crossed) & (wt > 0)
            ax.scatter(cx[u], cy[u], cz[u], c=w[u], cmap=cmap, vmin=vmin, vmax=vmax,
                       s=14.0 + 130.0 * (w[u] / vmax), depthshade=False, edgecolors="k", linewidths=1.3, zorder=3)
        ax.scatter([0], [0], [0], color="red", s=60, marker="v", depthshade=False, zorder=5)
        ax.view_init(elev=12, azim=-58)
        ax.set_xlim(cx.min(), cx.max()); ax.set_ylim(cy.min(), cy.max()); ax.set_zlim(cz.min(), cz.max())
        ax.set_title(title, fontsize=11, pad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}"))
        ax.tick_params(labelsize=9, pad=-2)
        ax.set_xlabel("East [km]", fontsize=9.5, labelpad=-4)
        ax.set_ylabel("North [km]", fontsize=9.5, labelpad=-4)
        ax.set_zlabel("Height [km]", fontsize=9.5, labelpad=-4)
        return sc

    s_a = panel(axes3[0], wt, "(a) truth", cmap_seq, 0, VMAX, show_uncrossed=False, outline_unc=True)
    panel(axes3[1], wm, "(b) MLEM", cmap_seq, 0, VMAX)
    panel(axes3[2], wg, r"(c) MLEM$\to$GN", cmap_seq, 0, VMAX)
    s_d = panel(axes3[3], err, r"(d) (c) $-$ (a)", cmap_div, -ecap, ecap,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-ecap, vmax=ecap))
    # the signed-error marker area must follow |dw| on the scale of the difference
    s_d.set_sizes(14.0 + 130.0 * (np.abs(err[crossed]) / ecap))
    cb1 = fig.colorbar(s_a, cax=cax1, orientation="horizontal")
    cb1.set_label(r"water mass fraction $w$, panels (a)$-$(c)", fontsize=10)
    cb2 = fig.colorbar(s_d, cax=cax2, orientation="horizontal")
    cb2.set_label(r"$\Delta w$ (reconstruction $-$ truth), panel (d)", fontsize=10)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=9)
    # legends: one for the grey cells, one for the outlined truth cells, detector marker
    hl_b = [Line2D([], [], color=GREY, marker="o", ms=4, lw=0, label="not crossed by rays"),
            Line2D([], [], color="red", marker="v", ms=6, lw=0, label="detector")]
    axes3[1].legend(handles=hl_b, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8.5,
                    framealpha=0.9, borderpad=0.3, handletextpad=0.3)
    hl_a = [Line2D([], [], mfc="none", mec="k", mew=1.3, marker="o", ms=6, lw=0,
                   label="nonzero truth,\nnot crossed")]
    axes3[0].legend(handles=hl_a, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8.5,
                    framealpha=0.9, borderpad=0.3, handletextpad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_s7_field." + ext), dpi=200)
    plt.close(fig)
    print("fig B: max |dw| over crossed cells = %.4f; limit used +-%.2f; max |dw| all cells %.4f" % (np.max(np.abs(err[crossed])), ecap, np.max(np.abs(err))))
    print("fig B written; difference limit +-%.2f" % ecap)
else:
    print("fig B skipped: reconstruction vectors not on disk")
