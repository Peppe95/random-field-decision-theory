from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "static"
OUT = ROOT / "figures" / "static"

A = pd.read_csv(RESULTS / "figure4_panelA_hab22_scores.csv")
B = pd.read_csv(RESULTS / "figure4_panelB_hab22_cell_differences.csv")
C = pd.read_csv(RESULTS / "figure4_panelC_choices13k_fold_differences.csv")

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linewidth=0.6, alpha=0.20)
    ax.tick_params(axis="y", length=0)

fig = plt.figure(figsize=(8.4, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.14], hspace=0.58, wspace=0.50)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# A: HAB22 participant-generalization benchmark
y = np.arange(len(A))[::-1]
rf = A["model"].eq("RFDT-P")
axA.scatter(A.loc[~rf, "mse"], y[~rf], s=28, zorder=3)
axA.scatter(A.loc[rf, "mse"], y[rf], s=52, marker="D", zorder=4)
axA.set_yticks(y)
axA.set_yticklabels(A["model"], fontsize=9)
axA.set_xlim(0.0295, 0.0405)
axA.set_xlabel("Held-out MSE")
clean(axA)
axA.text(-0.05, 1.03, "A", transform=axA.transAxes, fontweight="bold", fontsize=12)

def delta_panel(ax, df, order, letter, xmin, xmax):
    rng = np.random.default_rng(11)
    ymap = {lab:y for lab,y in zip(order, np.arange(len(order))[::-1])}
    xs, ys = [], []
    for lab in order:
        if "restriction" in df.columns:
            vals = df.loc[df["restriction"] == lab, "delta_mse"].to_numpy()
        else:
            vals = df.loc[df["other"].map({
                   "RFDT-P kappa=0":"No interaction",
                   "RFDT-P alpha=1":"Adaptive only",
                   "RFDT-P alpha=0":"Persistent only",
                   "RFDT-L":"RFDT-L",
                   "Power-EU logit":"Power-EU",
                   "Linear-EU logit":"Linear-EU"
               }) == lab, "delta_mse_other_minus_full"].to_numpy()
        xs.extend(vals)
        ys.extend(np.full(len(vals), ymap[lab]) + rng.uniform(-0.10,0.10,len(vals)))
    ax.scatter(xs, ys, s=14, alpha=0.42, zorder=2)
    means=[]
    for lab in order:
        if "restriction" in df.columns:
            vals=df.loc[df["restriction"]==lab,"delta_mse"].to_numpy()
        else:
            rev={"No interaction":"RFDT-P kappa=0","Adaptive only":"RFDT-P alpha=1","Persistent only":"RFDT-P alpha=0",
                 "RFDT-L":"RFDT-L","Power-EU":"Power-EU logit","Linear-EU":"Linear-EU logit"}
            vals=df.loc[df["other"]==rev[lab],"delta_mse_other_minus_full"].to_numpy()
        means.append(vals.mean())
    ax.scatter(means, [ymap[x] for x in order], s=48, marker="D", zorder=4)
    ax.axvline(0, linewidth=0.8, linestyle="--", alpha=0.35)
    ax.set_yticks([ymap[x] for x in order])
    ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("Held-out MSE difference (model − RFDT-P)", fontsize=8.5)
    clean(ax)
    ax.text(-0.15, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)

delta_panel(axB, B, ["No interaction","Adaptive only","Persistent only","Linear utility"], "B", -0.008, 0.052)
delta_panel(axC, C, ["No interaction","Adaptive only","Persistent only","RFDT-L","Power-EU","Linear-EU"], "C", -0.0012, 0.016)

fig.savefig(OUT / "Figure4_static.pdf", bbox_inches="tight")
fig.savefig(OUT / "Figure4_static.png", dpi=200, bbox_inches="tight")
