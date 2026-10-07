#!/usr/bin/env python3
"""Rebuild the main dynamic RFDT manuscript figures from compact saved summaries.

This script performs no fitting and no decision-process simulation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "dynamic"
OUT = ROOT / "figures" / "dynamic"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure5():
    df = pd.read_csv(RESULTS / "figure5_primary_comparisons.csv")
    targets = ["all_single_click_RTs", "0.3-10s_refitted"]
    comparators = ["DDM", "uncoupled_RFDT"]
    versions = ["independent_evaluation", "larger_bank_refit"]

    ybase = {
        ("DDM", "all_single_click_RTs"): 3.0,
        ("DDM", "0.3-10s_refitted"): 2.0,
        ("uncoupled_RFDT", "all_single_click_RTs"): 1.0,
        ("uncoupled_RFDT", "0.3-10s_refitted"): 0.0,
    }
    offsets = {"independent_evaluation": 0.11, "larger_bank_refit": -0.11}
    markers = {"independent_evaluation": "o", "larger_bank_refit": "s"}

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for version in versions:
        sub = df[df["version"].eq(version)]
        for _, row in sub.iterrows():
            y = ybase[(row["comparator"], row["target"])] + offsets[version]
            lo = row["estimate"] - row["ci_low"]
            hi = row["ci_high"] - row["estimate"]
            ax.errorbar(
                row["estimate"], y,
                xerr=np.array([[lo], [hi]]),
                fmt=markers[version],
                capsize=2.5,
                label=version if (row["comparator"], row["target"]) == ("DDM", "all_single_click_RTs") else None,
            )

    ax.axvline(0, linestyle="--", linewidth=0.8)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([
        "RFDT − nonlinear DDM\nAll single-click RTs",
        "RFDT − nonlinear DDM\n0.3–10 s (refitted)",
        "RFDT − uncoupled RFDT\nAll single-click RTs",
        "RFDT − uncoupled RFDT\n0.3–10 s (refitted)",
    ])
    ax.set_xlabel("Mean held-out log-score difference (nats/trial)")
    ax.legend(
        ["Original bank: independent evaluation", "Larger bank: refitted models"],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    clean(ax)
    fig.tight_layout()
    fig.savefig(OUT / "Figure5_dynamic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure5_dynamic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figure6():
    cal = pd.read_csv(RESULTS / "figure6_primary_rt_calibration.csv")
    regions = pd.read_csv(RESULTS / "figure6_primary_score_regions.csv")
    plot_regions = regions[~regions["region"].eq("all")].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(cal))
    width = 0.26
    ax1.bar(x - width, 100 * cal["observed_fraction"], width=width, label="Observed")
    ax1.bar(x, 100 * cal["rfdt_predicted"], width=width, label="RFDT")
    ax1.bar(x + width, 100 * cal["ddm_predicted"], width=width, label="Nonlinear DDM")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cal["label"], rotation=42, ha="right")
    ax1.set_ylabel("Fraction of responses (%)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.text(-0.10, 1.03, "A", transform=ax1.transAxes, fontweight="bold")

    y = np.arange(len(plot_regions))[::-1]
    ax2.axvline(0, linestyle="--", linewidth=0.8)
    ax2.hlines(y, 0, plot_regions["score_contribution"])
    ax2.plot(plot_regions["score_contribution"], y, "o")
    ax2.set_yticks(y)
    ax2.set_yticklabels([
        f"{lab}\n{n:,} trials"
        for lab, n in zip(plot_regions["label"], plot_regions["n"])
    ])
    ax2.set_xlabel("RFDT − DDM score contribution\n(nats/trial in full sample)")
    total = float(regions.loc[regions["region"].eq("all"), "score_contribution"].iloc[0])
    ax2.text(
        0.02, 0.03, f"Total: {total:+.3f} nats/trial",
        transform=ax2.transAxes, fontsize=9,
    )
    ax2.text(-0.15, 1.03, "B", transform=ax2.transAxes, fontweight="bold")

    clean(ax1)
    clean(ax2)
    fig.tight_layout()
    fig.savefig(OUT / "Figure6_dynamic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Figure6_dynamic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure5()
    make_figure6()
    print("Wrote Figure5_dynamic and Figure6_dynamic (PDF and PNG).")
