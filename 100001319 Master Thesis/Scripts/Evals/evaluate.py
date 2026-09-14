"""Stage 5, Step 2: validity/recall figures, rho-distribution figure, and
baseline comparison table on the held-out test split.

    python scripts/evaluate.py

Reads out_narratives_ollama_v3/t085/calibration_manifest_v2.json (frozen
lambda_hat per alpha, both loss policies -- Stage 4's output, not
recalculated here) and out_narratives_ollama_v3/t085/admission_eval_v2.json
(rho / self_consistency / fact_match_score per sentence -- Stage 3's output,
not rescored here). Reconstructs the SAME test split calibration used
(split_by_user, same seed/fraction read from the manifest, never from a
hardcoded duplicate) and evaluates on it via src.evaluation.evaluate().

Writes three figures (out_narratives_ollama_v3/t085/fig_*.png) and prints the
baseline comparison table. Nothing here re-scores or re-calibrates anything;
per docs/stage_5.md this step only measures and reports.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this runs from a script, not a notebook
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibrate import split_by_user  # noqa: E402
from src.evaluation import evaluate  # noqa: E402

MANIFEST_JSON = Path("out_narratives_ollama_v3/t085/calibration_manifest_v2.json")
ADMISSION_EVAL_JSON = Path("out_narratives_ollama_v3/t085/admission_eval_v2.json")
LABELLING_CSV = Path("out_narratives_ollama_v3/t085/labelling_final_v2.csv")
OUT_DIR = Path("out_narratives_ollama_v3/t085")

# Fixed-threshold baseline (docs/stage_5.md: "pick a plausible cutoff by
# inspection, apply it uniformly, no formal guarantee"). 0.5 chosen as the
# natural midpoint of rho's [0,1] range -- not tuned against these results;
# see the printed report for how this cutoff performs against the calibrated
# lambda_hat=0.00.
FIXED_THRESHOLD = 0.5

# The three test-split sentences human-labelled wrong (hallucinated or false
# precision) that Finding 14 / the Step 1 evaluation names explicitly.
WRONG_TEST_SENTENCES = {
    "30e15748717a_s001": "hallucinated",
    "7038b42bbdd3_s009": "hallucinated",
    "526ddf572cfc_s011": "false precision",
}


def _load_all_rows_with_metadata() -> tuple[list[dict], list[str]]:
    """admission_eval_v2.json rows, each augmented with 'user'/'malicious'/
    'rules' from labelling_final_v2.csv -- and the parallel user list
    split_by_user needs. Kept as one function so scenario-breakdown code and
    the test-split reconstruction can never drift apart on which CSV columns
    they pulled.
    """
    admission = json.loads(ADMISSION_EVAL_JSON.read_text(encoding="utf-8"))
    rows = admission["rows"]
    with open(LABELLING_CSV, encoding="utf-8-sig") as fh:
        meta_by_cluster = {r["cluster_id"]: r for r in csv.DictReader(fh)}
    for r in rows:
        meta = meta_by_cluster[r["cluster_id"]]
        r["user"] = meta["user"]
        r["malicious"] = meta["malicious"]
        r["rules"] = meta["rules"]
    users = [r["user"] for r in rows]
    return rows, users


def _load_test_split() -> list[dict]:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    m = manifest["manifests_by_loss_policy"]["primary"]
    calibrate_fraction, seed = m["calibrate_fraction"], m["seed"]

    rows, users = _load_all_rows_with_metadata()
    split = split_by_user(users, calibrate_fraction=calibrate_fraction, seed=seed)
    test_idx = sorted(split.test_indices)
    print(f"reconstructed test split: {len(test_idx)} rows "
          f"(calibrate_fraction={calibrate_fraction}, seed={seed})", file=sys.stderr)
    return [rows[i] for i in test_idx]


def _load_calibration_split() -> list[dict]:
    """Calibration-side rows -- used ONLY for the supplementary,
    explicitly-labelled malicious-scenario note below. Never used for any
    validity/recall number that claims to be a held-out measurement.
    """
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    m = manifest["manifests_by_loss_policy"]["primary"]
    calibrate_fraction, seed = m["calibrate_fraction"], m["seed"]
    rows, users = _load_all_rows_with_metadata()
    split = split_by_user(users, calibrate_fraction=calibrate_fraction, seed=seed)
    return [rows[i] for i in sorted(split.calibrate_indices)]


def _lambda_hats_by_alpha(policy: str) -> dict[float, float]:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    m = manifest["manifests_by_loss_policy"][policy]
    return {r["alpha"]: r["lambda_hat"] for r in m["ltt_results"]}


# ---------------------------------------------------------------------------
# Figure 1 + 2: validity (error vs alpha) and recall vs alpha
# ---------------------------------------------------------------------------

def make_validity_and_recall_figures(test_rows: list[dict]) -> None:
    labels = [r["human_label"] for r in test_rows]
    rho = [r["rho"] for r in test_rows]

    alpha_grid = sorted(_lambda_hats_by_alpha("primary"))
    results = {}
    for policy in ("primary", "secondary"):
        lam_by_alpha = _lambda_hats_by_alpha(policy)
        results[policy] = [
            evaluate(rho, labels, lambda_hat=lam_by_alpha[a], loss_policy=policy)
            for a in alpha_grid
        ]

    # --- Figure 1: validity ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(alpha_grid, alpha_grid, "k--", linewidth=1, label="y = x (guarantee boundary)")
    for policy, marker, color in (("primary", "o", "tab:blue"), ("secondary", "s", "tab:orange")):
        errors = [res.error for res in results[policy]]
        ax.plot(alpha_grid, errors, marker=marker, color=color,
                label=f"{policy} (hallucinated"
                      + ("+misweighted+false precision" if policy == "primary" else " only")
                      + ")")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("empirical error L(lambda_hat) on test split")
    ax.set_title("Validity: empirical error vs nominal alpha\n"
                  "(n=567, v3/llama3:latest/k=10/threshold=0.85, delta=0.10)")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.annotate("both policies flat: lambda_hat=0.00 at every alpha\n"
                "(vacuous calibration -- see Finding 14)",
                xy=(0.15, max(res.error for res in results["primary"])),
                xytext=(0.05, 0.20),
                fontsize=8, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_validity.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: recall ---
    # primary and secondary are BOTH exactly 1.0 at every alpha (vacuous
    # calibration accepts every sentence regardless of policy), so the two
    # lines fully overlap -- drawn with different linewidths/markers so the
    # underlying blue (primary) line is still visible under the orange
    # (secondary) one, plus an explicit annotation, rather than leaving one
    # line silently hidden with no explanation.
    fig, ax = plt.subplots(figsize=(6, 5))
    recalls_primary = [res.recall for res in results["primary"]]
    recalls_secondary = [res.recall for res in results["secondary"]]
    ax.plot(alpha_grid, recalls_primary, marker="o", color="tab:blue",
            linewidth=4, markersize=10, label="primary")
    ax.plot(alpha_grid, recalls_secondary, marker="s", color="tab:orange",
            linewidth=1.5, markersize=5, label="secondary")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("recall of grounded content on test split")
    ax.set_title("Usefulness: grounded-content recall vs nominal alpha\n"
                  "(both policies flat at 1.0 -- vacuous calibration accepts everything)")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.annotate("primary and secondary recall are IDENTICAL (both 1.0) at\n"
                "every alpha -- lines fully overlap by construction, not\n"
                "a rendering artifact (thicker blue drawn under orange)",
                xy=(0.15, 1.0), xytext=(0.05, 0.5),
                fontsize=8, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_recall.png", dpi=150)
    plt.close(fig)

    print("\n=== Figure 1/2 data (validity + recall vs alpha) ===")
    for policy in ("primary", "secondary"):
        for a, res in zip(alpha_grid, results[policy]):
            print(f"  policy={policy:10s} alpha={a:.2f}  error={res.error:.4f}  "
                  f"recall={res.recall:.4f}  (n_accepted={res.n_accepted}/{res.n_test})")


# ---------------------------------------------------------------------------
# Figure 3: rho distribution, wrong test-split sentences highlighted
# ---------------------------------------------------------------------------

def make_rho_distribution_figure(test_rows: list[dict]) -> None:
    grounded_rho = [r["rho"] for r in test_rows if r["human_label"] == "grounded"]
    other_rho = [r["rho"] for r in test_rows
                 if r["human_label"] != "grounded"
                 and r["cluster_id"] not in WRONG_TEST_SENTENCES]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [i / 20 for i in range(21)]  # 0.05-wide bins over [0,1]
    ax.hist(grounded_rho, bins=bins, alpha=0.6, color="tab:green",
            label=f"grounded (n={len(grounded_rho)})")
    ax.hist(other_rho, bins=bins, alpha=0.6, color="tab:gray",
            label=f"unverifiable / other non-wrong (n={len(other_rho)})")

    # The three wrong test-split sentences, each individually marked and
    # labelled -- not pooled into one "wrong" category.
    colors = ["tab:red", "tab:purple", "tab:brown"]
    for (cid, lbl), color in zip(WRONG_TEST_SENTENCES.items(), colors):
        row = next(r for r in test_rows if r["cluster_id"] == cid)
        ax.axvline(row["rho"], color=color, linewidth=2, linestyle="--")
        ax.annotate(f"{cid}\n({lbl}, rho={row['rho']:.3f})",
                    xy=(row["rho"], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10),
                    xytext=(row["rho"], 0), rotation=90,
                    fontsize=7, color=color, ha="right", va="bottom")

    ax.set_xlabel("rho(s) on test split")
    ax.set_ylabel("count")
    ax.set_title("rho distribution, test split (n=265)\n"
                  "3 human-labelled-wrong sentences marked individually")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_rho_distribution.png", dpi=150)
    plt.close(fig)

    print("\n=== Figure 3 data (wrong test-split sentences vs grounded population) ===")
    n_grounded = len(grounded_rho)
    for cid, lbl in WRONG_TEST_SENTENCES.items():
        row = next(r for r in test_rows if r["cluster_id"] == cid)
        rho_val = row["rho"]
        n_grounded_below = sum(1 for g in grounded_rho if g < rho_val)
        n_grounded_at_or_above = n_grounded - n_grounded_below
        print(f"  {cid:25s} label={lbl:16s} rho={rho_val:.4f}  "
              f"grounded sentences at/above this rho: {n_grounded_at_or_above}/{n_grounded} "
              f"({100*n_grounded_at_or_above/n_grounded:.1f}%) -- "
              f"any threshold excluding this sentence also excludes at least that many grounded ones")


# ---------------------------------------------------------------------------
# Baseline comparison table
# ---------------------------------------------------------------------------

def baseline_table(test_rows: list[dict]) -> None:
    labels = [r["human_label"] for r in test_rows]
    rho = [r["rho"] for r in test_rows]
    self_consistency = [r["self_consistency"] for r in test_rows]
    fact_match = [r["fact_match_score"] for r in test_rows]
    n_nan_fact_match = sum(1 for x in fact_match if x != x)

    print(f"\n=== Baseline comparison (fixed threshold T={FIXED_THRESHOLD}, no calibration) ===")
    print(f"n_test={len(test_rows)}  "
          f"rows with NaN fact_match_score (no numeric/entity content): "
          f"{n_nan_fact_match}/{len(test_rows)} -- never accepted under fact_match_alone at any threshold")
    print()
    print(f"{'score':24s} {'policy':10s} {'n_accepted':>10s} {'error':>8s} {'recall':>8s}")

    rows_out = []
    for score_name, scores in (
        ("LTT-calibrated rho", None),  # special-cased below, uses per-alpha lambda_hat
        ("fixed-threshold combined_rho", rho),
        ("fixed-threshold self_consistency_alone", self_consistency),
        ("fixed-threshold fact_match_alone", fact_match),
    ):
        for policy in ("primary", "secondary"):
            if score_name == "LTT-calibrated rho":
                # alpha=0.10 as the representative point (delta=0.10, matching
                # the calibration manifest); all alphas give the same vacuous
                # lambda_hat=0.00, so any alpha would show the same numbers.
                lam = _lambda_hats_by_alpha(policy)[0.10]
                res = evaluate(rho, labels, lambda_hat=lam, loss_policy=policy)
                label_str = f"{score_name} (alpha=0.10, lambda_hat={lam:.2f})"
            else:
                res = evaluate(scores, labels, lambda_hat=FIXED_THRESHOLD, loss_policy=policy)
                label_str = score_name
            print(f"{label_str:44s} {policy:10s} {res.n_accepted:10d} "
                  f"{res.error:8.4f} {res.recall:8.4f}")
            rows_out.append({
                "score": label_str, "policy": policy,
                "n_accepted": res.n_accepted, "error": res.error, "recall": res.recall,
            })

    (OUT_DIR / "baseline_table.json").write_text(
        json.dumps({"fixed_threshold": FIXED_THRESHOLD, "rows": rows_out}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Per-scenario breakdown: malicious vs. benign, and by detection rule(s)
# ---------------------------------------------------------------------------

def per_scenario_breakdown(test_rows: list[dict]) -> None:
    """docs/stage_5.md: "Do not report one pooled recall number across all
    malicious signals... report recall of correct explanations conditional
    on the anomaly having been correctly flagged in the first place."

    Malicious-vs-benign is reported here as a GAP, not a number: this run's
    frozen test split (split_by_user, seed=42, calibrate_fraction=0.5) put
    all 3 malicious users -- and all 75 of their sentences -- entirely in the
    CALIBRATION split. Zero malicious sentences exist in the held-out test
    split to measure recall against. This is not computed around; a
    calibration-side substitute is shown separately below, explicitly
    labelled as non-held-out and not a validity measurement, because
    reporting it as if it were would misrepresent what "held-out" means.
    """
    labels = [r["human_label"] for r in test_rows]
    rho = [r["rho"] for r in test_rows]

    print("\n=== Per-scenario breakdown: malicious vs. benign ===")
    n_malicious_test = sum(1 for r in test_rows if r["malicious"] == "yes")
    print(f"  malicious sentences in TEST split: {n_malicious_test}/{len(test_rows)}")
    if n_malicious_test == 0:
        print("  GAP: the seed=42 user-level split placed all 3 malicious users "
              "(CDE1846, ACM2278, CMP2946; 75 sentences) entirely in the "
              "calibration split. No malicious-scenario recall can be computed "
              "on held-out data for this split -- reported as a limitation, "
              "not silently omitted.")
        cal_rows = _load_calibration_split()
        cal_labels = [r["human_label"] for r in cal_rows]
        cal_rho = [r["rho"] for r in cal_rows]
        mal_idx = [i for i, r in enumerate(cal_rows) if r["malicious"] == "yes"]
        ben_idx = [i for i, r in enumerate(cal_rows) if r["malicious"] == "no"]
        print(f"\n  SUPPLEMENTARY, CALIBRATION-SIDE ONLY (NOT held-out, NOT a "
              f"validity measurement -- these sentences were part of what "
              f"lambda_hat was calibrated against):")
        for name, idx in (("malicious", mal_idx), ("benign", ben_idx)):
            n = len(idx)
            n_grounded = sum(1 for i in idx if cal_labels[i] == "grounded")
            n_grounded_accepted = sum(
                1 for i in idx if cal_labels[i] == "grounded" and cal_rho[i] >= 0.0
            )
            recall = n_grounded_accepted / max(n_grounded, 1)
            print(f"    {name:10s} n={n:4d}  n_grounded={n_grounded:4d}  "
                  f"recall_at_lambda_0.00={recall:.4f} (trivial -- lambda_hat=0.00 "
                  f"accepts everyone on the calibration side too)")

    print("\n=== Per-scenario breakdown: by detection rule(s) fired (test split) ===")
    rule_groups: dict[str, list[int]] = {}
    for i, r in enumerate(test_rows):
        rule_groups.setdefault(r["rules"], []).append(i)

    print(f"  {'rules':12s} {'n':>5s} {'n_grounded':>11s} {'error(primary)':>15s} "
          f"{'error(secondary)':>17s} {'recall':>8s}")
    for rules_key in sorted(rule_groups, key=lambda k: -len(rule_groups[k])):
        idx = rule_groups[rules_key]
        sub_rho = [rho[i] for i in idx]
        sub_labels = [labels[i] for i in idx]
        res_primary = evaluate(sub_rho, sub_labels, lambda_hat=0.0, loss_policy="primary")
        res_secondary = evaluate(sub_rho, sub_labels, lambda_hat=0.0, loss_policy="secondary")
        print(f"  {rules_key:12s} {len(idx):5d} {res_primary.n_grounded:11d} "
              f"{res_primary.error:15.4f} {res_secondary.error:17.4f} "
              f"{res_primary.recall:8.4f}")


def main() -> None:
    test_rows = _load_test_split()
    make_validity_and_recall_figures(test_rows)
    make_rho_distribution_figure(test_rows)
    baseline_table(test_rows)
    per_scenario_breakdown(test_rows)
    print(f"\nwrote {OUT_DIR / 'fig_validity.png'}")
    print(f"wrote {OUT_DIR / 'fig_recall.png'}")
    print(f"wrote {OUT_DIR / 'fig_rho_distribution.png'}")
    print(f"wrote {OUT_DIR / 'baseline_table.json'}")


if __name__ == "__main__":
    main()
