"""Stage 4, Step 2: LTT (and CRC) calibration against the real admission-
function output and the frozen human ground truth.

    python scripts/calibrate.py

Reads out_narratives_ollama_v3/t085/admission_eval.json (rho(s) per pooled
sentence, from scripts/evaluate_admission.py) and
out_narratives_ollama_v3/t085/labelling_final.csv (frozen ground truth, for
each row's user -- the exchangeability unit). Writes a run manifest
(out_narratives_ollama_v3/t085/calibration_manifest.json) that Stage 5 reads.

Per docs/stage_4.md, six decisions must be FROZEN before calibrating, not
tuned after seeing results -- alpha grid, delta, loss policy, exchangeability
unit, calibration set size, and the pooling-threshold/rendering-version tuple
that must accompany any quoted guarantee. This script takes the ones that are
genuine judgment calls (loss policy, calibration fraction, delta, alpha grid)
as EXPLICIT CLI arguments with no silent default that could be mistaken for
an already-frozen decision -- see --loss-policy and --calibrate-fraction
below.

CALIBRATION SET SIZE, stated up front because it is not a code decision:
docs/stage_4.md flags 197 labelled clusters as too small for a tight
high-probability bound ("500-1000 as a practical floor") and offers two
legitimate paths -- scale generation, or report a wide bound and say so.
Scaling generation to 300-500 signals was NOT done in this pass (out of
scope for this step); this script takes the "report a wide bound and say so"
path. The run manifest and printed report both state n_calibrate explicitly
so this is never silently mistaken for a tight-sample-size result.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibrate import (  # noqa: E402
    CalibrationManifest,
    crc_calibrate_grid,
    ltt_calibrate_grid,
)

ADMISSION_EVAL_JSON = Path("out_narratives_ollama_v3/t085/admission_eval.json")
LABELLING_CSV = Path("out_narratives_ollama_v3/t085/labelling_final.csv")
NARRATIVES_JSONL = Path("out_narratives_ollama_v3/t085/narratives.jsonl")
DEFAULT_OUT = Path("out_narratives_ollama_v3/t085/calibration_manifest.json")

# 0.00 -> 1.00 in 0.05 steps, DESCENDING (most conservative first) --
# lambda_grid must be pre-specified before looking at results, per
# docs/stage_4.md's fixed-sequence testing requirement.
DEFAULT_LAMBDA_GRID = [round(1.0 - 0.05 * i, 2) for i in range(21)]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--admission-eval", default=str(ADMISSION_EVAL_JSON))
    p.add_argument("--labelling-csv", default=str(LABELLING_CSV))
    p.add_argument("--narratives-jsonl", default=str(NARRATIVES_JSONL),
                   help="source of the frozen generation-config tuple "
                       "(model/k/temperature/pooling) recorded in the manifest")
    p.add_argument("--ldap-csv", default=r"raw data/r6.2/LDAP/2011-05.csv",
                   help="LDAP data for stratified user splitting")
    p.add_argument("--out", default=str(DEFAULT_OUT))

    p.add_argument("--alpha-grid", required=True,
                   help="comma-separated target risk levels, e.g. "
                       "'0.05,0.10,0.15,0.20,0.25' -- must be frozen before "
                       "running, docs/stage_4.md decision 1")
    p.add_argument("--delta", type=float, required=True,
                   help="fixed-sequence significance level (0.05 or 0.10 "
                       "per docs/stage_4.md decision 2) -- no default, must "
                       "be stated explicitly")
    p.add_argument("--loss-policy", choices=["primary", "secondary", "both"],
                   required=True,
                   help="'primary': hallucinated+misweighted+false_precision "
                       "count as wrong. 'secondary': hallucinated only. "
                       "docs/stage_4.md decision 3 says report BOTH -- pass "
                       "'both' to do so in one run; no default, this is a "
                       "judgment call that must be made consciously")
    p.add_argument("--calibrate-fraction", type=float, required=True,
                   help="fraction of USERS assigned to the calibration split "
                       "(the rest go to test) -- docs/stage_4.md decision 5. "
                       "No default: with only 197 labelled rows total, this "
                       "number directly determines how few calibration "
                       "points the bound is based on, and must be a "
                       "conscious choice, not an inherited default")
    p.add_argument("--lambda-grid", default=None,
                   help="comma-separated, sorted DESCENDING (most "
                       f"conservative first). Default: {DEFAULT_LAMBDA_GRID}")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the user-level split (reproducibility, "
                       "not a tuning knob)")
    return p.parse_args(argv)


def _load_rho_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _load_user_by_cluster(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8-sig") as fh:
        return {row["cluster_id"]: row["user"] for row in csv.DictReader(fh)}


def _load_generation_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        rec = json.loads(fh.readline())
    return {
        "generation_model": rec["model"],
        "k": rec["k"],
        "temperature": rec["temperature"],
        "pooling_threshold": rec["pooling"]["threshold"],
        "rendering_version": rec["prompt_version"],
    }


def _manifest_to_jsonable(manifest: CalibrationManifest) -> dict:
    d = dataclasses.asdict(manifest)
    return d


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
    lambda_grid = (
        [float(x) for x in args.lambda_grid.split(",")]
        if args.lambda_grid is not None else DEFAULT_LAMBDA_GRID
    )
    if lambda_grid != sorted(lambda_grid, reverse=True):
        raise SystemExit(
            f"--lambda-grid must be sorted DESCENDING (most conservative "
            f"first), got {lambda_grid}"
        )

    rho_rows = _load_rho_rows(Path(args.admission_eval))
    user_by_cluster = _load_user_by_cluster(Path(args.labelling_csv))
    gen_config = _load_generation_config(Path(args.narratives_jsonl))

    # Load LDAP for stratified splitting
    user_groups = {}
    if Path(args.ldap_csv).exists():
        try:
            with open(Path(args.ldap_csv), encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    dept = row.get("department", "Unknown").strip()
                    role = row.get("role", "Unknown").strip()
                    user_groups[row.get("user_id", "")] = f"{dept}_{role}"
        except Exception as e:
            print(f"WARNING: Failed to parse LDAP csv: {e}", file=sys.stderr)

    rhos = [r["rho"] for r in rho_rows]
    labels = [r["human_label"] for r in rho_rows]
    user_ids = [user_by_cluster[r["cluster_id"]] for r in rho_rows]
    n_unique_users = len(set(user_ids))

    print(f"n_rows: {len(rhos)}  n_unique_users: {n_unique_users}", file=sys.stderr)
    print(f"generation config: {gen_config}", file=sys.stderr)
    if len(rhos) < 500:
        print(
            f"NOTE: calibration input has only {len(rhos)} labelled rows -- "
            "docs/stage_4.md flags 500-1000 as a practical floor for a "
            "meaningful high-probability bound. This run takes the "
            "'report a wide bound and say so' path (generation was not "
            "scaled up in this pass). Expect wide bounds / small lambda "
            "grid coverage; that is expected given calibration set size, "
            "not a bug.",
            file=sys.stderr,
        )

    policies = ["primary", "secondary"] if args.loss_policy == "both" else [args.loss_policy]

    manifests: dict[str, dict] = {}
    for policy in policies:
        ltt_results = ltt_calibrate_grid(
            rhos=rhos, labels=labels, user_ids=user_ids,
            alpha_grid=alpha_grid, delta=args.delta, lambda_grid=lambda_grid,
            loss_policy=policy, calibrate_fraction=args.calibrate_fraction,
            user_groups=user_groups,
            seed=args.seed,
        )
        crc_results = crc_calibrate_grid(
            rhos=rhos, labels=labels, user_ids=user_ids,
            alpha_grid=[max(1, round(a * len(rhos) * args.calibrate_fraction)) for a in alpha_grid],
            delta=args.delta, lambda_grid=lambda_grid,
            loss_policy=policy, calibrate_fraction=args.calibrate_fraction,
            user_groups=user_groups,
            seed=args.seed,
        )

        manifest = CalibrationManifest(
            alpha_grid=alpha_grid,
            delta=args.delta,
            loss_policy=policy,
            calibrate_fraction=args.calibrate_fraction,
            lambda_grid=lambda_grid,
            pooling_threshold=gen_config["pooling_threshold"],
            rendering_version=gen_config["rendering_version"],
            generation_model=gen_config["generation_model"],
            k=gen_config["k"],
            temperature=gen_config["temperature"],
            seed=args.seed,
            ltt_results=ltt_results,
            crc_results=crc_results,
        )
        manifest_json = _manifest_to_jsonable(manifest)
        # lambda_hat=0.0 is ambiguous between "genuine full-permissive pass"
        # and "nothing certified" -- attach an explicit status per alpha so
        # Stage 5 (which reads this manifest) cannot make that mistake.
        for r_json, r in zip(manifest_json["ltt_results"], ltt_results):
            r_json["status"] = "pass" if r.p_value <= args.delta else "fail_nothing_certified"
        manifests[policy] = manifest_json

        print(f"\n=== loss_policy={policy} ===")
        print(f"{'alpha':>6}  {'status':>16}  {'lambda_hat':>10}  {'achieved_risk':>14}  "
              f"{'n_selected':>10}  {'n_wrong':>8}  {'p_value':>10}")
        for r in ltt_results:
            # lambda_hat alone is AMBIGUOUS at the grid floor: 0.0 means
            # either "fully permissive genuinely certified" (p_value <=
            # delta) or "nothing certified, not even the strictest lambda"
            # (default fallback -- p_value > delta at every grid point).
            # Both print the same lambda_hat=0.00; only p_value vs delta
            # tells them apart, so make that explicit rather than let a
            # reader mistake a calibration FAILURE for a permissive PASS.
            status = "PASS" if r.p_value <= args.delta else "FAIL (nothing certified)"
            print(f"{r.alpha:6.2f}  {status:>16}  {r.lambda_hat:10.2f}  {r.achieved_risk:14.4f}  "
                  f"{r.n_selected:10d}  {r.n_wrong:8.0f}  {r.p_value:10.4g}")
        print(f"n_calibrate (this policy's split): {ltt_results[0].n_calibrate if ltt_results else 'n/a'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_rows": len(rhos),
        "n_unique_users": n_unique_users,
        "calibration_set_size_note": (
            "197-row calibration input is below docs/stage_4.md's 500-1000 "
            "practical floor; this run reports the bound as-is rather than "
            "scaling generation, per the 'report a wide bound and say so' "
            "path documented there."
        ),
        "generation_config": gen_config,
        "manifests_by_loss_policy": manifests,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
