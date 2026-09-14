"""CLI: CERT r6.2 extraction -> signals.jsonl.

    python scripts/prepare_cert.py --cert-root /path/to/r6.2 --out build/

Writes signals.jsonl, features_userweek_raw.csv, and run_summary.json into
--out. Extraction is checkpointed (see src/cert_features.py), so a rerun
against the same --out resumes rather than restarting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/prepare_cert.py` from anywhere -- src/ is a package
# (relative imports inside it require this), but scripts/ is not on sys.path
# by default when the script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cert_features import extract_raw_features, label_user_weeks, load_insiders  # noqa: E402
from src import signals as sg  # noqa: E402

FLAG_RATE_WARNING = 0.05  # no SOC could triage more than this


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cert-root", required=True, help="directory containing logon.csv, http.csv, LDAP/, answers/, ...")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--sample-nrows", type=int, default=None,
                   help="cap rows read per log, independently per file -- covers a different "
                       "calendar date range per log (row density varies hugely across logs), "
                       "so flag rate/recall from a --sample-nrows run are not meaningful. Use "
                       "only to smoke-test that extraction runs; prefer --sample-until otherwise.")
    p.add_argument("--sample-until", default=None,
                   help="read each log only up to this date (e.g. 2010-06-01), so all five logs' "
                       "sampled windows end on the same calendar date -- use this for a bounded "
                       "trial run whose flag rate/recall should mean something.")
    p.add_argument("--exclude-periods", default=None,
                   help="comma-separated ISO week labels to drop entirely (e.g. "
                       "2011-W20,2011-W21,2011-W22). Use for periods whose features are "
                       "known-incomplete -- this copy of http.csv is truncated at "
                       "2011-05-13, so those three weeks have structurally-zero http "
                       "features that are not genuine zeros. Dropped before baselines, so "
                       "excluded weeks never enter a peer group or a trailing window.")
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--self-z", type=float, default=3.0, help="self-baseline z threshold used by rules")
    p.add_argument("--peer-z", type=float, default=4.0, help="peer-baseline z threshold used by rules")
    p.add_argument("--release", default="6.2", help="dataset release to filter insiders.csv to")
    p.add_argument("--date-fixes", default=None,
                   help="JSON file of {user: {malformed_value: corrected_value}} audited date repairs")
    return p.parse_args(argv)


def _load_date_fixes(path: str | None) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {(user, raw): fixed for user, fixes in data.items() for raw, fixed in fixes.items()}


def _load_ldap_snapshots(cert_root: Path):
    """All LDAP/YYYY-MM.csv snapshots, keyed by (year, month) -> DataFrame
    indexed by user_id. Employees who leave the company drop out of later
    snapshots -- using only the single most recent file would silently blank
    the identity of anyone who departed before it, which is exactly the
    population an insider-threat dataset is full of. Confirmed on this
    dataset: ACM2278 (a real r6.2 insider, malicious 2010-W33/W34) appears in
    LDAP/2009-12.csv through LDAP/2010-08.csv and nowhere after.
    """
    import pandas as pd

    files = sorted((cert_root / "LDAP").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no LDAP/*.csv files found under {cert_root}")

    snapshots = {}
    for f in files:
        year, month = (int(x) for x in f.stem.split("-"))
        snapshots[(year, month)] = pd.read_csv(f, dtype=str).set_index("user_id")
    return snapshots


def _latest_ldap(cert_root: Path):
    """Single most recent snapshot -- used only for role/peer-grouping, which
    needs one consistent role-per-user across the whole run. (Roles can also
    change over time in principle; treating peer groups as time-varying is a
    larger change than this fix and out of scope here.)"""
    snapshots = _load_ldap_snapshots(cert_root)
    latest_key = max(snapshots)
    return snapshots[latest_key].reset_index()


def _identity_lookup(cert_root: Path, feature_index: "pd.MultiIndex"):
    """Per-(user, period) identity, resolved from the LDAP snapshot closest
    to that period's month among those that actually CONTAIN the user -- not
    just the nearest snapshot regardless of membership. The nearest snapshot
    to a signal's month can itself postdate the user's departure (confirmed:
    PLJ1771's August 2010 signal has an August LDAP snapshot, but PLJ1771 only
    appears through July -- taking "nearest" without checking membership would
    still blank them). Search outward from the target month -- all at-or-before
    months nearest first, then all after-months nearest first -- and use the
    first snapshot that actually has the user.
    """
    from src.cert_features import week_to_range

    import pandas as pd

    snapshots = _load_ldap_snapshots(cert_root)
    months_sorted = sorted(snapshots)
    # business_unit is rendered by signal_to_prompt_context and was previously
    # absent from this list, so every signal reported "Business unit: unknown".
    cols = ["employee_name", "role", "business_unit", "functional_unit",
            "department", "team", "supervisor"]

    def _clean(value) -> str:
        # LDAP blanks read back as NaN under dtype=str; str(NaN) is "nan",
        # which would otherwise reach the prompt as a literal fact.
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    lookup = {}
    for user, period in feature_index:
        week_start, _ = week_to_range(period)
        target = (week_start.year, week_start.month)
        before = [m for m in months_sorted if m <= target]
        after = [m for m in months_sorted if m > target]
        candidates = list(reversed(before)) + after  # nearest-before first, then nearest-after

        identity = {c: "" for c in cols}
        for month in candidates:
            snap = snapshots[month]
            if user in snap.index:
                row = snap.loc[user]
                identity = {c: _clean(row.get(c)) for c in cols}
                break
        lookup[(user, period)] = identity
    return lookup


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    args = _parse_args(argv)
    cert_root = Path(args.cert_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_nrows is not None and args.sample_until is None:
        print(
            f"WARNING: --sample-nrows {args.sample_nrows} caps rows PER LOG INDEPENDENTLY. "
            "The five logs have very different row density (http.csv ~90 GB vs logon.csv "
            "~240 MB), so the same row count reaches a different calendar date in each -- "
            "a user-week can show file/device activity with zero http/email activity purely "
            "because those logs' sampled windows didn't reach that week. Flag rate and recall "
            "from this run are NOT meaningful. Use --sample-until DATE for a bounded run whose "
            "results mean something.",
            file=sys.stderr,
        )

    date_fixes = _load_date_fixes(args.date_fixes)

    feats = extract_raw_features(cert_root, out_dir, chunk_size=args.chunk_size,
                                 sample_nrows=args.sample_nrows,
                                 sample_until=args.sample_until)

    excluded_periods = [p.strip() for p in args.exclude_periods.split(",")
                       if p.strip()] if args.exclude_periods else []
    if excluded_periods:
        # Dropped before baselines so an excluded week never contributes to a
        # peer group or to a later week's trailing self-baseline.
        before = len(feats)
        keep = ~feats.index.get_level_values("period").isin(excluded_periods)
        feats = feats[keep]
        print(f"excluded periods {excluded_periods}: dropped {before - len(feats):,} "
             f"of {before:,} user-weeks", file=sys.stderr)

    ldap = _latest_ldap(cert_root)
    roles = ldap.set_index("user_id")["role"]

    insiders = load_insiders(cert_root, release=args.release, date_fixes=date_fixes)
    labels = label_user_weeks(feats.index, insiders)

    cfg = sg.SignalConfig(self_z_threshold=args.self_z, peer_z_threshold=args.peer_z)
    enriched = sg.add_self_baselines(feats, cfg)
    enriched = sg.add_peer_baselines(enriched, roles, cfg)
    flagged = sg.apply_rules(enriched, cfg)

    # Period-aware: a signal's identity comes from the LDAP snapshot nearest
    # that signal's own week, not always the single latest file (see
    # _identity_lookup -- an insider who left the company shortly after their
    # malicious activity would otherwise get blank identity precisely when it
    # matters most).
    identity_lookup = _identity_lookup(cert_root, flagged.index)

    signals = []
    for (user, period), row in flagged.iterrows():
        identity = identity_lookup[(user, period)]
        lbl_row = labels.loc[(user, period)]
        label = {"is_malicious": bool(lbl_row["is_malicious"]), "scenario": lbl_row["scenario"]}
        signals.append(sg.build_signal(user, period, row, identity, label, cfg))

    sg.write_signals(signals, str(out_dir / "signals.jsonl"))
    feats.to_csv(out_dir / "features_userweek_raw.csv")

    n_user_weeks = len(feats)
    n_flagged = len(flagged)
    flag_rate = n_flagged / n_user_weeks if n_user_weeks else float("nan")

    malicious_index = labels.index[labels["is_malicious"]]
    flagged_index = flagged.index
    n_malicious = len(malicious_index)
    n_malicious_flagged = len(malicious_index.intersection(flagged_index))
    recall = n_malicious_flagged / n_malicious if n_malicious else float("nan")

    summary = {
        "cert_root": str(cert_root),
        "release": args.release,
        "chunk_size": args.chunk_size,
        "self_z_threshold": args.self_z,
        "peer_z_threshold": args.peer_z,
        "sample_nrows": args.sample_nrows,
        "sample_until": args.sample_until,
        "excluded_periods": excluded_periods,
        "insiders_found": int(len(insiders)),
        "user_week_count": int(n_user_weeks),
        "flagged_count": int(n_flagged),
        "flag_rate": flag_rate,
        "malicious_user_weeks": int(n_malicious),
        "malicious_user_weeks_flagged": int(n_malicious_flagged),
        "recall_on_malicious_user_weeks": recall,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"insiders found:            {summary['insiders_found']}")
    print(f"user-week count:           {summary['user_week_count']}")
    print(f"flagged user-weeks:        {summary['flagged_count']} ({flag_rate:.2%})")
    print(f"recall on malicious weeks: {n_malicious_flagged}/{n_malicious} "
         f"({recall:.2%})" if n_malicious else "recall on malicious weeks: n/a (no malicious weeks in range)")
    if flag_rate > FLAG_RATE_WARNING:
        print(f"WARNING: flag rate {flag_rate:.2%} exceeds {FLAG_RATE_WARNING:.0%} -- "
             "no SOC could triage this.", file=sys.stderr)

    if signals:
        malicious_signals = [s for s in signals if s["label"]["is_malicious"]]
        example = malicious_signals[0] if malicious_signals else signals[0]
        print("\nExample prompt context:\n")
        print(sg.signal_to_prompt_context(example))

    print(f"\nwrote {out_dir / 'signals.jsonl'}")
    print(f"wrote {out_dir / 'features_userweek_raw.csv'}")
    print(f"wrote {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
