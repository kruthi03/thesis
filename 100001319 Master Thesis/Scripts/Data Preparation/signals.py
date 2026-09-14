"""
Baselines, anomaly rules, and construction of the structured anomaly signal.

The signal is the ground-truth object of this thesis. Everything downstream
depends on one property: **every claim a narrative could make must correspond
to an enumerable, typed fact in the signal.** That is what lets the admission
function decide entailment mechanically rather than heuristically.

Hence the `facts` array: each fact has an id, a field name, a numeric value, a
unit, and a natural-language gloss. A narrative sentence is grounded iff every
number and named entity it contains can be matched to some fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cert_features import FEATURE_COLUMNS

# Features that carry a rule. Restricting baselines to these keeps the signal
# small enough for an analyst to read in one screen.
BASELINE_FEATURES: list[str] = [
    "total_logons", "after_hours_logons", "weekend_logons", "distinct_pcs",
    "n_active_days", "n_weekend_days",
    "total_device_connects", "after_hours_device_connects",
    "total_file_events", "removable_media_copies", "after_hours_file_events",
    "distinct_files",
    "total_emails_sent", "external_emails", "external_recipients",
    "attachment_bytes", "after_hours_emails",
    "total_http_requests", "distinct_domains", "after_hours_http",
    "job_search_visits", "cloud_storage_visits", "leak_site_visits",
]

_missing = set(BASELINE_FEATURES) - set(FEATURE_COLUMNS)
if _missing:
    raise ValueError(
        f"BASELINE_FEATURES references columns absent from cert_features.FEATURE_COLUMNS: "
        f"{sorted(_missing)} -- extract_raw_features would never produce them."
    )

FEATURE_GLOSS: dict[str, str] = {
    "total_logons": "logon events",
    "after_hours_logons": "logons outside working hours",
    "weekend_logons": "logons at the weekend",
    "distinct_pcs": "distinct workstations used",
    "n_active_days": "days with recorded activity",
    "n_weekend_days": "weekend days with recorded activity",
    "earliest_hour": "hour of earliest logon in the week",
    "latest_hour": "hour of latest logon in the week",
    "total_device_connects": "removable device connections",
    "after_hours_device_connects": "removable device connections outside working hours",
    "total_file_events": "file operations",
    "removable_media_copies": "files copied to removable media",
    "after_hours_file_events": "file operations outside working hours",
    "distinct_files": "distinct files touched",
    "distinct_file_exts": "distinct file types touched",
    "total_emails_sent": "emails sent",
    "total_attachments": "files attached to sent email",
    "external_emails": "emails sent to at least one external recipient",
    "external_recipients": "external email recipients",
    "attachment_bytes": "bytes of email attachments sent",
    "after_hours_emails": "emails sent outside working hours",
    "total_http_requests": "web requests",
    "distinct_domains": "distinct web domains visited",
    "after_hours_http": "web requests outside working hours",
    "job_search_visits": "visits to job-search sites",
    "cloud_storage_visits": "visits to cloud-storage sites",
    "leak_site_visits": "visits to leak-publication sites",
}

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma under normality


@dataclass
class SignalConfig:
    # Window is in PERIODS (weeks here). 12 trailing weeks with a minimum
    # of 6 balances responsiveness against a stable median/MAD estimate;
    # r6.2 spans ~72 weeks per user, so ~60 remain usable.
    self_window: int = 12
    min_self_history: int = 6
    robust: bool = True
    eps: float = 1e-9
    self_z_threshold: float = 3.0
    peer_z_threshold: float = 4.0
    max_facts: int = 40
    # A user who has never performed an action has zero historical spread, so a
    # naive z-score divides by ~0 and explodes to ~1e9. For count features we
    # floor the spread at the Poisson standard deviation sqrt(max(center, 1)),
    # since for counts the variance is on the order of the mean. Non-count
    # features are floored at a fraction of their global spread.
    min_spread_frac: float = 0.05
    z_cap: float = 50.0
    # Peer statistics over a handful of colleagues are unstable; below this the
    # peer baseline is withheld (NaN) rather than reported unreliably.
    min_peer_group: int = 5


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

# Features that are NOT integer counts. Everything else is a count and gets
# the Poisson spread floor. Listing the exceptions rather than pattern-matching
# a name prefix avoids silent misclassification when the schema is renamed --
# an earlier prefix-based version treated every weekly count as continuous,
# giving spreads near zero and z-scores in the double digits for a difference
# of one event.
NON_COUNT_FEATURES: frozenset[str] = frozenset({
    "earliest_hour", "latest_hour", "attachment_bytes",
})


def is_count_feature(name: str) -> bool:
    """Count-valued features admit a Poisson spread floor; others do not."""
    return name not in NON_COUNT_FEATURES


def _spread_floor(center: float, field_name: str, global_scale: float,
                  cfg: SignalConfig) -> float:
    if is_count_feature(field_name):
        return float(np.sqrt(max(center, 1.0)))
    return max(cfg.min_spread_frac * global_scale, cfg.eps)


def _robust_center_spread(values: np.ndarray, robust: bool, eps: float) -> tuple[float, float]:
    """Centre and raw (unfloored) spread. Flooring is applied by the caller,
    which knows the feature name and its global scale."""
    if values.size == 0:
        return 0.0, 0.0
    if robust:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return med, MAD_SCALE * mad
    return float(values.mean()), float(values.std(ddof=0))


def _global_scale(series: pd.Series) -> float:
    """Robust spread of a feature across the whole population, used as the
    reference scale when flooring non-count features."""
    v = series.to_numpy(dtype=float)
    if v.size == 0:
        return 1.0
    med = float(np.median(v))
    mad = MAD_SCALE * float(np.median(np.abs(v - med)))
    return mad if mad > 0 else max(float(v.std(ddof=0)), 1.0)


def add_self_baselines(features: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """
    Trailing per-user baseline over the user's own prior active days.

    The window is shifted by one day so the current observation never
    contributes to its own baseline. Without that shift a single extreme day
    inflates its own centre and suppresses its z-score — the anomaly hides
    itself.
    """
    out = features.copy()
    new_cols: dict[str, list] = {}
    for col in BASELINE_FEATURES:
        scale = _global_scale(features[col])
        centers, spreads, zs, firsts, hist_max = [], [], [], [], []
        for _, grp in features.groupby(level="user", sort=False):
            series = grp[col].to_numpy(dtype=float)
            for i in range(len(series)):
                lo = max(0, i - cfg.self_window)
                hist = series[lo:i]
                if hist.size < cfg.min_self_history:
                    centers.append(np.nan); spreads.append(np.nan); zs.append(np.nan)
                    firsts.append(False); hist_max.append(np.nan)
                    continue
                c, raw = _robust_center_spread(hist, cfg.robust, cfg.eps)
                s = max(raw, _spread_floor(c, col, scale, cfg))
                z = float(np.clip((series[i] - c) / s, -cfg.z_cap, cfg.z_cap))
                centers.append(c); spreads.append(s); zs.append(z)
                # "Never done before" is far more useful to an analyst — and far
                # more checkable in a narrative — than an enormous z-score.
                firsts.append(bool(hist.max() == 0 and series[i] > 0))
                hist_max.append(float(hist.max()))
        new_cols[f"{col}__self_center"] = centers
        new_cols[f"{col}__self_spread"] = spreads
        new_cols[f"{col}__self_z"] = zs
        new_cols[f"{col}__self_first"] = firsts
        new_cols[f"{col}__self_hist_max"] = hist_max
    # One concat rather than ~115 inserts: repeated single-column assignment
    # fragments the block manager and is markedly slower at r6.2 scale.
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


# Known limitation, not fixed here: small peer groups (n close to
# min_peer_group) can produce spurious peer-deviation flags when one group
# member simply has a persistently elevated personal baseline. That member's
# own activity skews the group median/MAD, so their peers -- and the member
# themself -- end up compared against a center that's already pulled toward
# them. This is not a floor-formula artifact (see the zero-center handling
# above); it's a general limitation of estimating a robust center/spread from
# very few points. Concrete case from the mock calibration run: Manager role,
# n=6, total_http_requests -- HGW8295 fired R5 (peer_deviation) in two
# separate weeks on this field, which looks like a stable personal baseline
# running above their five peers rather than two anomalous weeks. Consider
# either a larger minimum group size for peer_z specifically, or a
# leave-one-out center (excluding the subject from their own peer statistic)
# if this keeps surfacing on the real dataset.
def add_peer_baselines(features: pd.DataFrame, roles: pd.Series, cfg: SignalConfig) -> pd.DataFrame:
    """
    Same-role, same-week peer baseline.

    Conditioning on the week absorbs organisation-wide effects (holidays,
    outages) that would otherwise register as individual anomalies.
    """
    out = features.copy()
    role_of = features.index.get_level_values("user").map(roles).fillna("UNKNOWN")
    out["_role"] = role_of.to_numpy()
    peer_cols: dict[str, np.ndarray] = {}
    period_of = features.index.get_level_values("period")

    for col in BASELINE_FEATURES:
        scale = _global_scale(features[col])
        frame = pd.DataFrame({"v": features[col].to_numpy(dtype=float),
                              "role": out["_role"].to_numpy(),
                              "period": period_of})
        grp = frame.groupby(["role", "period"])["v"]
        size = grp.transform("size")
        if cfg.robust:
            center = grp.transform("median")
            raw = grp.transform(lambda s: MAD_SCALE * (s - s.median()).abs().median())
        else:
            center = grp.transform("mean")
            raw = grp.transform("std").fillna(0.0)

        floor = center.map(lambda c: _spread_floor(float(c), col, scale, cfg))
        spread = np.maximum(raw.to_numpy(), floor.to_numpy())
        z = np.clip((frame["v"].to_numpy() - center.to_numpy()) / spread,
                    -cfg.z_cap, cfg.z_cap)

        too_small = (size < cfg.min_peer_group).to_numpy()

        # When a role-week's peer center is exactly 0, the Poisson floor for
        # count features is sqrt(max(0,1)) = 1 exactly, so any user posting a
        # raw count of precisely peer_z_threshold (e.g. 4 device connects
        # against peer_z_threshold=4.0) gets peer_z == threshold deterministically
        # -- a floor artifact, not a data-driven deviation. Withhold peer_z (fall
        # back to self_z, which the rules already OR against) rather than let a
        # trivially-reachable raw count trip a "peer deviation" rule.
        zero_center = (center.to_numpy() == 0.0)
        z[zero_center] = np.nan

        center_arr = center.to_numpy().astype(float)
        center_arr[too_small] = np.nan
        z[too_small] = np.nan

        peer_cols[f"{col}__peer_center"] = center_arr
        peer_cols[f"{col}__peer_spread"] = spread
        peer_cols[f"{col}__peer_z"] = z
        peer_cols[f"{col}__peer_n"] = size.to_numpy()

    out = out.drop(columns=["_role"])
    return pd.concat([out, pd.DataFrame(peer_cols, index=out.index)], axis=1)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str
    name: str
    description: str
    fields: tuple[str, ...]

    def evaluate(self, row: pd.Series, cfg: SignalConfig) -> bool:  # pragma: no cover
        raise NotImplementedError

    def triggering_fields(self, row: pd.Series, cfg: SignalConfig) -> tuple[str, ...]:
        """The fields that actually SATISFIED this rule's condition for this row.

        Deliberately narrower than `fields`, which also lists context features
        worth showing an analyst. A fact drawn from `fields` is nearby context;
        a fact named here is the reason the rule fired. Stage 2 must be able to
        tell them apart -- presenting eight facts with equal weight lets a
        narrative pad itself with confident claims about unremarkable numbers
        and dilute the one real anomaly, which is exactly the failure this
        thesis is built to bound.
        """  # pragma: no cover
        raise NotImplementedError


def _z(row: pd.Series, col: str, kind: str) -> float:
    val = row.get(f"{col}__{kind}_z", np.nan)
    return float(val) if pd.notna(val) else 0.0


class AfterHoursSpike(Rule):
    def evaluate(self, row, cfg):
        return (row["after_hours_logons"] >= 3
                and _z(row, "after_hours_logons", "self") >= cfg.self_z_threshold)

    def triggering_fields(self, row, cfg):
        return ("after_hours_logons",) if self.evaluate(row, cfg) else ()


class RemovableMediaExfil(Rule):
    def evaluate(self, row, cfg):
        return (row["removable_media_copies"] >= 10
                and (_z(row, "removable_media_copies", "self") >= cfg.self_z_threshold
                     or _z(row, "removable_media_copies", "peer") >= cfg.peer_z_threshold))

    def triggering_fields(self, row, cfg):
        return ("removable_media_copies",) if self.evaluate(row, cfg) else ()


# R2 requires removable_media_copies >= 10 alongside its z-test, so a user
# whose own baseline is a handful of copies can't trip the rule on a small
# absolute jump that's merely large *relative to them*. R3's self_z-only gate
# had the same gap: a user who rarely attaches anything trips self_z on a
# jump that's trivial in absolute terms. Brought in line with R2's pattern.
# 50 MB/week sits comfortably above every self_z-crossing benign week observed
# in this dataset's mock calibration run (max ~31 MB) and far below the
# genuine exfiltration case it recovers (~934 MB) -- revisit against the real
# r6.2 corpus once available; this is a floor calibrated on synthetic data.
EXTERNAL_ATTACHMENT_BYTES_FLOOR = 50_000_000


class ExternalEmailVolume(Rule):
    def evaluate(self, row, cfg):
        return (row["external_recipients"] >= 1
                and row["attachment_bytes"] >= EXTERNAL_ATTACHMENT_BYTES_FLOOR
                and _z(row, "attachment_bytes", "self") >= cfg.self_z_threshold)

    def triggering_fields(self, row, cfg):
        # Both are conditions of the rule: the volume that spiked, and the
        # external recipient that makes it exfiltration-shaped rather than
        # a large internal send.
        return ("attachment_bytes", "external_recipients") if self.evaluate(row, cfg) else ()


# Real-data calibration run (--sample-until 2011-01-01) found R4 and R5's
# peer_z-only gates produced 98% of all flags between them, concentrated in a
# small population firing on nearly every observed week (top R5 users: 53/53).
# Peer comparison alone tells you a user differs from their role peers; it
# cannot tell you whether that difference is this week's anomaly or just who
# that person is. R1-R3 already require self_z (compare the user to their own
# history) and stayed under 2% combined -- self_z is what actually locates an
# unusual WEEK. Both rules require self_z as the primary trigger, with peer_z
# as a supporting condition, not the reverse.
#
# A tiered variant was tried (an extra trigger for an extreme one-off peer_z,
# gated on the user not having a history of clearing peer_z_threshold on the
# same feature -- meant to recover two genuine positives, MBG3183 and
# PLJ1771, whose malicious week had a low self_z but a very high peer_z).
# Reverted: both users' peer_z on the relevant feature was ALSO habitually
# elevated (not just during their malicious week), so the persistence check
# correctly classified them as persistent and excluded them from the new path
# too -- it recovered neither. Worse, the persistence window has to start
# empty for every user, so a persistently-elevated user's early weeks (before
# their own hit count accumulates) passed the extreme-peer_z gate anyway:
# 97% of the users who fired R5 on >=30/53 weeks under a peer-only rule still
# fired at least once under the tiered version. Flag rate went UP (3.2% ->
# 8.6%) for no recall gain. self_z AND peer_z, jointly, is what's shipped.
class OffRoleBrowsing(Rule):
    def evaluate(self, row, cfg):
        return bool(self.triggering_fields(row, cfg))

    def triggering_fields(self, row, cfg):
        # The two paths are different evidence, and conflating them is exactly
        # the problem: on ACM2278's malicious weeks the leak-site visit is the
        # whole anomaly while job_search_visits sits at z=0.0, unremarkable for
        # that user. Only the path that actually fired is named as evidence.
        hits = []
        if row["leak_site_visits"] > 0:
            hits.append("leak_site_visits")
        if (row["job_search_visits"] >= 5
                and _z(row, "job_search_visits", "self") >= cfg.self_z_threshold
                and _z(row, "job_search_visits", "peer") >= cfg.peer_z_threshold):
            hits.append("job_search_visits")
        return tuple(hits)


class PeerDeviation(Rule):
    # self_z and peer_z must both clear on the SAME feature -- a user who is
    # simply elevated on one metric relative to peers while being elevated on a
    # different metric relative to their own history is not evidence of one
    # coherent anomaly.
    PEER_FIELDS = ("total_logons", "total_file_events",
                   "total_http_requests", "total_device_connects")

    def evaluate(self, row, cfg):
        return bool(self.triggering_fields(row, cfg))

    def triggering_fields(self, row, cfg):
        return tuple(c for c in self.PEER_FIELDS
                     if _z(row, c, "self") >= cfg.self_z_threshold
                     and _z(row, c, "peer") >= cfg.peer_z_threshold)


RULES: list[Rule] = [
    AfterHoursSpike("R1", "after_hours_spike",
                    "Unusual volume of logons outside the user's working hours",
                    ("after_hours_logons", "total_logons", "weekend_logons",
                     "n_active_days")),
    RemovableMediaExfil("R2", "removable_media_exfiltration",
                        "Elevated number of files copied to removable media",
                        ("removable_media_copies", "total_device_connects",
                         "after_hours_device_connects", "total_file_events")),
    ExternalEmailVolume("R3", "external_email_volume",
                        "Unusual volume of attachment data sent to external recipients",
                        ("attachment_bytes", "external_recipients",
                         "external_emails", "total_emails_sent")),
    OffRoleBrowsing("R4", "off_role_browsing",
                    "Browsing to job-search or leak-publication sites atypical for the role",
                    ("job_search_visits", "leak_site_visits",
                     "cloud_storage_visits", "distinct_domains")),
    PeerDeviation("R5", "peer_deviation",
                  "Overall activity volume far outside the same-role peer distribution",
                  ("total_logons", "total_file_events", "total_http_requests",
                   "total_device_connects")),
]


def apply_rules(enriched: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """Return only user-days firing at least one rule, with a triggered list."""
    if enriched.empty:
        return enriched.assign(triggered=[])

    triggered: list[list[str]] = []
    for _, row in enriched.iterrows():
        hits = [r.rule_id for r in RULES if r.evaluate(row, cfg)]
        triggered.append(hits)

    out = enriched.copy()
    out["triggered"] = triggered
    return out[out["triggered"].str.len() > 0]


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def _fmt(value: float, field_name: str) -> str:
    if field_name == "attachment_bytes":
        return f"{value/1_000_000:.2f} MB" if value >= 1_000_000 else f"{int(value)} bytes"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def build_signal(
    user: str,
    day: Any,
    row: pd.Series,
    identity: dict[str, str],
    label: dict[str, Any],
    cfg: SignalConfig,
) -> dict[str, Any]:
    """Assemble one anomaly signal record."""
    hits = [r for r in RULES if r.rule_id in row["triggered"]]

    # Which fields are the REASON each rule fired, as opposed to context the
    # rule merely lists. Facts carry this as triggering_rule_ids so Stage 2 can
    # foreground evidence instead of treating all facts as equally salient.
    evidence: dict[str, list[str]] = {}
    for r in hits:
        for fname in r.triggering_fields(row, cfg):
            evidence.setdefault(fname, []).append(r.rule_id)

    relevant = list(dict.fromkeys(f for r in hits for f in r.fields))
    for extra in ("distinct_pcs", "n_active_days", "earliest_hour",
                  "latest_hour"):
        if extra not in relevant:
            relevant.append(extra)
    # Evidence first, context after, so the low fact ids are the ones that
    # matter and the prompt's two blocks stay in a stable order.
    relevant.sort(key=lambda f: f not in evidence)

    facts = []
    for i, fname in enumerate(relevant[: cfg.max_facts], start=1):
        if fname not in row.index:
            continue
        value = float(row[fname])
        fact = {
            "fact_id": f"F{i:03d}",
            "field": fname,
            "value": value,
            "display": _fmt(value, fname),
            "gloss": FEATURE_GLOSS.get(fname, fname),
            "triggering_rule_ids": evidence.get(fname, []),
        }
        sz, pz = row.get(f"{fname}__self_z"), row.get(f"{fname}__peer_z")
        if pd.notna(sz):
            fact["self_center"] = round(float(row[f"{fname}__self_center"]), 3)
            fact["self_z"] = round(float(sz), 2)
            # z_cap is an artificial ceiling, not a measurement. Without this
            # flag a narrative can report "50 standard deviations", which is
            # fabricated precision -- the true value is unbounded.
            fact["self_z_capped"] = bool(abs(float(sz)) >= cfg.z_cap - 1e-9)
            hm = row.get(f"{fname}__self_hist_max")
            if pd.notna(hm):
                fact["self_recent_max"] = float(hm)
            fact["never_before"] = bool(row.get(f"{fname}__self_first", False))
        if pd.notna(pz):
            fact["peer_center"] = round(float(row[f"{fname}__peer_center"]), 3)
            fact["peer_z"] = round(float(pz), 2)
            fact["peer_z_capped"] = bool(abs(float(pz)) >= cfg.z_cap - 1e-9)
            fact["peer_group_size"] = int(row.get(f"{fname}__peer_n", 0))
        facts.append(fact)

    signal = {
        "signal_id": _signal_id(user, day),
        "user": user,
        "period": str(day),
        "period_type": "iso_week",
        "identity": identity,
        "triggered_rules": [
            {"rule_id": r.rule_id, "name": r.name, "description": r.description}
            for r in hits
        ],
        "facts": facts,
        "label": label,
    }
    return signal


def _week_span(year_week: str) -> str:
    from .cert_features import week_to_range
    mon, sun = week_to_range(year_week)
    return f"{mon.date()} to {sun.date()}"


def _signal_id(user: str, day: Any) -> str:
    return hashlib.sha1(f"{user}|{day}".encode()).hexdigest()[:12]


# Band edges for the qualitative deviation phrase that REPLACES the numeric z
# in the rendered prompt. Deliberately coarse: they preserve the ordering
# information a narrative needs ("is this a lot?") without handing it a number
# to restate as a statistic.
_DEVIATION_STRONG = 10.0
_DEVIATION_CLEAR = 3.0


def _deviation_phrase(z: float | None, capped: bool | None, kind: str) -> str:
    """Qualitative stand-in for the numeric z-score.

    The numeric z is suppressed from the rendered context entirely. Two
    measured reasons, both rendering-induced rather than model failures:

      1. It invites false precision. The v1 batch produced "a z-score of 21.0",
         "more than 20 standard deviations away", "3.44 standard deviations".
         These are traceable to the rendered text, so naive fact-matching
         accepts them -- but a median/MAD z with a Poisson spread floor is not
         a standard-deviation count, so the claim is statistically wrong
         (LIMITATIONS.md finding 7).
      2. In the common case the number is not even informative. When a peer
         group's centre is 0 the floor is sqrt(max(0,1)) = 1, so z = value / 1
         = value. That is why v1 shows z=21.0 beside 21 visits, z=34.0 beside
         34, z=7.0 beside 7 -- the "statistic" is just the raw count restated.

    The z stays in facts[] for scoring; this is a rendering change only.
    """
    if z is None and not capped:
        return ""
    scope = "their usual range" if kind == "own" else "the peer range"
    if capped:
        return f"; this week is far outside {scope}"
    z = float(z)
    if z >= _DEVIATION_STRONG:
        return f"; this week is far above {scope}"
    if z >= _DEVIATION_CLEAR:
        return f"; this week is clearly above {scope}"
    if z <= -_DEVIATION_CLEAR:
        return f"; this week is clearly below {scope}"
    return f"; this week is within {scope}"


def signal_to_prompt_context(signal: dict[str, Any]) -> str:
    """
    Render a signal as the compact block handed to the LLM.

    Facts are given with their ids so generated narratives can, in the
    instrumented condition, be asked to cite them — useful as an upper bound
    on what the admission function could achieve.
    """
    ident = signal["identity"]

    def _id(field: str) -> str:
        # Missing, empty, and the string "nan" (an LDAP blank read as NaN then
        # stringified) must all render as "unknown" -- a literal "nan" in the
        # prompt is something the LLM can and will copy into a narrative.
        value = ident.get(field)
        if value is None:
            return "unknown"
        text = str(value).strip()
        return "unknown" if text == "" or text.lower() == "nan" else text

    # business_unit is deliberately omitted: in r6.2 it is the bare string "1"
    # for every user in the company (one distinct value across all LDAP
    # snapshots), so it identifies nothing and only invites a narrative to
    # treat "Business Unit 1" as meaningful. functional_unit carries the
    # information an analyst would actually want ("1 - Adminstration").
    lines = [
        f"User: {signal['user']} ({_id('employee_name')})",
        f"Role: {_id('role')} | Team: {_id('team')} "
        f"| Functional unit: {_id('functional_unit')}",
        f"Week: {signal['period']} ({_week_span(signal['period'])})",
        "",
        "Triggered detection rules:",
    ]
    lines += [f"  - [{r['rule_id']}] {r['description']}" for r in signal["triggered_rules"]]

    def _render(f: dict) -> str:
        parts = [f"  - [{f['fact_id']}] {f['gloss']}: {f['display']}"]
        if f.get("triggering_rule_ids"):
            parts.append(f"[evidence for {', '.join(f['triggering_rule_ids'])}]")
        if f.get("never_before"):
            parts.append("(not observed for this user in the prior 12 weeks)")
        elif "self_center" in f:
            # The 12-week high is NOT rendered. Three renderings were measured:
            #
            #   v1 "max: N"                      -> the model read N as a
            #      threshold and claimed the value "exceeds the maximum" even
            #      when it was BELOW it (leak visits 3 vs max 4: 5/10 false).
            #   v2 "highest in the prior 12 weeks: N" -> relabelling fixed the
            #      two classes that really were rendering artifacts (numeric z,
            #      peer-size-as-denominator both went to zero) but NOT this one.
            #      False exceedance claims fell 11 -> 5 and stalled there, on
            #      sentences that quote the current value and the high, both
            #      correctly labelled, in one clause and still compare them
            #      wrongly ("12 files copied ... above their highest count in
            #      the prior 12 weeks of 13"). That is an arithmetic failure by
            #      an 8B model, not an ambiguous label, and no wording fixes it.
            #   v3 (here) omit it -> the model cannot restate a number it was
            #      never given, so a surviving exceedance claim cites an
            #      unsupported figure and fact-matching catches it.
            #
            # The deliberate consequence: this converts a traceable-but-wrong
            # comparison, which naive fact-matching accepts because both numbers
            # are real, into a plain hallucination it rejects. self_recent_max
            # stays in facts[] for scoring -- this is a rendering change only.
            parts.append(
                f"(own recent typical: {_fmt(f['self_center'], f['field'])}"
                f"{_deviation_phrase(f.get('self_z'), f.get('self_z_capped'), 'own')})"
            )
        if "peer_center" in f:
            # "this week", not "today": the aggregation unit is the ISO week
            # (see the scope guardrail in CLAUDE.md). Saying "today" invites a
            # narrative to make day-level claims that no fact supports.
            #
            # "across N peers in this role", never a bare "n=N": rendered as
            # n=70 the model narrated it as a fraction -- "3 out of 70",
            # "1 visit among 65", "328 employees falling within the same peer
            # distribution" -- treating the group size as a count of events.
            parts.append(
                f"(typical among same-role peers this week: "
                f"{_fmt(f['peer_center'], f['field'])}, across "
                f"{f['peer_group_size']} peers in this role"
                f"{_deviation_phrase(f.get('peer_z'), f.get('peer_z_capped'), 'peer')})"
            )
        return " ".join(parts)

    ev = [f for f in signal["facts"] if f.get("triggering_rule_ids")]
    ctx = [f for f in signal["facts"] if not f.get("triggering_rule_ids")]

    # Two blocks, not one flat list: the evidence facts are why the rules
    # fired; the context facts are nearby values that were not themselves
    # anomalous. Presenting them identically lets a narrative spend its
    # confident sentences on unremarkable numbers.
    if ev:
        lines += ["", "Evidence for triggered rules:"]
        lines += [_render(f) for f in ev]
    if ctx:
        lines += ["", "Additional context (not anomalous; do not present as findings):"]
        lines += [_render(f) for f in ctx]
    return "\n".join(lines)


def write_signals(signals: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        for s in signals:
            fh.write(json.dumps(s) + "\n")