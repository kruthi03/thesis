"""CERT r6.2 activity logs -> raw per user-week feature matrix.

Streams each log once, in chunks, so http.csv (~90 GB on the real dataset)
is never re-read to add a column. Checkpointing lets a crash mid-run resume
without redoing finished work. Features are raw counts and extrema --
normalization happens downstream in signals.py, never here.
"""
from __future__ import annotations

import csv
import os
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATE_FMT = "%m/%d/%Y %H:%M:%S"
INTERNAL_DOMAIN = "dtaa.com"

# After-hours: before 07:00 or at/after 19:00.
AFTER_HOURS_START_HOUR = 7
AFTER_HOURS_END_HOUR = 19

DEFAULT_CHUNK_SIZE = 1_000_000
CHECKPOINT_EVERY_CHUNKS = 50

# Cap on distinct-value sets (pcs, files, domains, ...) per user-week, so one
# pathological user cannot exhaust memory across a 90 GB pass.
MAX_DISTINCT = 500

# earliest_hour/latest_hour stay NaN (not 0) for a user-week with no logons --
# a default of 0 would read as midnight activity in a narrative.
_BLANK_ON_MERGE = {"earliest_hour", "latest_hour"}

# Crude but transparent and auditable keyword lists, matched against the
# request's domain (case-insensitive substring).
JOB_SEARCH_KEYWORDS = [
    "job-hunt", "careerbuilder", "monster.com", "indeed.com",
    "linkedin.com", "simplyhired", "dice.com",
]
CLOUD_STORAGE_KEYWORDS = [
    "dropbox.com", "box.com", "drive.google.com", "mega.co.nz", "4shared.com",
]
LEAK_SITE_KEYWORDS = ["wikileaks", "pastebin.com", "cryptome"]

_ATTACH_BYTES_RE = re.compile(r"\((\d+)\)$")


# ---------------------------------------------------------------------------
# Shared per-chunk prep
# ---------------------------------------------------------------------------

def _prep(chunk: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({
        "user": chunk["user"].values,
        "ts": pd.to_datetime(chunk["date"].values, format=DATE_FMT),
    })
    iso = d["ts"].dt.isocalendar()
    d["period"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    d["hour"] = d["ts"].dt.hour
    d["after_hours"] = (d["hour"] < AFTER_HOURS_START_HOUR) | (d["hour"] >= AFTER_HOURS_END_HOUR)
    d["weekend"] = d["ts"].dt.dayofweek >= 5
    return d


def _distinct_chunk(sub: pd.DataFrame, values, field: str) -> pd.Series:
    """Per-(user,period) set of unique values in this chunk, for later capped merge."""
    tmp = pd.DataFrame({"user": sub["user"].values, "period": sub["period"].values, field: values})
    return tmp.groupby(["user", "period"])[field].apply(lambda s: set(s.dropna().unique()))


def _domain_of(url: str) -> str:
    try:
        rest = url.split("://", 1)[-1]
        return rest.split("/", 1)[0]
    except Exception:
        return ""


def _keyword_mask(domains_lower: pd.Series, keywords: list[str]) -> pd.Series:
    pattern = "|".join(re.escape(k) for k in keywords)
    return domains_lower.str.contains(pattern, regex=True, na=False)


def _external_recipient_counts(series: pd.Series) -> np.ndarray:
    """External-recipient count per row, anchored on '@' so
    x@dtaa.com.evil.net is not mistaken for internal."""
    counts = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series.fillna("").astype(str)):
        if not val:
            continue
        n = 0
        for r in val.split(";"):
            r = r.strip()
            if not r or "@" not in r:
                continue
            domain = r.rsplit("@", 1)[-1].lower()
            if domain != INTERNAL_DOMAIN:
                n += 1
        counts[i] = n
    return counts


def parse_attachments_field(value: str) -> tuple[int, int]:
    """Parse one email ``attachments`` cell: ``path(bytes);path2(bytes2)``.

    ``size`` is the whole message (body + attachments combined), so
    attachment volume must come from parsing this field. Split on ``;``
    first, then anchor the byte-count regex to the end of each entry --
    a global findall over the raw string would misparse a filename that
    itself contains parentheses not at the end (e.g. ``report(v2).doc(4553)``
    would otherwise contribute a spurious ``(2)`` byte count).
    """
    value = value.strip()
    if not value:
        return 0, 0
    entries = [e.strip() for e in value.split(";") if e.strip()]
    total_bytes = 0
    for entry in entries:
        m = _ATTACH_BYTES_RE.search(entry)
        if m:
            total_bytes += int(m.group(1))
    return len(entries), total_bytes


def _parse_attachments(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(count, total_bytes) per row, via parse_attachments_field. Empty/NaN
    cells are a genuine zero, not a missing value."""
    n = np.zeros(len(series), dtype=np.int64)
    nbytes = np.zeros(len(series), dtype=np.int64)
    for i, val in enumerate(series.fillna("").astype(str)):
        n[i], nbytes[i] = parse_attachments_field(val)
    return n, nbytes


# ---------------------------------------------------------------------------
# Per-log chunk processors
# ---------------------------------------------------------------------------

def _logon_chunk(chunk: pd.DataFrame, acc: "_Accumulator") -> None:
    d = _prep(chunk)
    is_logon = (chunk["activity"].values == "Logon")
    d["total_logons"] = is_logon.astype(int)
    d["after_hours_logons"] = (is_logon & d["after_hours"].values).astype(int)
    d["weekend_logons"] = (is_logon & d["weekend"].values).astype(int)

    grouped = d.groupby(["user", "period"])[
        ["total_logons", "after_hours_logons", "weekend_logons"]
    ].sum()
    acc.add_sum(grouped)

    day = d["ts"].dt.date.astype(str)
    acc.merge_distinct(_distinct_chunk(d, chunk["pc"].values, "pcs"), "pcs")
    acc.merge_distinct(_distinct_chunk(d, day.values, "active_days"), "active_days")

    weekend_mask = d["weekend"].values
    if weekend_mask.any():
        acc.merge_distinct(
            _distinct_chunk(d.loc[weekend_mask], day.values[weekend_mask], "weekend_days"),
            "weekend_days",
        )

    if is_logon.any():
        ext = d.loc[is_logon].groupby(["user", "period"])["hour"].agg(["min", "max"])
        acc.merge_extrema(ext)


def _device_chunk(chunk: pd.DataFrame, acc: "_Accumulator") -> None:
    d = _prep(chunk)
    is_connect = (chunk["activity"].values == "Connect")
    d["total_device_connects"] = is_connect.astype(int)
    d["after_hours_device_connects"] = (is_connect & d["after_hours"].values).astype(int)

    grouped = d.groupby(["user", "period"])[
        ["total_device_connects", "after_hours_device_connects"]
    ].sum()
    acc.add_sum(grouped)


def _file_chunk(chunk: pd.DataFrame, acc: "_Accumulator") -> None:
    d = _prep(chunk)
    to_removable = chunk["to_removable_media"].astype(str).str.strip().str.lower() == "true"
    d["total_file_events"] = 1
    d["removable_media_copies"] = to_removable.astype(int).values
    d["after_hours_file_events"] = d["after_hours"].astype(int)

    grouped = d.groupby(["user", "period"])[
        ["total_file_events", "removable_media_copies", "after_hours_file_events"]
    ].sum()
    acc.add_sum(grouped)

    acc.merge_distinct(_distinct_chunk(d, chunk["filename"].values, "files"), "files")
    ext = chunk["filename"].astype(str).str.extract(r"(\.[A-Za-z0-9]+)$", expand=False)
    acc.merge_distinct(_distinct_chunk(d, ext.values, "file_exts"), "file_exts")


def _email_chunk(chunk: pd.DataFrame, acc: "_Accumulator") -> None:
    d = _prep(chunk)
    is_send = (chunk["activity"].values == "Send")

    ext_to = _external_recipient_counts(chunk["to"])
    ext_cc = _external_recipient_counts(chunk["cc"])
    ext_bcc = _external_recipient_counts(chunk["bcc"])
    ext_total = ext_to + ext_cc + ext_bcc
    n_attach, attach_bytes = _parse_attachments(chunk["attachments"])

    d["total_emails_sent"] = is_send.astype(int)
    d["total_attachments"] = np.where(is_send, n_attach, 0)
    d["external_emails"] = np.where(is_send & (ext_total > 0), 1, 0)
    d["external_recipients"] = np.where(is_send, ext_total, 0)
    d["attachment_bytes"] = np.where(is_send, attach_bytes, 0)
    d["after_hours_emails"] = np.where(is_send & d["after_hours"].values, 1, 0)

    cols = [
        "total_emails_sent", "total_attachments", "external_emails",
        "external_recipients", "attachment_bytes", "after_hours_emails",
    ]
    grouped = d.groupby(["user", "period"])[cols].sum()
    acc.add_sum(grouped)


def _http_chunk(chunk: pd.DataFrame, acc: "_Accumulator") -> None:
    d = _prep(chunk)
    domains = chunk["url"].astype(str).map(_domain_of)
    domains_lower = domains.str.lower()

    d["total_http_requests"] = 1
    d["after_hours_http"] = d["after_hours"].astype(int)
    d["job_search_visits"] = _keyword_mask(domains_lower, JOB_SEARCH_KEYWORDS).astype(int).values
    d["cloud_storage_visits"] = _keyword_mask(domains_lower, CLOUD_STORAGE_KEYWORDS).astype(int).values
    d["leak_site_visits"] = _keyword_mask(domains_lower, LEAK_SITE_KEYWORDS).astype(int).values

    cols = [
        "total_http_requests", "after_hours_http", "job_search_visits",
        "cloud_storage_visits", "leak_site_visits",
    ]
    grouped = d.groupby(["user", "period"])[cols].sum()
    acc.add_sum(grouped)

    acc.merge_distinct(_distinct_chunk(d, domains.values, "domains"), "domains")


# Canonical raw feature column order produced by extract_raw_features -- the
# outer-join order of logon, device, file, email, then http columns, each in
# the order their processor sums/appends them. Kept explicit (not derived at
# import time) so a change to a processor's column order is a visible diff
# here rather than a silent reshuffle picked up by downstream consumers.
FEATURE_COLUMNS: list[str] = [
    "total_logons", "after_hours_logons", "weekend_logons", "distinct_pcs",
    "n_active_days", "n_weekend_days", "earliest_hour", "latest_hour",
    "total_device_connects", "after_hours_device_connects",
    "total_file_events", "removable_media_copies", "after_hours_file_events",
    "distinct_files", "distinct_file_exts",
    "total_emails_sent", "total_attachments", "external_emails",
    "external_recipients", "attachment_bytes", "after_hours_emails",
    "total_http_requests", "after_hours_http", "job_search_visits",
    "cloud_storage_visits", "leak_site_visits", "distinct_domains",
]


# n_fields is the log's full column count; field_idx maps the columns we read
# to their 0-based position. Positions (not names) because rows are tokenized
# by splitting raw lines rather than by the CSV parser -- see _read_chunks.
LOG_SPECS = {
    "logon": {
        "n_fields": 5,
        "field_idx": {"date": 1, "user": 2, "pc": 3, "activity": 4},
        "processor": _logon_chunk,
    },
    "device": {
        "n_fields": 6,
        "field_idx": {"date": 1, "user": 2, "activity": 5},
        "processor": _device_chunk,
    },
    "file": {
        "n_fields": 9,
        "field_idx": {"date": 1, "user": 2, "filename": 4, "activity": 5,
                      "to_removable_media": 6},
        "processor": _file_chunk,
    },
    "email": {
        "n_fields": 12,
        "field_idx": {"date": 1, "user": 2, "to": 4, "cc": 5, "bcc": 6,
                      "activity": 8, "attachments": 10},
        "processor": _email_chunk,
    },
    "http": {
        "n_fields": 7,
        "field_idx": {"date": 1, "user": 2, "url": 4},
        "processor": _http_chunk,
    },
}


def _read_chunks(csv_path: Path, spec: dict, chunk_size: int, skip_rows: int):
    """Yield chunks with just the columns in spec["field_idx"].

    Reads each line as ONE opaque field (a separator byte that never occurs in
    the data) and splits it positionally, instead of letting the CSV parser
    interpret quotes. Confirmed necessary on the real dataset: http.csv
    contains a `content` field with an unbalanced quote, which the quote-aware
    C parser reads as "the rest of the file is one unterminated string",
    raising ParserError: EOF inside string ~113.27M rows in. A bare resume
    replays into the same row and fails identically.

    Rejected alternatives: quoting=QUOTE_NONE on the normal parser silently
    shifts columns on every row whose content legitimately contains a comma
    (pandas absorbs the surplus leading fields into the index); engine='python'
    with on_bad_lines='skip' tolerates the bad row but is far too slow to skip
    to a 113M-row offset in a 90 GB file.

    Safe because `content` is the LAST column in every schema that has one, so
    a split bounded at n_fields-1 leaves content as an untouched remainder --
    its commas and quotes never reach the fields we read. Verified against the
    normal parser on 300k-row samples of all five logs: identical row counts
    (so no content field contains an embedded newline) and identical values
    for every column read (so no earlier field contains a comma), modulo empty
    string vs NaN, normalised below.
    """
    n_fields = spec["n_fields"]
    field_idx = spec["field_idx"]
    with pd.read_csv(
        csv_path, sep="\x01", header=None, names=["_raw"],
        skiprows=1 + skip_rows, chunksize=chunk_size, dtype=str,
        quoting=csv.QUOTE_NONE, engine="c",
    ) as reader:
        for raw_chunk in reader:
            parts = raw_chunk["_raw"].str.split(",", n=n_fields - 1, expand=True)
            out = pd.DataFrame(
                {name: parts[i] for name, i in field_idx.items()},
                index=raw_chunk.index,
            )
            # The CSV parser yields NaN for an empty field; splitting yields "".
            yield out.replace("", np.nan)


# ---------------------------------------------------------------------------
# Accumulator + checkpointing
# ---------------------------------------------------------------------------

class _Accumulator:
    """Picklable running state for one log's streaming aggregation."""

    def __init__(self) -> None:
        self.sums = pd.DataFrame()
        self.distinct: dict[tuple, dict[str, set]] = {}
        self.extrema: dict[tuple, list] = {}
        self.rows_processed = 0
        self.chunks_processed = 0

    def add_sum(self, grouped: pd.DataFrame) -> None:
        self.sums = grouped.copy() if self.sums.empty else self.sums.add(grouped, fill_value=0)

    def merge_distinct(self, series: pd.Series, field: str, cap: int = MAX_DISTINCT) -> None:
        for key, vals in series.items():
            s = self.distinct.setdefault(key, {}).setdefault(field, set())
            if len(s) < cap:
                remaining = cap - len(s)
                s.update(list(vals)[:remaining])

    def merge_extrema(self, grouped: pd.DataFrame) -> None:
        for key, row in grouped.iterrows():
            if key in self.extrema:
                e = self.extrema[key]
                e[0] = min(e[0], row["min"])
                e[1] = max(e[1], row["max"])
            else:
                self.extrema[key] = [row["min"], row["max"]]


def _atomic_write_pickle(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def _set_len_col(distinct: dict, field: str, index: pd.MultiIndex) -> list[int]:
    return [len(distinct.get(key, {}).get(field, set())) for key in index]


def _extrema_col(extrema: dict, index: pd.MultiIndex, pos: int) -> list[float]:
    out = []
    for key in index:
        e = extrema.get(key)
        out.append(e[pos] if e is not None else np.nan)
    return out


def _finalize(name: str, acc: _Accumulator) -> pd.DataFrame:
    sums = acc.sums.copy()
    if not isinstance(sums.index, pd.MultiIndex):
        sums.index = pd.MultiIndex.from_tuples(list(sums.index), names=["user", "period"])

    if name == "logon":
        sums["distinct_pcs"] = _set_len_col(acc.distinct, "pcs", sums.index)
        sums["n_active_days"] = _set_len_col(acc.distinct, "active_days", sums.index)
        sums["n_weekend_days"] = _set_len_col(acc.distinct, "weekend_days", sums.index)
        sums["earliest_hour"] = _extrema_col(acc.extrema, sums.index, 0)
        sums["latest_hour"] = _extrema_col(acc.extrema, sums.index, 1)
    elif name == "file":
        sums["distinct_files"] = _set_len_col(acc.distinct, "files", sums.index)
        sums["distinct_file_exts"] = _set_len_col(acc.distinct, "file_exts", sums.index)
    elif name == "http":
        sums["distinct_domains"] = _set_len_col(acc.distinct, "domains", sums.index)

    return sums


def _aggregate_log(name: str, csv_path: Path, out_dir: Path, chunk_size: int,
                   sample_nrows: int | None, sample_until: "pd.Timestamp | None" = None) -> pd.DataFrame:
    agg_path = out_dir / f"{name}_agg.csv"
    if agg_path.exists():
        # Cache each completed per-log aggregation as CSV; skip it entirely on rerun.
        return pd.read_csv(agg_path, index_col=[0, 1])

    ckpt_path = out_dir / f"{name}_partial.pkl"
    spec = LOG_SPECS[name]

    if ckpt_path.exists():
        with open(ckpt_path, "rb") as fh:
            acc = pickle.load(fh)
    else:
        acc = _Accumulator()

    # Resume by physically skipping the rows already aggregated, so a resumed
    # run does not re-read what it already processed.
    for chunk in _read_chunks(csv_path, spec, chunk_size,
                              skip_rows=acc.chunks_processed * chunk_size):
        if sample_nrows is not None and acc.rows_processed >= sample_nrows:
            break

        if sample_until is not None:
            # Each CERT r6.2 log is written in chronological order, so once a
            # chunk contains any row at/after the cutoff, every later chunk is
            # entirely past it too -- filter this chunk down to the rows before
            # the cutoff, process that partial chunk, then stop. This keeps
            # every log's sampled window ending on the SAME calendar date,
            # unlike sample_nrows (a fixed row count covers a wildly different
            # date range per log -- http.csv is ~90 GB, logon.csv ~240 MB).
            row_ts = pd.to_datetime(chunk["date"], format=DATE_FMT)
            past_cutoff = row_ts >= sample_until
            if past_cutoff.any():
                chunk = chunk[~past_cutoff]
                if not chunk.empty:
                    spec["processor"](chunk, acc)
                    acc.rows_processed += len(chunk)
                    acc.chunks_processed += 1
                break

        spec["processor"](chunk, acc)
        # Count rows only for chunks actually processed here, or a resumed
        # run double-counts the ones skipped via skip_rows.
        acc.rows_processed += len(chunk)
        acc.chunks_processed += 1
        if acc.chunks_processed % CHECKPOINT_EVERY_CHUNKS == 0:
            _atomic_write_pickle(acc, ckpt_path)

    final = _finalize(name, acc)

    tmp = agg_path.with_suffix(agg_path.suffix + ".tmp")
    final.to_csv(tmp)
    os.replace(tmp, agg_path)
    if ckpt_path.exists():
        ckpt_path.unlink()

    return final


def extract_raw_features(cert_root: str | Path, out_dir: str | Path,
                         chunk_size: int = DEFAULT_CHUNK_SIZE,
                         sample_nrows: int | None = None,
                         sample_until: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Stream logon/device/file/email/http once each into a raw, unnormalized
    (user, period) feature matrix. Merge is an outer join: missing means zero
    activity (a genuine zero), except earliest_hour/latest_hour which stay
    NaN for a user-week with no logons.

    sample_nrows caps rows read PER LOG INDEPENDENTLY -- fine for a quick raw
    smoke test, but the five logs have wildly different row density (http.csv
    is ~90 GB, logon.csv ~240 MB), so the same row count reaches a different
    calendar date in each. That makes any downstream detection-quality read
    (flag rate, recall) meaningless: a user-week can show real file/device
    activity but zero http/email activity purely because those logs' sampled
    windows didn't reach that week, not because nothing happened. Use
    sample_until for a bounded trial run whose results should mean something;
    reserve sample_nrows for checking that extraction runs at all.
    """
    cert_root = Path(cert_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_until is not None:
        sample_until = pd.Timestamp(sample_until)

    frames = {}
    for name in LOG_SPECS:
        frames[name] = _aggregate_log(name, cert_root / f"{name}.csv", out_dir,
                                      chunk_size, sample_nrows, sample_until)

    merged = None
    for df in frames.values():
        merged = df if merged is None else merged.join(df, how="outer")
    merged = merged.sort_index()
    merged.index = merged.index.set_names(["user", "period"])

    for col in merged.columns:
        if col not in _BLANK_ON_MERGE:
            merged[col] = merged[col].fillna(0)

    count_cols = [c for c in merged.columns if c not in _BLANK_ON_MERGE]
    negative = merged[count_cols].lt(0).any()
    if negative.any():
        raise ValueError(f"negative counts in columns: {list(negative[negative].index)}")

    if list(merged.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"extract_raw_features produced columns {list(merged.columns)}, "
            f"which no longer match FEATURE_COLUMNS {FEATURE_COLUMNS} -- a "
            "processor's column order changed; update FEATURE_COLUMNS to match."
        )

    return merged


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

# The one known r6.2 case: CDE1846's start is recorded as
# "/21/2011 11:43:39" (month missing, leading slash), verified as
# "02/21/2011 11:43:39" against the scenario-4 detail file. Do NOT generalise
# this by prepending "02" to any malformed value -- see the bug list in
# CLAUDE.md: on a different user, whose true month was e.g. August, that
# guess silently produces February and corrupts the label.
KNOWN_DATE_FIXES: dict[tuple[str, str], str] = {
    ("CDE1846", "/21/2011 11:43:39"): "02/21/2011 11:43:39",
}

# A malicious window spanning this many weeks or more almost certainly means
# a date-repair or grouping bug, not a genuinely long campaign.
MAX_PLAUSIBLE_WINDOW_WEEKS = 10


def _parse_insider_dt(user: str, raw: str, fixes: dict[tuple[str, str], str]) -> datetime:
    raw = raw.strip()
    try:
        return datetime.strptime(raw, DATE_FMT)
    except ValueError:
        fix = fixes.get((user, raw))
        if fix is None:
            raise ValueError(
                f"unparseable insiders.csv date {raw!r} for user {user!r} has no "
                "registered fix. Add an audited entry to KNOWN_DATE_FIXES (or "
                "--date-fixes), keyed by (user, exact malformed value) -- do not "
                "guess a repair."
            ) from None
        return datetime.strptime(fix, DATE_FMT)


def load_insiders(cert_root: str | Path, release: str = "6.2",
                  date_fixes: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    """Ground-truth malicious windows: one row per user, min(start)/max(end)
    across all their rows in the target release."""
    path = Path(cert_root) / "answers" / "insiders.csv"
    raw = pd.read_csv(path, dtype=str)

    fixes = dict(KNOWN_DATE_FIXES)
    if date_fixes:
        fixes.update(date_fixes)

    rows = []
    for _, r in raw.iterrows():
        if str(r["dataset"]).strip() != release:
            continue
        user = r["user"].strip()
        rows.append({
            "user": user,
            "scenario": str(r["scenario"]).strip(),
            "start": _parse_insider_dt(user, r["start"], fixes),
            "end": _parse_insider_dt(user, r["end"], fixes),
        })

    if not rows:
        return pd.DataFrame(columns=["start", "end", "scenario"]).rename_axis("user")

    df = pd.DataFrame(rows)
    grouped = df.groupby("user").agg(start=("start", "min"), end=("end", "max"),
                                     scenario=("scenario", "first"))

    width_weeks = (grouped["end"] - grouped["start"]).dt.total_seconds() / (7 * 86400)
    implausible = width_weeks[width_weeks >= MAX_PLAUSIBLE_WINDOW_WEEKS]
    if not implausible.empty:
        raise ValueError(
            f"malicious window >= {MAX_PLAUSIBLE_WINDOW_WEEKS} weeks for users "
            f"{list(implausible.index)} -- check for a date-repair or grouping bug "
            "before trusting these labels."
        )

    return grouped


def week_to_range(period: str) -> tuple[datetime, datetime]:
    """ISO week label (e.g. ``2010-W25``) -> (Monday 00:00:00, Sunday 23:59:59)."""
    year, week = period.split("-W")
    start = datetime.strptime(f"{year}-W{week}-1", "%G-W%V-%w")
    end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end


def label_user_weeks(index: pd.MultiIndex, insiders: pd.DataFrame) -> pd.DataFrame:
    """Label malicious if the week's Monday-Sunday range overlaps the user's
    malicious window: a_start <= b_end and b_start <= a_end."""
    rows = []
    for user, period in index:
        week_start, week_end = week_to_range(period)
        is_malicious = False
        scenario = None
        if user in insiders.index:
            b_start, b_end = insiders.loc[user, "start"], insiders.loc[user, "end"]
            if week_start <= b_end and b_start <= week_end:
                is_malicious = True
                scenario = insiders.loc[user, "scenario"]
        rows.append({"user": user, "period": period, "is_malicious": is_malicious,
                    "scenario": scenario})
    return pd.DataFrame(rows).set_index(["user", "period"])
