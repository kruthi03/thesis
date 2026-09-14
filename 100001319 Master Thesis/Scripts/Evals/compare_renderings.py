"""Before/after contrast for the rendering fixes (LIMITATIONS finding 10).

    python scripts/compare_renderings.py \
        --before out_narratives_ollama_v2/t085/narratives.jsonl \
        --after  out_narratives_ollama_v3/t085/narratives.jsonl

Labels come from each file's recorded prompt_version, so the same script serves
v1->v2 and v2->v3. Counts are over the RAW sample text (not the pooled
sentences -- pooling collapses duplicates and would understate how often a
phrasing occurred):

  * "exceeds the maximum" claims: any sentence asserting the current value went
    over a limit. Reported separately for signals where the claim was FALSE,
    since those are the ones finding 10a is about.
  * numeric-sigma restatements: "z-score of N", "N standard deviations", "N
    sigma".
  * peer-group-as-denominator phrasings: "out of N" / "among N" where N is the
    signal's peer_group_size.
  * unrendered-max citations: the trailing-12-week high quoted verbatim. Only
    meaningful once the render stops showing it (v3), where it is the direct
    test of the v3 rationale -- a number the model was never given can only be
    hallucinated, which is what makes fact-matching able to reject the claim.
    Under v1/v2 the figure IS in the prompt, so a nonzero count there is just
    faithful restatement and is reported for contrast, not as an error.

Comparison is restricted to signal_ids present in BOTH files, so a partial run
cannot make the later rendering look better by having fewer signals.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# "exceeded its maximum", "exceeds the max of 12", "above the maximum", "surpassed
# the peak". Deliberately broad -- false positives here are visible on inspection,
# whereas a narrow pattern would silently undercount and flatter v2.
EXCEEDS = re.compile(
    r"\b(exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|above|over|beyond|higher than)\b"
    r"[^.]{0,60}?\b(maximum|max|peak|highest|previous high|prior high|limit|threshold)\b",
    re.I,
)
SIGMA = re.compile(
    r"(z[- ]?score[^.]{0,20}?\d|\d+(?:\.\d+)?\s*(?:standard deviations?|sigma|σ)"
    r"|\d+(?:\.\d+)?\s*(?:sd|s\.d\.)\b)",
    re.I,
)


def signal_facts(signal_ids, signals_path):
    """Per signal: peer group sizes, and whether ANY fact genuinely exceeded
    its own trailing-12-week maximum.

    The second is what makes the exceeds-max count interpretable. A raw count
    of "exceeds" phrasings conflates true statements with false ones -- and the
    v2 rendering deliberately still SHOWS the 12-week high, so a correct
    narrative may legitimately say "above the highest in the prior 12 weeks".
    On a signal where no fact exceeds its max, every such claim is false.
    """
    out = {}
    for line in open(signals_path, encoding="utf-8"):
        s = json.loads(line)
        if s["signal_id"] not in signal_ids:
            continue
        sizes, exceeds = set(), False
        maxes, other = set(), set()
        # Digits in the identity block are rendered too ("Team: 19 - Materials
        # Engineering" -> "Team 19", "the 13th team"). Found the same way as
        # the "12 weeks" false positives: the two surviving v3 max-figure hits
        # were both team numbers, not fact citations.
        for field in ("role", "team", "functional_unit"):
            for d in re.findall(r"\d+", str(s.get("identity", {}).get(field, ""))):
                other.add(float(d))
        for f in s["facts"]:
            if f.get("peer_group_size") is not None:
                sizes.add(int(f["peer_group_size"]))
                other.add(float(f["peer_group_size"]))
            for key in ("value", "self_center", "peer_center"):
                if f.get(key) is not None:
                    other.add(float(f[key]))
            if f.get("self_recent_max") is not None and not f.get("never_before"):
                maxes.add(float(f["self_recent_max"]))
                if float(f["value"]) > float(f["self_recent_max"]):
                    exceeds = True
        # A max that coincides with a figure the prompt DOES render (the value
        # itself, the centre, a peer count) is not attributable: seeing it in
        # the text proves nothing about where it came from. Only maxes unique
        # to the max are counted, so this undercounts rather than inflates.
        out[s["signal_id"]] = {"sizes": sizes, "true_exceedance": exceeds,
                               "maxes": maxes - other,
                               "user": s["user"], "period": s["period"]}
    return out


# Numbers that are structural boilerplate, not restated facts:
#   "the prior 12 weeks" / "past 12-week"  -- the window length, in ~every text
#   "R4", "[R2]"                           -- rule ids
#   "2010-W33"                             -- period labels
_BOILERPLATE = re.compile(
    r"\[?\bR\d+\]?"
    r"|\b\d{4}-W\d{1,2}\b"
    r"|\b\d+[- ]?weeks?\b",
    re.I,
)


def _scrub(text: str) -> str:
    """Blank out boilerplate numerals so they cannot be read as fact citations."""
    return _BOILERPLATE.sub(" ", text)


def scan(records, meta):
    out = {}
    for r in records:
        sid = r["signal_id"]
        ns = meta.get(sid, {}).get("sizes", set())
        # "(?! *peers)" matters: the v2 fix renders "across 85 peers in this
        # role", which is the CORRECT phrasing. Counting it would score the fix
        # as a regression against itself. Only a bare "out of 85" / "among 85"
        # with no "peers" head noun is the denominator misreading.
        denom = re.compile(
            r"\b(?:out of|of|among|amongst|across)\s+(?:the\s+)?(" +
            "|".join(str(n) for n in sorted(ns)) +
            r")\b(?!\s+(?:peers|same-role|colleagues|others))"
        ) if ns else None

        # Word-boundary match on the max as rendered, so 13 does not match
        # inside 130 or 2013. Integers are matched bare ("13"), non-integers in
        # the one-decimal form the renderer uses ("1.5").
        #
        # Boilerplate is scrubbed BEFORE matching. Without this the metric is
        # almost entirely false positives, as the first v3 run showed: the
        # phrase "the prior 12 weeks" appears in nearly every narrative and
        # matched any fact whose max happened to be 12, and rule ids ("R4",
        # "[R2]") matched any max of 2/3/4/5. That produced an apparent 8/10 ->
        # 10/10 RISE on ACM2278 2010-W33 where not one sample cited a max.
        mx = meta.get(sid, {}).get("maxes", set())
        maxnum = re.compile(
            r"(?<![\d.])(" + "|".join(
                re.escape(f"{m:g}" if m != int(m) else str(int(m)))
                for m in sorted(mx)
            ) + r")(?![\d.])"
        ) if mx else None

        hits = Counter()
        per_sample = {"exceeds": 0, "sigma": 0, "denom": 0, "maxnum": 0}
        for n in r["narratives"]:
            text = n["text"]
            e = EXCEEDS.findall(text)
            g = SIGMA.findall(text)
            d = denom.findall(text) if denom else []
            x = maxnum.findall(_scrub(text)) if maxnum else []
            hits["exceeds"] += len(e)
            hits["sigma"] += len(g)
            hits["denom"] += len(d)
            hits["maxnum"] += len(x)
            per_sample["exceeds"] += bool(e)
            per_sample["sigma"] += bool(g)
            per_sample["denom"] += bool(d)
            per_sample["maxnum"] += bool(x)
        m = meta.get(sid, {})
        out[sid] = {"user": m.get("user", ""), "period": m.get("period", ""),
                    "true_exceedance": m.get("true_exceedance", False),
                    "k": len(r["narratives"]), "hits": hits, "samples": per_sample}
    return out


def dist(records):
    return Counter(s["n_samples"] for r in records for s in r["sentences"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="out_narratives_ollama_v2/t085/narratives.jsonl")
    p.add_argument("--after", default="out_narratives_ollama_v3/t085/narratives.jsonl")
    p.add_argument("--signals", default="out_full/signals.jsonl")
    args = p.parse_args()

    v1 = [json.loads(l) for l in open(args.before, encoding="utf-8")]
    v2 = [json.loads(l) for l in open(args.after, encoding="utf-8")]
    common = {r["signal_id"] for r in v1} & {r["signal_id"] for r in v2}
    v1 = [r for r in v1 if r["signal_id"] in common]
    v2 = [r for r in v2 if r["signal_id"] in common]
    # Labels track what the files actually record, so a mislabelled path shows
    # up as (say) "v2 -> v2" rather than silently comparing a version to itself.
    LA, LB = v1[0]["prompt_version"], v2[0]["prompt_version"]
    print(f"{LA}: {args.before}\n{LB}: {args.after}")
    print(f"signals compared: {len(common)}  "
          f"({LA} file had {sum(1 for _ in open(args.before, encoding='utf-8'))}, "
          f"{LB} file had {sum(1 for _ in open(args.after, encoding='utf-8'))})")
    if LA == LB:
        print(f"WARNING: both files record prompt_version={LA} -- "
              "this is not a before/after contrast.")
    print(f"pooling: {v2[0]['pooling']}")

    meta = signal_facts(common, args.signals)
    a, b = scan(v1, meta), scan(v2, meta)

    # Signals where NO fact exceeds its own 12-week max: every exceedance claim
    # on these is necessarily false, so the count is unambiguous.
    clean = sorted(s for s in common if not meta[s]["true_exceedance"])
    dirty = sorted(s for s in common if meta[s]["true_exceedance"])

    print("\n" + "=" * 96)
    print(f"RENDERING-INDUCED ERROR CLASSES  {LA} -> {LB}  "
          "(samples affected, out of k per signal)")
    print("=" * 96)
    print(f"{'signal':<14}{'user':<9}{'period':<10}{'true?':<7}"
          f"{'exceeds-max':>14}{'numeric-sigma':>15}{'peer-as-denom':>14}"
          f"{'max-figure':>14}")
    tot = {"exceeds": [0, 0], "sigma": [0, 0], "denom": [0, 0], "maxnum": [0, 0]}
    false_only = {"exceeds": [0, 0]}
    for sid in clean + dirty:
        ra, rb = a[sid], b[sid]
        cells = []
        for key in ("exceeds", "sigma", "denom", "maxnum"):
            tot[key][0] += ra["samples"][key]
            tot[key][1] += rb["samples"][key]
            cells.append(f"{ra['samples'][key]}/{ra['k']} -> {rb['samples'][key]}/{rb['k']}")
        if not ra["true_exceedance"]:
            false_only["exceeds"][0] += ra["samples"]["exceeds"]
            false_only["exceeds"][1] += rb["samples"]["exceeds"]
        flag = "yes" if ra["true_exceedance"] else "NO"
        print(f"{sid:<14}{ra['user']:<9}{ra['period']:<10}{flag:<7}"
              f"{cells[0]:>14}{cells[1]:>15}{cells[2]:>14}{cells[3]:>14}")
    print("-" * 96)
    print(f"  {'exceeds-max, ALL signals':<38} {LA}: {tot['exceeds'][0]:>4}   "
          f"{LB}: {tot['exceeds'][1]:>4}   (mixes true and false claims)")
    print(f"  {'exceeds-max, FALSE only':<38} {LA}: {false_only['exceeds'][0]:>4}   "
          f"{LB}: {false_only['exceeds'][1]:>4}   "
          f"({len(clean)} signals where no fact exceeds its max)")
    for key, label, note in [
        ("sigma", "numeric-sigma restatements", ""),
        ("denom", "peer-size-as-denominator", ""),
        ("maxnum", "12-week-high figure quoted", "(unrendered from v3 on)"),
    ]:
        was, now = tot[key]
        print(f"  {label:<38} {LA}: {was:>4}   {LB}: {now:>4}   {note}")

    print("\n" + "=" * 78)
    print("n_samples DISTRIBUTION AT THRESHOLD 0.85")
    print("=" * 78)
    da, db = dist(v1), dist(v2)
    ta, tb = sum(da.values()), sum(db.values())
    k = v1[0]["k"]
    print(f"{'n':>4}{LA:>10}{LB:>10}")
    for n in range(1, k + 1):
        print(f"{n:>4}{da.get(n,0):>10}{db.get(n,0):>10}")
    for label, fn in [("unique sentences", lambda d, t: t),
                      ("singletons (1/k)", lambda d, t: d.get(1, 0)),
                      ("recurring (>=2/k)", lambda d, t: sum(c for n, c in d.items() if n >= 2)),
                      (f"majority (>={k//2+1}/k)", lambda d, t: sum(c for n, c in d.items() if n >= k//2+1))]:
        va, vb = fn(da, ta), fn(db, tb)
        pa = f"({va/ta:.1%})" if label != "unique sentences" else ""
        pb = f"({vb/tb:.1%})" if label != "unique sentences" else ""
        print(f"  {label:<22} {LA}: {va:>4} {pa:<9} {LB}: {vb:>4} {pb}")


if __name__ == "__main__":
    main()
