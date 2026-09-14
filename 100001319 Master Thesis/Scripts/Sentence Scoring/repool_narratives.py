"""Re-pool existing narratives semantically. No regeneration -- reads the raw
narrative text already in narratives.jsonl.

    python scripts/repool_narratives.py --narratives out_narratives_ollama/narratives.jsonl \
        --thresholds 0.80 0.85 0.90 0.95 --write-threshold 0.85

Reports the n_samples distribution at each threshold so the pooling parameter
is chosen from measured sensitivity rather than picked blind, then rewrites the
file at --write-threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sentences import (  # noqa: E402
    OllamaEmbedder, pool_sentences_semantic, split_sentences, threshold_slug,
)


def distribution(records: list[dict]) -> Counter:
    return Counter(s["n_samples"] for r in records for s in r["sentences"])


def report(label: str, records: list[dict], k: int) -> dict:
    dist = distribution(records)
    total = sum(dist.values())
    peak = max(dist.values()) if dist else 1
    print(f"\n--- {label} ---")
    print(f"    unique sentences: {total}   mean/signal: {total/len(records):.1f}")
    for n in range(1, k + 1):
        c = dist.get(n, 0)
        bar = "#" * int(48 * c / peak)
        print(f"      {n:>2}/{k} : {c:>4}  {bar}")
    singletons = dist.get(1, 0)
    recurring = sum(c for n, c in dist.items() if n >= 2)
    majority = sum(c for n, c in dist.items() if n >= (k // 2 + 1))
    print(f"    singletons (1/{k}):        {singletons:>4}  ({singletons/total:.1%})")
    print(f"    recurring  (>=2/{k}):      {recurring:>4}  ({recurring/total:.1%})")
    print(f"    majority   (>={k//2+1}/{k}):      {majority:>4}  ({majority/total:.1%})")
    return {"unique": total, "singletons": singletons, "recurring": recurring,
            "majority": majority, "dist": dict(dist)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--narratives", default="out_narratives_ollama/narratives.jsonl")
    p.add_argument("--embed-model", default="nomic-embed-text:latest")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.80, 0.85, 0.90, 0.95])
    # 0.85 is the FROZEN threshold (LIMITATIONS.md finding 9), applied by
    # default here so the working artifact for Stage 3 is produced without
    # anyone having to remember the number:
    #
    #   0.80 merges evidence and context claims -- in the ACM2278 2010-W33
    #        leak-site cluster, a CONTEXT statement about job-search visits
    #        inherits n_samples=10/10, undoing at the measurement layer the
    #        evidence/context separation Stage 1 built (finding 8).
    #   0.90 fragments the dominant claim across 3/10, 3/10, 2/10 clusters with
    #        nothing reaching majority, losing the consistency gradient.
    #
    # Only the CLI defaults it; pool_sentences_semantic still requires it.
    p.add_argument("--write-thresholds", nargs="*", type=float, default=[0.85],
                   help="write pooled output for these thresholds, each into "
                       "its own threshold-scoped directory (t080/, t085/, ...). "
                       "Frozen default: 0.85. Pass with no values to write none.")
    args = p.parse_args()

    path = Path(args.narratives)
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    k = records[0]["k"]
    print(f"{len(records)} signals, k={k}, embedder={args.embed_model}")

    # Re-split from the raw narrative text -- the authoritative source.
    per_signal = [[split_sentences(n["text"]) for n in r["narratives"]] for r in records]
    n_raw = sum(len(s) for sig in per_signal for s in sig)
    print(f"raw sentences across all samples: {n_raw}")

    print("\n" + "=" * 60)
    print("BASELINE: exact-string pooling (what shipped)")
    print("=" * 60)
    report("exact match", records, k)

    embedder = OllamaEmbedder(model=args.embed_model, host=args.host)
    summaries = {}
    pooled_by_threshold = {}
    print("\n" + "=" * 60)
    print("SEMANTIC POOLING -- threshold sweep")
    print("=" * 60)
    for t in args.thresholds:
        pooled = []
        for record, samples in zip(records, per_signal):
            sentences, ids = pool_sentences_semantic(samples, embedder, threshold=t)
            new = dict(record)
            new["sentences"] = sentences
            new["narratives"] = [
                {**n, "sentence_ids": ids[i]} for i, n in enumerate(record["narratives"])
            ]
            # Same key shape as GenerationConfig.pooling_metadata(), so an
            # artifact reads identically whether it was pooled at generation
            # time or re-pooled here.
            new["pooling"] = {"method": "semantic", "threshold": t,
                              "embed_model": args.embed_model}
            pooled.append(new)
        pooled_by_threshold[t] = pooled
        summaries[t] = report(f"cosine >= {t}", pooled, k)

    print("\n" + "=" * 60)
    print("SENSITIVITY")
    print("=" * 60)
    print(f"{'threshold':>10s} {'unique':>8s} {'singleton%':>11s} {'recurring%':>11s} {'majority%':>10s}")
    base = report.__self__ if False else None  # noqa
    ex = distribution(records); ex_tot = sum(ex.values())
    print(f"{'exact':>10s} {ex_tot:>8} {ex.get(1,0)/ex_tot:>10.1%} "
          f"{sum(c for n,c in ex.items() if n>=2)/ex_tot:>10.1%} "
          f"{sum(c for n,c in ex.items() if n>=k//2+1)/ex_tot:>9.1%}")
    for t in args.thresholds:
        s = summaries[t]
        print(f"{t:>10.2f} {s['unique']:>8} {s['singletons']/s['unique']:>10.1%} "
              f"{s['recurring']/s['unique']:>10.1%} {s['majority']/s['unique']:>9.1%}")

    if args.write_thresholds:
        # Threshold-scoped directories, never an in-place overwrite: results at
        # two thresholds must be able to coexist and stay attributable.
        for t in args.write_thresholds:
            out = pooled_by_threshold.get(t)
            if out is None:
                print(f"\n--write-thresholds {t} not in --thresholds; skipped.")
                continue
            dest_dir = path.parent / threshold_slug(t)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / path.name
            with open(dest, "w", encoding="utf-8") as fh:
                for r in out:
                    fh.write(json.dumps(r) + "\n")
            meta = {
                "source": str(path),
                "pooling": out[0]["pooling"],
                "model": out[0]["model"],
                "prompt_version": out[0]["prompt_version"],
                "temperature": out[0]["temperature"],
                "k": out[0]["k"],
                "signals": len(out),
                "unique_sentences": summaries[t]["unique"],
            }
            (dest_dir / "pooling_summary.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")
            print(f"wrote {dest}  ({summaries[t]['unique']} sentences)")


if __name__ == "__main__":
    main()
