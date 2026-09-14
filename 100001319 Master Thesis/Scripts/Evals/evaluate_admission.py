"""Stage 3, Step 3: evaluate src/admission.py's score_sentence against the
frozen human ground truth (out_narratives_ollama_v3/t085/labelling_final.csv).

    python scripts/evaluate_admission.py

Reads-only: labelling_final.csv, narratives.jsonl (raw k drafts + pooled
sentences), out_full/signals.jsonl (facts[] for fact-matching). Writes
out_narratives_ollama_v3/t085/admission_eval.json (machine-readable, every
per-row score kept) and prints the same report to stdout. Nothing in src/ is
touched and nothing is tuned based on what comes out -- docs/stage_3.md: "This
doesn't change the code, only the reported numbers."

NLI BACKEND: real entailment via src.scoring.CrossEncoderNLIClient
(cross-encoder/nli-deberta-v3-small, sentence-transformers, CPU, no API key),
replacing the token-overlap StubNLIClient used for the first pass of this
evaluation. --stub reverts to StubNLIClient (offline, no model download) for
comparison or when sentence-transformers is unavailable; the report and the
written JSON both record which backend produced the numbers so the two runs
are never confused for each other.

Entailment calls are memoized on the exact (premise, hypothesis) string pair
within one run: the same k raw drafts of a signal are reused as the
comparison set for every pooled sentence belonging to that signal, so caching
avoids re-scoring identical pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.admission import score_sentence  # noqa: E402
from src.scoring import StubNLIClient  # noqa: E402


class _MemoizingNLI:
    """Wraps any NLIClient with a (premise, hypothesis) -> bool cache."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, str], bool] = {}
        self.calls = 0
        self.cache_hits = 0

    def entails(self, premise: str, hypothesis: str) -> bool:
        self.calls += 1
        key = (premise, hypothesis)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        result = self._inner.entails(premise, hypothesis)
        self._cache[key] = result
        return result

LABELLING_CSV = Path("out_narratives_ollama_v3/t085/labelling_final.csv")
NARRATIVES_JSONL = Path("out_narratives_ollama_v3/t085/narratives.jsonl")
SIGNALS_JSONL = Path("out_full/signals.jsonl")
OUT_JSON = Path("out_narratives_ollama_v3/t085/admission_eval.json")

THRESHOLDS = (0.3, 0.5, 0.7)

# The three hallucination shapes docs/annotation_guideline.md names, keyed by
# the cluster_id whose `notes` column identifies it -- see the worked
# examples section. Hand-picked from labelling_final.csv, not pattern-matched
# from the notes text, so a future notes-wording edit cannot silently drop a
# row from this evaluation.
HALLUCINATION_SHAPES = {
    "30e15748717a_s001": "own_current_value_swap",
    "b1e9fc4b7ae7_s002": "category_conflation",
    "ea3d0678f325_s006": "fact_mixing",
}

KNOWN_KEYWORD_OVERLAP_MISS = "107dcedee69e_s002"


def _load_narratives(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["signal_id"]] = rec
    return out


def _load_signals(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            sig = json.loads(line)
            out[sig["signal_id"]] = sig
    return out


def _load_labelling(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def score_all_rows(nli, backend_name: str, *, labelling_csv: Path,
                    narratives_jsonl: Path, signals_jsonl: Path) -> list[dict]:
    narratives_by_signal = _load_narratives(narratives_jsonl)
    signals_by_id = _load_signals(signals_jsonl)
    rows = _load_labelling(labelling_csv)

    results = []
    for row in rows:
        cluster_id = row["cluster_id"]
        signal_id, sentence_id = cluster_id.split("_", 1)
        rec = narratives_by_signal[signal_id]
        signal = signals_by_id[signal_id]

        sentence = next(s for s in rec["sentences"] if s["sentence_id"] == sentence_id)
        text = sentence["text"]
        assert text == row["representative_text"], (
            f"{cluster_id}: labelling_final.csv representative_text does not "
            "match narratives.jsonl -- ground truth and scoring input have "
            "drifted apart"
        )
        draft_texts = [n["text"] for n in rec["narratives"]]

        # exclude_index=None: a pooled cluster's appears_in can span several
        # of the k samples, so there is no single "the draft this came from"
        # to exclude -- src.scoring.self_consistency's own documented case
        # for a pooled-cluster representative.
        result = score_sentence(
            text, draft_texts, signal, rec["evidence_fact_ids"], nli,
            exclude_index=None,
        )

        results.append({
            "cluster_id": cluster_id,
            "signal_id": signal_id,
            "human_label": row["label"],
            "hallucination_shape": HALLUCINATION_SHAPES.get(cluster_id),
            "rho": result.rho,
            "self_consistency": result.self_consistency,
            "fact_match_score": result.fact_match_score,
            "misweighted": result.misweighted.misweighted,
            "n_numbers": len(result.fact_match.numbers),
            "n_entities": len(result.fact_match.entities),
            "self_consistency_backend": backend_name,
        })
    return results


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def threshold_sweep(results: list[dict], score_key: str) -> dict:
    """Precision/recall of `score_key >= threshold` vs. human label=='grounded',
    for each threshold in THRESHOLDS. NaN scores never clear a positive
    threshold (a sentence with nothing to check is not asserted "grounded"
    by this admission function), so they count toward the negative side --
    see rows_with_nan_score below for how many that affects, per score_key.
    """
    out = {}
    n_nan = sum(1 for r in results if _is_nan(r[score_key]))
    for t in THRESHOLDS:
        tp = fp = fn = tn = 0
        for r in results:
            predicted_grounded = (not _is_nan(r[score_key])) and r[score_key] >= t
            actually_grounded = r["human_label"] == "grounded"
            if predicted_grounded and actually_grounded:
                tp += 1
            elif predicted_grounded and not actually_grounded:
                fp += 1
            elif not predicted_grounded and actually_grounded:
                fn += 1
            else:
                tn += 1
        precision, recall = _prf(tp, fp, fn)
        out[str(t)] = {
            "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    out["rows_with_nan_score"] = n_nan
    return out


def _is_nan(x: float) -> bool:
    return x != x


def hallucination_shape_report(results: list[dict]) -> dict:
    out = {}
    for r in results:
        shape = r["hallucination_shape"]
        if shape is None:
            continue
        out[shape] = {
            "cluster_id": r["cluster_id"],
            "human_label": r["human_label"],
            "rho": r["rho"],
            "self_consistency": r["self_consistency"],
            "fact_match_score": r["fact_match_score"],
            "misweighted": r["misweighted"],
        }
    return out


def known_miss_report(results: list[dict]) -> dict | None:
    # Only present in batch 1 -- a batch2-only run (or any other subset that
    # doesn't include this cluster_id) has nothing to report here.
    row = next((r for r in results if r["cluster_id"] == KNOWN_KEYWORD_OVERLAP_MISS), None)
    if row is None:
        return None
    return {
        "cluster_id": row["cluster_id"],
        "human_label": row["human_label"],
        "rho": row["rho"],
        "self_consistency": row["self_consistency"],
        "fact_match_score": row["fact_match_score"],
        "misweighted": row["misweighted"],
    }


def ablation_report(results: list[dict]) -> dict:
    return {
        "self_consistency_alone": threshold_sweep(results, "self_consistency"),
        "fact_match_alone": threshold_sweep(results, "fact_match_score"),
        "combined_rho": threshold_sweep(results, "rho"),
    }


def _fmt_pr(entry: dict) -> str:
    p = "nan" if entry["precision"] != entry["precision"] else f"{entry['precision']:.3f}"
    r = "nan" if entry["recall"] != entry["recall"] else f"{entry['recall']:.3f}"
    return (f"P={p} R={r}  (tp={entry['tp']} fp={entry['fp']} "
            f"fn={entry['fn']} tn={entry['tn']})")


def print_report(results: list[dict], ablation: dict, shapes: dict, known_miss: dict) -> None:
    print(f"rows scored: {len(results)}")
    print()

    print("=== 1. rho(s) >= threshold vs. human 'grounded' label ===")
    sweep = ablation["combined_rho"]
    for t in THRESHOLDS:
        print(f"  threshold={t}: {_fmt_pr(sweep[str(t)])}")
    print(f"  rows with NaN rho (no numeric/entity content and self_consistency "
          f"undefined -- see JSON for exact count): {sweep['rows_with_nan_score']}")
    print()

    print("=== 2. Hallucination-shape breakdown (reported separately, not pooled) ===")
    for shape, info in shapes.items():
        print(f"  {shape} ({info['cluster_id']}): rho={info['rho']:.4f}  "
              f"self_consistency={info['self_consistency']:.4f}  "
              f"fact_match_score={_fmt_score(info['fact_match_score'])}  "
              f"misweighted={info['misweighted']}  human_label={info['human_label']}")
    print()

    print("=== 3. Known keyword-overlap miss (107dcedee69e_s002) ===")
    if known_miss is None:
        print("  (not in this results set)")
    else:
        print(f"  rho={known_miss['rho']:.4f}  self_consistency={known_miss['self_consistency']:.4f}  "
              f"fact_match_score={_fmt_score(known_miss['fact_match_score'])}  "
              f"misweighted={known_miss['misweighted']}  human_label={known_miss['human_label']}")
    print()

    print("=== 4. Ablation: self-consistency alone vs. fact-match alone vs. combined ===")
    for name, sweep in ablation.items():
        print(f"  {name}:")
        for t in THRESHOLDS:
            print(f"    threshold={t}: {_fmt_pr(sweep[str(t)])}")
        print(f"    rows with NaN score: {sweep['rows_with_nan_score']}")


def _fmt_score(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stub", action="store_true",
                   help="use StubNLIClient's token-overlap fallback instead of "
                       "the real cross-encoder (no model download; matches "
                       "Step 3's original run)")
    p.add_argument("--gemini", action="store_true",
                   help="use GeminiNLIClient instead of the cross-encoder")
    p.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small",
                   help="sentence-transformers CrossEncoder model id, ignored with --stub or --gemini")
    p.add_argument("--gemini-model", default="gemini-2.5-flash",
                   help="Gemini model for GeminiNLIClient, used if --gemini is set")
    p.add_argument("--labelling-csv", default=str(LABELLING_CSV),
                   help="ground-truth CSV (must have cluster_id, label columns)")
    p.add_argument("--narratives-jsonl", default=str(NARRATIVES_JSONL),
                   help="source of raw k drafts + pooled sentences for the "
                       "signals labelling-csv's rows come from -- must be the "
                       "SAME batch (batch1 rows need batch1's narratives.jsonl, "
                       "batch2 rows need batch2's; a combined labelling CSV "
                       "spanning batches must be scored per-batch and merged, "
                       "not pointed at one narratives.jsonl for all rows)")
    p.add_argument("--signals-jsonl", default=str(SIGNALS_JSONL))
    p.add_argument("--out", default=str(OUT_JSON))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.stub:
        nli = _MemoizingNLI(StubNLIClient())
        backend_name = "StubNLIClient.token_overlap_fallback"
    elif args.gemini:
        from src.scoring import GeminiNLIClient
        print(f"loading Gemini NLI model: {args.gemini_model} ...", file=sys.stderr)
        nli = _MemoizingNLI(GeminiNLIClient(args.gemini_model))
        backend_name = f"GeminiNLIClient({args.gemini_model})"
    else:
        from src.scoring import CrossEncoderNLIClient
        print(f"loading NLI model: {args.nli_model} ...", file=sys.stderr)
        nli = _MemoizingNLI(CrossEncoderNLIClient(args.nli_model))
        backend_name = f"CrossEncoderNLIClient({args.nli_model})"

    results = score_all_rows(
        nli, backend_name,
        labelling_csv=Path(args.labelling_csv),
        narratives_jsonl=Path(args.narratives_jsonl),
        signals_jsonl=Path(args.signals_jsonl),
    )
    ablation = ablation_report(results)
    shapes = hallucination_shape_report(results)
    known_miss = known_miss_report(results)

    print_report(results, ablation, shapes, known_miss)
    print(f"\nnli calls: {nli.calls}  cache hits: {nli.cache_hits}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_rows": len(results),
        "self_consistency_backend": backend_name,
        "threshold_sweep_grounded_vs_rho": ablation["combined_rho"],
        "ablation": ablation,
        "hallucination_shapes": shapes,
        "known_keyword_overlap_miss": known_miss,
        "rows": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
