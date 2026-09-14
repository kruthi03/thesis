"""One-off diagnostic (not a pipeline deliverable): re-run of the same 10
examples after restructuring src.scoring.self_consistency to compare the
candidate sentence against each individual SENTENCE of the other draft
(src.sentences.split_sentences) rather than the whole draft paragraph as one
NLI premise.

The first run of this script (whole-paragraph premises) found
cross-encoder/nli-deberta-v3-small scoring a VERBATIM substring match as
neutral=0.999 -- a premise/hypothesis length mismatch relative to what the
model was trained on (SNLI/MNLI: short, roughly single-sentence pairs), not
genuine disagreement between drafts. This run shows the SAME 10 sentence/pair
selections, but now probes the model with sentence-level premises to see
whether that was in fact the fix, or whether the neutral-dominant,
near-zero-contradiction pattern persists even at the right premise length
(which would point to model capacity, not premise length, as the problem).

Still no model swap, no aggregation change to rho -- just the raw pairs and
scores, now at sentence granularity.

    python scripts/diagnose_self_consistency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder  # noqa: E402

from src.scoring import StubNLIClient  # noqa: E402
from src.sentences import split_sentences  # noqa: E402

REAL_JSON = Path("out_narratives_ollama_v3/t085/admission_eval.json")
STUB_JSON = Path("out_narratives_ollama_v3/t085/admission_eval_stub.json")
NARRATIVES_JSONL = Path("out_narratives_ollama_v3/t085/narratives.jsonl")

N_EXAMPLES = 10
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def _load_narratives() -> dict[str, dict]:
    out = {}
    with open(NARRATIVES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["signal_id"]] = rec
    return out


def main() -> None:
    real = json.loads(REAL_JSON.read_text(encoding="utf-8"))
    stub = json.loads(STUB_JSON.read_text(encoding="utf-8"))
    real_by_id = {r["cluster_id"]: r for r in real["rows"]}
    stub_by_id = {r["cluster_id"]: r for r in stub["rows"]}
    narratives_by_signal = _load_narratives()

    candidates = [
        cid for cid, sr in stub_by_id.items()
        if sr["human_label"] == "grounded"
        and sr["self_consistency"] >= 0.5
        and real_by_id[cid]["self_consistency"] < 0.3
    ]
    chosen = candidates[:N_EXAMPLES]
    print(f"{len(candidates)} candidate rows total; showing first {len(chosen)}\n")

    print(f"loading {MODEL_NAME} for raw probability inspection ...", file=sys.stderr)
    model = CrossEncoder(MODEL_NAME)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    id2label_inv = {v: k for k, v in id2label.items()}
    print(f"id2label: {id2label}\n")

    stub_client = StubNLIClient()

    for cid in chosen:
        signal_id, sentence_id = cid.split("_", 1)
        rec = narratives_by_signal[signal_id]
        sentence = next(s for s in rec["sentences"] if s["sentence_id"] == sentence_id)
        text = sentence["text"]
        drafts = [n["text"] for n in rec["narratives"]]

        # Two drafts the STUB judged as entailing (token-overlap >= 0.6).
        agreeing = [d for d in drafts if stub_client.entails(d, text)][:2]

        sr, rr = stub_by_id[cid], real_by_id[cid]
        print("=" * 100)
        print(f"{cid}  human_label=grounded  "
              f"stub_self_consistency={sr['self_consistency']:.2f}  "
              f"real_self_consistency={rr['self_consistency']:.2f}")
        print(f"SENTENCE: {text}")
        print()

        if not agreeing:
            print("  (stub found no agreeing draft via token overlap >= 0.6 -- skipping pairs)")
            continue

        for i, draft in enumerate(agreeing, 1):
            draft_sentences = split_sentences(draft) or [draft]
            pair_probs = model.predict(
                [(s, text) for s in draft_sentences], apply_softmax=True
            )
            # The draft-level signal self_consistency now uses: the ONE
            # sentence within this draft scoring highest on 'entailment',
            # i.e. the max, not the whole-paragraph premise.
            entail_idx = id2label_inv["entailment"]
            best_j = max(range(len(draft_sentences)), key=lambda j: pair_probs[j][entail_idx])
            best_sentence = draft_sentences[best_j]
            probs = pair_probs[best_j]
            ranked = sorted(
                ((id2label[j], float(p)) for j, p in enumerate(probs)),
                key=lambda x: -x[1],
            )
            entails_now = ranked[0][0] == "entailment"
            print(f"  --- draft {i} (stub: entails=True; "
                  f"{len(draft_sentences)} sentences split out) ---")
            print(f"  best-matching sentence: {best_sentence}")
            print(f"  cross-encoder probs: "
                  + "  ".join(f"{label}={p:.3f}" for label, p in ranked)
                  + f"   -> draft-level entails={entails_now}")
            print()


if __name__ == "__main__":
    main()
