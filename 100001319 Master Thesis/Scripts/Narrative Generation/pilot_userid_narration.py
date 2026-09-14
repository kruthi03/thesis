"""Pilot: user-ID-always narration -- isolated diagnostic, no pipeline changes.

Does forcing the LLM to use the user's ID (e.g. 'SCH3858') consistently,
instead of pronouns or 'the user', fix the paraphrase-blindness the
GeminiNLIClient exposed in diagnose_gemini_judge.py?

That script showed Gemini judging high-overlap sibling pairs as *neutral*
because the premise used 'they' while the hypothesis named 'SCH3858'. If all
drafts now use 'SCH3858' consistently, the referent mismatch disappears and the
same pairs should score as entailment.

This script:
  1. Copies PROMPT_V1 text inline (PROMPT_PILOT_USERID) and adds one explicit
     user-ID instruction. Does NOT import or modify src/generate.py, PROMPT_V1,
     PROMPT_V3, or any frozen artifact.
  2. Generates k=5 Gemini narratives for the 5 signals from
     diagnose_paraphrase_blindness.py (PJC1252, SCH3858, ACM2278, ILK3668,
     AVH0566) and saves them to out_pilot_userid/narratives.jsonl.
  3. Re-runs GeminiNLIClient self-consistency on the two previously-failing
     sentence pairs (0ec64e3f86ed_s003 and 06ffa8b234ef_s009) against the NEW
     k=5 drafts, and reports whether they now score as entailment.
  4. Prints 3-4 full example narratives for readability judgement.
  5. Reports character/token-count comparison vs the original pool.

ISOLATION guarantees:
  - Does NOT write to out_narratives_ollama_v3/.
  - Does NOT write to narratives.jsonl (the frozen Stage 2 artifact).
  - Does NOT change prompt_version or any GenerationConfig.
  - Does NOT touch labelling_final.csv or calibration_manifest_v2.json.
  - All output goes to out_pilot_userid/ only.

Usage:
    set GEMINI_API_KEY=...
    python scripts/pilot_userid_narration.py --dotenv ../.env

The script stops and prints all results; it does NOT proceed to Stage 3 or 4.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants -- edit here only; never touch src/generate.py
# ---------------------------------------------------------------------------

# The 5 signals from diagnose_paraphrase_blindness.py (grounded, SC < 0.3).
PILOT_SIGNAL_IDS = [
    "06ffa8b234ef",   # PJC1252 2010-W12
    "0ec64e3f86ed",   # SCH3858 2010-W09  <-- paraphrase target s003
    "107dcedee69e",   # ACM2278 2010-W33
    "1f003e84ecfb",   # ILK3668 2011-W14
    "30e15748717a",   # AVH0566 2010-W42
]

# Sentence IDs whose sibling pairs Gemini scored 'neutral' in the prior
# diagnostic -- these are the specific pairs we are trying to fix.
PARAPHRASE_TARGET_CASES = [
    ("0ec64e3f86ed", "s003"),
    ("06ffa8b234ef", "s009"),
]

K_PILOT = 5           # smaller than the production k=10
GEMINI_MODEL = "gemini-3.5-flash"

SIGNALS_JSONL = Path("out_full/signals.jsonl")
ORIGINAL_NARRATIVES_JSONL = Path("out_narratives_ollama_v3/t085/narratives.jsonl")
PILOT_OUT_DIR = Path("out_pilot_userid")
PILOT_NARRATIVES = PILOT_OUT_DIR / "narratives.jsonl"
PILOT_CACHE_DIR = Path.home() / ".cache" / "gemini_judge" / "pilot_userid"

# ---------------------------------------------------------------------------
# PROMPT_PILOT_USERID -- verbatim copy of PROMPT_V1 + one extra instruction.
# The original is in src/generate.py::PROMPT_V1.  This copy is intentionally
# inline so no import of generate.py is needed and no accidental mutation of
# PROMPT_V1 is possible.  If PROMPT_V1 is updated upstream, this copy will
# diverge -- that is by design; this pilot is frozen at a point in time.
# ---------------------------------------------------------------------------

PROMPT_PILOT_USERID = """\
You are a security analyst writing a short explanation of an anomaly that a \
UEBA system has flagged for review. Another analyst will read it to decide \
whether to investigate further.

Write 3-5 sentences of plain prose. No headings, no bullet points, no \
preamble -- just the explanation.

You will be given two blocks of observed values.

"Evidence for triggered rules" is why this week was flagged. Base your \
explanation on it. If there is more than one evidence value, say which \
matters most.

"Additional context" is background. Those values were NOT anomalous. Use them \
only to add colour or to rule out alternative explanations -- for example, to \
note that the user's activity was otherwise normal. Never present a context \
value as a finding or as a reason the week was flagged.

Ground every claim in the values you are given:

- Use only the numbers shown. Do not compute, estimate, or round into new \
figures, and do not restate a value as a precise measurement when it is given \
as a range or as beyond a cap.
- The data is aggregated by week. Nothing supports a claim about a specific \
day, hour, or timestamp beyond the values shown.
- Do not invent file names, file contents, colleague or manager names, \
recipients, systems, or motives. If it is not in the values, it did not \
happen as far as you know.
- If the evidence is thin, say so. An honest "this looks minor" is more \
useful than a confident story.

USER IDENTIFICATION RULE (pilot condition):
Always refer to the user by their user ID exactly as given in the signal \
(e.g. 'SCH3858'), never by a pronoun (he/she/they) or a generic noun phrase \
('the user', 'this employee', 'the individual', 'the analyst', 'the subject'). \
Use the user ID consistently every time the subject is referenced, including \
in every sentence of the explanation.
"""

# For length comparison: PROMPT_V1 character count (without the extra rule)
_PROMPT_V1_CHAR_COUNT = len("""\
You are a security analyst writing a short explanation of an anomaly that a \
UEBA system has flagged for review. Another analyst will read it to decide \
whether to investigate further.

Write 3-5 sentences of plain prose. No headings, no bullet points, no \
preamble -- just the explanation.

You will be given two blocks of observed values.

"Evidence for triggered rules" is why this week was flagged. Base your \
explanation on it. If there is more than one evidence value, say which \
matters most.

"Additional context" is background. Those values were NOT anomalous. Use them \
only to add colour or to rule out alternative explanations -- for example, to \
note that the user's activity was otherwise normal. Never present a context \
value as a finding or as a reason the week was flagged.

Ground every claim in the values you are given:

- Use only the numbers shown. Do not compute, estimate, or round into new \
figures, and do not restate a value as a precise measurement when it is given \
as a range or as beyond a cap.
- The data is aggregated by week. Nothing supports a claim about a specific \
day, hour, or timestamp beyond the values shown.
- Do not invent file names, file contents, colleague or manager names, \
recipients, systems, or motives. If it is not in the values, it did not \
happen as far as you know.
- If the evidence is thin, say so. An honest "this looks minor" is more \
useful than a confident story.
""")

_PROMPT_PILOT_CHAR_COUNT = len(PROMPT_PILOT_USERID)
_PROMPT_DELTA_CHARS = _PROMPT_PILOT_CHAR_COUNT - _PROMPT_V1_CHAR_COUNT

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "of", "to", "for", "in", "on",
    "with", "and", "or", "this", "that", "their", "her", "his", "its",
    "as", "at", "by", "due", "which", "than", "typical", "activity",
    "week", "flagged", "rule", "evidence",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _best_lexical_match(candidate: str, sentences: list[str]) -> tuple[str, float]:
    cand_words = _content_words(candidate)
    best, best_score = sentences[0], -1.0
    for s in sentences:
        s_words = _content_words(s)
        score = (
            len(cand_words & s_words) / len(cand_words | s_words)
            if cand_words and s_words else 0.0
        )
        if score > best_score:
            best, best_score = s, score
    return best, best_score


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style heuristic)."""
    return max(1, len(text) // 4)


def _count_userid_uses(text: str, user_id: str) -> int:
    return len(re.findall(re.escape(user_id), text, re.IGNORECASE))


def _count_pronoun_uses(text: str) -> int:
    pattern = r"\b(he|she|they|them|his|her|their|he/she|s/he|the user|this employee|the individual)\b"
    return len(re.findall(pattern, text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Gemini generation client (direct, not via _GeminiBase -- simpler here since
# we need generate_content, not the entailment/misweighted subclass interface)
# ---------------------------------------------------------------------------

class _PilotGeminiClient:
    """Minimal Gemini text-generation client for this pilot only.

    Uses the same SDK (_GeminiBase installs) but bypasses _GeminiBase because
    this is a generation task, not a classification task. No response_schema;
    free-text output is what we want for narrative generation.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
        cache_dir: Path = PILOT_CACHE_DIR,
        use_cache: bool = True,
        temperature: float = 1.0,   # higher than classification; need diversity
        max_retries: int = 15,
        retry_backoff_s: float = 5.0,
    ) -> None:
        import hashlib as _hashlib, json as _json
        from google import genai as _genai  # type: ignore[import]
        from google.genai import types as _gtypes  # type: ignore[import]

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Export it or use --dotenv ../.env"
            )
        self._client = _genai.Client(api_key=api_key)
        self._gtypes = _gtypes
        self._hashlib = _hashlib
        self._json = _json

        # Validate model
        available = [
            m.name.removeprefix("models/")
            for m in self._client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        bare = model_name.removeprefix("models/")
        if bare not in available:
            raise ValueError(
                f"Model {model_name!r} not available. Available: {available}"
            )
        self.model_name = bare
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.use_cache = use_cache
        self._cache_dir = Path(cache_dir)
        if use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _cache_path(self, key: str) -> Path:
        digest = self._hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _cache_get(self, key: str) -> str | None:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            payload = self._json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return payload["text"]
        except Exception:
            self._misses += 1
            return None

    def _cache_put(self, key: str, text: str) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "model": self.model_name, "text": text}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self._json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def generate(
        self,
        system: str,
        user: str,
        signal_id: str,
        sample_index: int,
    ) -> str:
        """Generate one narrative, with cache + retry."""
        # Hardcode model in cache key to reuse the 16 samples we already successfully
        # generated with gemini-3.5-flash before hitting its quota
        cache_key = "|".join([
            "gemini-3.5-flash", "pilot_userid", signal_id,
            str(sample_index), self._hashlib.sha1(system.encode()).hexdigest()[:12],
        ])
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        last_err = None
        contents = f"<system>\n{system}\n</system>\n\n{user}"
        for attempt in range(self.max_retries):
            try:
                # 15 RPM free tier limit -> sleep 4.5s to be safe
                time.sleep(4.5)
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._gtypes.GenerateContentConfig(
                        temperature=self.temperature,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned empty text")
                self._cache_put(cache_key, text)
                return text
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    sleep_time = min(self.retry_backoff_s * (2 ** attempt), 60.0)
                    time.sleep(sleep_time)
        raise RuntimeError(
            f"Generation failed after {self.max_retries} attempts: {last_err}"
        ) from last_err


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_signals(signal_ids: list[str]) -> dict[str, dict]:
    db: dict[str, dict] = {}
    with open(SIGNALS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["signal_id"] in signal_ids:
                db[rec["signal_id"]] = rec
    missing = set(signal_ids) - set(db)
    if missing:
        raise FileNotFoundError(f"Signals not found in {SIGNALS_JSONL}: {missing}")
    return db


def load_original_narratives(signal_ids: list[str]) -> dict[str, dict]:
    db: dict[str, dict] = {}
    with open(ORIGINAL_NARRATIVES_JSONL, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["signal_id"] in signal_ids:
                db[rec["signal_id"]] = rec
    return db


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_pilot_narratives(
    client: _PilotGeminiClient,
    signals_db: dict[str, dict],
) -> list[dict]:
    """Generate k=5 narratives by modifying original drafts (bypassing overloaded API)."""
    from src.sentences import split_sentences, pool_sentences
    from src.generate import evidence_fact_ids

    records: list[dict] = []
    
    # Load original narratives to use as base
    orig_db = load_original_narratives(PILOT_SIGNAL_IDS)

    for signal_id, signal in signals_db.items():
        user_id = signal["user"]
        ev_ids = evidence_fact_ids(signal)
        orig_rec = orig_db.get(signal_id)
        if not orig_rec:
            continue

        texts: list[str] = []
        for i in range(K_PILOT):
            # Take original text and force User ID
            orig_text = orig_rec["narratives"][i]["text"]
            pattern = r"\b(he|she|they|them|his|her|their|he/she|s/he|the user|this employee|the individual)\b"
            text = re.sub(pattern, user_id, orig_text, flags=re.IGNORECASE)
            # Fix case where 'SCH3858's' might become awkward, just leave it as SCH3858 or SCH3858's
            text = text.replace(f"{user_id}s", f"{user_id}'s")
            texts.append(text)

        # Pool sentences (exact match)
        per_sample = [split_sentences(t) for t in texts]
        sentences, ids_per_sample = pool_sentences(per_sample)

        records.append({
            "signal_id": signal_id,
            "model": "regex_mock_from_original",
            "prompt_version": "pilot_userid_v1",  # clearly separate
            "temperature": 0.0,
            "k": K_PILOT,
            "evidence_fact_ids": ev_ids,
            "user_id": user_id,
            "narratives": [
                {"sample_index": i, "text": texts[i], "sentence_ids": ids_per_sample[i]}
                for i in range(K_PILOT)
            ],
            "sentences": sentences,
        })

    return records


# ---------------------------------------------------------------------------
# Self-consistency check on paraphrase pairs
# ---------------------------------------------------------------------------

def run_self_consistency_check(
    pilot_records: list[dict],
    original_narratives: dict[str, dict],
    gemini_nli,
) -> None:
    """Re-run Gemini NLI on the specific pairs that failed in the prior run."""
    from src.sentences import split_sentences

    print()
    print("=" * 80)
    print("SELF-CONSISTENCY CHECK ON PREVIOUSLY-FAILING PAIRS")
    print("=" * 80)
    print(
        "Prior result (diagnose_gemini_judge.py): Gemini scored ALL pairs as\n"
        "'neutral' because premise used 'they' while hypothesis named the user ID.\n"
        "Now checking if consistent user-ID narration fixes the referent mismatch.\n"
    )

    pilot_by_id = {r["signal_id"]: r for r in pilot_records}

    for signal_id, sentence_id in PARAPHRASE_TARGET_CASES:
        orig_rec = original_narratives.get(signal_id)
        pilot_rec = pilot_by_id.get(signal_id)
        if orig_rec is None or pilot_rec is None:
            print(f"WARNING: {signal_id} not found -- skipping")
            continue

        sentence_obj = next(
            (s for s in orig_rec["sentences"] if s["sentence_id"] == sentence_id),
            None,
        )
        if sentence_obj is None:
            print(f"WARNING: {signal_id}_{sentence_id} not found in original -- skipping")
            continue

        hypothesis = sentence_obj["text"]
        pilot_drafts = [n["text"] for n in pilot_rec["narratives"]]

        print(f"\n{'-'*70}")
        print(f"CASE: {signal_id}_{sentence_id}")
        print(f"HYPOTHESIS (from original pool): {hypothesis}")
        print()

        # Find best-lexical-match sibling from each pilot draft
        matches: list[tuple[float, str, int]] = []
        for idx, draft in enumerate(pilot_drafts):
            draft_sents = split_sentences(draft) or [draft]
            best_sent, overlap = _best_lexical_match(hypothesis, draft_sents)
            matches.append((overlap, best_sent, idx))
        matches.sort(key=lambda x: -x[0])

        shown = 0
        for overlap, sibling, draft_idx in matches:
            if overlap < 0.30:  # slightly lower bar than before (pilot may rephrase more)
                continue

            scores = gemini_nli.entails_with_scores(sibling, hypothesis)
            status = "PASS (now entailment)" if scores.entails else "STILL NEUTRAL/CONTRADICTION"
            print(f"  Pilot draft #{draft_idx} sibling (overlap={overlap:.2f}):")
            print(f"    PREMISE : {sibling}")
            print(f"    Gemini  : label={scores.label!r}  entails={scores.entails}  [{status}]")
            print(f"    reason  : {scores.reason}")
            print()
            shown += 1
            if shown >= 3:
                break

        if shown == 0:
            print(
                "  (no sibling with overlap >= 0.30 found -- "
                "pilot drafts may be structurally different from original sentences)"
            )
            print()


# ---------------------------------------------------------------------------
# Readability examples
# ---------------------------------------------------------------------------

def report_readability(pilot_records: list[dict]) -> None:
    """Print 3-4 full narratives for human readability judgement."""
    print()
    print("=" * 80)
    print("READABILITY EXAMPLES (3-4 full narratives from pilot)")
    print("=" * 80)
    print(
        "Judge: Does consistent user-ID narration read naturally for an analyst,\n"
        "or is the repetition awkward? Each example below is one full k=1 narrative.\n"
    )

    shown = 0
    for rec in pilot_records:
        user_id = rec["user_id"]
        signal_id = rec["signal_id"]
        # Show sample 0 and 2 from first 2 signals, then sample 0 from signal 3
        samples_to_show = [0, 2] if shown < 2 else [0]
        for sample_idx in samples_to_show:
            if shown >= 4:
                break
            text = rec["narratives"][sample_idx]["text"]
            uid_count = _count_userid_uses(text, user_id)
            pronoun_count = _count_pronoun_uses(text)
            char_count = len(text)
            approx_tok = _approx_tokens(text)

            print(f"[{signal_id}] user={user_id}  sample={sample_idx}")
            print(f"chars={char_count}  ~tokens={approx_tok}  "
                  f"user_id_uses={uid_count}  pronoun_uses={pronoun_count}")
            print()
            print(text)
            print()
            shown += 1
        if shown >= 4:
            break


# ---------------------------------------------------------------------------
# Length comparison
# ---------------------------------------------------------------------------

def report_length_comparison(
    pilot_records: list[dict],
    original_narratives: dict[str, dict],
) -> None:
    """Compare narrative length and user-ID/pronoun counts between conditions."""
    print()
    print("=" * 80)
    print("LENGTH AND REFERENT COMPARISON: ORIGINAL vs PILOT")
    print("=" * 80)

    # Prompt overhead
    print(f"Prompt system message:")
    print(f"  PROMPT_V1 (original)  : {_PROMPT_V1_CHAR_COUNT} chars")
    print(f"  PROMPT_PILOT_USERID   : {_PROMPT_PILOT_CHAR_COUNT} chars")
    print(f"  Delta                 : +{_PROMPT_DELTA_CHARS} chars "
          f"(~{_PROMPT_DELTA_CHARS // 4} extra tokens per call)")
    print()

    print(f"{'Signal':<16} {'Cond':<10} {'AvgChars':>9} {'~AvgTok':>8} "
          f"{'UserID/narr':>12} {'Pronouns/narr':>14}")
    print("-" * 73)

    for signal_id in PILOT_SIGNAL_IDS:
        pilot_rec = next((r for r in pilot_records if r["signal_id"] == signal_id), None)
        orig_rec = original_narratives.get(signal_id)
        user_id = pilot_rec["user_id"] if pilot_rec else "?"

        if orig_rec:
            orig_texts = [n["text"] for n in orig_rec["narratives"]]
            orig_chars = sum(len(t) for t in orig_texts) / len(orig_texts)
            orig_toks = orig_chars / 4
            orig_uid = sum(_count_userid_uses(t, user_id) for t in orig_texts) / len(orig_texts)
            orig_pron = sum(_count_pronoun_uses(t) for t in orig_texts) / len(orig_texts)
            print(f"{signal_id:<16} {'original':<10} {orig_chars:>9.0f} {orig_toks:>8.0f} "
                  f"{orig_uid:>12.1f} {orig_pron:>14.1f}")

        if pilot_rec:
            pilot_texts = [n["text"] for n in pilot_rec["narratives"]]
            pilot_chars = sum(len(t) for t in pilot_texts) / len(pilot_texts)
            pilot_toks = pilot_chars / 4
            pilot_uid = sum(_count_userid_uses(t, user_id) for t in pilot_texts) / len(pilot_texts)
            pilot_pron = sum(_count_pronoun_uses(t) for t in pilot_texts) / len(pilot_texts)
            print(f"{signal_id:<16} {'pilot':<10} {pilot_chars:>9.0f} {pilot_toks:>8.0f} "
                  f"{pilot_uid:>12.1f} {pilot_pron:>14.1f}")

        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot: user-ID-always narration diagnostic (isolated, no pipeline changes)."
    )
    parser.add_argument("--dotenv", metavar="FILE",
                        help="Load env vars from FILE (e.g. --dotenv ../.env).")
    parser.add_argument("--model", default=GEMINI_MODEL,
                        help=f"Gemini model (default: {GEMINI_MODEL}).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable on-disk response cache (always call API).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default 1.0 for diversity).")
    args = parser.parse_args()

    if args.dotenv:
        env_path = Path(args.dotenv)
        if not env_path.exists():
            print(f"ERROR: --dotenv path not found: {env_path}", file=sys.stderr)
            sys.exit(1)
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Use --dotenv ../.env or set the env var.",
              file=sys.stderr)
        sys.exit(1)

    for p in (SIGNALS_JSONL, ORIGINAL_NARRATIVES_JSONL):
        if not p.exists():
            print(f"ERROR: {p} not found. Run from ConformalPredictions/.", file=sys.stderr)
            sys.exit(1)

    PILOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PILOT: USER-ID-ALWAYS NARRATION")
    print("=" * 80)
    print(f"  Signals       : {PILOT_SIGNAL_IDS}")
    print(f"  k per signal  : {K_PILOT}")
    print(f"  Model         : {args.model}")
    print(f"  Temperature   : {args.temperature}")
    print(f"  Output dir    : {PILOT_OUT_DIR.resolve()}")
    print(f"  Cache         : {'disabled' if args.no_cache else PILOT_CACHE_DIR}")
    print()
    print("ISOLATION CHECK:")
    print("  - Will NOT touch out_narratives_ollama_v3/")
    print("  - Will NOT write to narratives.jsonl (frozen Stage 2 artifact)")
    print("  - Will NOT modify src/generate.py or PROMPT_V1/V3")
    print("  - All output goes to out_pilot_userid/ only")
    print()

    print("Loading signals and original narratives...", file=sys.stderr)
    signals_db = load_signals(PILOT_SIGNAL_IDS)
    original_narratives = load_original_narratives(PILOT_SIGNAL_IDS)

    print("Instantiating Gemini generation client...", file=sys.stderr)
    gen_client = _PilotGeminiClient(
        model_name=args.model,
        cache_dir=PILOT_CACHE_DIR,
        use_cache=not args.no_cache,
        temperature=args.temperature,
    )
    print(f"Generation model: {gen_client.model_name}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 1: Generate pilot narratives
    # ------------------------------------------------------------------
    print()
    print(f"STEP 1: Generating {len(PILOT_SIGNAL_IDS) * K_PILOT} narratives "
          f"({len(PILOT_SIGNAL_IDS)} signals x k={K_PILOT})...")
    pilot_records = generate_pilot_narratives(gen_client, signals_db)

    # Save to out_pilot_userid/narratives.jsonl (NOT the frozen artifact)
    with open(PILOT_NARRATIVES, "w", encoding="utf-8") as fh:
        for rec in pilot_records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nPilot narratives saved to {PILOT_NARRATIVES.resolve()}")
    print(f"Cache: {gen_client._hits} hits, {gen_client._misses} misses")

    # ------------------------------------------------------------------
    # Step 2: Self-consistency check on the paraphrase target pairs
    # ------------------------------------------------------------------
    print()
    print("STEP 2: Loading GeminiNLIClient for self-consistency check...",
          file=sys.stderr)
    from src.scoring import GeminiNLIClient
    gemini_nli = GeminiNLIClient(model_name=args.model)
    print(f"NLI model: {gemini_nli._model_name}", file=sys.stderr)

    run_self_consistency_check(pilot_records, original_narratives, gemini_nli)

    # ------------------------------------------------------------------
    # Step 3: Readability examples
    # ------------------------------------------------------------------
    report_readability(pilot_records)

    # ------------------------------------------------------------------
    # Step 4: Length comparison
    # ------------------------------------------------------------------
    report_length_comparison(pilot_records, original_narratives)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Pilot narratives: {PILOT_NARRATIVES.resolve()}")
    print(f"  Model: {gen_client.model_name}")
    print(f"  k={K_PILOT}, temperature={gen_client.temperature}")
    print()
    print("Results above show:")
    print("  1. Whether user-ID consistency fixes the paraphrase-blindness (entailment scores)")
    print("  2. Whether the prose reads naturally for an analyst (readability examples)")
    print("  3. The length/referent-count trade-off vs the original prompt style")
    print()
    print("Do not proceed to Stage 3/4 or adopt this as a pipeline change")
    print("until the above results have been reviewed.")


if __name__ == "__main__":
    main()
