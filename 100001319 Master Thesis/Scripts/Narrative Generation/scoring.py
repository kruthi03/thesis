"""Stage 3: sentence scoring -- self-consistency (NLI) and fact-match extraction.

Read docs/stage_3.md before changing anything here; it is the brief this
module implements. Two independent signals are computed and kept separately
loggable -- the ablation (each alone vs. combined) is a required thesis
result, not optional, so nothing in this module should collapse them into one
number. Combining them into rho(s), and the evidence-weighting (misweighted)
check, belong to src/admission.py, not here -- keeping that split is
deliberate, matching the brief's own module boundary.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from .sentences import split_sentences

# ---------------------------------------------------------------------------
# Self-consistency: NLI entailment across the k raw draft narratives
# ---------------------------------------------------------------------------
#
# docs/stage_3.md is explicit that this must NOT reuse Stage 2's n_samples /
# appears_in: that count depends on the embedding threshold (0.85) and the
# rendering version (v3) -- LIMITATIONS.md finding 9 -- and it merges
# sentences that assert different things when they are phrasally similar
# (finding 9's ZCS0012 compound-sentence case, where a context clause and an
# evidence clause pooled into one cluster at 0.85). Self-consistency for
# scoring is recomputed fresh via NLI TEXTUAL ENTAILMENT: does an entire raw
# narrative draft entail a candidate sentence? That is a different question
# from "does this draft contain a similarly-worded sentence", and it is the
# one docs/stage_3.md and ConU / Conformal Language Modeling actually ask.


class NLIClient(Protocol):
    """Minimal surface an NLI backend must satisfy.

    Entailment, not similarity: does `premise` (a full narrative draft) entail
    `hypothesis` (one candidate sentence)? Embedding cosine similarity is
    explicitly ruled out by docs/stage_3.md for this purpose -- two sentences
    can be close in embedding space while one contradicts, or merely shares a
    topic with, the other. Semantic pooling (src/sentences.py) already uses
    embeddings for a different job (collapsing paraphrases); reusing it here
    would just be n_samples again with extra steps.
    """

    def entails(self, premise: str, hypothesis: str) -> bool: ...


@dataclass
class StubNLIClient:
    """Offline, deterministic entailment for the test suite.

    Truth-table driven: a test registers the exact (premise, hypothesis) ->
    entails relationships it wants via `set()`, then asserts self_consistency
    computes the correct fraction from them. This is deliberate -- a stub that
    "guesses" entailment from text similarity would validate the heuristic,
    not the arithmetic in self_consistency(), and every test in this module
    needs KNOWN relationships (docs/stage_3.md's own phrase) to be a real
    test of the formula rather than of the stub's guesswork.

    A crude token-overlap fallback exists for pairs no test registered, so the
    stub remains usable standalone (e.g. ad-hoc smoke checks) without every
    pair having to be enumerated -- but no test in this module should rely on
    the fallback's specific behaviour, only on registered pairs.

    Wiring in a real NLI model (docs/stage_3.md suggests
    microsoft/deberta-v3-base fine-tuned on MNLI) is a later step; this class
    exists so the rest of the pipeline is buildable and testable before that
    dependency exists, matching the precedent of StubLLMClient in generate.py
    and pysbd's offline segmenter in sentences.py.
    """

    fallback_overlap_threshold: float = 0.6
    calls: int = 0
    _table: dict[tuple[str, str], bool] = field(default_factory=dict)

    def set(self, premise: str, hypothesis: str, entails: bool) -> None:
        self._table[(premise, hypothesis)] = entails

    def entails(self, premise: str, hypothesis: str) -> bool:
        self.calls += 1
        key = (premise, hypothesis)
        if key in self._table:
            return self._table[key]
        return self._token_overlap(premise, hypothesis) >= self.fallback_overlap_threshold

    @staticmethod
    def _token_overlap(premise: str, hypothesis: str) -> float:
        p_tokens = set(re.findall(r"\w+", premise.lower()))
        h_tokens = set(re.findall(r"\w+", hypothesis.lower()))
        if not h_tokens:
            return 0.0
        return len(p_tokens & h_tokens) / len(h_tokens)


class CrossEncoderNLIClient:
    """Real entailment, via a sentence-transformers CrossEncoder fine-tuned on
    MNLI. Locally runnable (CPU, no API key) -- the standard lightweight
    choice docs/stage_3.md points at, not a hosted model.

    Lazy import, matching AnthropicClient/OllamaClient in src/generate.py:
    sentence-transformers (and its torch dependency) is only needed on the
    real-scoring path, so the offline test suite (StubNLIClient) never pays
    for it and never needs it installed.

    Default model is 'cross-encoder/nli-deberta-v3-small' -- small enough to
    run 197 sentences x ~10 drafts on CPU in a reasonable time, distilled from
    the same deberta-v3 family docs/stage_3.md names as its suggested MNLI
    model. `entails()` reduces the model's 3-way MNLI output (entailment /
    neutral / contradiction) to the boolean this module's Protocol needs:
    True iff entailment is the argmax label. Neutral and contradiction are
    both "does not entail" for this purpose -- self_consistency() counts
    drafts that actually assert the candidate sentence, not drafts that
    merely fail to contradict it.

    -base was tried and rejected (LIMITATIONS.md): it fixes -small's noisy
    ~40% contradiction misfires on same-fact/different-wording pairs, but not
    the dominant paraphrase-blindness failure (scripts/diagnose_paraphrase_
    blindness.py: 1/12 pairs flipped correctly, including a miss on the
    highest-overlap 0.81 sibling) -- the size increase's cost is not justified
    by that benefit. Do not swap back to -base without new evidence; see
    LIMITATIONS.md for the full record before reopening this axis.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        # id2label varies by checkpoint (some are 0=contradiction/1=entailment/
        # 2=neutral, others differ) -- read it from the model rather than
        # hardcoding an index, so a wrong assumption here cannot silently
        # invert every entailment call.
        id2label = {
            int(k): v.lower() for k, v in self._model.config.id2label.items()
        }
        entailment_ids = [i for i, label in id2label.items() if label == "entailment"]
        if len(entailment_ids) != 1:
            raise ValueError(
                f"{model_name}: expected exactly one 'entailment' label in "
                f"id2label, got {id2label}"
            )
        self._entailment_index = entailment_ids[0]

    def entails(self, premise: str, hypothesis: str) -> bool:
        scores = self._model.predict([(premise, hypothesis)])[0]
        return int(scores.argmax()) == self._entailment_index


# ---------------------------------------------------------------------------
# Gemini judge: shared base, NLI client
# ---------------------------------------------------------------------------
# Both GeminiNLIClient (here) and GeminiMisweightedClient (src/admission.py)
# inherit from _GeminiBase for shared key-read, model-validation, retry, and
# cache logic. The base class lives here because NLIClient (the Protocol) lives
# here; admission.py imports the base together with the rest of scoring.
#
# THE EXISTING CrossEncoderNLIClient AND is_misweighted() HEURISTIC ARE NOT
# TOUCHED. Both Gemini clients are opt-in: pass a GeminiNLIClient instance as
# `nli` to self_consistency() / score_sentence(), and pass a
# GeminiMisweightedClient instance as `gemini_misweighted_client` to
# score_sentence(). Defaults everywhere remain the keyword-heuristic path.


@dataclass
class GeminiScores:
    """Raw output from one Gemini NLI call, kept separately loggable.

    The NLIClient Protocol only requires the boolean `entails()` result; this
    dataclass is what `GeminiNLIClient.entails_with_scores()` returns for the
    diagnostic script, which needs the label and the model's one-sentence
    reason, not just a True/False.
    """

    label: str       # "entailment" | "contradiction" | "neutral"
    reason: str      # one-sentence explanation from Gemini

    @property
    def entails(self) -> bool:
        return self.label == "entailment"


class _GeminiBase:
    """Shared plumbing for GeminiNLIClient and GeminiMisweightedClient.

    Handles:
    - Lazy import of google.genai (never pays the import cost when not used).
    - API key read from GEMINI_API_KEY env var -- raises ValueError with a
      clear message if absent, rather than letting the SDK emit an opaque error.
    - Model validation at instantiation: calls client.models.list() and checks
      the requested model name exists among generateContent-capable models.
      Raises ValueError with the actual available model list if not found.
    - On-disk response cache keyed by SHA-1 of the relevant inputs, stored as
      JSON files in cache_dir.  Same temp-file + os.replace atomic-write
      pattern as NarrativeCache in generate.py: a crash mid-write cannot leave
      a half-parsed entry.
    - Exponential-backoff retry (same max_retries / retry_backoff_s shape as
      generate.py's _complete_with_retry).

    Not instantiated directly -- subclassed by GeminiNLIClient and
    GeminiMisweightedClient.
    """

    #: Model name validated at __init__ and reused for every call.
    _GEMINI_MODEL_DEFAULT = "gemini-3.6-flash"
    #: Prefix the list_models() endpoint returns; strip when comparing.
    _MODEL_PREFIX = "models/"

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> None:
        # Lazy import -- only needed on the Gemini code path.
        try:
            from google import genai as _genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini judge; "
                "install it with: pip install google-genai"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before using GeminiNLIClient or GeminiMisweightedClient."
            )

        self._client = _genai.Client(api_key=api_key)
        self._model_name = model_name or self._GEMINI_MODEL_DEFAULT

        # Validate the requested model exists -- fail loudly at instantiation,
        # not silently at first call (which could be thousands of calls in).
        available = [
            m.name.removeprefix(self._MODEL_PREFIX)
            for m in self._client.models.list()
            if "generateContent" in (m.supported_actions or [])
        ]
        bare = self._model_name.removeprefix(self._MODEL_PREFIX)
        if bare not in available:
            raise ValueError(
                f"Model {self._model_name!r} is not in the list of available "
                f"generateContent-capable models: {available}"
            )
        # Store the bare name (without prefix) for API calls.
        self._model_name = bare

        self.use_cache = use_cache
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._hits = 0
        self._misses = 0

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "gemini_judge"
        self._cache_dir = Path(cache_dir)
        if use_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal cache helpers (same SHA-1 + sharded-dir as NarrativeCache)
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _cache_get(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return payload["result"]
        except (json.JSONDecodeError, OSError, KeyError):
            self._misses += 1
            return None

    def _cache_put(self, key: str, result: dict) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "model": self._model_name, "result": result}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Internal retry wrapper
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        contents: str,
        response_schema: dict,
    ) -> dict:
        """Call Gemini with JSON-mode output (response_mime_type=application/json).

        Retries with exponential backoff on transient errors, same pattern as
        generate.py's _complete_with_retry. Returns the parsed JSON dict.
        """
        from google.genai import types as _gtypes  # type: ignore[import]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=_gtypes.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.0,  # deterministic for a classification task
                    ),
                )
                return json.loads(response.text)
            except Exception as exc:  # noqa: BLE001 -- transient API errors
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


class GeminiNLIClient(_GeminiBase):
    """NLI entailment via Gemini -- same NLIClient Protocol as CrossEncoderNLIClient.

    Opt-in, not a replacement: pass an instance of this class as the `nli`
    argument to self_consistency() or score_sentence() to use Gemini instead of
    the cross-encoder. Nothing else changes.

    Motivation (docs/stage_3.md / LIMITATIONS.md finding 11): the cross-encoder
    (cross-encoder/nli-deberta-v3-small) shows paraphrase-blindness on
    high-lexical-overlap sentence pairs -- it scores obvious paraphrases as
    neutral at ~0.999 confidence (scripts/diagnose_paraphrase_blindness.py:
    0ec64e3f86ed_s003 and 06ffa8b234ef_s009). Gemini is tested on the same pairs
    via scripts/diagnose_gemini_judge.py before any full-dataset run.

    API key: read from GEMINI_API_KEY environment variable at instantiation.
    Never hardcoded.

    Cache: on-disk JSON files under cache_dir, keyed by SHA-1 of
    (model_name, "nli", premise, hypothesis). A cache hit skips the Gemini call
    entirely, same as NarrativeCache in generate.py.

    Structured output: prompt requests a JSON object with `label` (one of
    "entailment"/"neutral"/"contradiction") and `reason` (one sentence).
    response_mime_type="application/json" + response_schema forces parseable
    output -- no regex extraction of free text.
    """

    # JSON schema for the structured NLI response
    _NLI_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["entailment", "neutral", "contradiction"],
            },
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
    }

    def _build_prompt(self, premise: str, hypothesis: str) -> str:
        return (
            "You are a precise textual entailment judge. "
            "Given a PREMISE sentence and a HYPOTHESIS sentence, determine whether "
            "the PREMISE entails, contradicts, or is neutral with respect to the HYPOTHESIS.\n\n"
            "Definitions:\n"
            "- entailment: the premise asserts the same fact as the hypothesis (even if "
            "worded differently, paraphrased, or with minor rewording).\n"
            "- contradiction: the premise asserts something that conflicts with the hypothesis.\n"
            "- neutral: the premise neither confirms nor denies the hypothesis.\n\n"
            f"PREMISE: {premise}\n"
            f"HYPOTHESIS: {hypothesis}\n\n"
            'Respond ONLY with a JSON object: {"label": <one of \'entailment\'/\'neutral\'/\'contradiction\'>, '
            '"reason": <one concise sentence explaining your judgment>}'
        )

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        return "|".join([self._model_name, "nli", premise, hypothesis])

    def entails_with_scores(self, premise: str, hypothesis: str) -> GeminiScores:
        """Full classification result for the diagnostic script.

        Returns GeminiScores with .label and .reason. The Protocol-required
        `entails()` method calls this and returns .entails.
        """
        key = self._cache_key(premise, hypothesis)
        cached = self._cache_get(key)
        if cached is not None:
            return GeminiScores(**cached)

        prompt = self._build_prompt(premise, hypothesis)
        result = self._call_gemini(prompt, self._NLI_RESPONSE_SCHEMA)
        # Normalise: the schema constrains the enum but be defensive
        label = result.get("label", "neutral").lower()
        if label not in ("entailment", "neutral", "contradiction"):
            label = "neutral"
        reason = result.get("reason", "")
        self._cache_put(key, {"label": label, "reason": reason})
        return GeminiScores(label=label, reason=reason)

    def entails(self, premise: str, hypothesis: str) -> bool:
        """NLIClient Protocol: True iff Gemini judges the premise as entailment."""
        return self.entails_with_scores(premise, hypothesis).entails


class GeminiFactExtractorClient(_GeminiBase):
    """Fact extraction via Gemini structured JSON output.
    
    Replaces the regex extraction in extract_numbers() with an LLM parser
    that handles complex numeric claims ("half a million", "100k").
    """
    
    _FACT_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "All numeric claims extracted from the sentence, converted to exact floats."
            }
        },
        "required": ["numbers"]
    }
    
    def _build_prompt(self, text: str) -> str:
        return (
            "You are an expert data parser. Your task is to extract every single numerical "
            "claim from the provided sentence and convert it to a standard floating-point number.\n"
            "For example, 'half a million' becomes 500000.0, '3.5k' becomes 3500.0, 'two' becomes 2.0.\n"
            "Ignore dates (e.g. 2010-08-16) and week IDs (2010-W33), but keep clock times (02:14 -> 2.0 and 14.0).\n"
            "Return ONLY a JSON object with a 'numbers' array. If there are no numbers, return an empty array.\n\n"
            f'SENTENCE: "{text}"'
        )
        
    def _cache_key(self, text: str) -> str:
        return "|".join([self._model_name, "fact_extract", text])
        
    def extract(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._cache_get(key)
        if cached is not None:
            return [float(x) for x in cached.get("numbers", [])]
            
        prompt = self._build_prompt(text)
        result = self._call_gemini(prompt, self._FACT_RESPONSE_SCHEMA)
        numbers = [float(x) for x in result.get("numbers", [])]
        self._cache_put(key, {"numbers": numbers})
        return numbers


def self_consistency(
    sentence: str,
    narratives: Sequence[str],
    nli: NLIClient,
    *,
    exclude_index: int | None = None,
) -> float:
    """Fraction of the OTHER k narrative drafts that entail `sentence`.

    docs/stage_3.md:  self_consistency(s) = 1/(k-1) * sum_{j != i} 1[narrative_j |= s]

    `exclude_index` is i -- the draft `sentence` was drawn from, if known.
    Excluding it matters: a narrative trivially "entails" a sentence lifted
    verbatim from itself, and counting that would inflate every sentence's
    consistency by exactly 1/(k-1) regardless of whether any OTHER draft
    agrees with it. Pass None when scoring a pooled-cluster representative not
    tied to one single draft (a cluster's `appears_in` can span several
    samples); the denominator is then all k narratives rather than k-1.

    Raises rather than returning a silent 0.0 or 0/0 when there is nothing to
    compare against (k < 2 with exclude_index set, or narratives empty) --
    "no peers to check agreement against" and "every peer disagreed" are
    different findings, and the former must not be reported as the latter.

    NLI PREMISE IS ONE SENTENCE OF THE OTHER DRAFT, NOT THE WHOLE DRAFT
    PARAGRAPH. A diagnostic run of a real cross-encoder NLI model
    (cross-encoder/nli-deberta-v3-small) against whole-paragraph premises
    found it scoring VERBATIM substring matches as "neutral" at ~0.999
    confidence -- not a real disagreement signal, a premise/hypothesis length
    mismatch relative to the short, roughly single-sentence pairs the model
    was trained on (SNLI/MNLI). Splitting each OTHER draft into its own
    sentences (src.sentences.split_sentences -- the SAME segmenter
    src.sentences.pool_sentences_semantic uses for pooling, not a second
    implementation) and taking the max entailment signal across that draft's
    sentences restores the premise/hypothesis length distribution the model
    was actually trained on. A draft counts as "entailing" `sentence` if ANY
    one of its own sentences does -- max, not average, because one sentence
    within a multi-claim draft asserting `sentence`'s claim is exactly what
    "this draft agrees" should mean; the draft's other, unrelated sentences
    saying nothing about this claim must not dilute that.
    """
    others = [n for j, n in enumerate(narratives) if j != exclude_index]
    if not others:
        raise ValueError(
            "self_consistency needs at least one comparison narrative "
            f"(got {len(narratives)} narratives, exclude_index={exclude_index})"
        )
    entailed = 0
    for narrative in others:
        draft_sentences = split_sentences(narrative) or [narrative]
        if any(nli.entails(s, sentence) for s in draft_sentences):
            entailed += 1
    return entailed / len(others)


# ---------------------------------------------------------------------------
# Fact matching: numeric and entity extraction against facts[]
# ---------------------------------------------------------------------------

_LEADING_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_leading_number(display: str) -> float:
    match = _LEADING_NUMBER_RE.search(display)
    if not match:
        raise ValueError(f"no numeric token found in rendered display: {display!r}")
    return float(match.group())


@dataclass
class RenderedFact:
    """The numbers one fact actually put in front of the model, per
    src.signals.signal_to_prompt_context -- NOT everything facts[] carries.

    This distinction is load-bearing (LIMITATIONS.md finding 10a and the v2 ->
    v3 DECISION). self_recent_max, self_z, and peer_z stay in facts[] for
    other purposes but are deliberately never rendered in the v3 prompt -- the
    entire point of the v3 rendering change was that a narrative stating the
    withheld 12-week high cites a number the model never saw, and can
    therefore be caught as unsupported rather than accepted as a
    traceable-but-wrong comparison (LIMITATIONS.md: "any residual claim would
    now cite an unsupported number", "0 of 200 v3 samples" reconstructed it).
    If this fact-matcher matched against self_recent_max, it would silently
    reopen exactly that leak and the measured 0/200 result would no longer
    mean what LIMITATIONS.md says it means. Every number exposed here is one
    signal_to_prompt_context() actually wrote into the prompt; that is the
    contract the rest of this module depends on.
    """

    fact_id: str
    gloss: str
    numbers: dict[str, float]


def rendered_facts(signal: dict[str, Any]) -> list[RenderedFact]:
    """Rebuild, from facts[], exactly the numbers rendered for each fact.

    Reuses signals._fmt (the same function signal_to_prompt_context calls)
    rather than reimplementing its rounding/unit-conversion logic -- CLAUDE.md
    and generate.py's build_prompt() both already establish the rule that the
    rendering must not be reconstructed a second time, since the two copies
    would drift apart. This matters concretely here: attachment_bytes >=
    1,000,000 renders in MB (`_fmt`), so a fact's raw f['value'] (bytes) is
    NOT what the model saw and matching against it directly would be wrong --
    f['display'] (already `_fmt`-formatted) is used instead. self_center and
    peer_center get the SAME treatment via an explicit _fmt call, mirroring
    exactly what signal_to_prompt_context._render does for them.
    """
    # Lazy import, matching generate.py's build_prompt(): signals.py pulls in
    # pandas/numpy via cert_features, and this module has no other reason to
    # force that at import time. Not a circular-import concern (signals.py
    # does not import scoring.py) -- purely a load-weight / precedent match.
    from .signals import _fmt

    out = []
    for f in signal["facts"]:
        numbers: dict[str, float] = {"value": _parse_leading_number(f["display"])}
        # never_before suppresses the self_center line entirely in the
        # rendered prompt (an "elif" in _render, not an independent branch --
        # see signals.py) in favour of a qualitative "(not observed ...)"
        # phrase with no number. Mirror that exactly, or a never_before fact's
        # self_center would be treated as grounded when it was never shown.
        if not f.get("never_before") and "self_center" in f:
            numbers["self_center"] = _parse_leading_number(
                _fmt(f["self_center"], f["field"]))
        if "peer_center" in f:
            numbers["peer_center"] = _parse_leading_number(
                _fmt(f["peer_center"], f["field"]))
            # Rendered as the bare int directly (signals.py: "across {N} peers
            # in this role"), not run through _fmt.
            numbers["peer_group_size"] = float(f["peer_group_size"])
        out.append(RenderedFact(fact_id=f["fact_id"], gloss=f["gloss"], numbers=numbers))
    return out


# ISO calendar dates (2010-08-16) and ISO week ids (2010-W33) correspond to
# the week span the model IS shown (signals.py: _week_span), but facts[] does
# not carry a period fact for these to be checked against, and
# docs/annotation_guideline.md already treats a malformed date as
# "not checkable against facts[]" (its e4f5ca67cebc_s001 worked example) --
# i.e. date matching is explicitly out of scope for the fact-matcher, not an
# oversight. Stripped before number extraction so "2010" / "33" are not
# scored as spurious unmatched (or worse, coincidentally matched) numbers.
#
# Deliberately NOT stripped: clock-time-shaped tokens ("02:14"). A fabricated
# sub-week timestamp (annotation guideline Rule 2) is exactly the kind of
# claim this function exists to flag as unsupported, so each half is
# extracted as an ordinary number rather than removed. Known limitation: if
# either half coincidentally equals a real fact value (e.g. an hour-of-day
# fact), that half will misleadingly score "matched" and mask part of the
# fabricated claim -- the fabricated day-of-week portion is a separate,
# unaffected problem this module does not attempt to catch (see
# docs/stage_3.md's evidence-weighting section / src/admission.py for
# claim-level checks beyond bare numbers).
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{4}-W\d{1,2}\b",
    re.IGNORECASE,
)

# \b\d+(?:\.\d+)?\b intentionally does not need extra exclusions for rule ids
# (R4, [R2]) or fact ids (F001): a digit run only matches at a WORD boundary,
# and there is no boundary between a letter and an adjacent digit in the same
# token, so "4" in "R4" or "001" in "F001" never starts a boundary-anchored
# match. Verified, not assumed -- see tests/test_scoring.py.
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def extract_numbers(text: str) -> list[float]:
    """Numeric tokens a sentence could be checked against facts[] for."""
    scrubbed = _DATE_RE.sub(" ", text)
    return [float(m.group()) for m in _NUMBER_RE.finditer(scrubbed)]


# Relative tolerance for a DIRECT numeric match: the model is expected to
# quote the rendered figure closely (its own re-rounding at most), not
# approximate it.
_MATCH_TOLERANCE = 0.01
# Wider tolerance for RATIO/PERCENT derivation: prose rounds an arithmetic
# result more loosely ("about 5 times", "5.4x") than a direct citation of a
# shown number.
_DERIVED_TOLERANCE = 0.05
# Unit-scale conversion (x1000/1e6/1e9) is tight, NOT the wider tolerance
# above -- it is exact arithmetic, not a prose approximation, so it gets the
# same precision as a direct match. Using _DERIVED_TOLERANCE here was
# measured to be a real false-positive source, not just a hypothetical one:
# with a fact rendering self_center as "10.00 MB" (10.0), an unrelated
# candidate number as far off as 9999 sits within 5% of 10.0 * 1000 (10000)
# and would be scored "derived" -- a materially different number laundered
# into a false-not-hallucinated verdict. A candidate that is genuinely a unit
# conversion reproduces the scale factor almost exactly; one that merely
# happens to land in a wide tolerance band should not.
_UNIT_TOLERANCE = _MATCH_TOLERANCE
# Metric-style unit scales checked in both directions (rendered/scale and
# rendered*scale) -- covers the byte-count-rendered-as-MB/KB/GB shape from
# LIMITATIONS.md finding 7 for any fact whose raw value fell under the 1e6
# threshold in signals._fmt and was therefore rendered in bytes, not MB.
_UNIT_SCALES = (1_000.0, 1_000_000.0, 1_000_000_000.0)


def _close(a: float, b: float, *, tol: float) -> bool:
    if b == 0:
        return abs(a - b) < 1e-9
    return abs(a - b) / abs(b) <= tol


@dataclass
class FactMatch:
    number: float
    status: str  # "matched" | "derived" | "unmatched"
    fact_id: str | None = None
    detail: str | None = None


def match_number(number: float, facts: Sequence[RenderedFact]) -> FactMatch:
    """Classify one extracted number against a signal's rendered facts.

    matched    -- equals (within tolerance) a number the fact actually shows.
    derived    -- not shown directly, but arithmetically reachable from
                  numbers the fact DOES show (a ratio, a percentage, or a
                  metric unit conversion). Per LIMITATIONS.md finding 7, these
                  are traceable but NOT grounded: they are exactly the false
                  precision the guideline's "false precision" label exists
                  for, and folding them into "matched" would hide that.
    unmatched  -- neither. The strongest hallucination signal.

    fact_id attribution for a match is best-effort: if two facts happen to
    share a rendered number (peer_group_size is often identical across every
    fact in one signal), the number alone cannot say which fact a narrative
    meant, so the first match in facts[] order (evidence before context, per
    build_signal) is reported. This does not affect the matched/derived/
    unmatched status, only which fact_id is attached to it.
    """
    for f in facts:
        for label, rendered in f.numbers.items():
            if _close(number, rendered, tol=_MATCH_TOLERANCE):
                return FactMatch(number, "matched", f.fact_id, label)

    for f in facts:
        vals = f.numbers
        for a_key, b_key in itertools.permutations(vals, 2):
            a, b = vals[a_key], vals[b_key]
            if b == 0:
                continue
            ratio = a / b
            if _close(number, ratio, tol=_DERIVED_TOLERANCE):
                return FactMatch(number, "derived", f.fact_id, f"{a_key}/{b_key} ratio")
            pct = ratio * 100
            if _close(number, pct, tol=_DERIVED_TOLERANCE):
                return FactMatch(number, "derived", f.fact_id, f"{a_key}/{b_key} percent")
            pct_diff = (a - b) / b * 100
            if _close(number, pct_diff, tol=_DERIVED_TOLERANCE):
                return FactMatch(number, "derived", f.fact_id,
                                 f"{a_key} vs {b_key} percent difference")
        for label, rendered in vals.items():
            if rendered == 0:
                continue
            for scale in _UNIT_SCALES:
                if _close(number, rendered / scale, tol=_UNIT_TOLERANCE):
                    return FactMatch(number, "derived", f.fact_id,
                                     f"{label} / {scale:g} unit conversion")
                if _close(number, rendered * scale, tol=_UNIT_TOLERANCE):
                    return FactMatch(number, "derived", f.fact_id,
                                     f"{label} * {scale:g} unit conversion")

    return FactMatch(number, "unmatched")


# Entity candidates: a Title-Case span of 2+ words ("Dana Whitfield") or an
# uppercase-letters+digits token shaped like a CERT user id ("ACM2278",
# "MBG3183"). Deliberately narrow -- a single capitalised word is usually just
# a sentence-initial ordinary word ("Specifically", "The"), and treating every
# one as a candidate named entity would make this check noise, not signal.
# Proper NER is future work; this is a regex stand-in, same spirit as the stub
# NLI client.
_ENTITY_CANDIDATE_RE = re.compile(
    r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)+\b"
    r"|\b[A-Z]{2,}\d{2,}\b"
)


def extract_entity_candidates(text: str) -> list[str]:
    return [m.group() for m in _ENTITY_CANDIDATE_RE.finditer(text)]


def known_entities(signal: dict[str, Any]) -> set[str]:
    """Lowercased strings a sentence may legitimately name: the user id, the
    employee's full name and each name part, role, team, functional unit, and
    every fact's gloss. Anything else naming a specific person, team, or role
    is unsupported by facts[].
    """
    ident = signal.get("identity", {})
    out: set[str] = set()
    for key in ("employee_name", "role", "team", "functional_unit"):
        val = ident.get(key)
        if not val:
            continue
        val = str(val)
        out.add(val.lower())
        # team/functional_unit render as "N - Name" (e.g. "3 - RegionalSales");
        # narratives commonly drop the numeric prefix, so the bare name must
        # match on its own too.
        out.add(re.sub(r"^\d+\s*-\s*", "", val).lower())
    if signal.get("user"):
        out.add(str(signal["user"]).lower())
    if ident.get("employee_name"):
        for part in str(ident["employee_name"]).split():
            out.add(part.lower())
    for f in signal.get("facts", []):
        if f.get("gloss"):
            out.add(f["gloss"].lower())
    return out


@dataclass
class EntityMatch:
    text: str
    status: str  # "matched" | "unmatched"


def entity_match(candidate: str, known: set[str]) -> bool:
    """Case-insensitive, two-directional substring check.

    Two-directional on purpose: a candidate may be a single name part
    ("Ariel") that is a substring of a known full name, or may itself be a
    longer phrase containing a shorter known fragment.
    """
    c = candidate.lower().strip()
    return bool(c) and any(c in k or k in c for k in known)


@dataclass
class FactMatchResult:
    """Everything the fact-match step found for one candidate sentence."""

    numbers: list[FactMatch]
    entities: list[EntityMatch]

    @property
    def fact_match_score(self) -> float:
        """Fraction of extracted numbers that are `matched` -- docs/stage_3.md:
        "Fact match score is the fraction of extracted numbers/entities that
        are matched (not derived, not unmatched)."

        NaN, not 1.0, when there are zero extracted numbers: a sentence with
        no numeric content has nothing for this score to check, so "fully
        grounded" would be a false signal, not a neutral one. Combining this
        into rho(s) and deciding how to treat that case is src/admission.py's
        job, not this property's -- it must not silently default to a value
        that looks like a real score.
        """
        total = len(self.numbers)
        if total == 0:
            return float("nan")
        matched = sum(1 for m in self.numbers if m.status == "matched")
        return matched / total


def score_fact_match(
    text: str, 
    signal: dict[str, Any],
    extracted_numbers: list[float] | None = None
) -> FactMatchResult:
    """Extract and classify every number and candidate entity in `text`
    against `signal`'s rendered facts."""
    facts = rendered_facts(signal)
    known = known_entities(signal)
    
    if extracted_numbers is not None:
        numbers = [match_number(n, facts) for n in extracted_numbers]
    else:
        numbers = [match_number(n, facts) for n in extract_numbers(text)]
        
    entities = [
        EntityMatch(c, "matched" if entity_match(c, known) else "unmatched")
        for c in extract_entity_candidates(text)
    ]
    return FactMatchResult(numbers=numbers, entities=entities)
