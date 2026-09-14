"""Stage 3: combine self-consistency and fact-match into rho(s), plus the
evidence-weighting (misweighted) check.

docs/stage_3.md splits Stage 3 into two modules on purpose: src/scoring.py
computes the two INDEPENDENT signals (self-consistency via NLI, fact-match
against facts[]) and keeps them separately loggable, because the ablation
(each alone vs. combined) is a required thesis result. This module is where
they get combined -- nothing in scoring.py should have done that.

COMPOUND-SENTENCE SPLIT AND CAUSAL-PHRASE DETECTION ARE PORTED FROM
scripts/prelabel_clusters.py, NOT REINVENTED. docs/stage_3.md is explicit
about why: "Reuse whatever heuristic the guideline settled on rather than
inventing a second one -- the two must agree, or your admission function and
your human labels are measuring different things." prelabel_clusters.py is
the script that actually produced the mechanical suggestions reviewed into
labelling_final.csv, so it is the concrete implementation of
docs/annotation_guideline.md's compound-sentence rule and Rule 4, not just
another reading of the prose. tests/test_admission.py cross-checks this
module's SUBORDINATE_LEADS/CAUSAL_PHRASES/split against
scripts.prelabel_clusters's directly, so "the two must agree" is an enforced
regression test, not a comment promising it.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence
import joblib

from .scoring import FactMatchResult, NLIClient, _GeminiBase, score_fact_match, self_consistency

# ---------------------------------------------------------------------------
# Compound-sentence split (ported from scripts/prelabel_clusters.py)
# ---------------------------------------------------------------------------

# Byte-identical to scripts.prelabel_clusters.SUBORDINATE_LEADS. Kept as a
# literal copy rather than an import because src/ modules do not depend on
# scripts/ (scripts import FROM src, never the reverse -- see generate.py,
# sentences.py, signals.py); tests/test_admission.py enforces agreement
# instead of a runtime import.
SUBORDINATE_LEADS = re.compile(
    r"^\s*(while|although|even though|though|whereas|despite|notwithstanding)"
    r"\b",
    re.IGNORECASE,
)

# Byte-identical to scripts.prelabel_clusters.CAUSAL_PHRASES.
CAUSAL_PHRASES = [
    "due to", "because of", "the reason", "triggered by",
    "is why", "primary driver", "primary reason", "main reason",
    "main evidence",
]


def split_compound_sentence(text: str) -> tuple[str, str | None]:
    """(main_clause, subordinate_clause | None).

    Ported algorithm, not just ported constants: scans for the first
    top-level comma (bracket-depth tracked, so a comma inside "(out of 5,
    typically)" does not end the subordinate clause early) after a leading
    While/Although/... word. Identical to
    scripts.prelabel_clusters._split_compound -- see tests/test_admission.py.
    """
    if SUBORDINATE_LEADS.match(text):
        depth = 0
        for i, ch in enumerate(text):
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0 and i > 5:
                sub = text[:i].strip()
                main = text[i + 1:].strip()
                if main:
                    return main, sub
    return text, None


def has_causal_language(text: str) -> bool:
    """Identical logic to scripts.prelabel_clusters._has_causal_language."""
    t = text.lower()
    return any(p in t for p in CAUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Evidence-weighting: does the main clause blame a context-only fact?
# ---------------------------------------------------------------------------

_MIN_KEYWORD_LEN = 5  # matches prelabel_clusters._clause_cites_context_only


@dataclass
class MisweightedResult:
    misweighted: bool
    main_clause: str
    subordinate_clause: str | None
    context_fact_id: str | None = None


def is_misweighted(
    text: str,
    facts: Sequence[dict[str, Any]],
    evidence_fact_ids: Sequence[str],
) -> MisweightedResult:
    """docs/stage_3.md: "A sentence whose main clause asserts a context fact
    as if it were the reason for the alert is misweighted regardless of
    fact-match score." Detected via the SAME two-part test as
    scripts.prelabel_clusters._classify's Rule 4:

      1. The MAIN clause (post compound-split) uses causal language
         (CAUSAL_PHRASES) -- "the number is real" is not the failure; stating
         it AS THE REASON is.
      2. The main clause's keyword overlap (>=5-char words, matching
         prelabel_clusters' threshold) hits a context-only fact's gloss and
         does NOT hit any evidence fact's gloss. Requiring the evidence-gloss
         absence matters: a main clause that mentions both an evidence fact
         and a context fact together is not this failure mode.

    `evidence_fact_ids` is the field docs/stage_3.md names explicitly
    ("narratives.jsonl already carries this") -- passed in rather than
    re-derived from `facts`, so this function's evidence/context split always
    matches whatever partition the record itself recorded, even if some
    future caller's notion of "evidence" diverges from
    fact.triggering_rule_ids for another reason.

    Known limitation, inherited from prelabel_clusters.py and explicitly
    acknowledged there (docs/annotation_guideline.md Rule 4: "heuristic and
    knowably imperfect"): a misweighted sentence that manufactures
    significance WITHOUT causal language (e.g. a bare peer-relative framing --
    docs/annotation_guideline.md's 107dcedee69e_s002 worked example, "Notably,
    ... 25 visits, significantly above their peers' average of 1 visit", human
    labelled Misweighted) will NOT be flagged here, because it contains none
    of CAUSAL_PHRASES. This is not a bug to fix independently -- doing so
    would break agreement with the human labels the mechanical pass and this
    function are both supposed to approximate. See tests/test_admission.py
    for a test that documents this miss explicitly rather than hiding it.
    """
    main, sub = split_compound_sentence(text)
    if not has_causal_language(main):
        return MisweightedResult(False, main, sub)

    evidence_ids = set(evidence_fact_ids)
    evidence = [f for f in facts if f["fact_id"] in evidence_ids]
    context = [f for f in facts if f["fact_id"] not in evidence_ids]

    ev_keywords = {
        w for f in evidence for w in f["gloss"].lower().split()
        if len(w) >= _MIN_KEYWORD_LEN
    }
    main_lower = main.lower()
    if any(kw in main_lower for kw in ev_keywords):
        # An evidence-fact gloss is also in scope -- not this failure mode.
        return MisweightedResult(False, main, sub)

    for f in context:
        ctx_keywords = [
            w for w in f["gloss"].lower().split() if len(w) >= _MIN_KEYWORD_LEN
        ]
        if any(kw in main_lower for kw in ctx_keywords):
            return MisweightedResult(True, main, sub, f["fact_id"])

    return MisweightedResult(False, main, sub)


# ---------------------------------------------------------------------------
# Gemini-based misweighted detector
# ---------------------------------------------------------------------------

class GeminiMisweightedClient(_GeminiBase):
    """Misweighted-sentence detection via Gemini -- opt-in alternative to is_misweighted().

    The keyword-heuristic is_misweighted() has a documented known miss: it
    cannot catch sentences that manufacture significance via a peer comparison
    WITHOUT causal language (107dcedee69e_s002, see test_admission.py and
    annotation_guideline.md's Rule 4 worked example). This client asks Gemini
    directly, enabling it to catch that failure mode.

    Wired into score_sentence() via the `gemini_misweighted_client` keyword
    argument (default None = use heuristic). Existing behaviour is unchanged
    unless the argument is explicitly passed.

    Returns a MisweightedResult -- same dataclass as is_misweighted() --
    so the rest of the pipeline (combine_rho, RhoResult) is unaffected.
    Note: context_fact_id is always None from this client; Gemini reasons
    holistically and does not attribute to a specific fact_id.

    Structured output: prompt requests JSON {"misweighted": bool, "reason": str}.
    response_mime_type="application/json" + response_schema forces parseable
    output -- no regex extraction.

    Cache: on-disk JSON keyed by SHA-1 of (model, "misweighted", text,
    sorted evidence_fact_ids, sorted fact_glosses). Same atomic-write pattern
    as GeminiNLIClient.
    """

    _MISWEIGHTED_RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "misweighted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["misweighted", "reason"],
    }

    def _build_misweighted_prompt(
        self,
        text: str,
        evidence_glosses: list[str],
        context_glosses: list[str],
    ) -> str:
        ev_block = "\n".join(f"  - {g}" for g in evidence_glosses) or "  (none)"
        ctx_block = "\n".join(f"  - {g}" for g in context_glosses) or "  (none)"
        return (
            "You are an expert analyst reviewing an anomaly-alert explanation.\n\n"
            "The anomaly was triggered by EVIDENCE facts:\n"
            f"{ev_block}\n\n"
            "Additional CONTEXT facts (background only, not the trigger):\n"
            f"{ctx_block}\n\n"
            f'SENTENCE: "{text}"\n\n'
            "Does this sentence present a NON-EVIDENCE (context) fact as if it "
            "were the REASON for the anomaly alert, even if it uses no explicit "
            "causal language (e.g., by framing a context value as surprisingly high "
            "compared with peers)? Answer true only if the sentence's main claim "
            "elevates a context fact to the role of the primary finding.\n\n"
            'Respond ONLY with JSON: {"misweighted": true/false, '
            '"reason": "<one sentence explaining your judgment>"}'
        )

    def _cache_key(
        self,
        text: str,
        evidence_fact_ids: Sequence[str],
        facts: Sequence[dict[str, Any]],
    ) -> str:
        ev_sorted = "__".join(sorted(evidence_fact_ids))
        glosses_sorted = "__".join(sorted(f.get("gloss", "") for f in facts))
        raw = "|".join([self._model_name, "misweighted", text, ev_sorted, glosses_sorted])
        return raw

    def classify(
        self,
        text: str,
        facts: Sequence[dict[str, Any]],
        evidence_fact_ids: Sequence[str],
    ) -> MisweightedResult:
        """Classify one sentence via Gemini.

        Returns MisweightedResult with the same shape as is_misweighted().
        context_fact_id is None -- Gemini's holistic judgment does not name a
        specific fact_id, only whether the sentence is misweighted overall.
        main_clause is set to `text` (no compound-split attempted; Gemini
        reasons over the whole sentence).
        """
        key = self._cache_key(text, evidence_fact_ids, facts)
        cached = self._cache_get(key)
        if cached is not None:
            return MisweightedResult(
                misweighted=cached["misweighted"],
                main_clause=text,
                subordinate_clause=None,
                context_fact_id=None,
            )

        evidence_ids = set(evidence_fact_ids)
        evidence_glosses = [f["gloss"] for f in facts if f["fact_id"] in evidence_ids]
        context_glosses = [f["gloss"] for f in facts if f["fact_id"] not in evidence_ids]

        prompt = self._build_misweighted_prompt(text, evidence_glosses, context_glosses)
        result = self._call_gemini(prompt, self._MISWEIGHTED_RESPONSE_SCHEMA)

        misweighted = bool(result.get("misweighted", False))
        reason = result.get("reason", "")
        self._cache_put(key, {"misweighted": misweighted, "reason": reason})

        return MisweightedResult(
            misweighted=misweighted,
            main_clause=text,
            subordinate_clause=None,
            context_fact_id=None,
        )

    # Make the reason available for diagnostic scripts without requiring
    # callers to parse the private cache.
    def classify_with_reason(
        self,
        text: str,
        facts: Sequence[dict[str, Any]],
        evidence_fact_ids: Sequence[str],
    ) -> tuple[MisweightedResult, str]:
        """(MisweightedResult, reason_string) for the diagnostic script."""
        key = self._cache_key(text, evidence_fact_ids, facts)
        cached = self._cache_get(key)
        if cached is not None:
            return (
                MisweightedResult(
                    misweighted=cached["misweighted"],
                    main_clause=text,
                    subordinate_clause=None,
                    context_fact_id=None,
                ),
                cached.get("reason", ""),
            )

        evidence_ids = set(evidence_fact_ids)
        evidence_glosses = [f["gloss"] for f in facts if f["fact_id"] in evidence_ids]
        context_glosses = [f["gloss"] for f in facts if f["fact_id"] not in evidence_ids]
        prompt = self._build_misweighted_prompt(text, evidence_glosses, context_glosses)
        result = self._call_gemini(prompt, self._MISWEIGHTED_RESPONSE_SCHEMA)

        misweighted = bool(result.get("misweighted", False))
        reason = result.get("reason", "")
        self._cache_put(key, {"misweighted": misweighted, "reason": reason})
        return (
            MisweightedResult(
                misweighted=misweighted,
                main_clause=text,
                subordinate_clause=None,
                context_fact_id=None,
            ),
            reason,
        )


#
# docs/stage_3.md specifies rho(s) in [0,1] "combining two independent
# signals" but does not fix the combination formula -- that is this module's
# design decision, and it is documented here rather than left implicit:
#
#   rho(s) = 0.0                       if misweighted
#          = fact_match_score(s)       if fact_match_score is available (not NaN)
#          = self_consistency(s)       if fact_match_score is NaN (no numeric/entity content)
#
# FACT-MATCH IS THE PRIMARY SIGNAL. Justification: across three independent
# runs (StubNLIClient, cross-encoder/nli-deberta-v3-small,
# cross-encoder/nli-deberta-v3-base) the ablation in scripts/evaluate_admission.py
# showed fact-match alone achieving 4-8x the recall of self-consistency alone at
# every threshold (fact_match_alone: P~0.88 R~0.63 at threshold=0.5;
# self_consistency_alone: P~0.93 R~0.16 at threshold=0.5). The weakest-link
# product caused combined rho to inherit self-consistency's confirmed
# paraphrase-blindness (LIMITATIONS.md finding 11) instead of benefiting from
# fact-match's stronger signal. This reweighting is decided from existing
# ablation evidence and frozen BEFORE any Stage 4 calibration touches data.
#
# MISWEIGHTED FORCES ZERO, not a discount, because docs/stage_3.md says so
# explicitly: "regardless of fact-match score -- every number in it can be
# real and it's still misleading." A high fact_match_score on a misweighted
# sentence is not partial credit; the sentence's claim about WHY the week was
# flagged is false regardless of how real its numbers are, so nothing else
# should be able to buy back trust.
#
# NaN FACT_MATCH_SCORE FALLS BACK TO self_consistency ALONE, not to zero and
# not to treating the sentence as fully grounded. src/scoring.py's
# FactMatchResult.fact_match_score is NaN specifically when a sentence has no
# extracted numbers to check (a purely qualitative sentence, e.g. "The
# activity is worth a quick look by an analyst.") -- there is nothing for
# fact-matching to confirm OR contradict, so substituting 0.0 (silently
# punishing every non-numeric sentence to zero regardless of how consistent
# the model is about it) is wrong. Falling back to self_consistency alone lets
# the ONE signal that has something to say decide.


def _combined_fact_match_score(result: FactMatchResult) -> float:
    """docs/stage_3.md, literally: "Fact match score is the fraction of
    extracted numbers/entities that are matched (not derived, not
    unmatched)." -- numbers AND entities together, not numbers alone.

    src/scoring.py's FactMatchResult.fact_match_score property deliberately
    computes numbers only, with its own docstring flagging that a combined
    numbers+entities score was left to this module. This function is that
    deferred decision, now made: entities are folded in because leaving them
    out has a real failure mode, not just a wording mismatch -- a sentence
    that invents a colleague's name ("Dana Whitfield") but cites only real
    numbers would score a numbers-only fact_match_score of 1.0, missing the
    entity hallucination completely (this is exactly StubLLMClient's
    "hallucinated_entity" scenario in generate.py; see
    tests/test_admission.py).

    NaN under the same rule as the numbers-only version: nothing extracted at
    all (zero numbers AND zero entity candidates) means nothing for
    fact-matching to check, not a perfect score.
    """
    total = len(result.numbers) + len(result.entities)
    if total == 0:
        return float("nan")
    matched = (sum(1 for m in result.numbers if m.status == "matched")
               + sum(1 for e in result.entities if e.status == "matched"))
    return matched / total


_RHO_MODEL = None
_RHO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "rho_model.pkl")

def combine_rho(
    consistency: float,
    fact_match_score: float,
    *,
    misweighted: bool,
) -> float:
    """The formula documented above, now replaced by a trained Logistic Regression
    model (trained on the human labels) to provide a smooth, continuous risk surface
    rather than a harsh binary collapse.
    
    Dynamically loads the trained model from scripts/rho_model.pkl.
    """
    global _RHO_MODEL
    if _RHO_MODEL is None:
        _RHO_MODEL = joblib.load(_RHO_MODEL_PATH)

    if math.isnan(fact_match_score):
        fact_match_score = consistency
        
    mw = 1.0 if misweighted else 0.0
    
    # model.predict_proba returns [[prob_0, prob_1]]
    # We want prob_1 (the probability of being 'grounded')
    prob = _RHO_MODEL.predict_proba([[consistency, fact_match_score, mw]])[0][1]
    
    return prob


@dataclass
class RhoResult:
    """Every component that went into rho(s), kept individually loggable --
    docs/stage_3.md requires the ablation (each signal alone vs. combined) as
    a reported result, which is only possible if nothing here is discarded
    once rho is computed.
    """

    rho: float
    self_consistency: float
    fact_match_score: float
    fact_match: FactMatchResult
    misweighted: MisweightedResult


def score_sentence(
    text: str,
    narratives: Sequence[str],
    signal: dict[str, Any],
    evidence_fact_ids: Sequence[str],
    nli: NLIClient,
    *,
    exclude_index: int | None = None,
    gemini_misweighted_client: "GeminiMisweightedClient | None" = None,
    gemini_fact_extractor_client: "GeminiFactExtractorClient | None" = None,
) -> RhoResult:
    """End-to-end: self-consistency (src.scoring) + fact-match (src.scoring)
    + evidence-weighting (this module) -> rho(s) (this module).

    This is the function scripts/score_narratives.py (not yet written) will
    call per pooled sentence. `signal` supplies facts[] for fact-matching and
    for is_misweighted's gloss lookup; `evidence_fact_ids` is the
    narratives.jsonl field docs/stage_3.md names for the evidence/context
    partition.

    `gemini_misweighted_client` -- opt-in Gemini misweighted detector
    (GeminiMisweightedClient). When None (default), is_misweighted() keyword
    heuristic is used unchanged. When set, its .classify() is called instead.
    The rest of the combination (combine_rho, RhoResult) is identical either
    way -- this is a drop-in backend swap, not a pipeline change.
    """
    consistency = self_consistency(text, narratives, nli, exclude_index=exclude_index)
    
    if gemini_fact_extractor_client is not None:
        extracted_numbers = gemini_fact_extractor_client.extract(text)
        fact_result = score_fact_match(text, signal, extracted_numbers=extracted_numbers)
    else:
        fact_result = score_fact_match(text, signal)
        
    fact_score = _combined_fact_match_score(fact_result)
    if gemini_misweighted_client is not None:
        misweighted = gemini_misweighted_client.classify(
            text, signal["facts"], evidence_fact_ids
        )
    else:
        misweighted = is_misweighted(text, signal["facts"], evidence_fact_ids)
    rho = combine_rho(consistency, fact_score, misweighted=misweighted.misweighted)
    return RhoResult(
        rho=rho,
        self_consistency=consistency,
        fact_match_score=fact_score,
        fact_match=fact_result,
        misweighted=misweighted,
    )
