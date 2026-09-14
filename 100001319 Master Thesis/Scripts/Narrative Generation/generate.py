"""Stage 2: narrative generation.

Prompt construction, k-sampling, and caching. No provider SDK is imported here
-- the LLM is behind the small LLMClient interface below, so the whole stage
runs and is tested offline against StubLLMClient.

The k samples are the uncertainty estimate, not redundancy: a sentence in 9 of
10 samples is one the model is confident about, a sentence in 1 is where
hallucinations concentrate. Temperature must be > 0 or there is no
nonconformity signal and no conformal method (ConU; Conformal Language
Modeling).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

class DraftResponse(BaseModel):
    drafts: list[str]

from .sentences import (
    pool_sentences, pool_sentences_semantic, split_sentences, threshold_slug,
)

# The SYSTEM text below is byte-identical across v1, v2 and v3 -- only the
# rendered context changed (signals.signal_to_prompt_context), which isolates
# the rendering as the single variable in every comparison:
#
#   v1 -> v2: "max: N" relabelled to "highest in the prior 12 weeks", numeric z
#             suppressed in favour of a qualitative phrase, peer group size
#             rendered as "across N peers in this role".
#   v2 -> v3: the 12-week high dropped from the render entirely. v2 zeroed the
#             z and peer-denominator errors but only halved false exceedance
#             claims (11 -> 5); see the comment at the render site for why no
#             wording fixes the residue.
#
# The version must change even though PROMPT_V1's text did not, because it is
# what the cache keys on: the prompt AS SENT includes the rendered context, so
# leaving it unchanged would serve v2 narratives for a v3 prompt.
PROMPT_VERSION = "v7"

# Versioned and kept in one place: every narrative records which prompt
# produced it, and results are not reproducible if this text drifts silently.

PROMPT_V4 = """\
You are a security analyst writing a short explanation of an anomaly that a \
UEBA system has flagged for review. Another analyst will read it to decide \
whether to investigate further.

Write {k} different ways to explain this, ordered by likelihood. Each explanation \
should be 3-5 sentences of plain prose. No headings, no bullet points, no \
preamble -- just the explanation.

You must respond with a JSON object containing a single key "drafts" which is a list of your {k} explanations.

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
- Always refer to the user by their User ID. Do not use their employee name \
or pronouns (he, she, they).
- Make every single sentence fully self-contained. Do not use reference \
phrases like 'this finding' or 'these visits'. Always explicitly repeat the \
User ID and the specific action in every sentence so that the sentence makes \
complete sense in isolation.
- If the evidence is thin, say so. An honest "this looks minor" is more \
useful than a confident story.
"""


@dataclass
class GenerationConfig:
    model: str = "gemini-3.7-flash"
    k: int = 10
    # docs/stage_2.md mandates temperature > 0, because at temperature 0 every
    # sample is identical, appears_in is k for everything, and the
    # self-consistency signal carries no information.
    #
    # But temperature/top_p/top_k were REMOVED from the current Claude models
    # (claude-opus-5, opus-4.8/4.7; non-default values rejected on sonnet-5) --
    # sending one returns a 400. The requirement the brief actually depends on
    # is *sampling diversity across the k draws*, and that still holds: these
    # models sample stochastically by default, which is why the k narratives
    # differ at all. What is lost is the ability to SET or REPORT a specific
    # value, so `temperature` here is a record of intent, and
    # temperature_supported says whether it reached the API. AnthropicClient
    # omits it from the request; the thesis must say temperature was not a
    # settable parameter on this model rather than cite 0.7.
    temperature: float = 0.7
    temperature_supported: bool = False
    prompt_version: str = PROMPT_VERSION
    seed: int | None = None
    max_retries: int = 3
    retry_backoff_s: float = 1.0

    # Pooling is a first-class, threaded parameter -- NOT a module constant.
    # n_samples is a function of this threshold (LIMITATIONS.md finding 9), so
    # any downstream guarantee inherits it and every artifact must record it.
    # There is deliberately no default threshold: a stage that silently falls
    # back to one would make a single value the de-facto frozen choice, which
    # is exactly the corner this design avoids. Semantic pooling without an
    # explicit threshold is an error, not a defaulted call.
    pooling_method: str = "exact"          # "exact" | "semantic"
    pooling_threshold: float | None = None
    embed_model: str | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0 for k-sampling to produce a "
                f"nonconformity signal; got {self.temperature}"
            )
        if self.k < 1:
            raise ValueError(f"k must be >= 1; got {self.k}")
        if self.pooling_method not in ("exact", "semantic"):
            raise ValueError(f"unknown pooling_method: {self.pooling_method}")
        if self.pooling_method == "semantic" and self.pooling_threshold is None:
            raise ValueError(
                "pooling_method='semantic' requires an explicit "
                "pooling_threshold -- it must be passed down, never defaulted"
            )
        if self.pooling_method == "exact" and self.pooling_threshold is not None:
            raise ValueError(
                "pooling_threshold is meaningless with pooling_method='exact'; "
                "set pooling_method='semantic' or drop the threshold"
            )

    def pooling_metadata(self) -> dict[str, Any]:
        """The block every artifact must carry so its threshold is recoverable
        from the file itself."""
        return {
            "method": self.pooling_method,
            "threshold": self.pooling_threshold,
            "embed_model": self.embed_model,
        }


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal surface a provider must satisfy.

    Strings in, string out -- deliberately narrow so a real client cannot come
    to depend on signal internals, and so the stub is a faithful substitute.
    """

    name: str

    def complete(self, system: str, user: str, temperature: float,
                 seed: int | None = None) -> list[str]: ...


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(signal: dict[str, Any], k: int) -> tuple[str, str]:
    """(system, user) for one signal.

    The user message is signal_to_prompt_context() verbatim. The evidence /
    context split is NOT reconstructed here -- the signal layer already renders
    it, and duplicating that logic would let the two drift apart.
    """
    from .signals import signal_to_prompt_context
    return PROMPT_V4.format(k=k), signal_to_prompt_context(signal)


def evidence_fact_ids(signal: dict[str, Any]) -> list[str]:
    return [f["fact_id"] for f in signal["facts"] if f.get("triggering_rule_ids")]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class NarrativeCache:
    """On-disk cache keyed on (signal_id, model, prompt_version, temperature,
    sample_index).

    Mandatory, not an optimisation: generation costs money and takes hours, and
    a crash must not repeat completed work -- same reasoning as the
    checkpointing in cert_features.py. Written via a temp file + os.replace so
    a crash mid-write cannot leave a half-parsed entry behind.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(signal_id: str, model: str, prompt_version: str,
                 temperature: float) -> tuple:
        # Temperature is formatted, not raw: 0.7 and 0.7000000000000001 are the
        # same request and must not produce two cache entries.
        return (signal_id, model, prompt_version, f"{float(temperature):.4f}")

    def _path(self, key: tuple) -> Path:
        digest = hashlib.sha1("|".join(str(part) for part in key).encode()).hexdigest()
        # Shard by prefix: one flat directory of ~80k files per run is slow to
        # list and unpleasant on Windows.
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: tuple) -> list[str] | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt entry is a miss, not a crash -- regenerate it.
            self.misses += 1
            return None
        self.hits += 1
        return payload["text"]

    def put(self, key: tuple, text: list[str], meta: dict | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": list(key), "text": text, "meta": meta or {}}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _complete_with_retry(client: LLMClient, system: str, user: str,
                         cfg: GenerationConfig, seed: int | None) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            return client.complete(system, user, cfg.temperature, seed)
        except Exception as exc:  # provider errors are transient often enough
            last_error = exc
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_backoff_s * (2 ** attempt))
    raise RuntimeError(
        f"LLM call failed after {cfg.max_retries} attempts: {last_error}"
    ) from last_error


def generate_for_signal(signal: dict[str, Any], client: LLMClient,
                        cfg: GenerationConfig,
                        cache: NarrativeCache | None = None,
                        embedder: Any = None) -> dict[str, Any]:
    """Sample k narratives for one signal and pool their sentences.

    Pooling follows cfg.pooling_method; semantic pooling requires both an
    embedder and cfg.pooling_threshold, and raises rather than falling back.
    """
    system, user = build_prompt(signal, cfg.k)
    signal_id = signal["signal_id"]

    key = NarrativeCache.make_key(signal_id, cfg.model, cfg.prompt_version, cfg.temperature)
    texts = cache.get(key) if cache is not None else None
    
    if texts is None:
        texts = _complete_with_retry(client, system, user, cfg, cfg.seed)
        if cache is not None:
            cache.put(key, texts, meta={"seed": cfg.seed, "signal_id": signal_id})

    # The model may return fewer or more drafts than requested
    if len(texts) > cfg.k:
        texts = texts[:cfg.k]
        
    actual_k = len(texts)

    per_sample = [split_sentences(t) for t in texts]
    if cfg.pooling_method == "semantic":
        if embedder is None:
            raise ValueError(
                "pooling_method='semantic' requires an embedder; refusing to "
                "silently fall back to exact pooling, which would change what "
                "n_samples means without recording it"
            )
        sentences, ids_per_sample = pool_sentences_semantic(
            per_sample, embedder, threshold=cfg.pooling_threshold)
    else:
        sentences, ids_per_sample = pool_sentences(per_sample)

    return {
        "signal_id": signal_id,
        "model": cfg.model,
        "prompt_version": cfg.prompt_version,
        "temperature": cfg.temperature,
        # The threshold that produced this record, recoverable from the file
        # itself -- an artifact whose pooling cannot be identified is a bug.
        "pooling": cfg.pooling_metadata(),
        # False means the value above was NOT sent to the API and did not shape
        # sampling -- see GenerationConfig. Recorded per narrative so a later
        # reader cannot mistake the requested value for an applied one.
        "temperature_supported": cfg.temperature_supported,
        "k": actual_k,
        "seed": cfg.seed,
        "evidence_fact_ids": evidence_fact_ids(signal),
        "narratives": [
            {"sample_index": i, "text": texts[i], "sentence_ids": ids_per_sample[i]}
            for i in range(actual_k)
        ],
        "sentences": sentences,
    }


def write_narratives(records: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Real client for Google Gemini models.
    
    Lazy import of google.genai so the offline suite doesn't need it.
    """
    
    def __init__(self, model: str = "gemini-3.7-flash", max_tokens: int = 2048,
                 effort: str = "medium", rate_limiter: Any = None, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini generator; "
                "install it with: pip install google-genai"
            ) from exc

        if api_key is None:
            # Ensure .env is loaded if it exists, so local test runs pick up GEMINI_API_KEY
            try:
                import dotenv
                dotenv.load_dotenv()
            except ImportError:
                pass
                
            import os
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError(
                "No API key provided and GEMINI_API_KEY environment variable is not set."
            )
            
        self.name = model
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort  # Unused by Gemini currently, kept for compatibility
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.rate_limiter = rate_limiter

    def complete(self, system: str, user: str, temperature: float,
                 seed: int | None = None) -> list[str]:
        
        kwargs = {
            "temperature": temperature,
            "max_output_tokens": self.max_tokens,
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": DraftResponse,
        }
        
        config = self._types.GenerateContentConfig(**kwargs)
        
        if self.rate_limiter:
            self.rate_limiter.wait()
            
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=config,
        )
        
        if not response.text:
            raise RuntimeError("Gemini returned an empty narrative. Check safety block reasons.")
            
        import json
        text = response.text.strip()
        if text.startswith("```"):
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}\nText: {response.text}")
            
        return data["drafts"]



