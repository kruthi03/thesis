"""Sentence segmentation and normalization for LLM narratives.

Stage 3 scores and Stage 4 calibrates at the SENTENCE level, so a bad split is
not cosmetic: a fragment like "3 a.m. vs. their norm." is not a claim anyone
can verify against facts[], and two claims fused into one sentence cannot be
accepted and rejected independently. Segmentation quality bounds the whole
guarantee.

# Segmenter choice: SpaCy transformer. A naive .split(".") is excluded by the
# brief and by the data -- narratives about this dataset are full of decimals
# (66.5 MB), times (02:14), filenames (dump.zip) and abbreviations (e.g.).
# The en_core_web_trf model uses grammatical context to solve edge cases that
# defeat rule-based segmenters.
# NOTE: Requires `python -m spacy download en_core_web_trf`.
"""
from __future__ import annotations

import math
import re
from typing import Sequence

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        from spacy.symbols import ORTH
        try:
            spacy.require_gpu()
        except Exception:
            pass
        try:
            _nlp = spacy.load("en_core_web_trf")
        except OSError as exc:
            raise RuntimeError(
                "Failed to load the SpaCy transformer model. "
                "Did you run 'python -m spacy download en_core_web_trf'?"
            ) from exc
            
        # SpaCy TRF models occasionally over-split on trailing lowercase abbreviations
        # like "no." (number). We explicitly prevent tokenization (and thus sentence
        # splitting) on these known corpus abbreviations.
        for abbrev in ("no.", "approx.", "cf.", "est.", "avg.", "max.", "ca.", "min."):
            _nlp.tokenizer.add_special_case(abbrev, [{ORTH: abbrev}])
            
    return _nlp

def split_sentences(text: str) -> list[str]:
    """Split a narrative into sentences using a SpaCy transformer model.

    Preserves the surface form of the sentences (digits, casing, etc.)
    while stripping leading/trailing whitespace.
    """
    if not text or not text.strip():
        return []

    sentences = []
    # Pre-split on newlines to ensure bullet points and distinct paragraphs 
    # are never fused into a single sentence by the transformer.
    for line in text.split("\n"):
        if not line.strip():
            continue
        doc = _get_nlp()(line)
        for sent in doc.sents:
            cleaned = str(sent).strip()
            if cleaned:
                sentences.append(cleaned)
            
    return sentences


def normalize_for_identity(sentence: str) -> str:
    """Key used to decide whether two sentences are 'the same' when pooling.

    Whitespace and casing only, per the brief. Punctuation is deliberately
    significant: "58 files were copied" and "58 files were copied?" are not
    interchangeable claims, and digits obviously must not be normalised away --
    the count IS the claim being verified.
    """
    return re.sub(r"\s+", " ", sentence).strip().lower()


class OllamaEmbedder:
    """Sentence embeddings from a local Ollama model, for semantic pooling.

    urllib rather than a new dependency, matching OllamaClient. nomic-embed-text
    is prefix-aware; "clustering: " is the prefix its authors specify for
    symmetric similarity, and it is applied to BOTH sides (every sentence is
    embedded the same way), so no query/document asymmetry is introduced.
    """

    def __init__(self, model: str = "nomic-embed-text:latest",
                 host: str = "http://localhost:11434",
                 prefix: str = "clustering: ", batch_size: int = 64,
                 timeout_s: float = 180.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.prefix = prefix
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import json
        import urllib.error
        import urllib.request

        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [self.prefix + t for t in texts[start:start + self.batch_size]]
            payload = json.dumps({"model": self.model, "input": chunk}).encode("utf-8")
            request = urllib.request.Request(
                f"{self.host}/api/embed", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Ollama embedding request failed ({self.host}): {exc}. "
                    f"Is `{self.model}` pulled?"
                ) from exc
            out.extend(body["embeddings"])
        return out


class GeminiPoolingJudge:
    """Sentence grouping via LLM-as-a-judge (Batch Prompting) for semantic pooling."""
    def __init__(self, model: str = "gemini-3.6-flash", rate_limiter=None, api_key: str | None = None) -> None:
        self.model = model
        self.rate_limiter = rate_limiter
        self.api_key = api_key

    def group_sentences(self, texts: Sequence[str]) -> list[list[int]]:
        from google import genai
        from google.genai import types
        from pydantic import BaseModel, Field
        import json

        class Cluster(BaseModel):
            sentence_indices: list[int] = Field(description="The 0-based indices of the sentences that assert the exact same facts.")

        class PoolingResponse(BaseModel):
            clusters: list[Cluster]

        if self.api_key:
            client = genai.Client(api_key=self.api_key)
        else:
            client = genai.Client()
        prompt = (
            "Here is a list of sentences generated by an AI. Group them into clusters based on whether they assert the same core facts.\n"
            "Ignore minor editorial differences (e.g., 'this is notable', 'this is the primary evidence', or mentions of triggering a specific rule) "
            "as long as the underlying hard data and actions described (e.g., '4 visits to leak-publication sites') are the same.\n"
            "Return the 0-based indices of the sentences in each cluster.\n\n"
        )
        for i, text in enumerate(texts):
            prompt += f"[{i}] {text}\n"

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PoolingResponse,
            temperature=0.0
        )

        if self.rate_limiter:
            self.rate_limiter.wait()

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        data = json.loads(response.text)
        return [cluster["sentence_indices"] for cluster in data.get("clusters", [])]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def threshold_slug(threshold: float) -> str:
    """0.85 -> 't085'. Used to scope output paths so runs at different
    thresholds cannot overwrite each other."""
    return "t" + f"{float(threshold):.2f}".replace("0.", "0").replace(".", "")


def pool_sentences_semantic(
    per_sample_sentences: Sequence[Sequence[str]],
    embedder: Any,
    *,
    threshold: float | None = None,
) -> tuple[list[dict], list[list[str]]]:
    """Pool sentences by MEANING rather than exact string match.

    Exact-string pooling assumes a model repeats a claim it is confident about
    in the same words. Measured against llama3:latest at temperature 0.7 that
    is false: 99.5% of sentences appeared in exactly 1 of 10 samples, because
    the claims recur while the wording never does ("a total of 4 such visits"
    vs "there were four visits"). That collapses n_samples to a constant and
    destroys the self-consistency signal Stage 3 and Stage 4 depend on.
    Semantic pooling restores it without touching the sampling design -- the
    fix belongs in the measurement, not in a lowered temperature.

    Two passes:
      1. Exact-duplicate collapse via normalize_for_identity (free, and keeps
         the embedding call smaller).
      2. Grouping. If embedder has `group_sentences` (LLM Judge), it groups via prompt.
         Otherwise, it does Greedy clustering by cosine similarity against each 
         cluster's REPRESENTATIVE embedding.

    Cluster order follows first appearance, so sentence ids stay stable for a
    fixed set of samples. `variants` retains every distinct surface form seen --
    Stage 3 extracts numbers and entities from surface text, and the paraphrase
    set is itself evidence about what the model treats as the same claim.
    """
    # Pass 1: exact collapse, preserving first-appearance order.
    order: list[str] = []
    exact: dict[str, dict] = {}
    for sample_index, sentences in enumerate(per_sample_sentences):
        for sentence in sentences:
            key = normalize_for_identity(sentence)
            if not key:
                continue
            record = exact.get(key)
            if record is None:
                record = {"text": sentence, "appears_in": [], "variants": [sentence]}
                exact[key] = record
                order.append(key)
            if sample_index not in record["appears_in"]:
                record["appears_in"].append(sample_index)

    if not order:
        return [], [[] for _ in per_sample_sentences]

    # Pass 2: semantic clustering over the exact-collapsed set.
    reps = [exact[k]["text"] for k in order]
    clusters: list[dict] = []
    assignment: dict[str, int] = {}

    if hasattr(embedder, "group_sentences"):
        # LLM Judge approach
        index_groups = embedder.group_sentences(reps)
        for indices in index_groups:
            keys = [order[i] for i in indices if i < len(order)]
            if keys:
                clusters.append({"keys": keys})
                for k in keys:
                    assignment[k] = len(clusters) - 1
                    
        # Ensure any sentences not clustered by the LLM get their own cluster
        for key in order:
            if key not in assignment:
                clusters.append({"keys": [key]})
                assignment[key] = len(clusters) - 1
    else:
        # Vector embedding approach
        vectors = embedder.embed(reps)
        for key, vector in zip(order, vectors):
            best_i, best_sim = None, threshold
            for i, cluster in enumerate(clusters):
                sim = _cosine(vector, cluster["vector"])
                if sim >= best_sim:
                    best_i, best_sim = i, sim
            if best_i is None:
                clusters.append({"keys": [key], "vector": vector})
                assignment[key] = len(clusters) - 1
            else:
                clusters[best_i]["keys"].append(key)
                assignment[key] = best_i

    sentences_out: list[dict] = []
    for i, cluster in enumerate(clusters, start=1):
        members = [exact[k] for k in cluster["keys"]]
        appears: list[int] = []
        variants: list[str] = []
        for m in members:
            for s in m["appears_in"]:
                if s not in appears:
                    appears.append(s)
            for v in m["variants"]:
                if v not in variants:
                    variants.append(v)
        sentences_out.append({
            "sentence_id": f"s{i:03d}",
            # First surface form seen stays the representative.
            "text": members[0]["text"],
            "variants": variants,
            "n_variants": len(variants),
            "appears_in": sorted(appears),
            "n_samples": len(appears),
        })

    ids_per_sample: list[list[str]] = []
    for sample_index, sentences in enumerate(per_sample_sentences):
        seen: list[str] = []
        for sentence in sentences:
            key = normalize_for_identity(sentence)
            if not key or key not in assignment:
                continue
            sid = f"s{assignment[key] + 1:03d}"
            if sid not in seen:
                seen.append(sid)
        ids_per_sample.append(seen)

    return sentences_out, ids_per_sample


def pool_sentences(
    per_sample_sentences: Sequence[Sequence[str]],
) -> tuple[list[dict], list[list[str]]]:
    """Pool sentences across the k samples of one signal.

    Returns (sentences, sentence_ids_per_sample):

      sentences  -- one record per unique sentence, with the sample indices it
                    appeared in. n_samples / k is the raw self-consistency
                    signal Stage 3 builds on.
      sentence_ids_per_sample -- the ids each sample contributed, in order, so
                    a narrative can be reconstructed and cited.

    Ids are assigned in order of first appearance and are therefore stable for
    a fixed set of samples, which the reproducibility requirement depends on.
    A sentence repeated within one sample is recorded once for that sample:
    appears_in counts how many of the k samples asserted it, so a model that
    repeats itself must not inflate its own confidence score.
    """
    order: list[str] = []
    by_key: dict[str, dict] = {}
    ids_per_sample: list[list[str]] = []

    for sample_index, sentences in enumerate(per_sample_sentences):
        this_sample: list[str] = []
        for sentence in sentences:
            key = normalize_for_identity(sentence)
            if not key:
                continue
            record = by_key.get(key)
            if record is None:
                # Keep the FIRST surface form seen; normalization is only an
                # identity key, never a replacement for the original text.
                order.append(key)
                record = {
                    "sentence_id": f"s{len(order):03d}",
                    "text": sentence,
                    "appears_in": [],
                }
                by_key[key] = record
            if sample_index not in record["appears_in"]:
                record["appears_in"].append(sample_index)
            if record["sentence_id"] not in this_sample:
                this_sample.append(record["sentence_id"])
        ids_per_sample.append(this_sample)

    sentences_out = [{
        "sentence_id": by_key[key]["sentence_id"],
        "text": by_key[key]["text"],
        "appears_in": by_key[key]["appears_in"],
        "n_samples": len(by_key[key]["appears_in"]),
    } for key in order]
    return sentences_out, ids_per_sample
