"""CLI: signals.jsonl -> narratives.jsonl.

    python scripts/generate_narratives.py --signals out_full/signals.jsonl \
        --out out_narratives/ --n-signals 20

Caching is keyed on (signal_id, model, prompt_version, temperature,
sample_index), so a re-run costs nothing for work already done and a crash
mid-run resumes rather than restarting.

--stub uses the offline stub client (no API calls, no key) for wiring checks.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate import (  # noqa: E402
    GenerationConfig, NarrativeCache, generate_for_signal,
    write_narratives,
)
from src.sentences import threshold_slug  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--signals", required=True, help="path to signals.jsonl")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--k", type=int, default=10, help="samples per signal")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="recorded for provenance; NOT sent to models that "
                       "reject sampling parameters (claude-opus-5 and family)")
    p.add_argument("--effort", default="medium",
                   choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--n-signals", type=int, default=None,
                   help="how many signals to sample from the corpus")
    p.add_argument("--sampling", default="stratified", choices=["stratified", "uniform"],
                   help="'stratified' (default): malicious-first + round-robin "
                       "across rule combos, for annotation-guideline coverage. "
                       "'uniform': plain random sample, NOT stratified toward "
                       "malicious or any rule -- for a calibration batch that "
                       "must reflect the corpus's real distribution")
    p.add_argument("--exclude-narratives", default=None,
                   help="path to an existing narratives.jsonl whose signal_ids "
                       "are excluded from the sampling pool -- use when adding "
                       "a batch to an already-generated, frozen set so no "
                       "signal is generated twice")
    p.add_argument("--provider", default="gemini", choices=["gemini"],
                   help="model provider to use (default: gemini)")
    p.add_argument("--api-keys", default=None,
                   help="comma-separated list of API keys for concurrent execution")
    p.add_argument("--threads", type=int, default=10,
                   help="number of worker threads (default 10)")
    p.add_argument("--rpm", type=int, default=1000,
                   help="rate limit per minute for the API key (default 1000)")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="host for the embedder (used if --pooling semantic)")

    # Pooling. 0.85 is the FROZEN threshold, chosen from the measured sweep in
    # LIMITATIONS.md finding 9 -- not a guess, and not a value that drifted in
    # as a convenient default:
    #
    #   0.80 merges evidence and context claims into one cluster. In the
    #        ACM2278 2010-W33 leak-site cluster, variant 14 ("job-search site
    #        visits remained consistent with their typical activity...") is a
    #        CONTEXT statement pooled into the evidence cluster, inheriting
    #        n_samples=10/10. That would undo at the measurement layer the
    #        evidence/context separation Stage 1 built and which finding 8
    #        records the prompt as having preserved.
    #   0.90 fragments the dominant claim across three clusters (3/10, 3/10,
    #        2/10) with nothing reaching majority, losing the consistency
    #        gradient that Stage 3 and Stage 4 consume.
    #
    # This is the ONLY place the value is defaulted. Library code
    # (pool_sentences_semantic, GenerationConfig) still requires it explicitly,
    # so no stage can silently fall back to it.
    p.add_argument("--pooling", default="semantic", choices=["exact", "semantic"])
    p.add_argument("--threshold", type=float, default=0.85,
                   help="cosine threshold for semantic pooling (frozen default: "
                       "0.85; see LIMITATIONS.md finding 9). Ignored with "
                       "--pooling exact")
    p.add_argument("--embed-model", default="nomic-embed-text:latest")
    p.add_argument("--seed", type=int, default=20260816,
                   help="seeds signal selection (and per-sample seeds if the "
                       "provider honours them), so a run is reproducible")
    args = p.parse_args(argv)

    if args.pooling == "exact":
        if "--threshold" in (argv if argv is not None else sys.argv[1:]):
            print("NOTE: --threshold is meaningless with --pooling exact; ignoring.",
                  file=sys.stderr)
        args.threshold = None
    return args


def uniform_random_signals(signals: list[dict], n: int | None, seed: int) -> list[dict]:
    """Sample n signals uniformly at random -- NOT stratified toward
    malicious or any rule combination, unlike select_signals().

    Exists as a separate function rather than a branch bolted onto
    select_signals(): the two have different, non-interchangeable
    justifications (select_signals ensures failure-mode coverage for writing
    the annotation guideline; this one is for a calibration-set batch that
    must reflect the corpus's real label/rule distribution un-distorted, so
    the two must never be silently swapped for each other).
    """
    if n is None or n >= len(signals):
        return signals
    rng = random.Random(seed)
    return rng.sample(signals, n)


def select_signals(signals: list[dict], n: int | None, seed: int) -> list[dict]:
    """Sample n signals spanning malicious/benign and different rules.

    Stratified rather than random: the annotation guideline is written from
    what these narratives contain, so a sample that happened to be all-R4 or
    all-benign would hide failure modes the later stages must handle. Malicious
    signals are taken first because there are only a handful in the corpus.
    """
    if n is None or n >= len(signals):
        return signals

    rng = random.Random(seed)
    malicious = [s for s in signals if s["label"]["is_malicious"]]
    benign = [s for s in signals if not s["label"]["is_malicious"]]

    chosen = malicious[:n]
    if len(chosen) >= n:
        return chosen[:n]

    # Fill the rest round-robin across rule combinations so no single rule
    # dominates the sample.
    by_rules: dict[tuple, list[dict]] = {}
    for s in benign:
        key = tuple(sorted(r["rule_id"] for r in s["triggered_rules"]))
        by_rules.setdefault(key, []).append(s)
    for bucket in by_rules.values():
        rng.shuffle(bucket)

    keys = sorted(by_rules)
    i = 0
    while len(chosen) < n and any(by_rules[k] for k in keys):
        bucket = by_rules[keys[i % len(keys)]]
        if bucket:
            chosen.append(bucket.pop())
        i += 1
    return chosen[:n]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Threshold-scoped output: two thresholds must never write to one path.
    # The cache sits ABOVE the scope on purpose -- raw narrative text does not
    # depend on pooling, so re-pooling at a new threshold costs no generation.
    root = Path(args.out)
    out_dir = (root / threshold_slug(args.threshold) if args.pooling == "semantic"
               else root / "exact")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache"

    with open(args.signals, encoding="utf-8") as fh:
        signals = [json.loads(line) for line in fh]

    if args.exclude_narratives:
        with open(args.exclude_narratives, encoding="utf-8") as fh:
            exclude_ids = {json.loads(line)["signal_id"] for line in fh}
        before = len(signals)
        signals = [s for s in signals if s["signal_id"] not in exclude_ids]
        print(f"excluded {before - len(signals)} already-generated signals "
              f"(from {args.exclude_narratives})", file=sys.stderr)

    selected = (
        uniform_random_signals(signals, args.n_signals, args.seed)
        if args.sampling == "uniform"
        else select_signals(signals, args.n_signals, args.seed)
    )

    import queue
    import concurrent.futures
    from src.generate import GeminiClient
    from src.utils import RateLimiter
    
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
    import os

    if args.api_keys:
        keys = [k.strip() for k in args.api_keys.split(',')]
    else:
        # Collect GEMINI_API_KEY, GEMINI_API_KEY1, GEMINI_API_KEY2, etc.
        keys = [v.strip().strip('"').strip("'") for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
        if not keys:
            keys = [None]
    worker_queue = queue.Queue()
    
    key_rate_limiters = {k: RateLimiter(max_calls=args.rpm, period_seconds=60) for k in keys}
    
    worker_queue = queue.Queue()
    for i in range(args.threads):
        key = keys[i % len(keys)]
        if key: key = key.strip()
        rl = key_rate_limiters[key]
        c = GeminiClient(model=args.model, max_tokens=args.max_tokens, effort=args.effort, rate_limiter=rl, api_key=key)
        
        e = None
        if args.pooling == "semantic":
            if args.embed_model.startswith("gemini"):
                from src.sentences import GeminiPoolingJudge
                e = GeminiPoolingJudge(model=args.embed_model, rate_limiter=rl, api_key=key)
            else:
                from src.sentences import OllamaEmbedder
                e = OllamaEmbedder(model=args.embed_model, host=args.ollama_host)
        worker_queue.put((c, e))

    model_name = args.model
    temperature_supported = True

    cfg = GenerationConfig(
        model=model_name, k=args.k, temperature=args.temperature,
        temperature_supported=temperature_supported, seed=args.seed,
        pooling_method=args.pooling, pooling_threshold=args.threshold,
        embed_model=args.embed_model if args.pooling == "semantic" else None,
    )
    cache = NarrativeCache(cache_dir)

    print(f"signals: {len(selected)} of {len(signals)}  model: {cfg.model}  "
          f"k: {cfg.k}  effort: {args.effort}", file=sys.stderr)
    print(f"pooling: {cfg.pooling_method}"
          + (f" @ cosine>={cfg.pooling_threshold} ({cfg.embed_model})"
             if cfg.pooling_method == "semantic" else "")
          + f"   -> {out_dir}", file=sys.stderr)

    records, failures = [], []
    
    def process_signal(args_tuple):
        idx, signal = args_tuple
        c, e = worker_queue.get()
        try:
            label = "MAL" if signal["label"]["is_malicious"] else "ben"
            rules = ",".join(r["rule_id"] for r in signal["triggered_rules"])
            print(f"  [{idx:>2}/{len(selected)}] {label} {signal['user']:9s} "
                  f"{signal['period']:9s} {rules}", file=sys.stderr, flush=True)
            
            record = generate_for_signal(signal, c, cfg, cache, embedder=e)
            return ("success", record)
        except Exception as exc:
            print(f"        FAILED [{signal['signal_id']}]: {exc}", file=sys.stderr)
            return ("error", {"signal_id": signal["signal_id"], "error": str(exc)})
        finally:
            worker_queue.put((c, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(process_signal, enumerate(selected, 1)))

    for status, result in results:
        if status == "success":
            records.append(result)
        else:
            failures.append(result)

    path = out_dir / "narratives.jsonl"
    if not records:
        # A run where every signal failed must not overwrite a good artifact.
        # This is not hypothetical: an Ollama run lost its embedding model
        # mid-batch, every signal failed at the pooling step, and the empty
        # write replaced 15 already-generated records. The generations
        # themselves survived only because the cache sits above the output.
        print(f"\nALL {len(selected)} signals failed -- refusing to overwrite "
              f"{path} with an empty file. Nothing written.", file=sys.stderr)
        for f in failures[:3]:
            print(f"  {f['signal_id']}: {f['error']}", file=sys.stderr)
        raise SystemExit(1)
    write_narratives(records, path)

    n_sentences = sum(len(r["sentences"]) for r in records)
    summary = {
        "model": cfg.model,
        "prompt_version": cfg.prompt_version,
        "k": cfg.k,
        "temperature": cfg.temperature,
        "temperature_supported": cfg.temperature_supported,
        "pooling": cfg.pooling_metadata(),
        "seed": cfg.seed,
        "sampling": args.sampling,
        "excluded_narratives_source": args.exclude_narratives,
        "effort": args.effort,
        "signals_requested": len(selected),
        "signals_succeeded": len(records),
        "malicious_signals": sum(1 for r in records
                                 if any(s["signal_id"] == r["signal_id"]
                                        and s["label"]["is_malicious"]
                                        for s in selected)),
        "unique_sentences": n_sentences,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "failures": failures,
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nwrote {path}", file=sys.stderr)
    print(f"  {len(records)} signals, {n_sentences} unique sentences, "
          f"cache {cache.hits} hit / {cache.misses} miss", file=sys.stderr)
    if failures:
        print(f"  {len(failures)} FAILED (see generation_summary.json)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
