"""CLI: Batch Judge (Scoring) for narratives.

python scripts/batch_score.py --prepare --narratives out_1000_narratives/t085/narratives.jsonl --signals out_full/1000_signals.jsonl
python scripts/batch_score.py --submit
python scripts/batch_score.py --download --job-name batchJobs/456

This script generates Batch API requests for all NLI entailment pairs and Misweighted checks.
When downloaded, it populates the local Gemini cache. Then, you can run `score_narratives.py`
and it will complete instantly with 100% cache hits.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import uuid
from pathlib import Path
import dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import GeminiNLIClient, GeminiFactExtractorClient
from src.admission import GeminiMisweightedClient
from src.sentences import split_sentences

try:
    from google import genai
except ImportError:
    pass


def _load_signals(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            sig = json.loads(line)
            out[sig["signal_id"]] = sig
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true", help="Prepare the batch requests file")
    p.add_argument("--submit", action="store_true", help="Upload file and submit batch job")
    p.add_argument("--download", action="store_true", help="Download and process a completed batch job")
    p.add_argument("--manage-queue", action="store_true", help="Run the queue manager to submit/download chunks sequentially")
    
    p.add_argument("--narratives", help="path to narratives.jsonl (required for --prepare)")
    p.add_argument("--signals", help="path to signals.jsonl (required for --prepare)")
    p.add_argument("--out", default="out_batch_score", help="output directory")
    p.add_argument("--model", default="gemini-3.1-flash-lite")
    p.add_argument("--job-name", help="Job name (e.g. batchJobs/456) required for --download")
    return p.parse_args(argv)


def get_client() -> 'genai.Client':
    dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY") and v.strip():
                api_key = v.strip().strip('"').strip("'")
                break
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found.")
    return genai.Client(api_key=api_key)


def do_prepare(args: argparse.Namespace, out_dir: Path) -> None:
    if not args.narratives or not args.signals:
        print("ERROR: --narratives and --signals required for --prepare")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    req_file = out_dir / "score_requests.jsonl"
    keys_file = out_dir / "score_keys.jsonl"
    
    # Instantiate clients just to access their prompt/cache key builders
    # They check for GEMINI_API_KEY inside, so make sure it's loaded
    get_client() # Loads env vars
    nli_client = GeminiNLIClient(args.model)
    misweighted_client = GeminiMisweightedClient(args.model)
    fact_client = GeminiFactExtractorClient(args.model)
    
    # FAST CACHE LOOKUP
    import hashlib
    print("Loading cache directory index into memory...")
    cache_dir = Path.home() / ".cache" / "gemini_judge"
    if cache_dir.exists():
        cache_set = set(os.listdir(cache_dir))
    else:
        cache_set = set()
    print(f"Loaded {len(cache_set)} items from cache.")
    
    signals_by_id = _load_signals(Path(args.signals))
    
    req_count = 0
    cache_hits = 0
    
    print("Scanning narratives for scoring requests...")
    with open(args.narratives, encoding="utf-8") as fin, \
         open(req_file, "w", encoding="utf-8") as freq, \
         open(keys_file, "w", encoding="utf-8") as fkeys:
         
        lines = fin.readlines()
        for line in tqdm(lines):
            record = json.loads(line)
            signal_id = record["signal_id"]
            if signal_id not in signals_by_id:
                continue
                
            signal = signals_by_id[signal_id]
            facts = signal.get("facts", [])
            evidence_fact_ids = record.get("evidence_fact_ids", [])
            draft_texts = [n["text"] for n in record["narratives"]]
            
            # BYPASS SPACY: Build a map of full text -> list of sentence texts
            sentences_by_id = {s["sentence_id"]: s["text"] for s in record["sentences"]}
            draft_sentences_cache = []
            for n in record["narratives"]:
                draft_sentences_cache.append([sentences_by_id[sid] for sid in n["sentence_ids"]])
            
            for sentence in record["sentences"]:
                hypothesis = sentence["text"]
                
                # 1. NLI requests
                # We need to evaluate NLI for ALL sentences in ALL OTHER drafts for this hypothesis!
                # Wait! self_consistency uses exclude_index=None for semantic clusters,
                # meaning it compares against ALL drafts!
                # others = [n for j, n in enumerate(narratives) if j != exclude_index]
                # exclude_index is None here, so others = narratives
                
                for draft_sentences in draft_sentences_cache:
                    for premise in draft_sentences:
                        cache_key = nli_client._cache_key(premise, hypothesis)
                        key_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + ".json"
                        if key_hash in cache_set:
                            cache_hits += 1
                            continue
                            
                        prompt = nli_client._build_prompt(premise, hypothesis)
                        req_id = str(uuid.uuid4())
                        req = {
                            "request_id": req_id,
                            "request": {
                                "model": f"models/{args.model}",
                                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                                "generationConfig": {
                                    "temperature": 0.0,
                                    "responseMimeType": "application/json",
                                    "responseSchema": nli_client._NLI_RESPONSE_SCHEMA
                                }
                            }
                        }
                        freq.write(json.dumps(req) + "\n")
                        fkeys.write(json.dumps({
                            "request_id": req_id,
                            "type": "nli",
                            "cache_key": cache_key
                        }) + "\n")
                        req_count += 1
                        
                # 2. Misweighted request
                cache_key = misweighted_client._cache_key(hypothesis, evidence_fact_ids, facts)
                key_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + ".json"
                if key_hash in cache_set:
                    cache_hits += 1
                    continue
                    
                evidence_ids = set(evidence_fact_ids)
                evidence_glosses = [f["gloss"] for f in facts if f["fact_id"] in evidence_ids]
                context_glosses = [f["gloss"] for f in facts if f["fact_id"] not in evidence_ids]
                
                prompt = misweighted_client._build_misweighted_prompt(hypothesis, evidence_glosses, context_glosses)
                req_id = str(uuid.uuid4())
                req = {
                    "request_id": req_id,
                    "request": {
                        "model": f"models/{args.model}",
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "responseMimeType": "application/json",
                            "responseSchema": misweighted_client._MISWEIGHTED_RESPONSE_SCHEMA
                        }
                    }
                }
                freq.write(json.dumps(req) + "\n")
                fkeys.write(json.dumps({
                    "request_id": req_id,
                    "type": "misweighted",
                    "cache_key": cache_key
                }) + "\n")
                req_count += 1
                
                # 3. Fact extraction request
                cache_key = fact_client._cache_key(hypothesis)
                key_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest() + ".json"
                if key_hash in cache_set:
                    cache_hits += 1
                    continue
                    
                prompt = fact_client._build_prompt(hypothesis)
                req_id = str(uuid.uuid4())
                req = {
                    "request_id": req_id,
                    "request": {
                        "model": f"models/{args.model}",
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "responseMimeType": "application/json",
                            "responseSchema": fact_client._FACT_RESPONSE_SCHEMA
                        }
                    }
                }
                freq.write(json.dumps(req) + "\n")
                fkeys.write(json.dumps({
                    "request_id": req_id,
                    "type": "fact_extract",
                    "cache_key": cache_key
                }) + "\n")
                req_count += 1
                
    print(f"Prepared {req_count} requests in {req_file}")
    print(f"Skipped {cache_hits} already cached.")


def do_manage_queue(args: argparse.Namespace, out_dir: Path) -> None:
    client = get_client()
    state_file = out_dir / "score_job.txt"
    active_job = None
    
    if state_file.exists():
        lines = [l.strip() for l in state_file.read_text().splitlines() if l.strip()]
        if lines:
            active_job = lines[0]
            
    # 1. Check active job
    if active_job:
        print(f"Active job found: {active_job}. Checking status...")
        try:
            job = client.batches.get(name=active_job)
            print(f"Job state: {job.state}")
            
            state_str = str(job.state)
            if "RUNNING" in state_str or "ENQUEUED" in state_str or "PENDING" in state_str or "INITIALIZING" in state_str:
                print("Job is still running. Will check again later.")
                sys.exit(0)
            elif "SUCCEEDED" in state_str:
                print("Job SUCCEEDED! Triggering download and process...")
                # Download and process the results
                file_name = job.dest.file_name if job.dest else None
                if not file_name:
                    print("ERROR: No output file name found in batch job.")
                    sys.exit(1)
                
                output_data = client.files.download(file=file_name)
                temp_out = out_dir / f"score_raw_responses_{active_job.split('/')[-1]}.jsonl"
                temp_out.write_bytes(output_data)
                print(f"Saved raw responses to {temp_out}")
                
                # Load keys
                keys_by_id = {}
                with open(out_dir / "score_keys.jsonl", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        keys_by_id[r["request_id"]] = r
                
                nli_client = GeminiNLIClient(args.model)
                misweighted_client = GeminiMisweightedClient(args.model)
                fact_client = GeminiFactExtractorClient(args.model)
                
                count = 0
                with open(temp_out, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if not line.strip(): continue
                        resp = json.loads(line)
                        req_id = resp.get("request_id")
                        try:
                            content = resp["response"]["candidates"][0]["content"]["parts"][0]["text"]
                            data = json.loads(content)
                        except Exception as e:
                            continue
                            
                        key_info = keys_by_id.get(req_id)
                        if not key_info: continue
                            
                        req_type, cache_key = key_info["type"], key_info["cache_key"]
                        if req_type == "nli":
                            label = data.get("label", "neutral").lower()
                            if label not in ("entailment", "neutral", "contradiction"): label = "neutral"
                            nli_client._cache_put(cache_key, {"label": label, "reason": data.get("reason", "")})
                            count += 1
                        elif req_type == "misweighted":
                            misweighted_client._cache_put(cache_key, {"misweighted": bool(data.get("misweighted", False)), "reason": data.get("reason", "")})
                            count += 1
                        elif req_type == "fact_extract":
                            try:
                                numbers = [float(x) for x in data.get("numbers", [])]
                            except Exception: numbers = []
                            fact_client._cache_put(cache_key, {"facts": data.get("facts", []), "numbers": numbers})
                            count += 1
                
                print(f"Successfully processed and cached {count} responses!")
                # Clear active job
                state_file.write_text("")
                active_job = None
            else:
                print(f"Job is in an unexpected state: {job.state}. Please investigate.")
                sys.exit(1)
        except Exception as e:
            print(f"Error checking job status: {e}")
            sys.exit(1)

    # 2. Submit next chunk if no active job
    if not active_job:
        print("No active job. Looking for next available chunk...")
        import glob
        pending_chunks = sorted(glob.glob(str(out_dir / "score_requests_chunk_*.jsonl")))
        
        if not pending_chunks:
            print("No pending chunks found! All chunks have been processed.")
            sys.exit(0)
            
        next_chunk = Path(pending_chunks[0])
        print(f"Submitting {next_chunk.name}...")
        
        file_obj = client.files.upload(
            file=str(next_chunk),
            config={"mime_type": "application/jsonl"}
        )
        print(f"Uploaded as: {file_obj.name}")
        
        job = client.batches.create(
            model=args.model,
            src=file_obj.name
        )
        print(f"Batch job created! JOB NAME: {job.name}")
        
        state_file.write_text(job.name + "\n")
        
        # Move processed chunk to avoid resubmitting
        completed_dir = out_dir / "completed_chunks"
        completed_dir.mkdir(exist_ok=True)
        next_chunk.rename(completed_dir / next_chunk.name)
        print(f"Moved {next_chunk.name} to {completed_dir.name}/")


def do_submit(args: argparse.Namespace, out_dir: Path) -> None:
    req_file = out_dir / "score_requests.jsonl"
    if not req_file.exists():
        print(f"ERROR: {req_file} not found. Run --prepare first.")
        sys.exit(1)
        
    client = get_client()
    state_file = out_dir / "score_job.txt"
    
    # We already split chunks and submitted chunk 0 in the previous run.
    # We won't re-split if chunks already exist.
    import glob
    existing = glob.glob(str(out_dir / "score_requests_chunk_*.jsonl"))
    if not existing:
        CHUNK_SIZE = 40000
        with open(req_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        total_chunks = (len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"Splitting {len(lines)} requests into {total_chunks} chunks of max {CHUNK_SIZE}...")
        
        for i in range(total_chunks):
            chunk_lines = lines[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            chunk_file = out_dir / f"score_requests_chunk_{i}.jsonl"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.writelines(chunk_lines)
    else:
        print(f"Found {len(existing)} chunks already split.")
        
    print("\nChunks are ready. Please use --manage-queue to process them sequentially.")


def do_download(args: argparse.Namespace, out_dir: Path) -> None:
    job_names = []
    if args.job_name:
        job_names = [args.job_name]
    else:
        state_file = out_dir / "score_job.txt"
        if state_file.exists():
            job_names = [line.strip() for line in state_file.read_text().splitlines() if line.strip()]
        else:
            print("ERROR: --job-name required or score_job.txt must exist.")
            sys.exit(1)
            
    if not job_names:
        print("ERROR: No jobs found in score_job.txt.")
        sys.exit(1)
            
    client = get_client()
    
    print(f"Checking status for {len(job_names)} jobs...")
    jobs_info = []
    for job_name in job_names:
        job = client.batches.get(name=job_name)
        jobs_info.append(job)
        if "SUCCEEDED" not in str(job.state):
            print(f"Job {job_name} is in state: {job.state}. Wait for all jobs to SUCCEED.")
            sys.exit(1)
    
    print("All jobs SUCCEEDED! Downloading results...")
    out_file = out_dir / "score_raw_responses.jsonl"
    with open(out_file, "wb") as f:
        for i, job in enumerate(jobs_info):
            file_name = None
            if job.dest and job.dest.file_name:
                file_name = job.dest.file_name
                
            if not file_name:
                print(f"ERROR: No output file name found in batch job {job.name}.")
                sys.exit(1)
                
            print(f"[{i+1}/{len(jobs_info)}] Downloading results from file: {file_name}...")
            output_data = client.files.download(file=file_name)
            f.write(output_data)
            
    print(f"\nSaved raw responses to {out_file}")
    
    # Load keys mapping
    keys_by_id = {}
    with open(out_dir / "score_keys.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys_by_id[r["request_id"]] = r
            
    # Instantiate clients for _cache_put
    nli_client = GeminiNLIClient(args.model)
    misweighted_client = GeminiMisweightedClient(args.model)
    fact_client = GeminiFactExtractorClient(args.model)
    
    print("Populating cache with results...")
    count = 0
    with open(out_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip(): continue
            resp = json.loads(line)
            req_id = resp.get("request_id")
            
            try:
                content = resp["response"]["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(content)
            except Exception as e:
                print(f"Failed to parse output for {req_id}: {e}")
                continue
                
            key_info = keys_by_id.get(req_id)
            if not key_info:
                print(f"Could not find key mapping for request {req_id}")
                continue
                
            req_type = key_info["type"]
            cache_key = key_info["cache_key"]
            
            if req_type == "nli":
                # ensure proper types
                label = data.get("label", "neutral").lower()
                if label not in ("entailment", "neutral", "contradiction"):
                    label = "neutral"
                reason = data.get("reason", "")
                nli_client._cache_put(cache_key, {"label": label, "reason": reason})
                count += 1
            elif req_type == "misweighted":
                mis = bool(data.get("misweighted", False))
                reason = data.get("reason", "")
                misweighted_client._cache_put(cache_key, {"misweighted": mis, "reason": reason})
                count += 1
            elif req_type == "fact_extract":
                try:
                    numbers = [float(x) for x in data.get("numbers", [])]
                except (ValueError, TypeError):
                    numbers = []
                fact_client._cache_put(cache_key, {"numbers": numbers})
                count += 1
                
    print(f"Successfully cached {count} responses!")
    print("You can now run `score_narratives.py` and it will use the cache.")


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    
    if args.prepare:
        do_prepare(args, out_dir)
    elif args.submit:
        do_submit(args, out_dir)
    elif args.download:
        do_download(args, out_dir)
    elif args.manage_queue:
        do_manage_queue(args, out_dir)
    else:
        print("Please specify --prepare, --submit, --download, or --manage-queue")
        sys.exit(1)


if __name__ == "__main__":
    main()
