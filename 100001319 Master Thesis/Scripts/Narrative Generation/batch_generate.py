"""CLI: Batch Generation for narratives.

python scripts/batch_generate.py --prepare --signals out_full/signals.jsonl
python scripts/batch_generate.py --submit
python scripts/batch_generate.py --download --job-name batchJobs/123
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import os
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate import build_prompt, GenerationConfig
from src.sentences import split_sentences

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prepare", action="store_true", help="Prepare the batch requests file")
    p.add_argument("--submit", action="store_true", help="Upload file and submit batch job")
    p.add_argument("--download", action="store_true", help="Download and process a completed batch job")
    
    p.add_argument("--signals", help="path to signals.jsonl (required for --prepare)")
    p.add_argument("--out", default="out_batch", help="output directory")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--k", type=int, default=10, help="samples per signal")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--job-name", help="Job name (e.g. batchJobs/123) required for --download")
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
    if not args.signals:
        print("ERROR: --signals required for --prepare")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    req_file = out_dir / "generation_requests.jsonl"
    
    cfg = GenerationConfig(model=args.model, k=args.k, temperature=args.temperature)
    
    count = 0
    with open(args.signals, encoding="utf-8") as fin, open(req_file, "w", encoding="utf-8") as fout:
        for line in fin:
            signal = json.loads(line)
            system, user = build_prompt(signal, cfg.k)
            
            # The google-genai batch JSONL format expects a raw JSON line that mirrors InlinedRequest
            req = {
                "request_id": signal["signal_id"],
                "request": {
                    "model": f"models/{args.model}",
                    "contents": [
                        {"role": "user", "parts": [{"text": user}]}
                    ],
                    "systemInstruction": {
                        "parts": [{"text": system}]
                    },
                    "generationConfig": {
                        "temperature": cfg.temperature,
                        "maxOutputTokens": args.max_tokens,
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "OBJECT",
                            "properties": {
                                "drafts": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"}
                                }
                            },
                            "required": ["drafts"]
                        }
                    }
                }
            }
            fout.write(json.dumps(req) + "\n")
            count += 1
            
    print(f"Prepared {count} requests in {req_file}")

def do_submit(args: argparse.Namespace, out_dir: Path) -> None:
    req_file = out_dir / "generation_requests.jsonl"
    if not req_file.exists():
        print(f"ERROR: {req_file} not found. Run --prepare first.")
        sys.exit(1)
        
    client = get_client()
    
    print(f"Uploading {req_file} to Gemini File API...")
    file_obj = client.files.upload(
        file=str(req_file),
        config={"mime_type": "application/jsonl"}
    )
    print(f"Uploaded as: {file_obj.name}")
    
    print("Creating batch job...")
    job = client.batches.create(
        model=args.model,
        src=file_obj.name
    )
    print(f"Batch job created successfully!")
    print(f"JOB NAME: {job.name}")
    print(f"State: {job.state}")
    
    state_file = out_dir / "generation_job.txt"
    state_file.write_text(job.name)
    print(f"Saved job name to {state_file}")

def do_download(args: argparse.Namespace, out_dir: Path) -> None:
    job_name = args.job_name
    if not job_name:
        state_file = out_dir / "generation_job.txt"
        if state_file.exists():
            job_name = state_file.read_text().strip()
        else:
            print("ERROR: --job-name required or generation_job.txt must exist.")
            sys.exit(1)
            
    client = get_client()
    print(f"Checking status for {job_name}...")
    job = client.batches.get(name=job_name)
    print(f"Current State: {job.state}")
    
    if job.state != "SUCCEEDED":
        print("Job has not succeeded yet. Please try again later.")
        sys.exit(1)
        
    file_name = None
    if job.dest and job.dest.file_name:
        file_name = job.dest.file_name
        
    if not file_name:
        print("ERROR: No output file name found in batch job.")
        sys.exit(1)
        
    print(f"Downloading results from file: {file_name}...")
    output_data = client.files.download(file=file_name)
    
    out_file = out_dir / "generation_raw_responses.jsonl"
    out_file.write_bytes(output_data)
    print(f"Saved raw responses to {out_file}")
    
    drafts_file = out_dir / "drafts.jsonl"
    count = 0
    with open(out_file, "r", encoding="utf-8") as fin, open(drafts_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            resp = json.loads(line)
            signal_id = resp.get("request_id")
            
            try:
                # The response object contains candidate -> content -> parts -> text
                content = resp["response"]["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(content)
                drafts = data.get("drafts", [])
            except Exception as e:
                print(f"Failed to parse generation for {signal_id}: {e}")
                continue
                
            per_sample = [split_sentences(t) for t in drafts]
            
            fout.write(json.dumps({
                "signal_id": signal_id,
                "drafts": drafts,
                "per_sample": per_sample
            }) + "\n")
            count += 1
            
    print(f"Processed {count} valid signal drafts to {drafts_file}")

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    
    if args.prepare:
        do_prepare(args, out_dir)
    elif args.submit:
        do_submit(args, out_dir)
    elif args.download:
        do_download(args, out_dir)
    else:
        print("Please specify one of: --prepare, --submit, --download")

if __name__ == "__main__":
    main()
