import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.admission import is_misweighted, score_fact_match, _combined_fact_match_score

def eval_generator(narratives_path: Path):
    """Evaluate the raw output of the LLM before pooling."""
    print("--- Evaluating Component 1: Generator Quality ---")
    if not narratives_path.exists():
        print(f"Error: Could not find {narratives_path}. Need narratives to evaluate generator.")
        return

    total_drafts = 0
    total_signals = 0
    unique_sentences = set()

    with open(narratives_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total_signals += 1
            if 'narratives' in data:
                total_drafts += len(data['narratives'])
                for draft in data['narratives']:
                    # Add the whole narrative text for uniqueness checking
                    unique_sentences.add(draft['text'].strip().lower())

    print(f"Total Signals Processed: {total_signals}")
    print(f"Total Drafts Generated: {total_drafts}")
    print(f"Total Unique Sentences Generated: {len(unique_sentences)}")
    print("Generator evaluation complete.\n")


def eval_pooling(narratives_path: Path):
    """Evaluate Gemini LLM-as-a-judge clustering."""
    print("--- Evaluating Component 2: Semantic Pooling ---")
    # For a real evaluation, we would compare the clusters against a gold standard.
    # Here we simulate evaluating the cluster sizes and purity from batch_judge output.
    if not narratives_path.exists():
        print(f"Error: Could not find {narratives_path}. Run batch_judge.py first.")
        return

    total_sentences = 0
    total_clusters = 0
    singletons = 0

    with open(narratives_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentences = data.get("sentences", [])
            total_clusters += len(sentences)
            for s in sentences:
                total_sentences += s.get("n_samples", 1)
                if s.get("n_samples", 1) == 1:
                    singletons += 1

    print(f"Total Raw Sentences Clustered: {total_sentences}")
    print(f"Total Clusters Formed: {total_clusters}")
    if total_sentences > 0:
        print(f"Compression Ratio (Clusters / Raw): {total_clusters / total_sentences:.2f}")
    print(f"Singleton Clusters (Appeared only once): {singletons}")
    print("Semantic Pooling evaluation complete.\n")


def eval_matcher():
    """Evaluate the Deterministic Fact-Matcher isolated from the pipeline."""
    print("--- Evaluating Component 3: Fact Matcher ---")
    # A small Trap Dataset to test the limits of qualitative numbers.
    trap_sentences = [
        ("The user downloaded exactly one thousand files.", 1000),
        ("User ACM2278 uploaded half a gigabyte to external media.", 500),
        ("They transferred roughly a dozen files.", 12),
        ("The user copied 104 files to a USB drive.", 104)
    ]
    
    mock_signal = {
        "facts": [
            {"fact_id": "F01", "field": "removable_media_copies", "value": 1000, "display": "1000", "gloss": "files copied to removable media"},
            {"fact_id": "F02", "field": "removable_media_copies", "value": 500, "display": "500", "gloss": "files copied to removable media"},
            {"fact_id": "F03", "field": "removable_media_copies", "value": 12, "display": "12", "gloss": "files copied to removable media"},
            {"fact_id": "F04", "field": "removable_media_copies", "value": 104, "display": "104", "gloss": "files copied to removable media"}
        ]
    }

    correct = 0
    for text, expected_val in trap_sentences:
        result = score_fact_match(text, mock_signal)
        matched = any(n.status == 'matched' for n in result.numbers)
        print(f"Sentence: '{text}'")
        print(f"  Expected Number: {expected_val} | Matched: {matched}")
        if matched:
            correct += 1
            print("  [SUCCESS]")
        else:
            print("  [FAIL - False Negative on qualitative extraction]")
            
    print(f"\nFact Matcher Recall on Trap Dataset: {correct}/{len(trap_sentences)} ({(correct/len(trap_sentences))*100:.1f}%)")
    print("Fact Matcher evaluation complete.\n")


def eval_parser():
    """Evaluate the Causal Misweighting Parser."""
    print("--- Evaluating Component 4: Causal Misweighting Parser ---")
    # Trap dataset: Sentences that have true facts but might be misweighted.
    # 1. Deliberately misweighted (uses causal phrase + context fact)
    # 2. Correctly weighted (uses causal phrase + evidence fact)
    # 3. Bare peer comparison (no causal language, known miss)
    
    test_cases = [
        {
            "text": "The primary reason for the alert is because the user is a SystemsEngineer.",
            "is_truly_misweighted": True,
            "desc": "Blatant misweighting with causal language"
        },
        {
            "text": "The alert was triggered by the user copying 104 files to removable media.",
            "is_truly_misweighted": False,
            "desc": "Correct attribution to evidence"
        },
        {
            "text": "Notably, the user transferred 104 files, which is above their peer average of 12.",
            "is_truly_misweighted": True,  # Based on annotation guideline
            "desc": "Bare peer comparison (Expected to bypass heuristic - Known Miss)"
        }
    ]

    mock_facts = [
        {"fact_id": "F_EVID", "gloss": "files copied to removable media", "value": 104},
        {"fact_id": "F_CTX1", "gloss": "user role is SystemsEngineer", "value": "SystemsEngineer"},
        {"fact_id": "F_CTX2", "gloss": "peer average of files copied", "value": 12}
    ]
    evidence_ids = ["F_EVID"]

    caught = 0
    false_positives = 0
    for case in test_cases:
        print(f"Sentence: '{case['text']}'")
        res = is_misweighted(case['text'], mock_facts, evidence_ids)
        flagged = res.misweighted
        print(f"  Expected Misweighted: {case['is_truly_misweighted']} | Heuristic Flagged: {flagged}")
        
        if case['is_truly_misweighted'] and flagged:
            caught += 1
            print("  [SUCCESS - True Positive]")
        elif case['is_truly_misweighted'] and not flagged:
            print("  [FAIL - False Negative (Bypass)]")
        elif not case['is_truly_misweighted'] and not flagged:
            print("  [SUCCESS - True Negative]")
        elif not case['is_truly_misweighted'] and flagged:
            false_positives += 1
            print("  [FAIL - False Positive]")
            
    print(f"\nMisweighting Parser Evaluation complete.")
    print(f"Caught {caught} true misweightings. Raised {false_positives} false positives.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component-level Evaluation Suite")
    parser.add_argument("--eval-generator", action="store_true", help="Evaluate LLM Generator")
    parser.add_argument("--eval-pooling", action="store_true", help="Evaluate Semantic Pooling")
    parser.add_argument("--eval-matcher", action="store_true", help="Evaluate Fact Matcher")
    parser.add_argument("--eval-parser", action="store_true", help="Evaluate Misweighting Parser")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--narratives-path", default="out_1000_narratives/t085/narratives.jsonl", help="Path to narratives.jsonl")
    
    args = parser.parse_args()
    narr_path = Path(args.narratives_path)
    
    if args.all or args.eval_generator:
        eval_generator(narr_path)
    if args.all or args.eval_pooling:
        eval_pooling(narr_path)
    if args.all or args.eval_matcher:
        eval_matcher()
    if args.all or args.eval_parser:
        eval_parser()
        
    if not any([args.all, args.eval_generator, args.eval_pooling, args.eval_matcher, args.eval_parser]):
        print("Please specify a component to evaluate. Use --help for options.")
