import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def evaluate_memory():
    print("--- Resource & Memory Profiling ---")
    print("Starting tracemalloc...")
    tracemalloc.start()
    
    # Snapshot 1: Baseline
    snapshot1 = tracemalloc.take_snapshot()
    
    # Import heavy components and load models
    print("Importing pipeline and loading models into memory...")
    from src.admission import is_misweighted, score_fact_match
    from src.scoring import CrossEncoderNLIClient
    
    nli_client = CrossEncoderNLIClient()
    
    # Snapshot 2: After models loaded
    snapshot2 = tracemalloc.take_snapshot()
    
    # Simulate processing a sentence
    print("Processing a simulated sentence...")
    test_sentence = "User XYZ123 copied 104 files to removable media."
    mock_signal = {
        "facts": [
            {"fact_id": "F01", "field": "removable_media_copies", "value": 104, "display": "104", "gloss": "files copied"},
        ]
    }
    score_fact_match(test_sentence, mock_signal)
    is_misweighted(test_sentence, mock_signal["facts"], ["F01"])
    nli_client.entails("User XYZ123 copied 104 files.", test_sentence)
    
    # Snapshot 3: Peak usage during execution
    snapshot3 = tracemalloc.take_snapshot()
    
    # Calculate Memory Footprint
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n[Memory Profiling Results]")
    print(f"  Current Memory Allocation: {current / 10**6:.2f} MB")
    print(f"  Peak Memory Footprint: {peak / 10**6:.2f} MB")
    
    if peak / 10**6 < 1000:
        print("\n  [PASS] The application consumes less than 1GB of RAM, suitable for lightweight containerized deployment.")
    else:
        print("\n  [WARN] The application requires significant memory resources.")

if __name__ == "__main__":
    evaluate_memory()
