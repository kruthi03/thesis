import json
import pandas as pd

def update():
    # 1. Load the regenerated features
    features = {}
    with open('out_batch/scored_final_pilot.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sig_id = data['signal_id']
            for sent in data.get('sentences', []):
                cluster_id = f"{sig_id}_{sent['sentence_id']}"
                features[cluster_id] = sent.get('hallucinated_context_flag', 0.0)

    # 2. Update gold_labels.csv
    df = pd.read_csv('out_batch/gold_labels.csv')
    
    hallucinated = []
    for cid in df['cluster_id']:
        hallucinated.append(features.get(cid, 0.0))
        
    df['hallucinated_context'] = hallucinated
    
    df.to_csv('out_batch/gold_labels.csv', index=False)
    print(f"Updated {len(df)} gold labels with hallucinated_context.")

if __name__ == "__main__":
    update()
