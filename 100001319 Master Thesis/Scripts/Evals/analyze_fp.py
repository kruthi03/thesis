import pandas as pd, json

df_labels = pd.read_csv(r'D:\CP_MTH\ConformalPredictions\out_narratives_ollama_v3\t085\labelling_final.csv')
human_target = {}
for _, row in df_labels.iterrows():
    val = row['label']
    if pd.isna(val): continue
    target = 1 if (str(val).strip().lower() == 'grounded' or str(val)=='1' or str(val)=='1.0') else 0
    human_target[row['cluster_id']] = target

false_pos = []
with open(r'D:\CP_MTH\ConformalPredictions\out_batch\scored_final_pilot.jsonl', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        for sent in data.get('sentences', []):
            cid = f"{data['signal_id']}_{sent['sentence_id']}"
            if cid in human_target:
                if human_target[cid] == 0:
                    false_pos.append(sent['text'])

print('Sample of Not Grounded (Human) sentences:')
for text in false_pos:
    print('- ' + text)
