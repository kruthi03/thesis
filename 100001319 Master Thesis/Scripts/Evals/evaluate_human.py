import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from pathlib import Path
import math

def evaluate_on_human_labels():
    model_path = r'D:\CP_MTH\ConformalPredictions\scripts\rho_model.pkl'
    human_labels_path = r'D:\CP_MTH\ConformalPredictions\out_narratives_ollama_v3\t085\labelling_final.csv'
    scored_features_path = r'D:\CP_MTH\ConformalPredictions\out_batch\scored_final_pilot.jsonl'
    
    # 1. Load the human labels
    df_labels = pd.read_csv(human_labels_path)
    # The 'label' column is the human label (1 for Grounded, 0 for Not Grounded)
    if 'label' not in df_labels.columns:
        print("Error: No 'label' column found.")
        return
        
    human_target = {}
    for idx, row in df_labels.iterrows():
        # Clean label strings if necessary, assume binary 1/0 or 'Grounded'/'Not Grounded'
        val = row['label']
        if pd.isna(val): continue
        if isinstance(val, str):
            target = 1 if val.strip().lower() == 'grounded' else 0
        else:
            target = int(val)
        human_target[row['cluster_id']] = target
        
    print(f"Loaded {len(human_target)} human labels.")
    
    # 2. Extract features for those clusters
    features = []
    y_true = []
    
    with open(scored_features_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sig_id = data['signal_id']
            for sent in data.get('sentences', []):
                cluster_id = f"{sig_id}_{sent['sentence_id']}"
                if cluster_id in human_target:
                    sc = sent.get('self_consistency', 0.0)
                    fm = sent.get('fact_match')
                    if fm is None or math.isnan(fm):
                        fm = sc
                    mw = 1.0 if sent.get('misweighted_flag') else 0.0
                    hc = sent.get('hallucinated_context_flag', 0.0)
                    
                    features.append([sc, fm, mw, hc])
                    y_true.append(human_target[cluster_id])

    if not features:
        print("No intersecting features found!")
        return

    # 3. Load model and predict
    clf = joblib.load(model_path)
    X = pd.DataFrame(features, columns=['self_consistency', 'fact_match', 'misweighted', 'hallucinated_context'])
    
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    # 4. Report
    print('\n--- ROC AUC Score on Human Labels ---')
    print(f'{roc_auc_score(y_true, y_prob):.4f}')
    
    print('\n--- Brier Score Loss ---')
    print(f'{brier_score_loss(y_true, y_prob):.4f}')
    
    print('\n--- Classification Report ---')
    print(classification_report(y_true, y_pred, target_names=["Not Grounded", "Grounded"]))

if __name__ == "__main__":
    evaluate_on_human_labels()
