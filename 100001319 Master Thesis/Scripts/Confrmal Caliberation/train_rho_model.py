import json
import math
import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load gold-labels and features
labels_path = Path(r'D:\CP_MTH\ConformalPredictions\out_batch\gold_labels.csv')
df = pd.read_csv(labels_path)

X_data = df[['self_consistency', 'fact_match', 'misweighted', 'hallucinated_context']].values
y_data = df['target'].values

print(f"Loaded {len(X_data)} gold-labeled sentences for training.")

if len(X_data) == 0:
    print("Error: No overlapping clusters found.")
    exit(1)

# 3. Train Logistic Regression
clf = LogisticRegression(random_state=42, class_weight='balanced')
clf.fit(X_data, y_data)

# 4. Save the model
model_path = r'D:\CP_MTH\ConformalPredictions\scripts\rho_model.pkl'
joblib.dump(clf, model_path)
print(f"\nModel saved to {model_path}")

# 5. Print results
print("\n--- Model Evaluation ---")
y_pred = clf.predict(X_data)
print(classification_report(y_data, y_pred, target_names=["Not Grounded", "Grounded"]))
