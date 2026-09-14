import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss, confusion_matrix, mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load ground truth gold labels
df = pd.read_csv(r'D:\CP_MTH\ConformalPredictions\out_batch\gold_labels.csv')

X_data = df[['self_consistency', 'fact_match', 'misweighted', 'hallucinated_context']].values
y_data = df['target'].values

clf = LogisticRegression(random_state=42, class_weight='balanced')

print('--- 5-Fold Cross-Validation ROC AUC Score ---')
roc_auc_scores = cross_val_score(clf, X_data, y_data, cv=5, scoring='roc_auc')
print(f'Mean ROC AUC: {roc_auc_scores.mean():.4f} (std: {roc_auc_scores.std():.4f})')

print('\n--- 5-Fold Cross-Validation Brier Score Loss ---')
brier_scores = cross_val_score(clf, X_data, y_data, cv=5, scoring='neg_brier_score')
print(f'Mean Brier Score Loss: {-brier_scores.mean():.4f} (std: {brier_scores.std():.4f})')

print('\n--- 5-Fold Cross-Validation Mean Absolute Error (MAE) ---')
mae_scores = cross_val_score(clf, X_data, y_data, cv=5, scoring='neg_mean_absolute_error')
print(f'Mean Absolute Error (MAE): {-mae_scores.mean():.4f} (std: {mae_scores.std():.4f})')

def recall_at_k(y_true, y_prob, k_list):
    """Calculate Recall@K by ranking lowest probability of being Grounded (i.e. highest probability of hallucination)"""
    sorted_indices = np.argsort(y_prob)
    y_true_not_grounded = (y_true == 0.0).astype(int)
    total_hallucinations = y_true_not_grounded.sum()
    
    print('\n--- Recall@K for "Not Grounded" (Hallucinations) ---')
    print(f'Total Hallucinations in dataset: {total_hallucinations}')
    for k in k_list:
        top_k_indices = sorted_indices[:k]
        hallucinations_caught = y_true_not_grounded[top_k_indices].sum()
        recall = hallucinations_caught / total_hallucinations
        print(f'Recall@{k:<3d}: {recall:.2%} ({hallucinations_caught}/{total_hallucinations} caught)')

y_prob_cv = cross_val_predict(clf, X_data, y_data, cv=5, method='predict_proba')[:, 1]
recall_at_k(y_data, y_prob_cv, k_list=[10, 50, 100, 200, 300, 500])


# Cross-validated predictions for confusion matrix
y_pred = cross_val_predict(clf, X_data, y_data, cv=5)
print('\n--- Confusion Matrix (Cross-Validated) ---')
print(confusion_matrix(y_data, y_pred))

print('\n--- Classification Report (Cross-Validated) ---')
print(classification_report(y_data, y_pred, target_names=['Not Grounded', 'Grounded']))
