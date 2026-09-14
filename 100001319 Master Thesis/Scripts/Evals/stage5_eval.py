import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.calibrate import ltt_calibrate_grid
from src.evaluation import evaluate

OUT_DIR = Path("out_batch")
OUT_DIR.mkdir(exist_ok=True)

def load_data():
    # 1. Load Gold Labels
    df = pd.read_csv('out_batch/gold_labels.csv')
    df['human_label'] = df['target'].map({1: 'grounded', 0: 'hallucinated'})
    df['signal_id'] = df['cluster_id'].apply(lambda x: x.split('_')[0])
    
    # 2. Get period & user_id
    cluster_to_period = {}
    cluster_to_user = {}
    with open(r'D:\CP_MTH\ConformalPredictions\out_full\1000_signals.jsonl', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sig_id = data['signal_id']
            cluster_to_period[sig_id] = data.get('period', 'Unknown')
            cluster_to_user[sig_id] = data.get('user', 'Unknown')
            
    df['period'] = df['signal_id'].map(lambda x: cluster_to_period.get(x, 'Unknown'))
    df['user_id'] = df['signal_id'].map(lambda x: cluster_to_user.get(x, 'Unknown'))
    
    # 3. Load LDAP for stratify
    user_groups = {}
    try:
        with open(r'D:\CP_MTH\ConformalPredictions\raw data\r6.2\LDAP\2011-05.csv', encoding='utf-8-sig') as f:
            import csv
            for row in csv.DictReader(f):
                dept = row.get("department", "Unknown").strip()
                role = row.get("role", "Unknown").strip()
                user_groups[row.get("user_id", "")] = f"{dept}_{role}"
    except Exception as e:
        print(f"Warning: LDAP not found, {e}")
        
    df['user_group'] = df['user_id'].map(lambda x: user_groups.get(x, 'Unknown'))
    
    # 4. Predict rho
    clf = joblib.load(r'D:\CP_MTH\ConformalPredictions\scripts\rho_model.pkl')
    X = df[['self_consistency', 'fact_match', 'misweighted', 'hallucinated_context']]
    df['rho'] = clf.predict_proba(X)[:, 1]
    
    return df, user_groups

def plot_validity_recall(df_test, results, alpha_grid):
    # Figure 1: Validity
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(alpha_grid, alpha_grid, "k--", linewidth=1, label="y = x (guarantee boundary)")
    errors = [res.error for res in results]
    ax.plot(alpha_grid, errors, marker="o", color="tab:blue", label="Fallback LTT Calibration")
    ax.set_xlabel("Nominal Alpha (Target Risk)")
    ax.set_ylabel("Empirical Error on Test Split")
    ax.set_title("Validity: Empirical Error vs Nominal Alpha")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_validity_gold.png", dpi=150)
    plt.close(fig)
    
    # Figure 2: Recall
    fig, ax = plt.subplots(figsize=(6, 5))
    recalls = [res.recall for res in results]
    ax.plot(alpha_grid, recalls, marker="o", color="tab:green", label="Recall of Grounded Content")
    ax.set_xlabel("Nominal Alpha (Target Risk)")
    ax.set_ylabel("Recall on Test Split")
    ax.set_title("Usefulness: Recall vs Nominal Alpha")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_recall_gold.png", dpi=150)
    plt.close(fig)

def plot_aci(df):
    periods = sorted([p for p in df['period'].unique() if p != 'Unknown'])
    alpha = 0.10
    gamma = 0.05
    lambda_t = 0.90
    
    lambda_history = []
    error_history = []
    
    for period in periods:
        df_period = df[df['period'] == period]
        admitted = df_period[df_period['rho'] >= lambda_t]
        n_admitted = len(admitted)
        n_wrong = len(admitted[admitted['target'] == 0])
        err_t = n_wrong / max(n_admitted, 1)
        
        lambda_history.append(lambda_t)
        error_history.append(err_t)
        
        lambda_t = max(0.0, min(1.0, lambda_t + gamma * (err_t - alpha)))
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(periods, lambda_history, color='tab:blue', marker='o', label='Lambda (Threshold)')
    ax1.set_xlabel("Time (Weeks)")
    ax1.set_ylabel("Threshold (Lambda)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticklabels(periods, rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.plot(periods, error_history, color='tab:red', marker='x', linestyle='--', label='Empirical Error')
    ax2.axhline(alpha, color='black', linestyle=':', label='Target Risk (0.10)')
    ax2.set_ylabel("Error Rate", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.suptitle("Adaptive Conformal Inference (ACI) Trajectory")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_aci_trajectory.png", dpi=150)
    plt.close(fig)

def plot_rho_distribution(df):
    grounded_rho = df[df['human_label'] == 'grounded']['rho'].tolist()
    hallucinated_rho = df[df['human_label'] == 'hallucinated']['rho'].tolist()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [i / 20 for i in range(21)]
    ax.hist(grounded_rho, bins=bins, alpha=0.6, color="tab:green", label=f"Grounded (n={len(grounded_rho)})")
    ax.hist(hallucinated_rho, bins=bins, alpha=0.6, color="tab:red", label=f"Hallucinated (n={len(hallucinated_rho)})")
    
    ax.set_xlabel("Predicted Probability: rho(s)")
    ax.set_ylabel("Count")
    ax.set_title("rho(s) Distribution on Gold Labels")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_rho_distribution_gold.png", dpi=150)
    plt.close(fig)

def main():
    df, user_groups = load_data()
    print(f"Loaded {len(df)} gold labels.")
    
    # Stratified Split (reusing src.calibrate logic)
    from src.calibrate import split_by_user
    user_ids = df['user_id'].tolist()
    split = split_by_user(user_ids, user_groups=user_groups, calibrate_fraction=0.5, seed=42)
    
    df_cal = df.iloc[split.calibrate_indices]
    df_test = df.iloc[split.test_indices]
    print(f"Calibration Split: {len(df_cal)} sentences")
    print(f"Test Split: {len(df_test)} sentences")
    
    # 1. Calibrate (Fallback LTT)
    alpha_grid = [round(0.05 * i, 2) for i in range(1, 7)]
    lambda_grid = [round(1.0 - 0.05 * i, 2) for i in range(21)]
    
    print("\nRunning Fallback LTT Calibration...")
    ltt_results = ltt_calibrate_grid(
        rhos=df_cal['rho'].tolist(),
        labels=df_cal['human_label'].tolist(),
        user_ids=df_cal['user_id'].tolist(),
        alpha_grid=alpha_grid,
        delta=0.10,
        lambda_grid=lambda_grid,
        loss_policy="secondary",
        user_groups=user_groups,
        seed=42
    )
    
    # 2. Evaluate on Test Split
    test_results = []
    for res in ltt_results:
        test_res = evaluate(
            df_test['rho'].tolist(),
            df_test['human_label'].tolist(),
            lambda_hat=res.lambda_hat,
            loss_policy="secondary"
        )
        test_results.append(test_res)
        print(f"Alpha {res.alpha:.2f} -> Lambda_hat {res.lambda_hat:.2f} | Test Error: {test_res.error:.4f} | Recall: {test_res.recall:.4f}")
        
    plot_validity_recall(df_test, test_results, alpha_grid)
    
    # 3. ACI
    plot_aci(df)
    
    # Plot Rho Distribution
    plot_rho_distribution(df)
    
    # 4. Baselines
    print("\n=== Baseline Ablation Comparison (Fixed Threshold = 0.80) ===")
    test_labels = df_test['human_label'].tolist()
    
    b_rho = evaluate(df_test['rho'].tolist(), test_labels, lambda_hat=0.80, loss_policy="secondary")
    b_sc = evaluate(df_test['self_consistency'].tolist(), test_labels, lambda_hat=0.80, loss_policy="secondary")
    b_fm = evaluate(df_test['fact_match'].fillna(0).tolist(), test_labels, lambda_hat=0.80, loss_policy="secondary")
    
    print(f"{'Feature':<25} | {'Error':<10} | {'Recall':<10}")
    print("-" * 50)
    print(f"{'4-Feature rho(s)':<25} | {b_rho.error:<10.4f} | {b_rho.recall:<10.4f}")
    print(f"{'Self-Consistency Alone':<25} | {b_sc.error:<10.4f} | {b_sc.recall:<10.4f}")
    print(f"{'Fact-Match Alone':<25} | {b_fm.error:<10.4f} | {b_fm.recall:<10.4f}")
    
    print("\nStage 5 Evaluation Complete. Plots saved to out_batch/")

if __name__ == "__main__":
    main()
