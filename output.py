"""
Stage-by-stage output inspection commands for the insider threat detection pipeline.
Run each block after its corresponding stage completes, before moving to the next stage.
"""

# ============================================================================
# STAGE 1 — Data Preparation
# Files: data/processed/user_week_features.csv, data/processed/user_splits.json
# ============================================================================
import pandas as pd
import json

df1 = pd.read_csv("data/processed/user_week_features.csv")
print("=== STAGE 1: user_week_features.csv ===")
print("Shape:", df1.shape)
print("Columns:", df1.columns.tolist())
print("\nSplit distribution:\n", df1["split"].value_counts())
print("\nLabel distribution:\n", df1["label"].value_counts())
print("\nMalicious rows (label==1):\n",
      df1[df1["label"] == 1][["user", "year_week", "split", "label"]])

with open("data/processed/user_splits.json") as f:
    splits = json.load(f)
print("\nSplit sizes: train={}, calibration={}, test={}".format(
    len(splits["train"]), len(splits["calibration"]), len(splits["test"])))

all_users = splits["train"] + splits["calibration"] + splits["test"]
print("No overlap check (should be True):", len(all_users) == len(set(all_users)))


# ============================================================================
# STAGE 2 — Feature Extraction
# Files: ExtractedData/week*.csv (or your configured output path)
# ============================================================================
df2 = pd.read_csv("ExtractedData/week6.2.csv")  # adjust dataset suffix as needed
print("\n=== STAGE 2: week-level extracted features ===")
print("Shape:", df2.shape)
print("Number of feature columns:", len(df2.columns))
print("Sample insider rows:\n", df2[df2["insider"] > 0].head())
print("\nPsychometric columns present:", all(c in df2.columns for c in ["O", "C", "E", "A", "N"]))


# ============================================================================
# STAGE 3 — Anomaly Detection
# Files: outputs/stage2_anomaly_detection/scored_user_weeks.csv, comparison_summary.json
# ============================================================================
df3 = pd.read_csv("outputs/stage2_anomaly_detection/scored_user_weeks.csv")
print("\n=== STAGE 3: scored_user_weeks.csv ===")
print("Shape:", df3.shape)
print("Score columns present:", "iforest_score" in df3.columns, "autoencoder_score" in df3.columns)

with open("outputs/stage2_anomaly_detection/comparison_summary.json") as f:
    comparison = json.load(f)
print("\nModel comparison summary:")
for row in comparison:
    print(f"  {row['Model']}: Test PR-AUC={row['Test PR-AUC']:.6f}, "
          f"Test Malicious Rank={row['Test Malicious Rank']}")

# Sanity check: confirm known malicious users still present with correct labels
print("\nKnown malicious rows in scored table:\n",
      df3[df3["label"] == 1][["user", "year_week", "split", "iforest_score", "autoencoder_score"]])


# ============================================================================
# STAGE 4 — Narrative Generation
# Files: data/processed/generated_narratives.jsonl
# ============================================================================
narratives = []
with open("data/processed/generated_narratives.jsonl") as f:
    for line in f:
        narratives.append(json.loads(line))

print(f"\n=== STAGE 4: generated_narratives.jsonl ===")
print(f"Total narratives generated: {len(narratives)}")

for rec in narratives:
    if rec["user_id"] in ["PLJ1771", "MBG3183"]:
        print(f"\n--- {rec['user_id']} ({rec['week_id']}) | score={rec['anomaly_score']:.6f} ---")
        print("Source log summary:", rec["source_log_summary"])
        print("Narrative:", rec["narrative_text"])


# ============================================================================
# STAGE 5 — Factual Verification
# Files: data/processed/verification_scores.jsonl
# ============================================================================
verifications = []
with open("data/processed/verification_scores.jsonl") as f:
    for line in f:
        verifications.append(json.loads(line))

print(f"\n=== STAGE 5: verification_scores.jsonl ===")
print(f"Total sentence-level verification records: {len(verifications)}")

nc_scores = [r["non_conformity_score"] for r in verifications]
print(f"Non-conformity score range: min={min(nc_scores):.4f}, max={max(nc_scores):.4f}, "
      f"mean={sum(nc_scores)/len(nc_scores):.4f}")

for target_u, target_yw in [("PLJ1771", "2010-W32"), ("MBG3183", "2010-W41")]:
    matches = [r for r in verifications if r["user_id"] == target_u and r["week_id"] == target_yw]
    print(f"\n--- {target_u} ({target_yw}): {len(matches)} sentences ---")
    for r in matches:
        print(f"  \"{r['sentence_text'][:60]}...\" -> NC score={r['non_conformity_score']:.4f}")


# ============================================================================
# STAGE 6 — Conformal Calibration
# Files: data/processed/conformal_calibration_results.json,
#        data/processed/coverage_sweep_robustness.json (robustness sweep)
# ============================================================================
with open("data/processed/conformal_calibration_results.json") as f:
    calib = json.load(f)

print(f"\n=== STAGE 6: conformal_calibration_results.json ===")
for alpha_key, vals in calib["threshold_summary"].items():
    print(f"  {alpha_key}: tau={vals['tau']:.6f}, target_coverage={vals['target_coverage']:.2%}, "
          f"empirical_coverage={vals['test_empirical_coverage']:.2%}, "
          f"n_calibration={vals['n_calibration']}, n_test={vals['n_test']}")

print("\nHigh-value claims analysis:")
for item in calib["target_claims_analysis"]:
    print(f"  {item['user_id']} ({item['week_id']}): NC={item['non_conformity_score']:.4f}, "
          f"verified@90={item['verified_at_tau_90']}, verified@95={item['verified_at_tau_95']}")

with open("data/processed/coverage_sweep_robustness.json") as f:
    robustness = json.load(f)

print(f"\n=== STAGE 6 ROBUSTNESS: 50-trial resampling ===")
print(f"Trials run: {robustness['num_trials']}")
verdict = robustness["verdict"]
print(f"Alphas where target fell inside 90% confidence band: "
      f"{verdict['num_in_band']}/{verdict['total_alphas']}")
if verdict["out_of_band_alphas"]:
    print("Out-of-band alphas:", verdict["out_of_band_alphas"])
else:
    print("All alpha values passed the robustness check.")


# ============================================================================
# STAGE 7 — Middleware (Shell commands, not Python — run these in terminal)
# ============================================================================
MIDDLEWARE_COMMANDS = """
# --- Check Docker/WSL status before attempting anything ---
docker --version
docker-compose --version
wsl --status                      # Windows only

# --- Start the full stack ---
cd siem_middleware
docker-compose up --build -d
docker-compose ps                 # confirm all services show "healthy" or "running"

# --- Health checks ---
curl http://localhost:8000/health
curl http://localhost:8001/health

# --- Run the actual end-to-end test (THE stage 7 checkpoint that matters) ---
python test_end_to_end.py --api-url http://localhost:8000 --soar-url http://localhost:8001

# --- Inspect what the mock SOAR actually received ---
curl http://localhost:8001/soar/alerts
docker-compose exec mock_soar cat /app/received_alerts.jsonl

# --- View live worker logs while a test alert processes ---
docker-compose logs -f worker

# --- Tear down when done ---
docker-compose down -v

# --- NATIVE FALLBACK if Docker/WSL is broken (no RabbitMQ/Redis needed) ---
# In celery_worker.py, temporarily add:
#   celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
# Then in separate terminals:
uvicorn mock_soar:app --host 0.0.0.0 --port 8001
uvicorn api:app --host 0.0.0.0 --port 8000
python test_end_to_end.py --api-url http://localhost:8000 --soar-url http://localhost:8001
"""
print("\n=== STAGE 7: Middleware shell commands (copy into terminal) ===")
print(MIDDLEWARE_COMMANDS)