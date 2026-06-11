# TODO.md — Lending Club SSL Project

## 📍 Current Status: SSL Pretraining Complete (200k samples)

**We are here:** ✅ SSL pretraining with 200,000 samples completed at 2:03 AM, June 12, 2026

**What's ready:**
- ✅ Baseline features (18 features) preprocessed
- ✅ Baseline models (LR, LightGBM, PCA+LR) run
- ✅ SSL pretraining with 10k, 50k, and 200k samples
- ✅ SSL few-shot evaluation with linear and MLP probes
- ✅ Enhanced embedding visualizations (t-SNE, UMAP, silhouette, etc.)
- ✅ All results documented and committed to git

---

## 📊 Current Results Summary

| Experiment | Pretraining Samples | Encoder Dims | SSL AUC (N=50) | LR AUC (N=50) | SSL Improvement |
|------------|-------------------|--------------|----------------|----------------|------------------|
| Exp 1 | 10,000 | 128→64 | 0.528 | 0.587 | -0.059 |
| Exp 2 | 50,000 | 256→128 | 0.548 | 0.587 | -0.039 |
| Exp 3 | 200,000 | 256→128 | 0.548 | 0.587 | -0.039 |

**Key finding:** SSL beats baseline at N=20 (+0.007 AUC), but underperforms at larger sample sizes.

---

## 📋 To Do List (June 12, 2026)

### Morning (1-2 hours)

```
□ 1. Review embedding visualizations
   □ Check t-SNE/UMAP for cluster separation
   □ Analyze silhouette scores and risk profiles
   □ Document findings in experiment report

□ 2. Run feature engineering experiments
   □ Add derived features: loan_to_income, credit_age
   □ Add more raw features (from 18 → 30+)
   □ Re-run baseline with new features
```

### Midday (2-3 hours)

```
□ 3. Try SCARF (Self-Supervised Contrastive Learning using Random Feature Corruption)
   □ Implement SCARF augmentations (feature swapping)
   □ Run pretraining with SCARF (100k samples)
   □ Compare with SimCLR results

□ 4. Explore different augmentation strategies
   □ Higher noise (0.2)
   □ Lower temperature (0.05)
   □ Feature swapping
```

### Afternoon (2-3 hours)

```
□ 5. Run ablation studies
   □ Vary embedding dimension (64, 128, 256)
   □ Vary hidden dimension (128, 256, 512)
   □ Vary batch size (256, 512, 1024)

□ 6. Generate final LinkedIn-ready visuals
   □ 3-plot carousel: t-SNE, learning curves, model comparison
   □ Write caption for finance data science audience
```

### Evening (1 hour)

```
□ 7. Update project documentation
   □ Update README.md with latest results
   □ Add experiment results to docs/experiment_results.md
   □ Commit all changes to git
```

---

## 🚨 Known Issues to Address

| Issue | Severity | Status |
|-------|----------|--------|
| SSL underperforms at N≥50 | High | 🔄 Need feature engineering |
| Embeddings are linear (MLP ≈ Linear) | Medium | 🔄 Try SCARF |
| Feature set limited to 18 | Medium | ⏳ Add more features |
| Pretraining data capped at 200k | Low | ⏳ Try full 2.26M dataset |

---

## 🔄 Long-term Roadmap

| Stage | Status | Target Completion |
|-------|--------|-------------------|
| Data ingestion & Parquet conversion | ✅ Done | Complete |
| Baseline EDA | ✅ Done | Complete |
| Feature selection (18 features) | ✅ Done | Complete |
| **Baseline models** | ✅ Done | Complete |
| **SSL pretraining (10k, 50k, 200k)** | ✅ Done | Complete |
| **SSL few-shot evaluation** | ✅ Done | Complete |
| **Embedding visualizations** | ✅ Done | Complete |
| **Feature engineering** | ⏳ Pending | Today |
| **SCARF pretraining** | ⏳ Pending | Today |
| **Ablation studies** | ⏳ Pending | Tomorrow |
| **Final analysis & write-up** | ⏳ Pending | Tomorrow |

---

## 📂 Project Structure

```
D:\contrastive-credit-representations\
├── data/
│   ├── raw/                  # Parquet files (2.26M rows)
│   └── processed/            # baseline_features_v2.npz (18 features)
├── src/
│   ├── baseline_features.py
│   ├── run_baselines.py
│   ├── ssl/
│   │   ├── contrastive_pretraining.py
│   │   └── few_shot_evaluation.py
│   └── visualizations/
│       └── embedding_visualizations_enhanced.py
├── results/
│   ├── baselines/
│   ├── ssl/
│   └── visualizations/
├── models/                   # SSL checkpoints
└── docs/
    └── experiment_results.md
```

---

## 🎯 Today's First Task

When you open your laptop tomorrow, **start here**:

```powershell
cd D:\contrastive-credit-representations
python src/baseline_features.py --add_derived_features
```

(You'll need to add the `--add_derived_features` flag or modify the script directly)

---

## 📝 Notes

- **Next experiment:** Add loan_to_income, credit_age, and total_credit_lines features
- **Try SCARF after feature engineering** if SimCLR still underperforms
- **Goal:** Achieve SSL AUC > 0.580 at N=50 to beat baseline