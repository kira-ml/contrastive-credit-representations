# TODO.md — Lending Club SSL Project

## 📍 Current Status: Baseline Feature Preprocessing Complete

**We are here:** ✅ Baseline features (18 features) have been preprocessed and saved to `data/processed/baseline_features_v2.npz`

**What's ready:**
- ✅ 2,260,668 samples with 0 missing values
- ✅ 14 continuous features (standardized)
- ✅ 4 categorical features (label-encoded)
- ✅ Target variable (default rate: 12.86%)
- ✅ Validation passed (no missing values)

---

## 🚨 Known Issues to Fix Tomorrow (Priority 1)

### Issue 1: Sparse features with extreme scaling
| Feature | Max z-score | Problem |
|---------|-------------|---------|
| `pub_rec` | 150.4 | Most zeros, rare extreme values |
| `collections_12_mths_ex_med` | 132.5 | Same |
| `acc_now_delinq` | 201.0 | Same |
| `delinq_2yrs` | 66.5 | Same |

**Fix:** Apply binary transformation (0/1) to these features before standardization

### Issue 2: Home ownership encoding
Only values 1, 4, 5 appear in output. Need to check:
- What categories map to 0, 2, 3?
- Are they rare or missing?

**Fix:** Run `Counter(data['home_ownership'])` after encoding

---

## 📋 To Do List (Tomorrow)

### Morning (1-2 hours)
```
□ 1. Fix sparse feature scaling
   □ Add binary transformation for: pub_rec, collections_12_mths_ex_med, acc_now_delinq, delinq_2yrs
   □ Re-run preprocessing
   □ Save as `baseline_features_v3.npz`

□ 2. Verify home ownership encoding
   □ Run Counter on encoded values
   □ Document rare categories (<0.1%)

□ 3. Validate the fixed dataset
   □ Check max z-scores are now < 10
   □ Confirm 0 missing values preserved
```

### Midday (2-3 hours)
```
□ 4. Run baseline experiments
   □ Navigate to: D:\contrastive-credit-representations
   □ python src/features/run_baselines.py
   □ Expected runtime: 10-30 minutes
   □ Check results in: results/baselines/baseline_results.csv

□ 5. Analyze baseline results
   □ Which baseline performs best at N=20, 50, 100, 200?
   □ Is there room for SSL to improve?
   □ Save summary to: results/baselines/baseline_summary.txt
```

### Afternoon (2-3 hours)
```
□ 6. Set up SSL pretraining environment
   □ Create: src/ssl/contrastive_pretraining.py
   □ Define encoder architecture (MLP with 64-dim embedding)
   □ Implement SimCLR-style contrastive loss

□ 7. Run SSL pretraining (first epoch only)
   □ Test on small subset (10,000 samples)
   □ Verify no collapse (embedding variance > 0.1)
   □ Save checkpoint: models/ssl_encoder_epoch1.pt
```

### Evening (1 hour)
```
□ 8. Document progress
   □ Update TODO.md with today's accomplishments
   □ Save experiment logs to: results/ssl/pretraining_log.txt
   □ Commit code to git with message: "Baseline features v3 + SSL pretraining start"
```

---

## 🔄 Long-term Roadmap

| Stage | Status | Target Completion |
|-------|--------|-------------------|
| Data ingestion & Parquet conversion | ✅ Done | Complete |
| Baseline EDA | ✅ Done | Complete |
| Feature selection (18 features) | ✅ Done | Complete |
| **Fix feature scaling issues** | 🔄 In progress | Tomorrow |
| **Run baseline models** | ⏳ Pending | Tomorrow |
| **SSL pretraining** | ⏳ Pending | This week |
| **Few-shot evaluation** | ⏳ Pending | This week |
| **Ablation studies** | ⏳ Pending | Next week |
| **Final analysis & write-up** | ⏳ Pending | Next week |

---

## 📂 Project Structure (Where to Find Things)

```
D:\contrastive-credit-representations\
├── data/
│   ├── raw/                  # Parquet files (2.26M rows)
│   └── processed/            # baseline_features_v2.npz (18 features)
├── src/
│   ├── baseline_features.py  # Your preprocessing script
│   └── features/
│       └── run_baselines.py  # Baseline model runner
├── results/
│   └── baselines/            # Where results will be saved
└── models/                   # Where SSL checkpoints will go
```

---

## 🎯 Tomorrow's First Task

When you open your laptop tomorrow, **start here**:

```powershell
cd D:\contrastive-credit-representations
python src/baseline_features.py --fix_sparse_features
```

(You'll need to add the `--fix_sparse_features` flag to your script, or modify the script directly.)
