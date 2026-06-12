# Contrastive Borrower Representations for Few-Shot Default Prediction

**An Experimental Investigation in Self-Supervised Representation Learning for Credit Risk**

---

## Overview

This project investigates whether self-supervised contrastive learning can extract meaningful credit risk structure from unlabeled loan application data, enabling accurate default prediction with very few labeled examples.

**Core Research Question:** Can a representation learned without observing any default outcomes improve few-shot default prediction compared to purely supervised approaches?

**Why This Matters:** In consumer lending, default labels are scarce (defaults are rare events that take months to manifest) and systematically missing (rejected applications never generate outcomes). If unlabeled applications contain sufficient structure to learn useful risk representations, lenders could make better decisions for applicants with limited credit histories while maintaining rigorous risk standards.

**ML Problem Type:** Self-supervised representation learning on heterogeneous tabular data, evaluated through few-shot supervised learning.

---

## Project Structure

```
contrastive-borrower-representations/
├── README.md                      # Project overview and documentation
├── data/
│   ├── processed/                 # Processed feature sets (Parquet format)
│   │   ├── advanced_plan1.parquet
│   │   ├── advanced_plan1.meta.json
│   │   ├── baseline_plan1.parquet
│   │   ├── baseline_plan1.meta.json
│   │   ├── baseline_plan2.parquet
│   │   ├── baseline_plan2.meta.json
│   │   └── feature_engineering.log
│   └── raw/                       # Raw Lending Club data
│       ├── accepted_2007_to_2018Q4.parquet
│       └── rejected_2007_to_2018Q4.parquet
├── docs/
│   └── experiment_results.md      # Detailed experiment documentation
├── eda_output/                    # EDA visualizations and reports
│   ├── cardinality_distribution.png
│   ├── column_types_distribution.png
│   ├── correlation_matrix.png
│   ├── default_distribution.png
│   ├── eda_report.txt
│   ├── key_numeric_distributions.png
│   ├── loan_status_distribution.png
│   ├── missing_percentage_bar.png
│   └── missing_values_heatmap.png
├── logs/
│   └── ssl/                       # Training logs
│       └── pretraining.log
├── models/                        # Saved model checkpoints
│   ├── ssl_encoder.pt
│   ├── ssl_encoder_epoch5.pt
│   ├── ssl_encoder_epoch10.pt
│   ├── ssl_encoder_epoch15.pt
│   ├── ssl_encoder_epoch20.pt
│   └── ssl_encoder_final.pt
├── results/                       # Experiment results
│   ├── baselines/                 # Baseline model comparisons
│   │   ├── baseline_comparison_results.csv
│   │   ├── baseline_results.csv
│   │   ├── best_feature_set.txt
│   │   └── temp_*.csv
│   ├── ssl/                       # SSL evaluation results
│   │   └── ssl_results.csv
│   └── visualizations/            # Embedding visualizations
│       ├── cluster_risk_profile.pdf
│       ├── cluster_risk_profile.png
│       ├── embeddings_tsne.pdf
│       ├── embeddings_tsne.png
│       ├── embeddings_umap.pdf
│       ├── embeddings_umap.png
│       ├── feature_correlation.pdf
│       ├── feature_correlation.png
│       ├── risk_score_distribution.pdf
│       ├── risk_score_distribution.png
│       ├── silhouette_analysis.pdf
│       └── silhouette_analysis.png
├── src/                           # Source code
│   ├── baseline_features.py       # Feature preprocessing
│   ├── data_ingest.py             # Data ingestion
│   ├── eda_baseline.py            # EDA scripts
│   ├── inspect_dataset.py         # Dataset inspection
│   ├── run_baselines.py           # Baseline model comparison runner
│   ├── features/                  # Feature engineering
│   │   ├── feature_engineering_advanced.py
│   │   └── feature_engineering_baseline.py
│   ├── ssl/                       # SSL implementation
│   │   ├── contrastive_pretraining.py
│   │   ├── diagnose_embeddings.py
│   │   ├── few_shot_evaluation.py
│   │   ├── augmentations/         # Augmentation strategies
│   │   │   └── scarf.py
│   │   └── losses/                # Loss functions
│   │       └── vicreg.py
│   └── visualizations/            # Visualization scripts
│       └── embedding_visualizations.py
└── requirements.txt               # Python dependencies
```

---

## Problem Framing

### Business Context

Consumer lending institutions face an information asymmetry: they observe extensive applicant data but only learn outcomes for the subset they approve. This creates a structural challenge where:
- **Label sparsity:** Defaults are rare and delayed, limiting training data for new models
- **Selection bias:** Historical models determined which applicants were funded, censoring the outcome distribution
- **Credit access tension:** Conservative approval rules protect against defaults but may exclude creditworthy applicants who don't resemble historically approved profiles

### Why Standard Supervised Learning Is Insufficient

Standard supervised models face two problems in this setting:

1. **Sample inefficiency:** With few labeled defaults, models cannot learn reliable feature interactions or non-linear decision boundaries
2. **Approval bias propagation:** Models trained only on funded loans learn to mimic past approval decisions, not to identify genuine default risk

### Why Self-Supervised Learning Is a Candidate Solution

The key insight: while default labels are scarce, loan application data is abundant. If the data itself contains discoverable structure that correlates with credit risk—even without outcome labels—then we can:
1. Learn this structure from unlabeled applications (pretraining)
2. Adapt it to default prediction with limited labels (few-shot fine-tuning)

---

## Research Hypothesis

**Primary Hypothesis:** Contrastive learning on unlabeled loan applications produces embeddings that capture latent credit risk structure, demonstrated by improved few-shot default prediction compared to supervised models trained on raw features.

**Null Hypothesis:** Contrastive pretraining provides no benefit beyond what supervised models can learn from the same limited labeled data.

---

## Data

### Source
[Lending Club Loan Data (2007-2020)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — publicly available on Kaggle.

### Dataset Construction
- **Temporal scope:** Loans issued 2012-2018 (ensures outcome maturity)
- **Size:** 2.26 million rows raw, subset to 200,000 for SSL pretraining
- **Filtering:** Remove loans with <12 months payment history, missing interest rate, or loan amount >$40,000

### Temporal Split
| Split | Time Period | Size | Labels Available? |
|-------|-------------|------|-------------------|
| SSL Pretraining | 2012-2016 | ~40,000 | Stripped (held out for analysis) |
| Few-Shot Training | 2017 | 50-200 | Yes |
| Validation | 2017 (remainder) | ~5,000 | Yes |
| Test | 2018 | ~5,000 | Yes (for final evaluation) |

### Features
| Category | Features |
|----------|----------|
| Continuous | Loan amount, annual income, debt-to-income ratio, revolving utilization, credit history length, number of open accounts, months since last delinquency, inquiries (6 months), revolving balance |
| Categorical | Homeownership, employment length bucket, loan purpose, verification status, grade (A-G) |
| Derived | Loan-to-income ratio, payment-to-income ratio, credit age, revolving utilization ratio, total credit lines, delinquency score, credit diversity, utilization pressure |

### Target Variable
Binary: 90+ days delinquent or charged off within 24 months of origination.

---

## Baseline Models Tested

This project compares SSL against three supervised baselines:

### Baseline 1: Logistic Regression (LR)
**Method:** L2-regularized logistic regression on standardized features with balanced class weights.

**Strengths:** Fast training, low variance, interpretable, strong when linear assumptions hold.
**Weaknesses:** Cannot capture feature interactions, degrades with very small samples.

**Implementation:** `sklearn.linear_model.LogisticRegression` with `C=1.0`, `class_weight='balanced'`, `max_iter=1000`.

**Performance at N=50 (Advanced Plan 1):** **0.5768 AUC** — This is the baseline that SSL consistently failed to beat.

---

### Baseline 2: LightGBM (LGBM)
**Method:** Gradient-boosted trees with default hyperparameters and early stopping.

**Strengths:** Handles mixed data types natively, captures non-linear interactions, robust to irrelevant features.
**Weaknesses:** Requires more labeled data than linear models, prone to overfitting with very few labels.

**Implementation:** `lightgbm.LGBMClassifier` with `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `class_weight='balanced'`.

**Performance at N=50 (Advanced Plan 1):** **0.5506 AUC**

---

### Baseline 3: PCA + Logistic Regression
**Method:** Principal Component Analysis on unlabeled applications (retain 95% variance) → logistic regression on PCA-transformed features.

**Strengths:** Leverages unlabeled data, reduces dimensionality (helps few-shot regime), computationally trivial.
**Weaknesses:** PCA finds variance-maximizing directions, not risk-discriminating directions; linear transformation only.

**Implementation:** `sklearn.decomposition.PCA(n_components=0.95)` → `LogisticRegression` on PCA features.

**Performance at N=50 (Advanced Plan 1):** **0.5821 AUC**

---

## Experimental Design

### Phase 1: Self-Supervised Pretraining

**Methods Tested:**
- **SimCLR:** Contrastive learning with NT-Xent loss
- **VICReg:** Variance-Invariance-Covariance regularization

**Augmentation Strategies:**
- **Gaussian Noise + Masking:** Noise applied to continuous features, random feature dropout
- **SCARF:** Random feature corruption with values from other samples

**Encoder Architecture:** 3-layer MLP with BatchNorm → 128-dimensional embedding

### Phase 2: Few-Shot Evaluation

**Procedure:**
1. Freeze pretrained encoder
2. Train linear classifier (logistic regression) on frozen embeddings using N labeled examples
3. Vary N ∈ {20, 50, 100, 200} defaults (balanced with non-defaults)
4. Repeat with 5 random seeds at each N to measure variance
5. Report mean ± std AUC on held-out test set

**Comparison:** All baselines receive identical labeled subsets. This ensures fair comparison—no method sees more labels than another.

### Feature Engineering Plans

| Plan | Description | Features |
|------|-------------|----------|
| **Baseline Plan 1** | 18 base + 5 ratio features | 23 features |
| **Baseline Plan 2** | 18 base + 10 raw features | 28 features |
| **Advanced Plan 1** | 18 base + 5 ratios + 10 raw + 7 risk scores | 39 features |

---

## Results

### Summary of All Experiments

| Experiment | Feature Set | Augmentation | SSL Method | SSL AUC (N=50) | LR Baseline (N=50) | Improvement |
|------------|-------------|--------------|------------|----------------|---------------------|--------------|
| Exp 1 | Baseline Plan 2 (28 features) | Gaussian + Mask | SimCLR | 0.5491 | 0.5768 | -0.0277 |
| Exp 2 | Advanced Plan 1 (39 features) | Gaussian + Mask | SimCLR | 0.5420 | 0.5768 | -0.0349 |
| Exp 3 | Advanced Plan 1 (39 features) | SCARF | SimCLR | 0.5392 | 0.5768 | -0.0376 |
| Exp 4 | Advanced Plan 1 (39 features) | SCARF | VICReg | 0.5179 | 0.5768 | -0.0590 |

### Baseline Performance (Advanced Plan 1, N=50)

| Model | AUC |
|-------|-----|
| **Logistic Regression (LR)** | **0.5768** |
| PCA + LR | 0.5821 |
| LightGBM | 0.5506 |

### Key Findings

1. **SSL consistently underperforms baseline LR** across all feature sets, augmentations, and SSL methods tested.

2. **The gap widens as labeled data increases** — SSL is worse at N=50 (-0.0277) than at N=20 (-0.0002).

3. **VICReg performed worst** among all SSL methods tested (0.5179 AUC at N=50).

4. **SCARF augmentation did not improve SSL performance** over Gaussian noise + masking.

5. **Embeddings are linear** — MLP probe did not outperform linear probe across any experiment.

6. **PCA + LR performed best among baselines** (0.5821 AUC at N=50), suggesting that linear dimensionality reduction is more effective than contrastive learning for this dataset.

### Conclusion

> **"Contrastive self-supervised learning on Lending Club loan data does not produce representations that improve few-shot default prediction compared to a supervised logistic regression baseline. Across all feature sets, augmentation strategies, and SSL methods tested, SSL consistently underperformed the baseline, with the gap widening as labeled data increased. The null hypothesis could not be rejected."**

---

## Failure Modes Observed

| Failure Mode | Detection | Observed |
|--------------|-----------|----------|
| Augmentation destroys risk signal | SSL linear probe worse than raw LR | ✅ All experiments |
| Representation collapse | Near-zero embedding variance | ❌ Variance healthy (0.45-0.65) |
| Embeddings are linear | MLP probe ≈ Linear probe | ✅ All experiments |
| No improvement over PCA baseline | SSL ≈ PCA + LR | ✅ All experiments |

---

## Computational Requirements

### Hardware
- **Target:** Standard laptop with Intel Core i5, 8-16 GB RAM
- **No GPU required:** All experiments designed for CPU execution

### Expected Runtime
| Component | Estimated Time |
|-----------|---------------|
| Data preprocessing | 5-10 minutes |
| SSL pretraining (20 epochs, 200k samples) | 2-4 hours |
| Baseline comparison (all feature sets) | 10-30 minutes |
| Few-shot evaluation (5 seeds × 4 sizes) | 30-60 minutes |
| **Total (full experiment suite)** | **4-8 hours** |

### Storage
- Raw data: ~200 MB
- Processed features: ~20 MB
- Model checkpoints: ~50 MB
- Experiment results: ~10 MB

---

## Repository Setup

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/contrastive-borrower-representations.git
cd contrastive-borrower-representations

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Download and preprocess data
python src/data_ingest.py
python src/baseline_features.py

# Run baseline comparison across all feature sets
python src/run_baselines.py

# Run SSL pretraining on best feature set
python src/ssl/contrastive_pretraining.py

# Evaluate SSL few-shot performance
python src/ssl/few_shot_evaluation.py

# Generate embedding visualizations
python src/visualizations/embedding_visualizations.py
```

---

## Results Interpretation Guide

### What Would Be Convincing Evidence?

**Strong positive result:** SSL-pretrained linear probe achieves AUC ≥ 0.03 higher than the best supervised baseline at N=50 labeled defaults.

**Moderate positive result:** SSL outperforms baselines at very small N (20-50) but the gap closes by N=200.

**Negative result:** No statistically significant improvement over PCA + supervised baselines. (✅ This project)

**Concerning result:** SSL underperforms raw-feature supervised models. (✅ This project)

### All Results Are Informative

This project is designed as an investigation, not a demonstration. A negative result that is well-documented and analyzed demonstrates stronger ML thinking than a positive result achieved through overfitting or cherry-picking.

---

## Limitations and Scope Boundaries

### This Project Does:
- Investigate whether contrastive pretraining helps in label-scarce credit settings
- Test specific augmentation strategies for tabular credit data
- Compare SSL against meaningful baselines in controlled experiments
- Produce learning curves that quantify sample efficiency gains
- **Provide a well-documented negative result** for the research community

### This Project Does NOT:
- Build a production underwriting system
- Address fair lending compliance or demographic bias
- Generalize across multiple datasets or economic conditions
- Claim state-of-the-art performance on credit default prediction

### Known Limitations
- **Single dataset:** Findings may not generalize to other credit products or markets
- **Historical data:** Relationships learned may change in different economic regimes
- **Feature completeness:** Real underwriting uses additional data sources not available in Lending Club
- **Selection bias unresolved:** The project cannot fully separate risk from policy without rejected application outcomes

---

## References

### Core Methods
- Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. (SimCLR)
- Bardes, A., et al. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022. (VICReg)
- Bahri, D., et al. "SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption." ICLR 2022. (SCARF)

### Credit Risk Context
- Lessmann, S., et al. "Benchmarking state-of-the-art classification algorithms for credit scoring." European Journal of Operational Research, 2015.

### Evaluation Methodology
- Khosla, P., et al. "Supervised Contrastive Learning." NeurIPS 2020.

---

## License

This project is for educational and portfolio demonstration purposes. The Lending Club dataset is publicly available under its original terms.

---

## Contact

Ken Ira Lacson — keniralacson@gmail.com — kira-ml

*This project was developed as a portfolio demonstration of applied machine learning research thinking, experimental design, and self-supervised representation learning for structured data. While the null hypothesis was not rejected, the systematic investigation and documentation of this negative result contributes valuable evidence to the understanding of SSL's limitations in credit risk.*
