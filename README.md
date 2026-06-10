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
│   ├── download_data.py           # Script to download Lending Club data
│   └── preprocess.py              # Data cleaning and feature engineering
├── models/
│   ├── encoder.py                 # MLP encoder architecture
│   ├── contrastive_pretraining.py # SimCLR-style contrastive learning
│   └── downstream_classifier.py   # Linear probe and fine-tuning
├── baselines/
│   ├── logistic_regression.py     # Baseline 1
│   ├── gradient_boosted_trees.py  # Baseline 2
│   └── pca_supervised.py          # Baseline 3
├── evaluation/
│   ├── few_shot_experiment.py     # Few-shot evaluation framework
│   ├── metrics.py                 # AUC, F1, learning curves
│   └── visualize.py               # t-SNE/UMAP embeddings
├── experiments/
│   └── run_all_experiments.py     # Main experiment runner
├── notebooks/
│   └── exploratory_analysis.ipynb # Initial data exploration
├── results/
│   └── (experiment outputs)
└── requirements.txt
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

### What Should the Model Learn?

If the hypothesis holds, the encoder should capture:
- **Credit profile similarity:** Applicants with similar credit histories should be close in embedding space
- **Risk-compensating relationships:** Strong income partially compensating for thin credit files
- **Interaction geometry:** High loan amount + low income creating distinct risk regions

### What Should the Model NOT Learn?
- Trivial input reconstruction
- Clustering by non-risk attributes (geography, loan purpose without risk differentiation)
- Historical approval propensity rather than fundamental risk

---

## Data

### Source
[Lending Club Loan Data (2007-2020)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — publicly available on Kaggle.

### Dataset Construction
- **Temporal scope:** Loans issued 2012-2018 (ensures outcome maturity)
- **Size:** ~50,000 loans for SSL pretraining, separate temporal holdout for evaluation
- **Filtering:** Remove loans with <12 months payment history, missing interest rate, or loan amount >$40,000

### Temporal Split (Critical for Realistic Evaluation)
| Split | Time Period | Size | Labels Available? |
|-------|-------------|------|-------------------|
| SSL Pretraining | 2012-2016 | ~40,000 | Stripped (held out for analysis) |
| Few-Shot Training | 2017 | 50-200 | Yes |
| Validation | 2017 (remainder) | ~5,000 | Yes |
| Test | 2018 | ~5,000 | Yes (for final evaluation) |

This temporal design simulates realistic deployment: representations are learned on historical applications, then adapted to recent labeled data and evaluated on future applications.

### Features
| Category | Features |
|----------|----------|
| Continuous | Loan amount, annual income, debt-to-income ratio, revolving utilization, credit history length, number of open accounts, months since last delinquency, inquiries (6 months), revolving balance |
| Categorical | Homeownership, employment length bucket, loan purpose, verification status, grade (A-G) |
| Derived | Income per account, revolving balance per account, loan-to-income ratio |

### Target Variable
Binary: 90+ days delinquent or charged off within 24 months of origination.

---

## Experimental Design

### Phase 1: Self-Supervised Pretraining

**Method:** SimCLR-style contrastive learning adapted for mixed-type tabular data.

**Augmentation Strategy (Core Design Decision):**
- **Continuous features:** Gaussian noise proportional to measurement error estimates
- **Ordinal features:** Small shifts along ordinal scale (e.g., employment length ±1 category)
- **Categorical features:** Replacement within semantic groups (homeownership types, loan purpose categories)
- **Feature masking:** Random 20% dropout to force inference from context

**Encoder Architecture:** 2-3 layer MLP → 64-dimensional embedding with L2 normalization.

**Loss:** NT-Xent (normalized temperature-scaled cross-entropy) with temperature 0.1.

**Monitoring:** Contrastive loss, embedding variance (collapse detection), t-SNE visualization colored by held-out credit grade.

### Phase 2: Few-Shot Evaluation

**Procedure:**
1. Freeze pretrained encoder
2. Train linear classifier (logistic regression) on frozen embeddings using N labeled examples
3. Vary N ∈ {20, 50, 100, 200} defaults (balanced with non-defaults)
4. Repeat with 10 random seeds at each N to measure variance
5. Report mean ± std AUC and F1 on held-out 2018 test set

**Comparison:** All baselines receive identical labeled subsets. This ensures fair comparison—no method sees more labels than another.

### Ablation Studies

1. **Augmentation sensitivity:** Vary noise level, masking probability, categorical swap rate
2. **Method comparison:** Contrastive vs. masked feature prediction (denoising autoencoder)
3. **Architecture sensitivity:** Embedding dimensions {32, 64, 128}
4. **PCA baseline strength:** How much does the PCA baseline improve with component count?

---

## Baseline Solutions

### Baseline 1: Supervised Logistic Regression on Raw Features

**Method:** L2-regularized logistic regression on standardized features.

**Why This Baseline:** Represents the simplest reasonable model. If SSL cannot beat linear classification on raw features, the pretraining adds no value.

**Expected Behavior:**
- **Strengths:** Fast training, low variance, interpretable, strong when linear assumptions hold
- **Weaknesses:** Cannot capture feature interactions, degrades with very small samples, no unlabeled data utilization

**Implementation:** `sklearn.linear_model.LogisticRegression` with `C` selected via cross-validation.

---

### Baseline 2: Gradient-Boosted Trees on Raw Features

**Method:** LightGBM classifier with default hyperparameters and early stopping.

**Why This Baseline:** Represents the dominant supervised approach for tabular data in practice. This is the strongest supervised baseline and the one SSL must convincingly outperform.

**Expected Behavior:**
- **Strengths:** Handles mixed data types natively, captures non-linear interactions, robust to irrelevant features
- **Weaknesses:** Requires more labeled data than linear models, prone to overfitting with very few labels, no unlabeled data mechanism

**Implementation:** `lightgbm.LGBMClassifier` with 100 trees, max depth 6, learning rate 0.1, early stopping on 20% validation split.

---

### Baseline 3: PCA + Supervised Classifier

**Method:** Principal Component Analysis on 40,000 unlabeled applications → retain components explaining 95% variance → train logistic regression and LightGBM on PCA-transformed features.

**Why This Baseline:** This is the most direct comparison to SSL pretraining. Both methods use unlabeled data to learn a transformation. If PCA + supervised performs comparably to SSL + supervised, then linear dimensionality reduction is sufficient and contrastive learning is unnecessary.

**Expected Behavior:**
- **Strengths:** Leverages unlabeled data, reduces dimensionality (helps few-shot regime), computationally trivial
- **Weaknesses:** PCA finds variance-maximizing directions, not risk-discriminating directions; linear transformation only; high-variance features (income scale) may dominate

**Implementation:** `sklearn.decomposition.PCA` on unlabeled set → `LogisticRegression` and `LGBMClassifier` on PCA features.

---

## Success Criteria

### Primary Metric
**AUC on 2018 test set** for the SSL-pretrained linear probe compared to the best supervised baseline at each few-shot sample size.

### Success Threshold
- Statistically significant improvement (p < 0.05, paired t-test across 10 seeds) at N ≤ 100 labeled defaults
- Minimum AUC improvement of ≥ 0.03 over the best baseline at N = 50

### Secondary Metrics
- **Few-shot learning curve:** Steeper initial slope for SSL-pretrained models indicates better sample efficiency
- **Embedding quality:** Silhouette score of default vs. non-default clusters in embedding space (using held-out outcomes)
- **Representation stability:** Lower variance across random seeds indicates more robust representations

---

## Failure Modes and Diagnostics

| Failure Mode | Detection | Potential Mitigation |
|--------------|-----------|---------------------|
| Augmentation destroys risk signal | SSL linear probe worse than raw logistic regression | Reduce augmentation strength, test alternative augmentations |
| Representation collapse | Near-zero embedding variance, uniform embeddings | Add redundancy reduction, switch to non-contrastive method |
| Approval bias dominates representations | Embeddings predict loan grade better than default risk | Acknowledge limitation; test on less policy-influenced features |
| Temporal degradation | Large validation-test performance gap | Investigate temporal feature shifts, reduce temporal gap |
| No improvement over PCA baseline | PCA + supervised ≈ SSL + supervised | Conclude linear structure sufficient; contrastive method unjustified |

---

## Computational Requirements

### Hardware
- **Target:** Standard laptop with Intel Core i5, 8-16 GB RAM
- **No GPU required:** All experiments designed for CPU execution

### Expected Runtime
| Component | Estimated Time |
|-----------|---------------|
| Data preprocessing | 5-10 minutes |
| SSL pretraining (200 epochs, 40k samples) | 1-3 hours |
| Baseline training (all baselines) | 10-30 minutes |
| Few-shot experiments (10 seeds × 4 sample sizes × all methods) | 30-60 minutes |
| Ablation studies | 1-2 hours |
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
python data/download_data.py
python data/preprocess.py

# Run baseline experiments
python experiments/run_all_experiments.py --mode baselines

# Run SSL pretraining + few-shot evaluation
python experiments/run_all_experiments.py --mode full

# Generate results and visualizations
python evaluation/visualize.py
```

### Dependencies

```
torch>=1.9.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
umap-learn>=0.5.0
```

---

## Results Interpretation Guide

### What Would Be Convincing Evidence?

**Strong positive result:** SSL-pretrained linear probe achieves AUC ≥ 0.03 higher than the best supervised baseline at N=50 labeled defaults, with the gap narrowing but persisting at larger N. This suggests the representation captures generalizable risk structure.

**Moderate positive result:** SSL outperforms baselines at very small N (20-50) but the gap closes by N=200. This suggests pretraining is valuable only in extreme label scarcity.

**Negative result:** No statistically significant improvement over PCA + supervised baselines. This suggests that the risk-relevant structure in the data is linear and PCA suffices, or that contrastive augmentations are not well-suited to credit data.

**Concerning result:** SSL underperforms raw-feature supervised models. This suggests augmentations destroyed useful signal or the pretraining objective learned misleading structure.

### All Results Are Informative

This project is designed as an investigation, not a demonstration. A negative result that is well-documented and analyzed (showing what was learned, why it failed, what diagnostics revealed) demonstrates stronger ML thinking than a positive result achieved through overfitting or cherry-picking.

---

## Limitations and Scope Boundaries

### This Project Does:
- Investigate whether contrastive pretraining helps in label-scarce credit settings
- Test specific augmentation strategies for tabular credit data
- Compare SSL against meaningful baselines in controlled experiments
- Produce learning curves that quantify sample efficiency gains

### This Project Does NOT:
- Build a production underwriting system
- Address fair lending compliance or demographic bias
- Generalize across multiple datasets or economic conditions
- Claim state-of-the-art performance on credit default prediction
- Handle the full complexity of real-world credit decisioning (interactive features, time-varying attributes, multi-product relationships)

### Known Limitations
- **Single dataset:** Findings may not generalize to other credit products or markets
- **Historical data:** Relationships learned may change in different economic regimes
- **Feature completeness:** Real underwriting uses additional data sources (bureau data, alternative data) not available in Lending Club
- **Selection bias unresolved:** The project cannot fully separate risk from policy without rejected application outcomes

---

## References

### Core Methods
- Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. (SimCLR)
- Chen, X., & He, K. "Exploring Simple Siamese Representation Learning." CVPR 2021. (SimSiam)
- Bahri, D., et al. "SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption." ICLR 2022. (Contrastive learning for tabular data)
- Yoon, J., et al. "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain." NeurIPS 2020.

### Credit Risk Context
- Lessmann, S., et al. "Benchmarking state-of-the-art classification algorithms for credit scoring." European Journal of Operational Research, 2015.
- Thomas, L.C., et al. "Credit Scoring and Its Applications." SIAM, 2017.

### Evaluation Methodology
- Khosla, P., et al. "Supervised Contrastive Learning." NeurIPS 2020.

---

## License

This project is for educational and portfolio demonstration purposes. The Lending Club dataset is publicly available under its original terms.

---

## Contact

Ken Ira Lacson — keniralacson@gmail.com — kira-ml

*This project was developed as a portfolio demonstration of applied machine learning research thinking, experimental design, and self-supervised representation learning for structured data.*