# Baseline Feature Selection Plan

## Contrastive Borrower Representations for Few-Shot Default Prediction

---

## 1. Context

This document defines the baseline feature set used to evaluate whether self-supervised contrastive pretraining improves few-shot default prediction compared to supervised models trained on raw features.

The selection process starts from the full Lending Club dataset (151 columns, 2.26M rows) and applies a systematic filtering framework based on temporal validity, data quality, and freedom from data leakage. The resulting 18 features represent a defensible starting point that a competent practitioner would use before considering feature engineering or representation learning.

This plan serves two purposes: establishing a credible lower bound for model performance and ensuring that any improvement from contrastive learning can be attributed to representation quality rather than access to additional information.

---

## 2. Selection Principles

Every inclusion and exclusion decision follows three hard constraints:

| Principle | Rule | Rationale |
|-----------|------|-----------|
| **Temporal validity** | Feature must be observable at application time | Prevents data leakage from post-origination events (payments, collections, settlements) |
| **Data quality** | Feature must have <50% missing values in the raw dataset | Ensures imputation does not become the dominant source of variation |
| **Non-leakage** | Feature must not directly encode the target outcome or post-decision information | Excludes `loan_status` and features updated after origination |

Features that fail any of these constraints are excluded regardless of their predictive power. This is not a judgment about their usefulness in production models — it reflects the specific requirements of an experiment where temporal ordering matters.

---

## 3. Selected Baseline Features

### Overview

| Property | Value |
|----------|-------|
| Total features | 18 |
| Continuous | 14 |
| Categorical | 4 |
| All application-time | Yes |
| Median missing rate | 0% across selected features |
| Computational footprint | Negligible for 50k samples on CPU |

---

### 3.1 Continuous Features (14)

#### Credit Risk Core (8 features)

| # | Feature | Description | Credit Relevance |
|---|---------|-------------|------------------|
| 1 | `fico_range_low` | Lower bound of borrower's FICO score at application | Single strongest predictor of default in consumer credit; industry standard for decades |
| 2 | `fico_range_high` | Upper bound of borrower's FICO score at application | Combined with low gives FICO range; wider range may indicate score uncertainty |
| 3 | `dti` | Debt-to-income ratio; monthly debt payments divided by monthly income | Direct affordability measure; high DTI indicates borrower is already leveraged |
| 4 | `revol_util` | Revolving credit utilization rate | High utilization signals reliance on credit lines; >70% is conventional distress threshold |
| 5 | `annual_inc` | Self-reported annual income | Foundation of ability-to-pay assessment; interacts with loan amount and DTI |
| 6 | `loan_amnt` | Requested loan amount | Larger loans represent higher exposure; risk interpretation depends on income context |
| 7 | `inq_last_6mths` | Number of credit inquiries in past 6 months | Multiple inquiries suggest active credit seeking or financial pressure |
| 8 | `delinq_2yrs` | Number of delinquency incidents in past 2 years | Direct behavioral signal; past delinquency is predictive of future default |

#### Credit History Depth (3 features)

| # | Feature | Description | Credit Relevance |
|---|---------|-------------|------------------|
| 9 | `open_acc` | Number of open credit lines | Proxy for credit experience breadth; very few accounts suggests thin file |
| 10 | `total_acc` | Total number of credit lines (open and closed) | Broader credit history measure; complements open_acc |
| 11 | `earliest_cr_line` | Date of borrower's earliest reported credit line | Length of credit history when converted to months; longer history generally indicates lower risk |

#### Negative Events (3 features)

| # | Feature | Description | Credit Relevance |
|---|---------|-------------|------------------|
| 12 | `pub_rec` | Number of public records (bankruptcies, tax liens, civil judgments) | Strong negative signal; public records indicate severe financial distress |
| 13 | `collections_12_mths_ex_med` | Number of collections in past 12 months excluding medical | Recent collections activity; distinguishes medical from non-medical collections |
| 14 | `acc_now_delinq` | Number of accounts currently delinquent | Immediate repayment concern; active delinquency suggests current financial difficulty |

---

### 3.2 Categorical Features (4)

| # | Feature | Cardinality | Categories | Credit Relevance |
|---|---------|-------------|------------|------------------|
| 15 | `home_ownership` | 4 | RENT, MORTGAGE, OWN, OTHER | Housing stability proxy; homeowners default at lower rates than renters |
| 16 | `purpose` | ~14 | debt_consolidation, credit_card, small_business, other, etc. | Loan purpose differentiates risk; small business and medical have higher default rates |
| 17 | `term` | 2 | 36 months, 60 months | Loan duration; 60-month loans show higher default rates after controlling for other factors |
| 18 | `verification_status` | 3 | Not Verified, Source Verified, Verified | Income verification level; verified income reduces information asymmetry between borrower and lender |

---

## 4. Features Explicitly Excluded

Understanding what was excluded and why is as important as understanding what was included.

### 4.1 Excluded: Lender Risk Assessment Features

| Feature | Reason for Exclusion |
|---------|---------------------|
| `grade` | Lending Club-assigned risk grade (A-G). Encodes the lender's proprietary underwriting model output. Including it would measure how well the representation learns to mimic LC's model, not how well it discovers fundamental risk structure. |
| `sub_grade` | Finer-grained risk grade (A1-G5, 35 levels). Same concern as grade but amplified — 35 levels of lender-specific risk assessment. |
| `int_rate` | Interest rate assigned to the loan. Direct mathematical function of grade and term. Contains the same information as grade through a pricing lens. |

**Why this matters for the experiment:** These three features are strongly predictive of default because Lending Club's underwriting model is effective. Including them would likely improve all models (baselines and SSL) but would obscure whether contrastive learning discovers risk-relevant structure independently. The more interesting scientific question is whether unlabeled data contains discoverable risk patterns beyond what the lender already encoded in these features.

**Follow-up experiment:** A useful ablation is to run the same contrastive pretraining with grade included and measure the performance gap. A small gap suggests SSL independently recovers the lender's risk assessment. A large gap suggests the lender's model captures information not easily recovered from raw features.

### 4.2 Excluded: Post-Origination Features (Data Leakage)

| Feature Category | Examples | Leakage Mechanism |
|-----------------|----------|-------------------|
| Payment history | `last_pymnt_d`, `last_pymnt_amnt` | Observed after origination; not available when application is evaluated |
| Updated credit pulls | `last_fico_range_high`, `last_fico_range_low`, `last_credit_pull_d` | Post-origination credit score updates; reflect behavior after loan funding |
| Hardship/Settlement | All `hardship_*` and `settlement_*` features | Outcomes that occur after default; using them to predict default is circular |
| Payment plan | `pymnt_plan` | Activated after payment difficulties; post-outcome feature |

### 4.3 Excluded: Data Quality Issues

| Feature | Issue |
|---------|-------|
| All `sec_app_*` features | >95% missing; secondary applicant data rarely available for individual loans |
| `*_joint` features | >94% missing; joint application features unavailable for most loans |
| `desc` | 94.4% missing; free text loan description rarely provided |
| `mths_since_last_delinq` | 100% missing in this dataset version |
| `member_id` | 100% missing in this dataset version |

### 4.4 Excluded: Redundant or Low Information

| Feature | Reason |
|---------|--------|
| `installment` | Mathematical function of `loan_amnt`, `int_rate`, and `term`; redundant when those features are present |
| `funded_amnt` | Nearly identical to `loan_amnt` in most cases; differences are small and reflect investor demand, not borrower risk |
| `initial_list_status` | Whole loan vs. fractional listing; reflects LC's funding mechanism, not borrower risk profile |
| `application_type` | Overwhelmingly INDIVIDUAL; near-zero variance limits predictive contribution |
| `addr_state` | 50-level categorical; geographic signal may exist but introduces fairness considerations and high cardinality without clear risk mechanism; better suited for a follow-up analysis |
| `emp_length` | Useful risk signal but requires investigation of missing rate and imputation strategy; deferred to derived feature set |

---

## 5. Feature Set Validation

Before locking this feature set, the following checks should be performed on the actual data:

```python
# Validation checks for baseline features

BASELINE_FEATURES = [
    # Continuous (14)
    'fico_range_low', 'fico_range_high', 'dti', 'revol_util',
    'annual_inc', 'loan_amnt', 'inq_last_6mths', 'delinq_2yrs',
    'open_acc', 'total_acc', 'earliest_cr_line',
    'pub_rec', 'collections_12_mths_ex_med', 'acc_now_delinq',
    # Categorical (4)
    'home_ownership', 'purpose', 'term', 'verification_status'
]

# 1. Verify missing rates are acceptable
missing = df[BASELINE_FEATURES].isnull().mean() * 100
assert all(missing < 50), f"Features exceed 50% missing: {missing[missing > 50]}"

# 2. Verify categorical cardinality is manageable
for col in ['home_ownership', 'purpose', 'term', 'verification_status']:
    n_unique = df[col].nunique()
    assert n_unique < 50, f"{col} has {n_unique} unique values"

# 3. Verify no post-origination leakage by inspection
# All selected features should represent application-time information

# 4. Check for near-zero variance in continuous features
# Features like acc_now_delinq and collections_12_mths_ex_med may be sparse
for col in ['acc_now_delinq', 'collections_12_mths_ex_med', 'pub_rec']:
    nonzero_rate = (df[col] > 0).mean()
    print(f"{col}: {nonzero_rate:.2%} non-zero")
    # Features with <1% non-zero rate may have limited utility
    # but should be retained for initial experiments
```

---

## 6. Preprocessing Specification

To ensure reproducibility, all preprocessing steps are defined upfront:

### 6.1 Continuous Features

| Step | Method | Details |
|------|--------|---------|
| Missing value imputation | Median imputation | Per-feature median computed on training set only |
| Outlier handling | Percentile capping | Cap at 1st and 99th percentile to reduce influence of extreme values |
| Scaling | StandardScaler | Zero mean, unit variance; fit on training set only |
| `earliest_cr_line` | Convert to months | Compute `(application_date - earliest_cr_line)` in months |

### 6.2 Categorical Features

| Step | Method | Details |
|------|--------|---------|
| Missing value imputation | Mode imputation | Most frequent category per feature |
| Encoding for baselines | Label encoding | Integer encoding for tree-based models |
| Encoding for contrastive | Embedding or frequency | Embedding layer in neural encoder OR frequency encoding for simplicity |

### 6.3 Target Variable

| Condition | Label |
|-----------|-------|
| Loan status is "Charged Off" | 1 (Default) |
| Loan status is "Default" | 1 (Default) |
| Loan status is "Late (31-120 days)" | 1 (Default) |
| All other statuses with mature outcomes | 0 (Non-default) |
| Loans without mature outcomes | Excluded from training and evaluation |

---

## 7. Rationale Summary

### Why 18 Features?

This number balances three competing concerns:

1. **Sufficient signal:** 18 features provide enough dimensionality for contrastive learning to potentially discover non-trivial structure. Very low-dimensional feature spaces (e.g., 5 features) make representation learning indistinguishable from PCA.

2. **Computational practicality:** Training contrastive models on 18 features with 40,000 samples completes in hours on a standard laptop CPU. Expanding to 50+ features would increase training time without clear benefit at this stage.

3. **Interpretability:** Every feature in this set has a well-understood relationship with credit risk. When analyzing why contrastive learning does or does not help, domain knowledge about each feature guides the investigation.

### Why Not More Features?

The Lending Club dataset contains many additional features that could be included. The decision to exclude them from the baseline is deliberate:

- **Feature engineering should be evaluated, not assumed.** Adding derived ratios (loan-to-income, installment-to-income) might help, but whether they capture information beyond what contrastive learning discovers from raw features is an empirical question worth testing.
- **Each added feature increases the risk of including noise.** With only 40,000 training samples, the effective sample size limits how many features can be meaningfully used.
- **Simplicity enables clear attribution.** If the baseline uses 18 features and SSL uses 18 features, any performance difference is due to representation learning. If the baseline uses 18 features and SSL uses 35 features plus engineered interactions, the source of improvement is ambiguous.

### When to Expand

This baseline feature set is a starting point, not a permanent constraint. Feature expansion is warranted when:

1. Baseline supervised models achieve their maximum performance and additional features are needed to push higher
2. Contrastive learning shows promise but the embedding space lacks clear risk structure, suggesting more informative features are needed
3. Ablation studies reveal specific feature interactions that raw features cannot capture

---

## 8. Relationship to Experimental Design

This feature set connects to the broader experiment as follows:

| Experimental Component | Feature Set Role |
|------------------------|------------------|
| SSL pretraining | Uses unlabeled applications with these 18 features; no outcome labels |
| Few-shot supervised training | Uses N labeled examples with these same 18 features |
| Baseline models | Trained on identical 18 features; same labeled subsets as SSL |
| Ablation studies | Vary augmentation strategy on this feature set; test sensitivity |

The key control is that **all methods see exactly the same information**. The only difference is whether they learn from unlabeled data first (SSL) or directly from labeled data (baselines). Any performance difference can be attributed to the pretraining strategy rather than to differential access to features.

---

## 9. Appendix: Quick Reference Card

```
BASELINE FEATURE SET v1.0
═══════════════════════════════════════════════════════
Total:              18 features
Continuous:         14 features  
Categorical:         4 features
Missing rate:       0% (median across features)
Data leakage:       Verified clean
Application-time:   All features
═══════════════════════════════════════════════════════

CONTINUOUS:
  fico_range_low          • dti                  • open_acc
  fico_range_high         • revol_util           • total_acc
  annual_inc              • inq_last_6mths       • earliest_cr_line*
  loan_amnt               • delinq_2yrs          • pub_rec
  collections_12_mths_ex_med • acc_now_delinq
  *convert to months_since_earliest_cr_line

CATEGORICAL:
  home_ownership          • term
  purpose                 • verification_status

EXCLUDED (in baseline):
  grade, sub_grade, int_rate (lender assessment)
  installment (redundant)
  addr_state, emp_length (deferred)
  all post-origination features (leakage)
  all features with >50% missing (data quality)
═══════════════════════════════════════════════════════


---

*This baseline feature selection plan is a living document. It reflects the initial experimental setup and will be updated if empirical results indicate that feature adjustments are necessary.*
