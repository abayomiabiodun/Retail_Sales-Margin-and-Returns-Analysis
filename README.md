# Retail Sales 2022–2025: Profit-Margin Prediction & Return-Risk Feasibility

An end-to-end, leakage-audited data-science project that (A) predicts the **profit margin** of a retail transaction *before* a discount is approved, and (B) tests honestly whether **product returns** can be predicted at the time of sale.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Validation](https://img.shields.io/badge/Validation-Chronological%20Split-brightgreen)

---

## Table of Contents
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Key Insights & Recommendations](#-key-insights--recommendations)
- [Leakage & Fit Safeguards](#-leakage--fit-safeguards)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Author](#-author)

---

## Business Problem

The company sells electronics, appliances, furniture and office supplies across **6 countries and 12 stores** through four channels (Online, Retail Store, B2B Portal, Phone Order). About 58% of lines carry a discount and about 10% are returned, yet there is no systematic way to see **what a discount will do to margin before it is granted**.

| Problem | Type | Question |
|---|---|---|
| **A (primary)** | Regression | Predict margin `m = Profit / Sales_Amount` using only information known at the moment of sale |
| **B (secondary)** | Classification | Can we predict `Return_Flag` at the time of sale? (run as a feasibility study) |

**Success criteria (defined before modelling)**

| | Metric | Target |
|---|---|---|
| A | MAE on the untouched 2025 hold-out | Well below the mean baseline and < 2 margin points |
| A | R² on 2025 hold-out | > 0.80 and within 0.05 of CV R² |
| B | ROC-AUC / PR-AUC | Materially above 0.5 / prevalence, otherwise declare "not predictable" |
| Both | Leakage | No post-sale fields; shuffled-target R² ≈ 0 |

---

## Dataset

- **Files:** `Sales_transactions_2022_2025.csv` and `sales_data_dictionary.csv`
- **Size:** 18,045 raw rows → **16,986 modelling rows** (94%) after cleaning

**Data issues found and fixed**
- Duplicate rows
- Six families of spelling and casing variants (gender, category, payment method, status, region, city)
- Impossible values and 117 rows with broken `Sales = Qty × Price × (1 − d)` / `Profit = Sales − Cost` arithmetic
- Cancelled orders (no realised margin, cannot be returned), excluded from both models
- `Order_ID` is **not** a true order key (shared across customers and dates), so no order-level features were built from it

---

## Approach

1. **Audit & clean** with explicit, logged rules
2. **Define population and target** carefully
3. **Feature engineering from first principles.** Margin is a hyperbolic function of discount, `m = 1 − c / (1 − d)`, where `c` is the product's cost-to-list-price ratio. This is *linear* in `1/(1 − d)`, so encoding it turns a non-linear problem into a nearly linear one.
4. **Feature reduction & scaling:** Cramér's V, VIF pruning for linear models, PCA for intrinsic dimensionality
5. **Honest validation:** chronological split (**train 2022–2024, test 2025**), time-series CV, all preprocessing fitted inside the pipeline
6. **Model comparison:** mean baseline, Ridge, Lasso, Random Forest, tuned Gradient Boosting
7. **Diagnostics:** learning curves, validation curves, train/CV/test gaps
8. **Leakage audit** and **business impact analysis**

**Other engineered features:** cyclical calendar encodings (sin/cos), past-only customer history, discount/price algebra, and a product cost-ratio encoder learned on training data only.

---

## Key Results

### Problem A: Margin regression (2025 hold-out)

| Item | Result |
|---|---|
| **Selected model** | Lasso on engineered features (chosen by CV only) |
| **R²** | **0.96** |
| **MAE** | **1.5 margin points** (vs 7.6 for the mean baseline, ~80% lower error) |
| **Train / CV / Test R²** | 0.959 / 0.958 / 0.961 (no over-fit, no under-fit) |

Ridge, Lasso, Random Forest and tuned Gradient Boosting all land at R² 0.958–0.961. The **features** do the work, so the simple, explainable linear model wins. Remaining error is consistent with noise in per-line cost (~2% SD of the cost ratio within a product).

### Problem B: Return prediction

| Item | Result |
|---|---|
| **Verdict** | **Not predictable** from pre-sale data |
| Hold-out ROC-AUC | 0.49 (95% CI includes 0.5) |
| PR-AUC | ≈ prevalence (0.11) |
| Permutation test | p ≈ 0.3 |
| Gradient Boosting | Over-fits (train AUC 0.82 vs test 0.48) |

Even post-sale rating and delivery days carry almost no signal.

---

##  Key Insights & Recommendations

1. **Add a margin guardrail to the discount workflow.** Discounts of 0–10% never lose money. At 20%, about 1 in 10 lines is loss-making, and at 25% almost 3 in 10; 44% of all loss-making lines sit at ≥ 25% discount. Show expected margin before approval and require manager sign-off above ~15%.
2. **Use the break-even discount `d* = 1 − c` per product.** Lowest headroom: Copy Paper Box (~18%), Printer Plus (~19%), EdgeTab Tablet (~20%), Meeting Table (~21%).
3. **Be realistic about the prize.** Capping all discounts at 20% recovers ~0.5% of profit, and at 15% ~2.6%, *assuming demand does not react*. Treat this as an upper bound and A/B test before roll-out.
4. **Deploy the simple model.** It is accurate, transparent and fast to retrain. Retrain annually, monitor MAE monthly, and alert above ~2.5 margin points.
5. **Do not build a returns model on current data.** Collect richer pre-sale and delivery data (defect/batch info, delivery events, customer return history, return reasons) and re-run the study.
6. **Fix data governance:** a true unique `Order_ID`, controlled vocabularies at entry, validation rules (quantity > 0, discount within the 0–30% grid, age 16–90), and a nightly Sales/Profit arithmetic check.

---

##  Leakage & Fit Safeguards

- Post-sale fields excluded; field-level availability audit
- **Shuffled-target test:** R² ≈ 0
- **Leaky-model demo:** adding `Cost/Sales` gives R² = 1.00, showing what leakage looks like
- **Random vs chronological CV** agree (0.959 vs 0.958)
- Customer-history features verified **past-only**
- An `Order_ID`-based feature was caught as leaky and **removed**
- Encoder-leakage check: target-derived encoder fitted on train only vs all data

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### 3. Add the data
Place these files next to the notebook:
- `Sales_transactions_2022_2025.csv`
- `sales_data_dictionary.csv`

### 4. Run the notebook
```bash
jupyter notebook Retail_Sales_Margin_and_Returns_Analysis.ipynb
```
Choose **Kernel → Restart & Run All**.

---

## Repository Structure

```
├── Retail_Sales_Margin_and_Returns_Analysis.ipynb  # Full end-to-end notebook
├── Sales_transactions_2022_2025.csv                # Transactions (add locally)
├── sales_data_dictionary.csv                       # Data dictionary (add locally)
└── README.md
```

---

##  Limitations

- The data shows synthetic-looking regularities (stable product cost ratios, near-uniform return rate), so results may be cleaner than a real retailer's. Validate on live data.
- The model predicts margin **given** the discount. It does not estimate how discounts change demand (no causal claim).
- The what-if cap analysis assumes unchanged volume.
- 2025 is a single hold-out year, so a rolling yearly re-evaluation is advisable.

---

##  Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
