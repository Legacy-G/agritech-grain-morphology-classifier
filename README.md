# AgriTech Grain Morphology Classifier: Multi-Class Dry Bean ML Benchmark

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![Institution](https://img.shields.io/badge/FUT--Minna-Computer%20Engineering-green.svg)](https://futminna.edu.ng/)
[![Dataset](https://img.shields.io/badge/UCI--ML--Repo-Dataset%20602-red.svg)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Final-Year Computer Engineering Project**  
> Department of Computer Engineering, Federal University of Technology, Minna (FUTMINNA)  
> *Course: Artificial Neural Networks & Machine Learning*

---

## 📌 Project Overview
Automated grain grading is a critical operation in agricultural processing and industrial packaging. This project establishes an empirical machine learning benchmark classifying **seven commercial dry bean cultivars** (*Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira*) using **16 computer vision morphological features** extracted from 13,611 high-resolution images.

The benchmark evaluates **four fundamental classifier families** under a strict **repeated nested stratified cross-validation protocol**, coupled with non-parametric statistical hypothesis testing and a domain-informed feature ablation study.

📖 **For an in-depth code audit, technical gaps, system architecture, and stakeholder guides, see [PROJECT_BREAKDOWN_AND_ARCHITECTURE.md](PROJECT_BREAKDOWN_AND_ARCHITECTURE.md).**

---

## 🔬 Dataset & Morphology Features
- **Source:** [UCI Machine Learning Repository: Dry Bean Dataset (ID: 602)](https://doi.org/10.24432/C50S4B)
- **Observations:** 13,611 bean seeds
- **Classes (7 Cultivars):** Dermason (26.05%), Sira (19.37%), Seker (14.89%), Horoz (14.17%), Cali (11.98%), Barbunya (9.71%), Bombay (3.84%)
- **Morphological Features (16):**
  - *Dimensional:* Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter
  - *Shape & Form:* AspectRatio, Eccentricity, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

![Feature Correlation Matrix](figures/feature_correlation_matrix.png)

---

## ⚙️ Algorithms & Hyperparameter Tuning
Evaluated across 4 diverse model families using nested grid search with macro-F1 optimization:

| Algorithm | Model Family | Hyperparameter Search Space | Preprocessing |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear / Probabilistic | $C \in \{0.1, 1, 10\}$, `solver='lbfgs'` | `StandardScaler` |
| **k-NN** | Instance-Based | `n_neighbors` $\in \{5, 11, 21\}$ | `StandardScaler` |
| **Decision Tree** | Tree-Based | `max_depth` $\in \{10, 20, \text{None}\}$ | None (Scale-invariant) |
| **RBF SVM** | Kernel Machine | $C \in \{1, 10, 100\}$, `gamma='scale'` | `StandardScaler` |

---

## 🧪 Validation & Statistical Methodology
- **Outer Validation:** 5-Fold Stratified Cross-Validation $\times$ 3 Repetitions (15 total outer evaluation folds per model)
- **Repetition Seeds:** `[42, 43, 44]`
- **Inner Tuning:** 2-Fold Stratified Cross-Validation to eliminate data leakage
- **Primary Tuning Metric:** Macro-averaged F1 score (`f1_macro`)
- **Statistical Significance:**
  - Omnibus **Friedman test** across all paired folds
  - **Kendall's W** concordance effect size
  - Post-hoc pairwise **Wilcoxon signed-rank tests** with **Holm-Bonferroni correction** and Wilcoxon effect size $r$

---

## 📊 Key Benchmark Results

### 1. Main Performance Comparison (15 Outer Folds)
| Algorithm | Macro-F1 (Mean ± SD) | Balanced Accuracy | Cohen's Kappa | Fit Time (s) | Predict Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF SVM (Winner)** | **0.9415 ± 0.0046** | **0.9401 ± 0.0046** | **0.9158 ± 0.0066** | 5.605s | 0.452s |
| **Logistic Regression** | 0.9358 ± 0.0040 | 0.9347 ± 0.0043 | 0.9082 ± 0.0066 | 4.169s | **0.006s** |
| **k-NN** | 0.9354 ± 0.0036 | 0.9334 ± 0.0038 | 0.9077 ± 0.0055 | 1.800s | 0.257s |
| **Decision Tree** | 0.9189 ± 0.0038 | 0.9177 ± 0.0040 | 0.8869 ± 0.0054 | **1.540s** | **0.002s** |

![Macro-F1 Distribution](figures/figure1_macro_f1_distribution.png)

### 2. Statistical Significance Verdict
- **Friedman Test:** $\chi^2 = 40.68, p = 7.65 \times 10^{-9}$ (Statistically significant difference among model families).
- **Kendall's W:** $0.904$ (Near-perfect ranking consensus across folds).
- **Holm-Adjusted Wilcoxon:** RBF SVM statistically outperforms Decision Tree, Logistic Regression, and k-NN ($p_{\text{adj}} < 0.001$). Logistic Regression and k-NN perform comparably ($p_{\text{adj}} = 0.978$).

### 3. Feature Ablation Insights
Removing four redundant/collinear size features (`Perimeter`, `ConvexArea`, `EquivalentDiameter`, `Compactness`) resulted in $<0.1\%$ performance variation, demonstrating that optical sorting hardware can operate with a leaner 12-feature configuration with negligible impact on sorting yield.

---

## 📁 Repository Structure
```
├── figures/                            # 10 diagnostic plots and confusion matrices
├── results/                            # 21 CSV/JSON fold-level logs and statistical tables
├── dry_bean_benchmark.ipynb            # Core computational benchmark notebook (40 cells)
├── PROJECT_BREAKDOWN_AND_ARCHITECTURE.md # Full architecture specification & engineering audit
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- `pip` or virtual environment manager

### Installation
```bash
# Clone the repository
git clone https://github.com/Legacy-G/agritech-grain-morphology-classifier.git
cd agritech-grain-morphology-classifier

# Install dependencies
pip install -r requirements.txt
```

### Running the Benchmark
Open and execute the Jupyter Notebook:
```bash
jupyter notebook dry_bean_benchmark.ipynb
```
The notebook automatically downloads the UCI dataset, runs the nested cross-validation loops, computes statistical tests, and updates all tables and figures.

---

## 👥 Authors & Academic Credits
- **Gbure G. Thomas** ([@Legacy-G](https://github.com/Legacy-G)) & Project Group
- Department of Computer Engineering, **Federal University of Technology, Minna (FUTMINNA)**
- *Final Year Project / Course Assignment — Artificial Neural Networks & Machine Learning*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
