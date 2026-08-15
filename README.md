# 🫀 Heart Disease Classification & End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

📝 **Full write-up:** *(Medium article coming soon)*
---
A rigorous, end-to-end Machine Learning pipeline designed for early screening and clinical classification of cardiovascular disease using patient physiological data.

---

## 📌 Project Overview & Clinical Objective

Cardiovascular disease remains the leading cause of mortality worldwide. In clinical diagnostic screening, model errors carry asymmetric real-world consequences:
* **False Positive:** Inconveniences the patient with secondary tests, but remains safe.
* **False Negative:** Fails to detect an existing cardiac risk condition, potentially endangering patient health.

This pipeline compares 5 distinct classification models under 5-Fold Stratified Cross-Validation, specifically optimizing for **F1-Score and Recall**, while embedding feature scaling inside a leak-free `ColumnTransformer` + `Pipeline` architecture.

---

## 📊 Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle, fedesoriano) — 918 patient records, 11 clinical features.

## 📂 Repository Structure

```text
├── heart.csv                        # Clinical diagnostic dataset (918 samples)
├── heart_disease_ml_pipeline.ipynb  # End-to-end data pipeline, EDA, modeling & analysis
├── requirements.txt                 # Pinned dependencies for reproducible execution
└── README.md                        # Project documentation
```

---

## ⚙️ Pipeline Architecture & Methodology

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Data Loading   │ ──> │  Targeted Imputation │ ──> │ Feature Engineering  │
│  (918 Patients) │     │ (Physiological Zeros)│     │ (Age x Oldpeak Int.) │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                │
                                                                ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Clinical Error  │ <── │ 5-Model Benchmarking │ <── │ Scaler + Pipeline    │
│ Analysis (CM)   │     │ (StratifiedKFold CV) │     │ (No Data Leakage)    │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
```

1. **Targeted Imputation:** Clinically impossible values (`0` in `Cholesterol` and `RestingBP`) were flagged as missing and imputed using class-conditional medians (`HeartDisease` partitions) to preserve physiological distinctions.
2. **Feature Engineering:**
   * Engineered `Age_Oldpeak_Interaction` to capture composite cardiovascular stress risk (Age × Oldpeak).
   * Screened experimental features (e.g. `Cholesterol_Age_Ratio`) via target correlation thresholds.
3. **Leak-Free Preprocessing:** Embedded continuous feature scaling (`StandardScaler`) in a `ColumnTransformer` inside `Pipeline`, ensuring fold isolation during Cross-Validation.
4. **Hyperparameter Tuning:** Systematic exploration using `GridSearchCV` with 5-fold `StratifiedKFold`, targeting `f1` metric optimization across 5 classifiers.

---

## 📊 Benchmark & Evaluation Results

All models were evaluated on an unseen test set (20% holdout split, $N=184$, stratified):

| Model | CV Best F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.8840** | **0.8641** | **0.8812** | **0.8725** | **0.8768** | **0.9328** |
| **KNN** | 0.8820 | 0.8804 | 0.9082 | 0.8725 | 0.8900 | 0.9316 |
| **SVM (RBF Kernel)** | 0.8779 | 0.8587 | 0.8725 | 0.8725 | 0.8725 | 0.9329 |
| **Logistic Regression** | 0.8712 | 0.8750 | 0.8911 | 0.8824 | 0.8867 | 0.9328 |
| **Decision Tree** | 0.8433 | 0.7772 | 0.7961 | 0.8039 | 0.8000 | 0.8210 |

### 🏆 Final Model Selection: Random Forest
* **Top Cross-Validation F1 (0.8840):** Best generalization stability across training folds.
* **Feature Importance Ranking:** Highlights `ST_Slope_Up`, `ST_Slope_Flat`, `MaxHR`, and `Age_Oldpeak_Interaction` as the strongest clinical predictors.
* **Test Performance:** Maintained 87.25% Recall and 0.9328 ROC-AUC on holdout clinical data.

---

## 🚀 Quickstart & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/sametozantopcu/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook heart_disease_ml_pipeline.ipynb
```

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning & Pipeline:** Scikit-Learn
* **Visualization:** Plotly Express & Plotly Graph Objects, Kaleido

---

## 👤 Author

**Samet Ozan Topcu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/samet-ozan-topcu-4328003a0/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sametozantopcu)
