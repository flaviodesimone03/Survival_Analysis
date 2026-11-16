[English](README.md) | [Italiano](README.it.md)
# Classical and Bayesian Probabilistic Methodologies for Survival Analysis in Oral Squamous Cell Carcinoma Patients

[English](README.md) | [Italiano](README.it.md)

Official repository of the Bachelor's Thesis in Computer Science — University of Salerno (Department of Computer Science, CAIS Laboratory, A.Y. 2024/2025)[cite: 1, 2].

- Candidate: Flavio de Simone[cite: 1]
- Supervisor: Prof. Genoveffa Tortora[cite: 1]
- Co-supervisors: Dr. Alessia Auriemma Citarella, Dr. Luigia Rizzo[cite: 1]

---

## Project Overview

Oral Squamous Cell Carcinoma (OSCC) is a malignancy characterized by a complex prognosis and significant clinical heterogeneity[cite: 5, 11]. While modern Machine Learning (ML) algorithms achieve strong predictive performance, they often operate as black-box models, making their integration into clinical decision-making workflows challenging[cite: 11, 15, 19].

This project provides a rigorous methodological comparison between classical and Bayesian probabilistic models (intrinsically interpretable) and Machine Learning models on longitudinal clinical data, evaluating not only predictive discrimination, but also probability calibration, structural transparency, and clinical utility[cite: 5, 18, 63].

---

## Dataset and Pre-processing Pipeline

The study analyzes a clinical cohort of 527 OSCC patients retrieved from the TCGA-HNSC project (The Cancer Genome Atlas - Head and Neck Squamous Cell Carcinoma)[cite: 33].

The data preparation pipeline, developed entirely in Python, includes[cite: 34]:
1. Data Integration and Restructuring: Alignment and merging of three data sources (`oral_cancer_dataset.xlsx`, `follow_up.xlsx`, `fattori_di_rischio.xlsx`) to consolidate over 2,000 longitudinal records into a single row per patient[cite: 33, 34].
2. Time-to-Event Outcome Definition: Extraction of observation time (`days_to_follow_up`) and right-censoring/event indicator (`vital_status`)[cite: 35].
3. Feature Engineering and Cleaning:
   - Removal of patient identifiers (`case_id`), zero-variance features, and variables with more than 70% missing values[cite: 39].
   - Integration of pathological TNM staging with clinical staging and grouping into clinically coherent macro-categories[cite: 39, 40].
   - Binary encoding of treatments (`Surgery`, `Radiation`, `Pharmaceutical`), treatment intents (`Intent_CURE`, `Intent_PALLIATIVE`, etc.), and clinical history events (`Hist_RECURRENCE`, `Hist_METASTASIS`)[cite: 40, 41].

---

## Implemented Methodologies

The workflow combines traditional statistical survival analysis with probabilistic graphical modeling[cite: 18, 30]:

- Non-Parametric Analysis (Python - `scikit-survival`): Estimation of the overall survival function via Kaplan-Meier (KM) curves with 95% confidence bands and univariate subgroup screening via Log-Rank Test[cite: 41, 42].
- Semi-Parametric Model (Python - `scikit-survival`): Multivariate Cox Proportional Hazards (CPH) regression for Hazard Ratio (HR) estimation[cite: 42, 43], validated via Likelihood Ratio Test (LRT)[cite: 43], Schoenfeld residuals diagnostics[cite: 43], and internal validation using 500 Bootstrap resamples[cite: 44].
- Discrete Bayesian Network (R - `bnlearn`, `gRain`): Structure learning via the hybrid Max-Min Hill-Climbing algorithm (BIC score) incorporating prior clinical constraints (temporal blacklists and whitelists), Bayesian parameter estimation (Dirichlet prior), and what-if conditional probability inference under partial evidence[cite: 45, 46, 47].

---

## Main Results and Benchmark

### 1. Key Prognostic Factors
- Cox Proportional Hazards: Tumor recurrence (`Hist_RECURRENCE`) nearly triples the hazard of an event (HR = 2.864, p = 1.53e-11), followed by advanced nodal involvement N2+ (HR = 1.852, p = 1.50e-05)[cite: 52]. Curative treatment intent (`Intent_CURE`) shows a strong protective effect (HR = 0.363, p = 1.26e-11)[cite: 52].
- Bayesian Network (What-If Inference): Starting from a data-driven baseline patient profile (P(Dead) = 50.5%), surgical treatment reduces the estimated probability of death to 8.7% (Delta = -41.9%), whereas recurrence raises it to 92.2% (Delta = +41.7%)[cite: 58, 59].

### 2. Comparison with Machine Learning Models
The probabilistic models were benchmarked on the exact same dataset against three state-of-the-art ML architectures (XGBoost, Random Survival Forest, and DeepSurv)[cite: 61, 62]:

| Model | Paradigm | C-index | IBS / Brier Score | Mean AUC |
| :--- | :--- | :---: | :---: | :---: |
| XGBoost | Machine Learning | 0.7248 | 0.1891 | 0.7339 |
| Random Survival Forest | Machine Learning | 0.7164 | 0.1887 | 0.6997 |
| DeepSurv | Deep Learning | 0.7075 | 0.2039 | 0.7064 |
| Cox PH (This study) | Classical Statistics | 0.6989 (0.693 corr.) | 0.1876 | 0.7111 |
| Bayesian Network (This study) | Probabilistic Graphical Model | 0.6723 | 0.2136 | 0.7060 |

Although XGBoost achieves the highest C-index, the Cox model obtains the best Integrated Brier Score (0.1876) and competitive discrimination without requiring post-hoc XAI techniques[cite: 62, 63]. The Bayesian Network provides stable calibration in 10-fold CV (AUC 0.706, Brier 0.214) alongside the unique capability to perform inference with incomplete clinical observations[cite: 28, 61].

---

## Technologies Used

- Languages: Python, R[cite: 34, 46]
- Python Libraries: `pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `matplotlib`, `seaborn`[cite: 42]
- R Libraries: `bnlearn`, `gRain`[cite: 46]

## Tecnologie Utilizzate

- Linguaggi: Python, R[cite: 34, 46]
- Librerie Python: `pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `matplotlib`, `seaborn`[cite: 42]
- Librerie R: `bnlearn`, `gRain`[cite: 46]
