[English](README.md) | [Italiano](README.it.md)
# Metodologie probabilistiche classiche e bayesiane per l'analisi di sopravvivenza in pazienti affetti da carcinoma del cavo orale

Repository ufficiale del progetto di Tesi di Laurea in Informatica — Università degli Studi di Salerno (Dipartimento di Informatica, CAIS Laboratory, A.A. 2024/2025)[cite: 1, 2].

- Candidato: Flavio de Simone[cite: 1]
- Relatore: Prof.ssa Genoveffa Tortora[cite: 1]
- Correlatori: Dott.ssa Alessia Auriemma Citarella, Dott.ssa Luigia Rizzo[cite: 1]

---

## Panoramica del Progetto

Il carcinoma squamocellulare del cavo orale (OSCC) è una neoplasia caratterizzata da una prognosi complessa e da una forte eterogeneità clinica[cite: 5, 11]. Sebbene i moderni algoritmi di Machine Learning (ML) offrano elevate capacità predittive, spesso operano come black-box, rendendo difficile l'integrazione nei processi decisionali medici[cite: 11, 15, 19].

Questo lavoro propone un confronto metodologico rigoroso tra modelli probabilistici classici e bayesiani (intrinsecamente interpretabili) e modelli di Machine Learning su dati clinici longitudinali, valutando non solo le prestazioni predittive, ma anche la calibrazione delle probabilità, la trasparenza strutturale e l'utilità clinica[cite: 5, 18, 63].

---

## Dataset e Pipeline di Pre-processing

Lo studio utilizza una coorte clinica di 527 pazienti affetti da OSCC estratta dal progetto TCGA-HNSC (The Cancer Genome Atlas - Head and Neck Squamous Cell Carcinoma)[cite: 33].

La pipeline di data preparation, interamente sviluppata in Python, comprende[cite: 34]:
1. Integrazione e Ristrutturazione: Allineamento e merge di tre sorgenti dati (`oral_cancer_dataset.xlsx`, `follow_up.xlsx`, `fattori_di_rischio.xlsx`) per ricondurre oltre 2.000 record longitudinali a una singola riga per paziente[cite: 33, 34].
2. Definizione dell'Outcome Time-to-Event: Estrazione del tempo di osservazione (`days_to_follow_up`) e dell'indicatore di evento/censura a destra (`vital_status`)[cite: 35].
3. Feature Engineering e Pulizia:
   - Rimozione di identificativi (`case_id`), variabili a varianza nulla e colonne con oltre il 70% di valori mancanti[cite: 39].
   - Integrazione della stadiazione TNM patologica con quella clinica e accorpamento in macro-gruppi clinicamente coerenti[cite: 39, 40].
   - Codifica binaria di terapie (`Surgery`, `Radiation`, `Pharmaceutical`), intenti terapeutici (`Intent_CURE`, `Intent_PALLIATIVE`, ecc.) ed eventi clinici (`Hist_RECURRENCE`, `Hist_METASTASIS`)[cite: 40, 41].

---

## Metodologie Implementate

Il workflow combina approcci statistici tradizionali e modellazione grafica probabilistica[cite: 18, 30]:

- Analisi Non Parametrica (Python - `scikit-survival`): Stima della funzione di sopravvivenza globale tramite curve di Kaplan-Meier (KM) con bande di confidenza al 95% e screening univariato tra sottogruppi tramite Log-Rank Test[cite: 41, 42].
- Modello Semi-Parametrico (Python - `scikit-survival`): Regressione multivariata a rischi proporzionali di Cox (CPH) per la stima degli Hazard Ratio (HR)[cite: 42, 43], validata mediante Likelihood Ratio Test (LRT)[cite: 43], analisi dei residui di Schoenfeld[cite: 43] e validazione interna tramite 500 ricampionamenti Bootstrap[cite: 44].
- Rete Bayesiana Discreta (R - `bnlearn`, `gRain`): Apprendimento della struttura tramite algoritmo ibrido Max-Min Hill-Climbing (score BIC) con vincoli clinici a priori (whitelist e blacklist temporali), stima bayesiana dei parametri (prior di Dirichlet) e analisi what-if per l'inferenza probabilistica condizionale anche in presenza di evidenza parziale[cite: 45, 46, 47].

---

## Risultati Principali e Benchmark

### 1. Fattori Prognostici Chiave
- Cox Proportional Hazards: La presenza di recidiva (`Hist_RECURRENCE`) quasi triplica il rischio di evento (HR = 2.864, p = 1.53e-11), seguita dal coinvolgimento linfonodale avanzato N2+ (HR = 1.852, p = 1.50e-05)[cite: 52]. L'intento terapeutico curativo (`Intent_CURE`) esercita un forte effetto protettivo (HR = 0.363, p = 1.26e-11)[cite: 52].
- Rete Bayesiana (Inferenze What-If): Partendo da un profilo clinico di riferimento (P(Dead) = 50.5%), l'intervento chirurgico riduce la probabilità stimata di decesso all'8.7% (Delta = -41.9%), mentre la recidiva la innalza al 92.2% (Delta = +41.7%)[cite: 58, 59].

### 2. Confronto con Modelli di Machine Learning
I modelli probabilistici sono stati confrontati sullo stesso dataset con tre architetture ML allo stato dell'arte (XGBoost, Random Survival Forest e DeepSurv)[cite: 61, 62]:

| Modello | Paradigma | C-index | IBS / Brier Score | Mean AUC |
| :--- | :--- | :---: | :---: | :---: |
| XGBoost | Machine Learning | 0.7248 | 0.1891 | 0.7339 |
| Random Survival Forest | Machine Learning | 0.7164 | 0.1887 | 0.6997 |
| DeepSurv | Deep Learning | 0.7075 | 0.2039 | 0.7064 |
| Cox PH (Questo studio) | Statistica Classica | 0.6989 (0.693 corr.) | 0.1876 | 0.7111 |
| Bayesian Network (Questo studio) | Modello Grafico Probabilistico | 0.6723 | 0.2136 | 0.7060 |

Sebbene XGBoost ottenga il C-index più elevato, il modello di Cox raggiunge il miglior Integrated Brier Score (0.1876) e una discriminazione competitiva senza richiedere tecniche XAI post-hoc[cite: 62, 63]. La Rete Bayesiana offre una calibrazione stabile in 10-fold CV (AUC 0.706, Brier 0.214) unita alla capacità unica di ragionare su osservazioni cliniche incomplete[cite: 28, 61].

---

## Tecnologie Utilizzate

- Linguaggi: Python, R[cite: 34, 46]
- Librerie Python: `pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `matplotlib`, `seaborn`[cite: 42]
- Librerie R: `bnlearn`, `gRain`[cite: 46]
