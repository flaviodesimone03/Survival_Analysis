import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pip install lifelines openpyxl
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# =========================
# CONFIG
# =========================
FILE_PATH=r"C:\Users\flavi\ProgettoTesiFLavio_de_Simone\data\dataset_survival_ready.xlsx"

time_col  = "follow_ups.days_to_follow_up"
event_col = "demographic.vital_status_DEAD"

# Log-rank: discretizzazione numeriche "continue" (NON usata per age_at_index, che è forzata <60/>=60)
NUMERIC_BINNING = "quartiles"   # "quartiles" | "tertiles" | "median"

# Soglie
MAX_UNIQUE_AS_CATEGORICAL = 10
MIN_GROUP_SIZE = 5
INCLUDE_MISSING_AS_GROUP = True

# Variabili richieste (solo queste)
SELECTED_VARS = [
    "Trt_SURGERY",
    "demographic.age_at_index",   # forzata in <60 e >=60
    "N_grp",
    "T_grp",
    "Hist_RECURRENCE",            # gestisce anche Hist_recurrence ecc.
]

OUTPUT_DIR = "km_plots_selected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD
# =========================
df = pd.read_excel(FILE_PATH)

# Pulizia minima time/event
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
df = df.dropna(subset=[time_col, event_col])
df = df[df[time_col] >= 0]
df[event_col] = df[event_col].astype(int).clip(0, 1)

# =========================
# Helpers
# =========================
def find_column_case_insensitive(df, target_name):
    """Trova una colonna anche se cambia maiuscole/minuscole."""
    target_lower = target_name.lower()
    for c in df.columns:
        if c.lower() == target_lower:
            return c
    return None

def find_hist_recurrence_column(df):
    """Gestione robusta per Hist_RECURRENCE (varie grafie)."""
    # match esatto case-insensitive
    for name in ["Hist_RECURRENCE", "Hist_recurrence", "HIST_recurrence", "HIST_RECURRENCE"]:
        c = find_column_case_insensitive(df, name)
        if c is not None:
            return c
    # fallback: contiene "hist" e "recurr"
    candidates = [c for c in df.columns if ("hist" in c.lower() and "recurr" in c.lower())]
    return candidates[0] if candidates else None

def collapse_rare_levels(s: pd.Series, min_size: int = 5) -> pd.Series:
    vc = s.value_counts(dropna=False)
    rare_levels = vc[vc < min_size].index
    return s.where(~s.isin(rare_levels), other="Other")

def bin_numeric(x: pd.Series, strategy: str) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    if strategy == "median":
        med = x_num.median(skipna=True)
        return pd.Series(np.where(x_num <= med, "<=median", ">median"), index=x.index)
    elif strategy == "tertiles":
        return pd.qcut(x_num, q=3, duplicates="drop").astype(str)
    else:  # quartiles
        return pd.qcut(x_num, q=4, duplicates="drop").astype(str)

def age_groups_under_over_60(age_series: pd.Series) -> pd.Series:
    """Forza i gruppi di età: <60 e >=60."""
    a = pd.to_numeric(age_series, errors="coerce")
    g = pd.Series(np.where(a < 60, "<60", ">=60"), index=age_series.index).astype("object")
    # se age è NaN:
    g[pd.isna(a)] = np.nan
    return g

def make_groups_for_logrank(series: pd.Series, series_name: str) -> tuple[pd.Series, str]:
    """Crea 'groups' per logrank + stringa tipo."""
    # Forza raggruppamento specifico per età
    if series_name.lower() == "demographic.age_at_index":
        g = age_groups_under_over_60(series)
        var_type = "age_binned(<60,>=60)"
    else:
        nun = series.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(series) and nun > MAX_UNIQUE_AS_CATEGORICAL:
            g = bin_numeric(series, NUMERIC_BINNING)
            var_type = f"numeric_binned({NUMERIC_BINNING})"
        else:
            g = series.astype("object")
            var_type = "categorical_or_low_unique_numeric"

    # Missing as group (se richiesto)
    if INCLUDE_MISSING_AS_GROUP:
        g = g.astype("object").where(~pd.isna(g), other="Missing")
    else:
        g = g.dropna()

    # collasso livelli rari (utile soprattutto per categoriali con tanti livelli)
    g = collapse_rare_levels(g.astype("object"), MIN_GROUP_SIZE)

    return g, var_type

def plot_km_by_variable(df_in: pd.DataFrame, col: str, out_dir: str, original_name: str = None):
    """Plot KM stratificato per colonna, senza CI, e salva PNG+PDF."""
    s = df_in[col].copy()

    # gruppi: per età forzo <60 / >=60, altrimenti logica standard
    if col.lower() == "demographic.age_at_index".lower():
        groups = age_groups_under_over_60(s)
        plot_type = "age_binned(<60,>=60)"
    else:
        nun = s.nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(s) and nun > MAX_UNIQUE_AS_CATEGORICAL:
            groups = bin_numeric(s, NUMERIC_BINNING)
            plot_type = f"numeric_binned({NUMERIC_BINNING})"
        else:
            groups = s.astype("object")
            plot_type = "categorical"

    # missing as group oppure drop
    if INCLUDE_MISSING_AS_GROUP:
        groups = groups.astype("object").where(~pd.isna(groups), other="Missing")
        df_plot = df_in[[time_col, event_col]].copy()
        df_plot["_group_"] = groups
    else:
        mask = ~pd.isna(groups)
        df_plot = df_in.loc[mask, [time_col, event_col]].copy()
        df_plot["_group_"] = groups[mask]

    # collassa livelli rari + filtra gruppi troppo piccoli
    df_plot["_group_"] = collapse_rare_levels(df_plot["_group_"].astype("object"), MIN_GROUP_SIZE)
    gc = df_plot["_group_"].value_counts()
    valid = gc[gc >= MIN_GROUP_SIZE].index
    df_plot = df_plot[df_plot["_group_"].isin(valid)]

    if df_plot["_group_"].nunique() < 2:
        print(f"[SKIP PLOT] {col}: gruppi insufficienti dopo filtri.")
        return

    # log-rank per titolo
    try:
        lr = multivariate_logrank_test(df_plot[time_col], df_plot["_group_"], df_plot[event_col])
        pval = float(lr.p_value)
    except Exception:
        pval = np.nan

    kmf = KaplanMeierFitter()

    plt.figure()
    ax = plt.gca()

    # Ordina gruppi per numerosità
    order = df_plot["_group_"].value_counts().index.tolist()

    for g in order:
        mask = df_plot["_group_"] == g
        kmf.fit(df_plot.loc[mask, time_col], event_observed=df_plot.loc[mask, event_col], label=str(g))
        kmf.plot_survival_function(ax=ax, ci_show=False)

    shown_name = original_name if original_name else col
    if not np.isnan(pval):
        ax.set_title(f"Kaplan–Meier stratificata per {shown_name}  (log-rank p={pval:.3g})")
    else:
        ax.set_title(f"Kaplan–Meier stratificata per {shown_name}")

    ax.set_xlabel("Tempo (giorni)")
    ax.set_ylabel("Sopravvivenza S(t)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # salva
    safe_name = shown_name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".", "_")
    png_path = os.path.join(out_dir, f"km_{safe_name}.png")
    pdf_path = os.path.join(out_dir, f"km_{safe_name}.pdf")
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.show()

    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")

# =========================
# LOG-RANK solo sulle variabili selezionate
# =========================
results = []

# risoluzione nomi colonne reali nel dataset
resolved = []
for v in SELECTED_VARS:
    if v.lower() == "hist_recurrence":
        col_real = find_hist_recurrence_column(df)
    else:
        col_real = find_column_case_insensitive(df, v)

    if col_real is None:
        print(f"[WARN] Colonna non trovata per '{v}'. Controlla il nome nel dataset.")
        continue

    resolved.append((v, col_real))

for original_name, col in resolved:
    s = df[col]
    if s.nunique(dropna=True) <= 1:
        results.append({
            "variable": original_name,
            "column_used": col,
            "type": "",
            "n_used": 0,
            "n_groups": 0,
            "chi2": np.nan,
            "p_value": np.nan,
            "note": "Skipped: no variability"
        })
        continue

    groups, var_type = make_groups_for_logrank(s, original_name)

    df_for_test = df[[time_col, event_col]].copy()
    df_for_test["_group_"] = groups

    # filtra gruppi piccoli
    gc = df_for_test["_group_"].value_counts()
    valid = gc[gc >= MIN_GROUP_SIZE].index
    df_for_test = df_for_test[df_for_test["_group_"].isin(valid)]

    k = df_for_test["_group_"].nunique()
    n = len(df_for_test)

    if k < 2 or n < 2 * MIN_GROUP_SIZE:
        results.append({
            "variable": original_name,
            "column_used": col,
            "type": var_type,
            "n_used": n,
            "n_groups": k,
            "chi2": np.nan,
            "p_value": np.nan,
            "note": "Skipped: insufficient groups/size"
        })
        continue

    try:
        test = multivariate_logrank_test(
            df_for_test[time_col],
            df_for_test["_group_"],
            df_for_test[event_col]
        )
        results.append({
            "variable": original_name,
            "column_used": col,
            "type": var_type,
            "n_used": n,
            "n_groups": k,
            "chi2": float(test.test_statistic),
            "p_value": float(test.p_value),
            "note": ""
        })
    except Exception as e:
        results.append({
            "variable": original_name,
            "column_used": col,
            "type": var_type,
            "n_used": n,
            "n_groups": k,
            "chi2": np.nan,
            "p_value": np.nan,
            "note": f"Error: {e}"
        })

res = pd.DataFrame(results).sort_values("p_value", na_position="last").reset_index(drop=True)
res.to_csv("logrank_selected_variables.csv", index=False)

print("\n=== LOG-RANK (solo variabili selezionate, ordinato per p-value) ===")
print(res[["variable", "column_used", "type", "p_value", "chi2", "n_groups", "n_used", "note"]].to_string(index=False))
print("\nSalvato: logrank_selected_variables.csv")

# =========================
# PLOT KM per le variabili selezionate
# =========================
print("\n=== PLOT KM (variabili selezionate) ===")
for original_name, col in resolved:
    plot_km_by_variable(df, col, OUTPUT_DIR, original_name=original_name)
