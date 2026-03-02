"""
Robust Cox PH with covariates:
- RECURRENCE (e.g., Hist_RECURRENCE / HIST_recurrence / recurrence ...)
- N_grp      (categorical)
- Intent_CURE (binary or categorical)

Includes:
- HR + CI95% (lifelines)
- LRT global + per-block (lifelines)
- C-index apparent (lifelines + scikit-survival check)
- Harrell bootstrap optimism-corrected C-index (penalizer + low-variance handling)

pip install pandas numpy openpyxl lifelines scikit-survival scipy
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines import CoxPHFitter

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis


# =========================
# CONFIG
# =========================
FILE_PATH = r"C:\Users\flavi\ProgettoTesiFLavio_de_Simone\data\dataset_survival_ready.xlsx"

TIME_CANDIDATES  = ["follow_ups.days_to_follow_up", "days_to_follow_up", "followup_time", "time"]
EVENT_CANDIDATES = ["demographic.vital_status_DEAD", "vital_status_dead", "event", "death"]

RECURRENCE_CANDIDATES = ["Hist_RECURRENCE", "HIST_RECURRENCE", "HIST_recurrence", "Hist_recurrence", "recurrence"]
NGRP_CANDIDATES       = ["N_grp", "n_grp", "N group", "N-group", "N_grp "]
INTENT_CURE_CANDIDATES = ["Intent_CURE", "Intent_cure", "intent_cure", "intent cure", "Intent Cure"]

BOOTSTRAP_B = 500
RANDOM_SEED = 42

LIFELINES_PENALIZER = 0.01
SKSURV_ALPHA = 1e-4


# =========================
# COLUMN PICKING (robust)
# =========================
def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\.]+", "", s)
    return s


def pick_col(df: pd.DataFrame, candidates):
    norm_map = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_map:
            return norm_map[key]
    raise KeyError(
        "Nessuna colonna trovata tra candidati:\n"
        f"  candidates={candidates}\n"
        f"  available(example)={list(df.columns)[:60]}"
    )


# =========================
# PREP / HELPERS
# =========================
def prepare_design(df: pd.DataFrame) -> pd.DataFrame:
    t_col = pick_col(df, TIME_CANDIDATES)
    e_col = pick_col(df, EVENT_CANDIDATES)

    r_col = pick_col(df, RECURRENCE_CANDIDATES)
    n_col = pick_col(df, NGRP_CANDIDATES)
    c_col = pick_col(df, INTENT_CURE_CANDIDATES)

    d = df[[t_col, e_col, r_col, n_col, c_col]].copy()
    d = d.dropna(subset=[t_col, e_col, r_col, n_col, c_col])

    d = d.rename(columns={
        t_col: "TIME",
        e_col: "EVENT",
        r_col: "RECURRENCE",
        n_col: "N_grp",
        c_col: "Intent_CURE",
    })

    # time/event types
    d["EVENT"] = pd.to_numeric(d["EVENT"], errors="coerce").astype(int)
    d["TIME"]  = pd.to_numeric(d["TIME"], errors="coerce")
    d = d.dropna(subset=["TIME"])
    d["TIME"] = d["TIME"].astype(float)

    # RECURRENCE as 0/1 int
    d["RECURRENCE"] = pd.to_numeric(d["RECURRENCE"], errors="coerce")
    d = d.dropna(subset=["RECURRENCE"])
    d["RECURRENCE"] = d["RECURRENCE"].astype(int)

    # N_grp always categorical -> dummy
    n_dum = pd.get_dummies(d[["N_grp"]].astype("category"), drop_first=True)

    # Intent_CURE: binary numeric => keep; else dummy encode
    intent_num = pd.to_numeric(d["Intent_CURE"], errors="coerce")
    is_binary_intent = intent_num.notna().all() and set(intent_num.unique()).issubset({0, 1})

    if is_binary_intent:
        d["Intent_CURE"] = intent_num.astype(int)
        d_out = pd.concat([d[["TIME", "EVENT", "RECURRENCE", "Intent_CURE"]], n_dum], axis=1)
    else:
        i_dum = pd.get_dummies(d[["Intent_CURE"]].astype("category"), drop_first=True)
        d_out = pd.concat([d[["TIME", "EVENT", "RECURRENCE"]], n_dum, i_dum], axis=1)

    return d_out.reset_index(drop=True)


def drop_near_zero_variance_cols(d: pd.DataFrame, threshold: float = 1e-12) -> pd.DataFrame:
    protected = {"TIME", "EVENT"}
    cols = [c for c in d.columns if c not in protected]
    variances = d[cols].var(numeric_only=True)
    to_drop = variances[variances <= threshold].index.tolist()
    if to_drop:
        return d.drop(columns=to_drop)
    return d


def fit_lifelines_cox(d: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=LIFELINES_PENALIZER)
    cph.fit(d, duration_col="TIME", event_col="EVENT", show_progress=False)
    return cph


def hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary.copy()
    return pd.DataFrame(
        {
            "coef": s["coef"],
            "HR": np.exp(s["coef"]),
            "CI95_low": np.exp(s["coef lower 95%"]),
            "CI95_high": np.exp(s["coef upper 95%"]),
            "p_value_Wald": s["p"],
        }
    )


def lrt_stat(ll_full: float, ll_reduced: float, df_diff: int):
    LR = 2.0 * (ll_full - ll_reduced)
    p = chi2.sf(LR, df=df_diff)
    return float(LR), float(p)


def blocks_for_original_vars(design_cols):
    blocks = {
        "RECURRENCE": ["RECURRENCE"],
        "N_grp": [c for c in design_cols if c.startswith("N_grp_")],
    }
    if "Intent_CURE" in design_cols:
        blocks["Intent_CURE"] = ["Intent_CURE"]
    else:
        blocks["Intent_CURE"] = [c for c in design_cols if c.startswith("Intent_CURE_")]
    return blocks


def cindex_sksurv(event_bool, time_arr, risk_scores) -> float:
    return float(concordance_index_censored(event_bool, time_arr, risk_scores)[0])


def bootstrap_harrell_cindex(d: pd.DataFrame, B=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(d)

    d_full = drop_near_zero_variance_cols(d)
    cph_full = fit_lifelines_cox(d_full)
    c_app_lifelines = float(cph_full.concordance_index_)

    risk_full = cph_full.predict_partial_hazard(d_full).values.reshape(-1)
    c_app_sks = cindex_sksurv(d_full["EVENT"].astype(bool).values, d_full["TIME"].values, risk_full)

    opt_life, opt_sks = [], []
    success = 0

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boot = d.iloc[idx].copy().reset_index(drop=True)
        boot = drop_near_zero_variance_cols(boot)

        # Align columns (rare levels may vanish)
        common_cols = [c for c in d_full.columns if c in boot.columns]
        boot = boot[common_cols]
        orig_aligned = d_full[common_cols]

        try:
            cph_b = fit_lifelines_cox(boot)
        except Exception:
            continue

        c_train = float(cph_b.concordance_index_)
        c_test = float(cph_b.score(orig_aligned, scoring_method="concordance_index"))
        opt_life.append(c_train - c_test)

        risk_boot = cph_b.predict_partial_hazard(boot).values.reshape(-1)
        risk_orig = cph_b.predict_partial_hazard(orig_aligned).values.reshape(-1)

        c_train_sks = cindex_sksurv(boot["EVENT"].astype(bool).values, boot["TIME"].values, risk_boot)
        c_test_sks  = cindex_sksurv(orig_aligned["EVENT"].astype(bool).values, orig_aligned["TIME"].values, risk_orig)
        opt_sks.append(c_train_sks - c_test_sks)

        success += 1

    if success == 0:
        raise RuntimeError("No successful bootstrap replicates. Try increasing penalizer or checking categories.")

    return {
        "B_requested": int(B),
        "B_success": int(success),
        "C_app_lifelines": float(c_app_lifelines),
        "optimism_mean_lifelines": float(np.mean(opt_life)),
        "C_corrected_lifelines": float(c_app_lifelines - np.mean(opt_life)),
        "C_app_sksurv": float(c_app_sks),
        "optimism_mean_sksurv": float(np.mean(opt_sks)),
        "C_corrected_sksurv": float(c_app_sks - np.mean(opt_sks)),
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = pd.read_excel(FILE_PATH)
    d = prepare_design(df)

    cph = fit_lifelines_cox(d)

    print("\n=== HR / CI95% / p (Wald) - lifelines ===")
    print(hr_table(cph))

    print("\n=== Likelihood Ratio Test (model full vs null) - lifelines ===")
    print(cph.log_likelihood_ratio_test())

    print("\n=== LRT per covariate (block) ===")
    blocks = blocks_for_original_vars(d.columns)
    for var, cols_block in blocks.items():
        if not cols_block:
            print(f"{var}: no columns found (maybe only one level present?).")
            continue

        reduced_cols = [c for c in d.columns if c not in cols_block]
        d_red = d[reduced_cols].copy().reset_index(drop=True)

        cph_red = fit_lifelines_cox(d_red)
        df_diff = len(cols_block)

        LR, p = lrt_stat(cph.log_likelihood_, cph_red.log_likelihood_, df_diff=df_diff)
        print(f"{var}: df={df_diff}, LR={LR:.4f}, p={p:.4g}  (removed: {cols_block})")

    print("\n=== C-index apparent (lifelines) ===")
    print(float(cph.concordance_index_))

    print("\n=== Fit scikit-survival CoxPHSurvivalAnalysis (check) ===")
    y = Surv.from_arrays(event=d["EVENT"].astype(bool).values, time=d["TIME"].values)
    X = d.drop(columns=["TIME", "EVENT"])

    sk_cox = CoxPHSurvivalAnalysis(alpha=SKSURV_ALPHA)
    sk_cox.fit(X, y)

    risk = sk_cox.predict(X)
    c_sks = concordance_index_censored(d["EVENT"].astype(bool).values, d["TIME"].values, risk)[0]
    print("C-index scikit-survival (apparent):", float(c_sks))

    print("\n=== Bootstrap Harrell (optimism-corrected C-index) ===")
    boot_res = bootstrap_harrell_cindex(d, B=BOOTSTRAP_B, seed=RANDOM_SEED)
    print(boot_res)