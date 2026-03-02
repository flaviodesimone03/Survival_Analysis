import pandas as pd
import matplotlib.pyplot as plt

# Se non hai lifelines:
# pip install lifelines openpyxl

from lifelines import KaplanMeierFitter

# === Caricamento dataset ===
path = r"C:\Users\flavi\ProgettoTesiFLavio_de_Simone\data\dataset_survival_ready.xlsx"   # oppure: r"/percorso/assoluto/dataset_survival_ready.xlsx"
df = pd.read_excel(path)

# === Colonne (dal tuo file) ===
time_col = "follow_ups.days_to_follow_up"      # tempo (in giorni)
event_col = "demographic.vital_status_DEAD"    # evento (1=deceduto, 0=censurato)

# === Pulizia minima ===
km_df = df[[time_col, event_col]].copy()
km_df[time_col] = pd.to_numeric(km_df[time_col], errors="coerce")
km_df[event_col] = pd.to_numeric(km_df[event_col], errors="coerce")

# Tieni solo righe valide
km_df = km_df.dropna(subset=[time_col, event_col])
km_df = km_df[km_df[time_col] >= 0]
km_df[event_col] = km_df[event_col].astype(int).clip(0, 1)

# === Kaplan–Meier globale ===
kmf = KaplanMeierFitter()
kmf.fit(durations=km_df[time_col], event_observed=km_df[event_col], label="KM globale")

ax = kmf.plot_survival_function(ci_show=True)
ax.set_title("Curva di sopravvivenza Kaplan–Meier (globale)")
ax.set_xlabel("Tempo (giorni)")
ax.set_ylabel("Sopravvivenza S(t)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
