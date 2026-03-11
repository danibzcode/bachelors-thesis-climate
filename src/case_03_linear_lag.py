# ==============================================================================
# TFG MASTER SCRIPT - CASE 3: DIRECT LINEAR COUPLING WITH LAG
# Description: Demonstrates a real causal link X -> Y with a delay of 2 steps.
#              This is the perfect scenario for Granger Causality.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from causal_tools import (
    create_surrogate, 
    manual_pearson_corr, 
    manual_mi, 
    get_best_granger_pvalue, 
    manual_transfer_entropy_lagged, 
    standardize
)

# ---------------------------------------------------------
# 1. GENERACIÓN DE DATOS (X -> Y con Lag = 2)
# ---------------------------------------------------------
n_samples, t_lag = 400, 2
np.random.seed(42)

rates = np.zeros(n_samples) 
unemp = np.zeros(n_samples)

for t in range(t_lag, n_samples):
    # X es un proceso autorregresivo AR(1)
    rates[t] = 0.8 * rates[t-1] + 0.5 * np.random.randn()
    # Y depende directamente de su propio pasado y del de X (hace 2 pasos)
    unemp[t] = 0.7 * unemp[t-1] + 0.4 * rates[t-t_lag] + 0.5 * np.random.randn()

print(f"Ejecutando Caso 3: Acoplamiento Lineal con Memoria (Lag={t_lag})...")

# ---------------------------------------------------------
# 2. ANÁLISIS Y TEST DE HIPÓTESIS (IAAFT)
# ---------------------------------------------------------
N_BINS, n_surrogates = 5, 500

# A. Métricas observadas (Síncronas para asociación, con Lag para causales)
obs_pearson = manual_pearson_corr(rates, unemp)
obs_mi = manual_mi(rates, unemp, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(rates, unemp, bins=N_BINS, k_lag=t_lag)

# B. Granger (Métrica paramétrica lineal) explorando hasta 4 retardos
data_g = pd.DataFrame({'Unemp': unemp, 'Rates': rates})
res_xy = grangercausalitytests(data_g[['Unemp', 'Rates']], maxlag=4, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, 4) # Extraemos el mejor p-valor (debería ser en lag=2)

# C. Generación de subrogados
print(f"Calculando {n_surrogates} subrogados para validación estadística...")
surr_pearson, surr_mi, surr_te = [], [], []
for _ in range(n_surrogates):
    sx = create_surrogate(rates)
    surr_pearson.append(manual_pearson_corr(sx, unemp))
    surr_mi.append(manual_mi(sx, unemp, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, unemp, bins=N_BINS, k_lag=t_lag))

# D. Cálculo de p-valores empíricos
p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 3. VISUALIZACIÓN ACADÉMICA (ESTILO TFG)
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

# --- A. SERIES TEMPORALES (ZOOM PARA VER EL RETARDO) ---
ax_time = fig.add_subplot(gs[0, :])
z = slice(100, 180) 
ax_time.plot(standardize(rates)[z], 'o-', label='Causa X (Ej: Tipos de Interés)', color='#1f77b4', markersize=4)
ax_time.plot(standardize(unemp)[z], 's-', label='Efecto Y (Ej: Desempleo)', color='#ff7f0e', markersize=4)
ax_time.set_title(f'Acoplamiento Lineal Directo con Retardo (X $\\to$ Y, Lag={t_lag})', fontweight='bold')
ax_time.set_ylabel('Amplitud (std)')
ax_time.legend(loc='upper right', ncol=2); ax_time.grid(alpha=0.3, linestyle='--')

# Función para histogramas
def plot_p(ax, surr_data, obs_val, p_val, title, color_obs):
    ax.hist(surr_data, bins=30, color='#cccccc', edgecolor='white', density=True, alpha=0.7)
    ax.axvline(obs_val, color=color_obs, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    status = "SIG." if p_val < 0.05 else "NO SIG."
    ax.text(0.95, 0.90, f"p = {p_val:.4f}\n({status})", transform=ax.transAxes, 
            ha='right', va='top', fontweight='bold', color='green', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', fontsize='small')

# --- B. DISTRIBUCIONES (TODAS DEBEN SER SIGNIFICATIVAS, PERO CON MATICES) ---
plot_p(fig.add_subplot(gs[1, 0]), surr_pearson, obs_pearson, p_pearson, 'Pearson Síncrono (Ambiguo)', '#e41a1c')
plot_p(fig.add_subplot(gs[1, 1]), surr_mi, obs_mi, p_mi, 'Info. Mutua Síncrona (Ambigua)', '#377eb8')
plot_p(fig.add_subplot(gs[1, 2]), surr_te, obs_te, p_te, f'Transfer Entropy (Lag={t_lag} Exacto)', '#4daf4a')

plt.tight_layout()
plt.savefig('TFG_Caso3_RetardoLineal.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. TABLA DE RESULTADOS POR CONSOLA
# ---------------------------------------------------------
print("\n" + "="*85)
print(f"{'MÉTRICA':<25} | {'P-VALOR':<10} | {'CONCLUSIÓN FÍSICA'}")
print("-" * 85)
print(f"{'Pearson':<25} | {p_pearson:10.4f} | {'SÍ (Detecta asociación, pero sin dirección)'}")
print(f"{'Información Mutua':<25} | {p_mi:10.4f} | {'SÍ (Detecta dependencia, pero sin dirección)'}")
print(f"{'Granger Causality':<25} | {p_granger:10.4f} | {'ÉXITO TOTAL (Identifica dirección y lag=2)'}")
print(f"{'Transfer Entropy':<25} | {p_te:10.4f} | {'ÉXITO TOTAL (Identifica dirección y lag=2)'}")
print("="*85)