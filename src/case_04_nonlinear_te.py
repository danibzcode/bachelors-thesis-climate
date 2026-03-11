# ==============================================================================
# TFG MASTER SCRIPT - CASE 4: NON-LINEAR COUPLING (QUADRATIC)
# Description: Demonstrates why non-linear metrics (MI/TE) are essential.
#              Linear metrics (Pearson/Granger) fail to detect the Y = X^2 link.
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
# 1. GENERACIÓN DE DATOS (NO LINEALES: Y proporcional a X^2)
# ---------------------------------------------------------
n_samples, t_lag = 500, 2
np.random.seed(42)

fertilizer = np.zeros(n_samples) # Variable X (Causa)
yield_crop = np.zeros(n_samples) # Variable Y (Efecto)

for t in range(t_lag, n_samples):
    # X: Proceso autorregresivo (Memoria propia)
    fertilizer[t] = 0.8 * fertilizer[t-1] + 0.5 * np.random.randn()
    
    # Y: Relación cuadrática con el pasado de X (hace 2 pasos)
    # Esta parábola es la trampa perfecta que "ciega" a los algoritmos lineales
    yield_crop[t] = (fertilizer[t-t_lag])**2 + 0.5 * np.random.randn()

print(f"Ejecutando Caso 4: Acoplamiento No Lineal (Y = X^2, Lag={t_lag})...")

# ---------------------------------------------------------
# 2. ANÁLISIS Y TEST DE HIPÓTESIS (IAAFT)
# ---------------------------------------------------------
N_BINS, n_surrogates = 6, 500

# A. Métricas observadas
obs_pearson = manual_pearson_corr(fertilizer, yield_crop)
obs_mi = manual_mi(fertilizer, yield_crop, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(fertilizer, yield_crop, bins=N_BINS, k_lag=t_lag)

# B. Granger (Test paramétrico lineal)
data_g = pd.DataFrame({'Yield': yield_crop, 'Fert': fertilizer})
res_xy = grangercausalitytests(data_g[['Yield', 'Fert']], maxlag=4, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, 4)

# C. Generación de subrogados IAAFT
print(f"Calculando {n_surrogates} subrogados para validación de no linealidad...")
surr_pearson, surr_mi, surr_te = [], [], []
for _ in range(n_surrogates):
    sx = create_surrogate(fertilizer)
    surr_pearson.append(manual_pearson_corr(sx, yield_crop))
    surr_mi.append(manual_mi(sx, yield_crop, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, yield_crop, bins=N_BINS, k_lag=t_lag))

# D. Cálculo de p-valores
p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 3. VISUALIZACIÓN ACADÉMICA (DETECCIÓN DE NO LINEALIDAD)
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

# --- A. SERIES TEMPORALES ---
ax_time = fig.add_subplot(gs[0, :])
z = slice(100, 180) 
ax_time.plot(standardize(fertilizer)[z], 'o-', label='Causa X (Ej: Forzamiento)', color='#9b59b6', markersize=4)
ax_time.plot(standardize(yield_crop)[z], 's-', label='Efecto Y ($Y \propto X^2$)', color='#2ecc71', markersize=4)
ax_time.set_title(f'Acoplamiento Causal No Lineal: Dinámica de Respuesta Cuadrática', fontweight='bold')
ax_time.set_ylabel('Amplitud (std)')
ax_time.legend(loc='upper right', ncol=2); ax_time.grid(alpha=0.3, linestyle='--')

# Función para histogramas
def plot_p(ax, surr_data, obs_val, p_val, title, color_obs):
    ax.hist(surr_data, bins=30, color='#cccccc', edgecolor='white', density=True, alpha=0.7)
    ax.axvline(obs_val, color=color_obs, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Lógica condicional: Si es lineal (debe fallar), si es no lineal (debe acertar)
    if 'Pearson' in title:
        status = "NO SIG. (Fallo)" if p_val >= 0.05 else "SIG."
        t_color = 'red' if p_val >= 0.05 else 'black'
    else:
        status = "SIG. (Éxito)" if p_val < 0.05 else "NO SIG."
        t_color = 'green' if p_val < 0.05 else 'black'

    ax.text(0.95, 0.90, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes, 
            ha='right', va='top', fontweight='bold', color=t_color, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', fontsize='small')

# --- B. DISTRIBUCIONES (PEARSON DEBE FALLAR EN ROJO, MI/TE ACERTAR EN VERDE) ---
plot_p(fig.add_subplot(gs[1, 0]), surr_pearson, obs_pearson, p_pearson, 'Pearson (Ciego a $X^2$)', '#e74c3c')
plot_p(fig.add_subplot(gs[1, 1]), surr_mi, obs_mi, p_mi, 'Info. Mutua (Acierta No-Lineal)', '#3498db')
plot_p(fig.add_subplot(gs[1, 2]), surr_te, obs_te, p_te, 'Transfer Entropy (Causal No-Lineal)', '#27ae60')

plt.tight_layout()
plt.savefig('TFG_Caso4_NoLineal.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. TABLA DE RESULTADOS POR CONSOLA
# ---------------------------------------------------------
print("\n" + "="*85)
print(f"{'MÉTRICA':<25} | {'P-VALOR':<10} | {'COMPORTAMIENTO ESPERADO'}")
print("-" * 85)
print(f"{'Pearson':<25} | {p_pearson:10.4f} | {'ERROR (Falso Negativo: asume independencia)'}")
print(f"{'Granger (Lineal)':<25} | {p_granger:10.4f} | {'ERROR (Falso Negativo: modelo lineal erróneo)'}")
print(f"{'Información Mutua':<25} | {p_mi:10.4f} | {'CORRECTO (Captura la dependencia funcional)'}")
print(f"{'Transfer Entropy':<25} | {p_te:10.4f} | {'ÉXITO TOTAL (Captura dependencia causal direccional)'}")
print("="*85)