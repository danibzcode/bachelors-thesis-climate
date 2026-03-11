# ==============================================================================
# TFG MASTER SCRIPT - CASE 1: INDEPENDENT WHITE NOISE (PURE SYNTHETIC)
# Description: Benchmarking linear, non-linear, and causal metrics against noise.
#              Optimized for LaTeX integration and academic clarity.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from causal_tools import (
    create_surrogate, 
    manual_pearson_corr, 
    manual_mi, 
    manual_transfer_entropy_lagged,
    get_best_granger_pvalue,
    standardize
)

# ---------------------------------------------------------
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ---------------------------------------------------------
# X_t = epsilon_x,  Y_t = epsilon_y (Procesos estocásticos independientes)
n_samples = 250
np.random.seed(123)

x = np.random.randn(n_samples) 
y = np.random.randn(n_samples)

print("Ejecutando Caso 1: Ruido Blanco Independiente...")
print(f"Muestras: {n_samples} | Semilla: 123")

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DEL ANÁLISIS ESTADÍSTICO
# ---------------------------------------------------------
N_BINS = 5           # Discretización para MI y TE
TEST_LAG = 1         # Retraso para causalidad
n_surrogates = 500   # Número de subrogados IAAFT para el test de hipótesis

# A. Cálculo de métricas observadas
obs_pearson = manual_pearson_corr(x, y)
obs_mi = manual_mi(x, y, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(x, y, bins=N_BINS, k_lag=TEST_LAG)

# B. Cálculo de Causalidad de Granger (Test paramétrico analítico)
data_g = pd.DataFrame({'Y': y, 'X': x})
# Test X -> Y
res_xy = grangercausalitytests(data_g[['Y', 'X']], maxlag=TEST_LAG, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, TEST_LAG)

# C. Generación de Distribuciones Nulas (IAAFT) para métricas no paramétricas
print(f"Generando {n_surrogates} subrogados IAAFT...")
surr_pearson, surr_mi, surr_te = [], [], []

for i in range(n_surrogates):
    sx = create_surrogate(x)
    sy = create_surrogate(y)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=TEST_LAG))

# D. Cálculo de p-valores empíricos
p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 3. VISUALIZACIÓN ACADÉMICA (LAYOUT HORIZONTAL)
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'}) # Fuente serif para LaTeX

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

# --- A. SERIES TEMPORALES (DETALLE) ---
ax_time = fig.add_subplot(gs[0, :])
zoom = slice(0, 60) # Visualizamos 60 pasos para claridad
ax_time.plot(standardize(x)[zoom], 'o-', label='Proceso X', color='#1f77b4', markersize=4, linewidth=1.2, alpha=0.8)
ax_time.plot(standardize(y)[zoom], 's-', label='Proceso Y', color='#ff7f0e', markersize=4, linewidth=1.2, alpha=0.8)
ax_time.set_title('Caso 1: Baseline (Ruido Blanco Independiente)', fontweight='bold', pad=10)
ax_time.set_xlabel('Paso de Tiempo (t)')
ax_time.set_ylabel('Amplitud (std)')
ax_time.legend(loc='upper right', ncol=2, frameon=True)
ax_time.grid(True, alpha=0.3, linestyle='--')

# Función para estandarizar los histogramas de significancia
def plot_stat(ax, surr_data, obs_val, p_val, title, color_line):
    ax.hist(surr_data, bins=30, color='#cccccc', edgecolor='white', density=True, alpha=0.7)
    ax.axvline(obs_val, color=color_line, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Texto de resultado
    res_text = f"p = {p_val:.4f}\n" + ("(Sig.)" if p_val < 0.05 else "(No Sig.)")
    ax.text(0.95, 0.90, res_text, transform=ax.transAxes, verticalalignment='top', 
            horizontalalignment='right', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_yticks([]) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', fontsize='small')

# --- B. DISTRIBUCIONES DE SIGNIFICANCIA ---
ax_p = fig.add_subplot(gs[1, 0])
plot_stat(ax_p, surr_pearson, obs_pearson, p_pearson, 'Correlación de Pearson', '#e41a1c')

ax_mi = fig.add_subplot(gs[1, 1])
plot_stat(ax_mi, surr_mi, obs_mi, p_mi, 'Información Mutua', '#377eb8')

ax_te = fig.add_subplot(gs[1, 2])
plot_stat(ax_te, surr_te, obs_te, p_te, 'Transfer Entropy', '#4daf4a')

plt.tight_layout()
plt.savefig('Benchmark_Caso1_Sintetico.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 4. TABLA DE RESULTADOS POR CONSOLA
# ---------------------------------------------------------
print("\n" + "="*75)
print(f"{'MÉTRICA':<25} | {'P-VALOR':<10} | {'RESULTADO TEST'}")
print("-" * 75)
def print_row(name, p):
    status = "CORRECTO (No Causal)" if p >= 0.05 else "ERROR (Falso Positivo)"
    print(f"{name:<25} | {p:10.4f} | {status}")

print_row("Pearson (Asociación)", p_pearson)
print_row("Info. Mutua (No Lineal)", p_mi)
print_row("Granger (Lineal)", p_granger)
print_row("Transfer Entropy (Causal)", p_te)
print("="*75)