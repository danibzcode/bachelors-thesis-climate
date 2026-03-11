# ==============================================================================
# TFG MASTER SCRIPT - CASE 2: SPURIOUS CORRELATION (WITH VISIBLE COMMON CAUSE)
# Description: Demonstrates how symmetric metrics fail in a "fork" structure,
#              while causal and conditional metrics (PCMCI logic) succeed.
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
    manual_cmi,
    get_best_granger_pvalue,
    standardize
)

# ---------------------------------------------------------
# 1. GENERACIÓN DE DATOS (U -> X  y  U -> Y)
# ---------------------------------------------------------
n_samples = 300
np.random.seed(10)

# U: La CAUSA COMÚN (Forzamiento externo, ej: Temperatura)
u_temp = 2.0 * np.random.randn(n_samples) 

# X e Y: Esclavos de la señal maestra (sincronizados, pero independientes entre sí)
ice_cream = u_temp + 0.8 * np.random.randn(n_samples) 
ac_usage = u_temp + 0.8 * np.random.randn(n_samples)

print("Ejecutando Caso 2: La Trampa de la Causa Común...")

# ---------------------------------------------------------
# 2. CÁLCULOS Y TEST DE HIPÓTESIS (P-VALORES)
# ---------------------------------------------------------
N_BINS, TEST_LAG, n_surrogates = 5, 1, 500

# A. Métricas observadas
obs_pearson = manual_pearson_corr(ice_cream, ac_usage)
obs_mi = manual_mi(ice_cream, ac_usage, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(ice_cream, ac_usage, bins=N_BINS, k_lag=TEST_LAG)
obs_cmi = manual_cmi(ice_cream, ac_usage, u_temp, bins=N_BINS) # El corazón de PCMCI

# B. Causalidad de Granger (Test Lineal Paramétrico)
data_g = pd.DataFrame({'Y': ac_usage, 'X': ice_cream})
res_xy = grangercausalitytests(data_g[['Y', 'X']], maxlag=TEST_LAG, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, TEST_LAG)

# C. Distribuciones Nulas (IAAFT)
print(f"Generando {n_surrogates} subrogados para validación...")
surr_pearson, surr_mi, surr_te, surr_cmi = [], [], [], []

for _ in range(n_surrogates):
    # Destruimos la relación de X manteniendo Y y U intactos
    sx = create_surrogate(ice_cream) 
    surr_pearson.append(manual_pearson_corr(sx, ac_usage))
    surr_mi.append(manual_mi(sx, ac_usage, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, ac_usage, bins=N_BINS, k_lag=TEST_LAG))
    surr_cmi.append(manual_cmi(sx, ac_usage, u_temp, bins=N_BINS))

p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)
p_cmi = np.mean(np.array(surr_cmi) >= obs_cmi)

# ---------------------------------------------------------
# 3. VISUALIZACIÓN ACADÉMICA OPTIMIZADA
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])

# --- A. SERIES TEMPORALES CON CAUSA COMÚN (U) ---
ax_time = fig.add_subplot(gs[0, :])
z = slice(0, 60) 

# Dibujamos U con una línea negra gruesa de fondo
ax_time.plot(standardize(u_temp)[z], color='black', linewidth=3, alpha=0.2, label='Causa Común (U: Temperatura)')
ax_time.plot(standardize(ice_cream)[z], 'o-', label='Variable X (Helados)', color='#e41a1c', markersize=4, alpha=0.8)
ax_time.plot(standardize(ac_usage)[z], 's-', label='Variable Y (A/A)', color='#377eb8', markersize=4, alpha=0.8)

ax_time.set_title('Mecanismo de Correlación Espuria', fontweight='bold', fontsize=13)
ax_time.set_ylabel('Amplitud (std)')
ax_time.legend(loc='upper right', ncol=3, frameon=True)
ax_time.grid(True, alpha=0.3, linestyle='--')

# Función para histogramas de p-valor
def plot_hist(ax, surr_data, obs_val, p_val, title, color_obs):
    ax.hist(surr_data, bins=30, color='#cccccc', edgecolor='white', density=True, alpha=0.7)
    ax.axvline(obs_val, color=color_obs, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    status = "SIG. (Fallo)" if p_val < 0.05 else "NO SIG. (Éxito)"
    color_text = 'red' if p_val < 0.05 else 'green'
    
    ax.text(0.95, 0.90, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes, 
            ha='right', va='top', fontweight='bold', color=color_text, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', fontsize='small')

# --- B. DISTRIBUCIONES DE SIGNIFICANCIA ---
# Los métodos de asociación fallan (detectan la correlación espuria) - ROJO
plot_hist(fig.add_subplot(gs[1, 0]), surr_pearson, obs_pearson, p_pearson, 'Pearson (Asociación)', '#e41a1c')
plot_hist(fig.add_subplot(gs[1, 1]), surr_mi, obs_mi, p_mi, 'Info. Mutua (No Lineal)', '#377eb8')

# Los métodos causales aciertan (ven que no hay predictibilidad cruzada) - VERDE
plot_hist(fig.add_subplot(gs[1, 2]), surr_te, obs_te, p_te, 'Transfer Entropy', '#4daf4a')

# PCMCI acierta condicionando (destruye el enlace al ver a U) - VERDE
plot_hist(fig.add_subplot(gs[1, 3]), surr_cmi, obs_cmi, p_cmi, 'CMI (Lógica PCMCI)', '#984ea3')

plt.tight_layout()
plt.savefig('TFG_Caso2_Mecanismo_U.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. TABLA DE RESULTADOS POR CONSOLA
# ---------------------------------------------------------
print("\n" + "="*85)
print(f"{'MÉTRICA':<25} | {'P-VALOR':<10} | {'COMPORTAMIENTO ESPERADO'}")
print("-" * 85)
print(f"{'Pearson':<25} | {p_pearson:10.4f} | {'ERROR (Falso Positivo por sincronía)'}")
print(f"{'Información Mutua':<25} | {p_mi:10.4f} | {'ERROR (Falso Positivo por sincronía)'}")
print(f"{'Granger Causality':<25} | {p_granger:10.4f} | {'CORRECTO (Rechaza enlace temporal)'}")
print(f"{'Transfer Entropy':<25} | {p_te:10.4f} | {'CORRECTO (Rechaza enlace temporal)'}")
print(f"{'CMI (PCMCI condicional)':<25} | {p_cmi:10.4f} | {'CORRECTO (Bloquea la causa común)'}")
print("="*85)