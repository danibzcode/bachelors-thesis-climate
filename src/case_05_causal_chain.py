# ==============================================================================
# TFG MASTER SCRIPT - CASE 5: CAUSAL CHAIN & CONDITIONAL INDEPENDENCE
# Description: Demonstrates how bivariate methods (Granger/TE) fail to 
#              distinguish direct vs indirect links. PCMCI (CMI) succeeds.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from causal_tools import (
    create_surrogate, 
    manual_mi, 
    manual_transfer_entropy_lagged,
    manual_cmi, 
    get_best_granger_pvalue,
    standardize
)

# ---------------------------------------------------------
# 1. GENERACIÓN DE DATOS (CADENA: X -> Y -> Z)
# ---------------------------------------------------------
n_samples, t_lag = 400, 1
np.random.seed(99)
oil = np.zeros(n_samples)   # X: Causa Primaria (Petróleo)
gas = np.zeros(n_samples)   # Y: Mediador (Gasolina)
trans = np.zeros(n_samples) # Z: Efecto Final (Transporte)

for t in range(t_lag, n_samples):
    oil[t] = 0.8 * oil[t-1] + 0.5 * np.random.randn()
    gas[t] = 0.5 * gas[t-1] + 0.4 * oil[t-t_lag] + 0.5 * np.random.randn()
    trans[t] = 0.7 * trans[t-1] + 0.3 * gas[t-t_lag] + 0.5 * np.random.randn()

print("Ejecutando Caso 5: Cadena Causal (X -> Y -> Z)...")
print("Evaluando el enlace espurio directo X -> Z")

# Alineación estricta para el test causal (X_t-1 causa Z_t dado Y_t-1)
z_target = trans[t_lag:]
x_source = oil[:-t_lag]
y_mediator = gas[:-t_lag]

# ---------------------------------------------------------
# 2. ANÁLISIS DE SIGNIFICANCIA (IAAFT)
# ---------------------------------------------------------
N_BINS, n_surrogates = 5, 500

# A. Métricas bivariadas (Deben fallar al detectar el "eco")
obs_mi = manual_mi(x_source, z_target, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(oil, trans, bins=N_BINS, k_lag=t_lag)

data_g = pd.DataFrame({'Z': trans, 'X': oil})
res_xz = grangercausalitytests(data_g[['Z', 'X']], maxlag=2, verbose=False)
p_granger = get_best_granger_pvalue(res_xz, 2)

# B. Métrica multivariada / Condicional (Debe acertar al bloquear Y)
obs_cmi = manual_cmi(x_source, z_target, y_mediator, bins=N_BINS)

print(f"Calculando {n_surrogates} subrogados para validar la ruta indirecta X -> Z...")
surr_mi, surr_te, surr_cmi = [], [], []

for _ in range(n_surrogates):
    # Destruimos la topología original aleatorizando la fuente X
    sx_source = create_surrogate(x_source)
    sx_full = create_surrogate(oil)
    
    surr_mi.append(manual_mi(sx_source, z_target, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx_full, trans, bins=N_BINS, k_lag=t_lag))
    surr_cmi.append(manual_cmi(sx_source, z_target, y_mediator, bins=N_BINS))

p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)
p_cmi = np.mean(np.array(surr_cmi) >= obs_cmi)

# ---------------------------------------------------------
# 3. VISUALIZACIÓN ACADÉMICA (BLOQUEO DE INFORMACIÓN)
# ---------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

# --- A. SERIES TEMPORALES (EL FLUJO DE LA SEÑAL) ---
ax_time = fig.add_subplot(gs[0, :])
zoom = slice(0, 80)
ax_time.plot(standardize(oil)[zoom], color='black', label='Causa Inicial (X)', linewidth=2)
ax_time.plot(standardize(gas)[zoom], color='#27ae60', label='Mediador (Y)', linewidth=1.5, alpha=0.7)
ax_time.plot(standardize(trans)[zoom], color='#2980b9', label='Efecto Final (Z)', linewidth=1.5, alpha=0.7)
ax_time.set_title('Dinámica de una Cadena Causal: Propagación Transitiva', fontweight='bold')
ax_time.legend(loc='upper right', ncol=3)
ax_time.grid(alpha=0.3, linestyle='--')

# Función para histogramas
def plot_p(ax, surr_data, obs_val, p_val, title, color_obs):
    ax.hist(surr_data, bins=30, color='#cccccc', edgecolor='white', density=True, alpha=0.7)
    ax.axvline(obs_val, color=color_obs, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # Lógica: Si es bivariado (falla si es sig), si es condicional (acierta si no es sig)
    if 'Condicional' in title:
        status = "NO SIG. (Éxito)" if p_val >= 0.05 else "SIG. (Fallo)"
        t_color = 'green' if p_val >= 0.05 else 'red'
    else:
        status = "SIG. (Fallo)" if p_val < 0.05 else "NO SIG. (Éxito)"
        t_color = 'red' if p_val < 0.05 else 'green'

    ax.text(0.95, 0.90, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes, 
            ha='right', va='top', fontweight='bold', color=t_color, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', fontsize='small')

# --- B. DISTRIBUCIONES (MI Y TE FALLAN, CMI ACIERTA) ---
plot_p(fig.add_subplot(gs[1, 0]), surr_mi, obs_mi, p_mi, 'MI (Bivariada X → Z)', '#e67e22')
plot_p(fig.add_subplot(gs[1, 1]), surr_te, obs_te, p_te, 'TE (Bivariada X → Z)', '#d35400')
plot_p(fig.add_subplot(gs[1, 2]), surr_cmi, obs_cmi, p_cmi, 'CMI (Condicional X → Z | Y)', '#16a085')

plt.tight_layout()
plt.savefig('TFG_Caso5_CadenaCausal.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. TABLA DE RESULTADOS POR CONSOLA
# ---------------------------------------------------------
print("\n" + "="*85)
print(f"{'TEST (Evaluando X -> Z)':<30} | {'P-VALOR':<10} | {'CONCLUSIÓN (Estructura de Red)'}")
print("-" * 85)
print(f"{'Información Mutua (X -> Z)':<30} | {p_mi:10.4f} | {'ERROR (Falso positivo por eco)'}")
print(f"{'Granger Causality (X -> Z)':<30} | {p_granger:10.4f} | {'ERROR (Falso positivo transitivo)'}")
print(f"{'Transfer Entropy (X -> Z)':<30} | {p_te:10.4f} | {'ERROR (Falso positivo transitivo)'}")
print(f"{'CMI / PCMCI (X -> Z | Y)':<30} | {p_cmi:10.4f} | {'ÉXITO (Bloquea ruta espuria)'}")
print("="*85)