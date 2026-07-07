# ==============================================================================
# LORENZ 63 - FIGURA 7: RECONSTRUCCIÓN FÍSICA DE LA RED (CMI) + CASO NEGATIVO
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ---------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS 
# ---------------------------------------------------------
def create_surrogate(ts):
    """Genera subrogado IAAFT para destruir relaciones no lineales"""
    f_coeffs = np.fft.rfft(ts)
    amplitudes = np.abs(f_coeffs)
    random_phases = np.random.uniform(0, 2 * np.pi, len(f_coeffs))
    new_f_coeffs = amplitudes * np.exp(1j * random_phases)
    new_f_coeffs[0] = f_coeffs[0]
    if len(ts) % 2 == 0:
        new_f_coeffs[-1] = f_coeffs[-1]
    return np.fft.irfft(new_f_coeffs, n=len(ts))

def manual_cmi(x, y, z, bins=5):
    """Calcula Información Mutua Condicional (Bits) con 1 condición"""
    data = np.stack([x, y, z], axis=1)
    p_xyz_counts, edges = np.histogramdd(data, bins=bins)
    p_xyz = p_xyz_counts / len(x)
    p_xz_counts, _, _ = np.histogram2d(x, z, bins=[edges[0], edges[2]])
    p_xz = p_xz_counts / len(x)
    p_yz_counts, _, _ = np.histogram2d(y, z, bins=[edges[1], edges[2]])
    p_yz = p_yz_counts / len(y)
    p_z_counts, _ = np.histogram(z, bins=edges[2])
    p_z = p_z_counts / len(z)

    cmi = 0.0
    epsilon = 1e-12
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                p_xyz_val = p_xyz[i, j, k]
                if p_xyz_val < epsilon: continue
                p_z_val = p_z[k]
                p_xz_val = p_xz[i, k]
                p_yz_val = p_yz[j, k]
                if p_z_val > epsilon and p_xz_val > epsilon and p_yz_val > epsilon:
                    log_term = np.log((p_xyz_val * p_z_val) / (p_xz_val * p_yz_val))
                    cmi += p_xyz_val * log_term
    return cmi / np.log(2)

def manual_cmi_2cond(x, y, z1, z2, bins=2):
    """Calcula CMI condicionando sobre DOS variables simultáneamente.
    Se usa bins=2 para aplicar Coarse-Graining y eliminar el eco de integración."""
    data = np.stack([x, y, z1, z2], axis=1)
    p_xyzz_counts, edges = np.histogramdd(data, bins=bins)
    p_xyzz = p_xyzz_counts / len(x)
    
    p_zz_counts, _ = np.histogramdd(np.stack([z1, z2], axis=1), bins=[edges[2], edges[3]])
    p_zz = p_zz_counts / len(x)
    
    p_xzz_counts, _ = np.histogramdd(np.stack([x, z1, z2], axis=1), bins=[edges[0], edges[2], edges[3]])
    p_xzz = p_xzz_counts / len(x)
    
    p_yzz_counts, _ = np.histogramdd(np.stack([y, z1, z2], axis=1), bins=[edges[1], edges[2], edges[3]])
    p_yzz = p_yzz_counts / len(x)
    
    cmi = 0.0
    epsilon = 1e-12
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                for l in range(bins):
                    p_xyzz_val = p_xyzz[i, j, k, l]
                    if p_xyzz_val < epsilon: continue
                    p_zz_val = p_zz[k, l]
                    p_xzz_val = p_xzz[i, k, l]
                    p_yzz_val = p_yzz[j, k, l]
                    if p_zz_val > epsilon and p_xzz_val > epsilon and p_yzz_val > epsilon:
                        cmi += p_xyzz_val * np.log((p_xyzz_val * p_zz_val) / (p_xzz_val * p_yzz_val))
    return cmi / np.log(2)

def standardize(ts):
    return (ts - np.mean(ts)) / np.std(ts)

# ---------------------------------------------------------
# 2. GENERAR DATOS (SISTEMA LORENZ 63)
# ---------------------------------------------------------
def lorenz_deriv(state, t):
    x, y, z = state
    return 10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z

print("Generando dinámica del Atractor de Lorenz 63...")
np.random.seed(42)
t_raw = np.arange(0.0, 50.0, 0.01) 
states_raw = odeint(lorenz_deriv, [1.0, 1.0, 1.0], t_raw)

# SOLUCIÓN FÍSICA: Reducimos el salto a ::2 para mitigar la expansión de Taylor
N_SAMPLES = 400
states_clean = states_raw[1500 : 1500 + N_SAMPLES * 2 : 2]

X = states_clean[:, 0]
Y = states_clean[:, 1]
Z = states_clean[:, 2]

# Definir variables origen, destino y condición (lag=1)
X_src, X_tgt = X[:-1], X[1:]
Y_src, Y_tgt = Y[:-1], Y[1:]
Z_src, Z_tgt = Z[:-1], Z[1:]

# ---------------------------------------------------------
# 3. CÁLCULO DE CMI Y SURROGATES
# ---------------------------------------------------------
N_SURR = 500 
print(f"Calculando CMI y {N_SURR} surrogates por enlace...")

# 1. Y -> X | Z (Fuerte acoplamiento real)
obs_yx = manual_cmi(Y_src, X_tgt, Z_src)
surr_yx = [manual_cmi(create_surrogate(Y_src), X_tgt, Z_src) for _ in range(N_SURR)]
p_yx = np.mean(np.array(surr_yx) >= obs_yx)

# 2. X -> Y | Z (Fuerte acoplamiento real)
obs_xy = manual_cmi(X_src, Y_tgt, Z_src)
surr_xy = [manual_cmi(create_surrogate(X_src), Y_tgt, Z_src) for _ in range(N_SURR)]
p_xy = np.mean(np.array(surr_xy) >= obs_xy)

# 3. X -> Z | Y (Fuerte acoplamiento real)
obs_xz = manual_cmi(X_src, Z_tgt, Y_src)
surr_xz = [manual_cmi(create_surrogate(X_src), Z_tgt, Y_src) for _ in range(N_SURR)]
p_xz = np.mean(np.array(surr_xz) >= obs_xz)

# 4. Z -> X | X_past, Y_past (ENLACE FALSO - Bloqueo en 4D con bins=2)
obs_zx = manual_cmi_2cond(Z_src, X_tgt, X_src, Y_src, bins=2)
surr_zx = [manual_cmi_2cond(create_surrogate(Z_src), X_tgt, X_src, Y_src, bins=2) for _ in range(N_SURR)]
p_zx = np.mean(np.array(surr_zx) >= obs_zx)

# ---------------------------------------------------------
# 4. VISUALIZACIÓN ACADÉMICA
# ---------------------------------------------------------

# Aplicamos la configuración de tamaños de fuente
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 14, 'font.family': 'serif'})

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2])

# --- PANEL SUPERIOR: Serie Temporal ---
ax_time = fig.add_subplot(gs[0, :])

zoom = slice(50, 120) 
time_axis = np.arange(70) * 0.05 + 12.5 

ax_time.plot(time_axis, standardize(X)[zoom], 'o-', label='$X_t$', color='#1f77b4', markersize=4, linewidth=1.2, alpha=0.8)
ax_time.plot(time_axis, standardize(Y)[zoom], 's-', label='$Y_t$', color='#ff7f0e', markersize=4, linewidth=1.2, alpha=0.8)
ax_time.plot(time_axis, standardize(Z)[zoom], 'd-', label='$Z_t$', color='#2ca02c', markersize=4, linewidth=1.2, alpha=0.8)

# Aumentamos ligeramente el título principal para que destaque aún más con los nuevos tamaños
ax_time.set_title('Lorenz 63: Reconstrucción Física de la Red (PCMI)', fontweight='bold', fontsize=17, pad=12)
ax_time.set_xlabel('Paso de Tiempo (t)')
ax_time.set_ylabel('Amplitud (std)')

y_min, y_max = ax_time.get_ylim()
y_range = y_max - y_min
ax_time.set_ylim(y_min, y_max + (y_range * 0.35))
ax_time.legend(loc='upper right', ncol=3, frameon=True)
ax_time.grid(True, alpha=0.3, linestyle='--')

# --- PANEL INFERIOR: Histogramas ---
def plot_stat(ax, surr_data, obs_val, p_val, title, color_obs, is_false_link=False):
    ax.hist(surr_data, bins=25, color='#cccccc', edgecolor='dimgray', density=True, alpha=0.8)
    max_surr, min_surr = np.max(surr_data), np.min(surr_data)
    rango_surr = max_surr - min_surr
    
    if obs_val > max_surr + 2 * rango_surr:
        ax.set_xlim(min_surr - 0.5 * rango_surr, max_surr + 1.5 * rango_surr)
        ax.annotate(f'Obs: {obs_val:.3f} \u2192', xy=(0.95, 0.4), xycoords='axes fraction',
                    color=color_obs, fontweight='bold', fontsize=13, ha='right', va='center')
    else:
        ax_max = max(max_surr, obs_val)
        ax.set_xlim(min_surr - 0.5 * rango_surr, ax_max + 0.5 * rango_surr)
        ax.axvline(obs_val, color=color_obs, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')
        ax.legend(loc='upper right', fontsize=12)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.45)
    
    # Aumentamos el tamaño de los títulos de los subgráficos
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    if is_false_link:
        status = "NO SIG. (Éxito)" if p_val >= 0.05 else "SIG. (Fallo)"
        color_text = 'green' if p_val >= 0.05 else 'red'
    else:
        status = "SIG. (Éxito)" if p_val < 0.05 else "NO SIG. (Fallo)"
        color_text = 'green' if p_val < 0.05 else 'red'

    ax.text(0.05, 0.95, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes, 
            ha='left', va='top', fontweight='bold', color=color_text, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
            
    ax.set_yticks([]) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plot_stat(fig.add_subplot(gs[1, 0]), surr_yx, obs_yx, p_yx, r'PCMI ($Y \rightarrow X | Z$)', '#d62728')
plot_stat(fig.add_subplot(gs[1, 1]), surr_xy, obs_xy, p_xy, r'PCMI ($X \rightarrow Y | Z$)', '#9467bd')
plot_stat(fig.add_subplot(gs[1, 2]), surr_xz, obs_xz, p_xz, r'PCMI ($X \rightarrow Z | Y$)', '#16a085')
plot_stat(fig.add_subplot(gs[1, 3]), surr_zx, obs_zx, p_zx, r'PCMI ($Z \rightarrow X | X_{t-1}, Y_{t-1}$)' + '\n(Falso Enlace)', '#34495e', is_false_link=True)

plt.tight_layout()
plt.show()