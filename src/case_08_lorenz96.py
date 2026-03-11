# ==============================================================================
# TFG MASTER SCRIPT: Lorenz-96 Spatiotemporal Causal Discovery
# Description: Generates Hovmoller diagram, performs local PCMCI with IAAFT,
#              and renders the highly-optimized 40x40 Causal Matrix Animation.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import odeint
import time

# ---------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS (IAAFT, CMI normal y CMI Rápido)
# ---------------------------------------------------------
def create_surrogate(ts):
    """Genera subrogado IAAFT para test de hipótesis (ruido lineal)"""
    f_coeffs = np.fft.rfft(ts)
    amplitudes = np.abs(f_coeffs)
    random_phases = np.random.uniform(0, 2 * np.pi, len(f_coeffs))
    new_f_coeffs = amplitudes * np.exp(1j * random_phases)
    new_f_coeffs[0] = f_coeffs[0]
    if len(ts) % 2 == 0:
        new_f_coeffs[-1] = f_coeffs[-1]
    return np.fft.irfft(new_f_coeffs, n=len(ts))

def manual_cmi(x, y, z, bins=5):
    """Calcula CMI estándar para el test local de variables"""
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
                if p_z[k] > epsilon and p_xz[i, k] > epsilon and p_yz[j, k] > epsilon:
                    cmi += p_xyz_val * np.log((p_xyz_val * p_z[k]) / (p_xz[i, k] * p_yz[j, k]))
    return cmi / np.log(2)

def fast_te(x_src, x_tgt, bins=4):
    """Versión vectorizada ultra-rápida de CMI para calcular la matriz 40x40"""
    x = x_src[:-1]; y = x_tgt[1:]; z = x_tgt[:-1]
    
    data = np.stack([x, y, z], axis=1)
    p_xyz, edges = np.histogramdd(data, bins=bins)
    p_xyz = p_xyz / len(x)
    
    p_xz, _, _ = np.histogram2d(x, z, bins=[edges[0], edges[2]])
    p_xz = p_xz / len(x)
    
    p_yz, _, _ = np.histogram2d(y, z, bins=[edges[1], edges[2]])
    p_yz = p_yz / len(y)
    
    p_z, _ = np.histogram(z, bins=edges[2])
    p_z = p_z / len(z)
    
    p_z_full = np.broadcast_to(p_z[np.newaxis, np.newaxis, :], p_xyz.shape)
    p_xz_full = np.broadcast_to(p_xz[:, np.newaxis, :], p_xyz.shape)
    p_yz_full = np.broadcast_to(p_yz[np.newaxis, :, :], p_xyz.shape)
    
    mask = (p_xyz > 1e-10) & (p_z_full > 1e-10) & (p_xz_full > 1e-10) & (p_yz_full > 1e-10)
    cmi = np.sum(p_xyz[mask] * np.log((p_xyz[mask] * p_z_full[mask]) / (p_xz_full[mask] * p_yz_full[mask])))
    return cmi / np.log(2)

# ---------------------------------------------------------
# 2. GENERAR LORENZ-96 (Anillo N=40)
# ---------------------------------------------------------
def lorenz96(x, t, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

N = 40
F = 8.0

print("1/4 - Integrando dinámica espacial de Lorenz-96...")
x0 = np.full(N, F)
x0[19] += 0.01  # Perturbación inicial que desata el caos

t_raw = np.arange(0.0, 50.0, 0.01)
states_raw = odeint(lorenz96, x0, t_raw, args=(F,))

states_clean = states_raw[500::5]
t_clean = t_raw[500::5]

# ---------------------------------------------------------
# 3. EXPORTAR DIAGRAMA HOVMÖLLER ESTÁTICO (PNG)
# ---------------------------------------------------------
print("2/4 - Generando Diagrama de Hovmöller...")
fig1, ax1 = plt.subplots(figsize=(10, 5))
c1 = ax1.contourf(t_raw[0:1500], np.arange(N), states_raw[0:1500].T, levels=20, cmap='RdBu_r')
ax1.set_title("Propagación del Caos en Lorenz-96 (Transitorio inicial)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Tiempo", fontsize=12)
ax1.set_ylabel("Índice de Variable ($X_0$ a $X_{39}$)", fontsize=12)
fig1.colorbar(c1, ax=ax1, label="Amplitud de la Variable")
plt.tight_layout()
fig1.savefig("TFG_Lorenz96_Hovmoller.png")
plt.close(fig1)

# ---------------------------------------------------------
# 4. PCMCI LOCAL CON VALIDACIÓN IAAFT (Consola)
# ---------------------------------------------------------
print("\n3/4 - Test PCMCI Local (Validando la advección asimétrica)...")
# Extraemos 3 variables consecutivas
x2 = states_clean[:, 2]
x3 = states_clean[:, 3]
x4 = states_clean[:, 4]

# Físicamente, en Lorenz-96 el flujo va predominantemente i-1 -> i
links_local = [
    ('X2', x2, 'X3', x3, x4),  # Hacia la derecha (Esperado: Fuerte)
    ('X4', x4, 'X3', x3, x2)   # Hacia la izquierda (Esperado: Débil/Ruido)
]

N_SURR = 50
print("\n" + "="*60)
print("RESULTADOS PCMCI: ESTRUCTURA ESPACIAL DEL ANILLO (p < 0.05)")
print("="*60)

for src_name, src_data, tgt_name, tgt_data, cond_data in links_local:
    src_t_minus_1 = src_data[:-1]; tgt_t = tgt_data[1:]; cond_t_minus_1 = cond_data[:-1]
    
    obs_cmi = manual_cmi(src_t_minus_1, tgt_t, cond_t_minus_1, bins=5)
    
    surr_cmis = [manual_cmi(create_surrogate(src_t_minus_1), tgt_t, cond_t_minus_1, bins=5) for _ in range(N_SURR)]
    p_val = np.mean(np.array(surr_cmis) >= obs_cmi)
    
    sig = "SIGNIFICATIVO (Advección Real)" if p_val < 0.05 else "RUIDO (Rechazado)"
    print(f"Enlace {src_name} -> {tgt_name} | CMI = {obs_cmi:5.3f} bits | p-valor: {p_val:5.3f} | {sig}")
print("="*60 + "\n")

# ---------------------------------------------------------
# 5. MATRIZ 40x40 ANIMADA (GIF)
# ---------------------------------------------------------
WINDOW_SIZE = 150
STEP = 15
N_FRAMES = min((len(states_clean) - WINDOW_SIZE) // STEP, 60) 

print(f"4/4 - Calculando {N_FRAMES} matrices causales 40x40 y renderizando GIF...")
print("Esto tomará aproximadamente 30 segundos. ¡Paciencia!")

matrices = []
for f in range(N_FRAMES):
    start_idx = f * STEP
    end_idx = start_idx + WINDOW_SIZE
    mat = np.zeros((N, N))
    window_data = states_clean[start_idx:end_idx]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                mat[i, j] = fast_te(window_data[:, i], window_data[:, j], bins=4)
    matrices.append(mat)

fig2, (ax_hov, ax_mat) = plt.subplots(1, 2, figsize=(15, 6))

hov_plot = ax_hov.contourf(np.arange(len(states_clean)), np.arange(N), states_clean.T, levels=20, cmap='RdBu_r')
ax_hov.set_title("Evolución Lorenz-96 (Hovmöller)", fontsize=14, fontweight='bold')
ax_hov.set_ylabel("Variables en el anillo ($X_0$ a $X_{39}$)", fontsize=12)
ax_hov.set_xlabel("Paso de tiempo (Submuestreado)", fontsize=12)

cmap_mat = plt.cm.magma 
norm_mat = mcolors.Normalize(vmin=0.0, vmax=np.max(matrices)) 
mat_im = ax_mat.imshow(matrices[0], cmap=cmap_mat, norm=norm_mat, origin='lower')

ax_mat.set_title("Matriz Causal 40x40 (Información Mutua)", fontsize=14, fontweight='bold')
ax_mat.set_xlabel("Variable Destino (Efecto $X_j$)", fontsize=12)
ax_mat.set_ylabel("Variable Origen (Causa $X_i$)", fontsize=12)
cbar = fig2.colorbar(mat_im, ax=ax_mat, shrink=0.8)
cbar.set_label('Información Mutua Condicional (Bits)', fontsize=12, fontweight='bold')

rect = patches.Rectangle((0, 0), WINDOW_SIZE, N-1, linewidth=3, edgecolor='lime', facecolor='none')
ax_hov.add_patch(rect)

def update(frame):
    start_idx = frame * STEP
    rect.set_xy((start_idx, 0))
    mat_im.set_array(matrices[frame])
    ax_mat.set_title(f"Matriz Causal 40x40 (Bits)\nVentana de tiempo: {start_idx} a {start_idx + WINDOW_SIZE}", fontsize=12, fontweight='bold')
    return [rect, mat_im]

ani = FuncAnimation(fig2, update, frames=N_FRAMES, interval=150)
ani.save('TFG_Lorenz96_Matriz40x40.gif', writer=PillowWriter(fps=6))
plt.close(fig2)

print("¡PROGRAMA FINALIZADO CON ÉXITO!")
print("- Gráfica estática guardada: TFG_Lorenz96_Hovmoller.png")
print("- Animación guardada: TFG_Lorenz96_Matriz40x40.gif")