# ==============================================================================
# Case 6: Lorenz Attractor Causal Discovery
# Description: Generates both the CMI Evolution Plot (PNG) and the 
#              Real-Time Network Animation (GIF) with Dynamic p-value thresholds.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import odeint

# ---------------------------------------------------------
# 1. FUNCIONES MATEMÁTICAS (IAAFT y CMI)
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
    """Calcula Información Mutua Condicional (Bits)"""
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

# ---------------------------------------------------------
# 2. GENERAR DATOS (SISTEMA LORENZ)
# ---------------------------------------------------------
def lorenz_deriv(state, t):
    x, y, z = state
    return 10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z

print("1/4 - Generando dinámica del Atractor de Lorenz...")
np.random.seed(42)
t_raw = np.arange(0.0, 50.0, 0.01)
states_raw = odeint(lorenz_deriv, [1.0, 1.0, 1.0], t_raw)

states_clean = states_raw[1500::5]
vars_dict = {'X': states_clean[:, 0], 'Y': states_clean[:, 1], 'Z': states_clean[:, 2]}

# ---------------------------------------------------------
# 3. PRE-CALCULAR CMI Y UMBRALES DE SIGNIFICANCIA (IAAFT)
# ---------------------------------------------------------
WINDOW_SIZE = 150
STEP = 5
N_FRAMES = 100 # Reducido a 100 para que el código termine en 1-2 minutos
N_SURR = 30    # Subrogados por ventana temporal

links = [('X', 'Y', 'Z'), ('Y', 'X', 'Z'), ('X', 'Z', 'Y'),
         ('Z', 'X', 'Y'), ('Y', 'Z', 'X'), ('Z', 'Y', 'X')]

cmi_history = {link: [] for link in links}
dynamic_threshold = []

print(f"2/4 - Pre-calculando flujos de información y p-valores para {N_FRAMES} ventanas...")
for i in range(N_FRAMES):
    start = i * STEP
    end = start + WINDOW_SIZE
    
    # Calcular CMI para las 6 posibles direcciones
    for source, target, cond in links:
        src = vars_dict[source][start : end-1]
        tgt = vars_dict[target][start+1 : end]
        cnd = vars_dict[cond][start : end-1]
        cmi_history[(source, target, cond)].append(manual_cmi(src, tgt, cnd))
        
    # Calcular Umbral de Ruido IAAFT (Percentil 95 -> p-valor 0.05)
    src_thresh = vars_dict['X'][start : end-1]
    tgt_thresh = vars_dict['Y'][start+1 : end]
    cnd_thresh = vars_dict['Z'][start : end-1]
    
    surr_vals = [manual_cmi(create_surrogate(src_thresh), tgt_thresh, cnd_thresh) for _ in range(N_SURR)]
    dynamic_threshold.append(np.percentile(surr_vals, 95))

# ---------------------------------------------------------
# 4. EXPORTAR GRÁFICA ESTÁTICA (PNG)
# ---------------------------------------------------------
print("3/4 - Generando Gráfica de Evolución (PNG)...")
fig1, ax1 = plt.subplots(figsize=(12, 6))
time_axis = np.arange(N_FRAMES)

# Banda de Ruido
ax1.fill_between(time_axis, 0, dynamic_threshold, color='silver', alpha=0.5, 
                 label='Umbral Ruido IAAFT (p > 0.05)')

colors_links = {
    ('X', 'Y', 'Z'): ('#1f77b4', '-'), ('Y', 'X', 'Z'): ('#ff7f0e', '--'),
    ('X', 'Z', 'Y'): ('#2ca02c', '-'), ('Z', 'X', 'Y'): ('#d62728', '--'),
    ('Y', 'Z', 'X'): ('#9467bd', '-'), ('Z', 'Y', 'X'): ('#8c564b', '--')
}

for link in links:
    src, tgt, _ = link
    color, linestyle = colors_links[link]
    ax1.plot(time_axis, cmi_history[link], label=f'{src} -> {tgt}', 
             color=color, linestyle=linestyle, linewidth=2.5, alpha=0.85)

ax1.set_title("Evolución Temporal de CMI con Validación Estadística (p < 0.05)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Paso de Tiempo (Ventana Móvil)", fontsize=12)
ax1.set_ylabel("Información Mutua Condicional (Bits)", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.4); ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

plt.tight_layout()
fig1.savefig('TFG_Evolucion_CMI.png')
plt.close(fig1)

# ---------------------------------------------------------
# 5. EXPORTAR ANIMACIÓN (GIF)
# ---------------------------------------------------------
print("4/4 - Generando Animación de Red (GIF)... Esto tardará aprox 1 minuto.")
fig2 = plt.figure(figsize=(15, 6))
ax_3d = fig2.add_subplot(1, 2, 1, projection='3d')
ax_net = fig2.add_subplot(1, 2, 2)

G = nx.DiGraph()
G.add_nodes_from(['X', 'Y', 'Z'])
pos = {'X': (0.2, 0.8), 'Y': (0.8, 0.8), 'Z': (0.5, 0.2)}

cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=0.0, vmax=1.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax_net, shrink=0.7, pad=0.05)
cbar.set_label('Información Mutua Condicional (Bits)', fontsize=12, fontweight='bold')

def update(frame):
    ax_3d.clear(); ax_net.clear()
    
    # Atractor 3D
    ax_3d.set_axis_on(); ax_3d.grid(True)
    ax_3d.set_xlabel('Eje X'); ax_3d.set_ylabel('Eje Y'); ax_3d.set_zlabel('Eje Z')
    end_idx = frame * STEP + WINDOW_SIZE
    start_idx = max(0, end_idx - WINDOW_SIZE)
    
    ax_3d.plot(vars_dict['X'][:end_idx], vars_dict['Y'][:end_idx], vars_dict['Z'][:end_idx], color='lightgray', alpha=0.3)
    ax_3d.plot(vars_dict['X'][start_idx:end_idx], vars_dict['Y'][start_idx:end_idx], vars_dict['Z'][start_idx:end_idx], color='red', lw=2.5)
    ax_3d.set_xlim(-20, 20); ax_3d.set_ylim(-30, 30); ax_3d.set_zlim(0, 50)
    ax_3d.view_init(elev=20., azim=-35)
    
    # Red Causal
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#333333', edgecolors='black', ax=ax_net)
    nx.draw_networkx_labels(G, pos, font_size=18, font_color='white', font_weight='bold', ax=ax_net)
    
    edges, weights, ecolors = [], [], []
    for link in links:
        val = cmi_history[link][frame]
        thresh = dynamic_threshold[frame] # Usamos el umbral dinámico exacto de este frame
        
        # SOLO DIBUJAMOS FLECHAS SI SUPERAN EL RUIDO (p < 0.05)
        if val > thresh: 
            edges.append((link[0], link[1]))
            weights.append(val * 8)
            ecolors.append(cmap(norm(val)))
            
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color=ecolors, 
                               arrowsize=25, arrowstyle='-|>', node_size=2000, 
                               connectionstyle='arc3,rad=0.15', ax=ax_net)
    ax_net.set_title(f"Red Causal Validada (p < 0.05)\nFrame {frame}/{N_FRAMES}", fontsize=12)
    ax_net.axis('off')

ani = FuncAnimation(fig2, update, frames=N_FRAMES, interval=100)
ani.save('TFG_Animacion_Red.gif', writer=PillowWriter(fps=10))
plt.close(fig2)

print("\n¡PROGRAMA FINALIZADO CON ÉXITO!")
print("- Gráfica guardada como: TFG_Evolucion_CMI.png")
print("- Animación guardada como: TFG_Animacion_Red.gif")