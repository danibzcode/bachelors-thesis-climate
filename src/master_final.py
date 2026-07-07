# ==============================================================================
# TFG MASTER SCRIPT: GENERADOR UNIVERSAL COMPLETO (CASOS 1-7) -- CORREGIDO
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import odeint

from causal_tools import (
    create_surrogate, manual_pearson_corr, manual_mi,
    manual_transfer_entropy_lagged, manual_te_test, manual_granger_test,
    manual_cmi, standardize
)

# ------------------------------------------------------------------------------
# 1. MOTORES DE VISUALIZACIÓN Y ETIQUETADO (Casos 1 al 6)
# ------------------------------------------------------------------------------

def obtener_etiqueta_estado(p_val, tipo_esperado):
    es_significativo = p_val < 0.05
    if tipo_esperado == 'Exito_SIG':
        status = "SIG.\n(Éxito)" if es_significativo else "NO SIG.\n(Falso Neg.)"
        color_text = 'green' if es_significativo else 'red'
    elif tipo_esperado == 'Exito_NO_SIG':
        status = "NO SIG.\n(Éxito)" if not es_significativo else "SIG.\n(Falso Pos.)"
        color_text = 'green' if not es_significativo else 'red'
    elif tipo_esperado == 'Falso_Positivo':
        status = "SIG.\n(Falso Pos.)" if es_significativo else "NO SIG.\n(Inesperado)"
        color_text = 'red' if es_significativo else 'green'
    elif tipo_esperado == 'Fallo_Ambiguedad':
        status = "SIG.\n(Ambigüedad)" if es_significativo else "NO SIG.\n(Fallo)"
        color_text = 'red'
    elif tipo_esperado == 'Fallo_Ceguera':
        status = "NO SIG.\n(Ceguera)" if not es_significativo else "SIG.\n(Inesperado)"
        color_text = 'red'
    elif tipo_esperado == 'Fallo_Ruido':
        status = "NO SIG.\n(Degradado)" if not es_significativo else "SIG.\n(Falso Pos.)"
        color_text = 'red'
    else:
        status = "SIG." if es_significativo else "NO SIG."
        color_text = 'black'
    return status, color_text

def render_tfg_dashboard(series_dict, obs_dict, surr_dict, expected_dict, title, filename):
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'legend.fontsize': 12, 'font.family': 'serif'})
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.2, 1.2], hspace=0.35)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

    ax_time = fig.add_subplot(gs[0, :])
    time_axis = None
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o-', 's-', 'd-']

    for idx, (label, data) in enumerate(series_dict.items()):
        zoom = slice(0, min(80, len(data)))
        if time_axis is None: time_axis = np.arange(len(data[zoom]))
        ax_time.plot(time_axis, standardize(data)[zoom], markers[idx % 3],
                     label=label, color=colors[idx % 3], markersize=4, linewidth=1.5, alpha=0.8)

    ax_time.set_xlabel('Paso de Tiempo (t)')
    ax_time.set_ylabel('Amplitud (std)')
    y_min, y_max = ax_time.get_ylim()
    ax_time.set_ylim(y_min, y_max * 1.35)
    ax_time.legend(loc='upper right', ncol=len(series_dict), frameon=True)
    ax_time.grid(True, alpha=0.3, linestyle='--')

    posiciones = [gs[1, 0:2], gs[1, 2:4], gs[1, 4:6], gs[2, 1:3], gs[2, 3:5]]

    for i, (nombre, val) in enumerate(obs_dict.items()):
        ax = fig.add_subplot(posiciones[i])
        surr_data = surr_dict[nombre]
        ax.hist(surr_data, bins=25, color='#cccccc', edgecolor='dimgray', density=True, alpha=0.8)

        if 'Pearson' in nombre:
            p_val = np.mean(np.abs(surr_data) >= np.abs(val))
        else:
            p_val = np.mean(np.array(surr_data) >= val)

        tipo_esperado = expected_dict.get(nombre, 'Exito_SIG')
        status, color_text = obtener_etiqueta_estado(p_val, tipo_esperado)

        max_surr, min_surr = np.max(surr_data), np.min(surr_data)
        rango = max_surr - min_surr
        if val > max_surr + 2 * rango:
            ax.set_xlim(min_surr - 0.5 * rango, max_surr + 1.5 * rango)
            ax.annotate(f'Obs: {val:.3f} \u2192', xy=(0.95, 0.4), xycoords='axes fraction',
                        color='black', fontweight='bold', fontsize=12, ha='right', va='center')
        else:
            ax_max = max(max_surr, val)
            ax.set_xlim(min_surr - 0.5 * rango, ax_max + 0.5 * rango)
            ax.axvline(val, color='black', linestyle='-', linewidth=2.5, label=f'Obs: {val:.3f}')
            ax.legend(loc='upper right', fontsize=11)

        ax.set_title(nombre, fontweight='bold')
        ax.text(0.05, 0.95, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes,
                ha='left', va='top', fontweight='bold', color=color_text, fontsize=11,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2.0))

        ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def render_tfg_highlight(series_dict, obs_dict, surr_dict, expected_dict, keys_to_plot, title, filename):
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'legend.fontsize': 13, 'font.family': 'serif'})
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

    ax_time = fig.add_subplot(gs[0, :])
    time_axis = None
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o-', 's-', 'd-']

    for idx, (label, data) in enumerate(series_dict.items()):
        zoom = slice(0, min(80, len(data)))
        if time_axis is None: time_axis = np.arange(len(data[zoom]))
        ax_time.plot(time_axis, standardize(data)[zoom], markers[idx % 3],
                     label=label, color=colors[idx % 3], markersize=4, linewidth=1.5, alpha=0.8)

    ax_time.set_xlabel('Paso de Tiempo (t)')
    ax_time.set_ylabel('Amplitud (std)')
    y_min, y_max = ax_time.get_ylim()
    ax_time.set_ylim(y_min, y_max * 1.35)
    ax_time.legend(loc='upper right', ncol=len(series_dict), frameon=True)
    ax_time.grid(True, alpha=0.3, linestyle='--')

    colores_linea = {'Pearson': '#e74c3c', 'Info. Mutua': '#d35400',
                     'Transfer Entropy': '#27ae60', 'Granger': '#3498db', 'PCMCI': '#8e44ad'}

    for i, target_key in enumerate(keys_to_plot):
        try:
            nombre = next(k for k in obs_dict.keys() if target_key in k)
        except StopIteration:
            print(f"⚠️ AVISO: No se encontró la métrica '{target_key}' en el diccionario. Se omite.")
            continue

        val = obs_dict[nombre]
        surr_data = surr_dict[nombre]

        ax = fig.add_subplot(gs[1, i])
        ax.hist(surr_data, bins=25, color='#cccccc', edgecolor='dimgray', density=True, alpha=0.8)

        if 'Pearson' in nombre:
            p_val = np.mean(np.abs(surr_data) >= np.abs(val))
        else:
            p_val = np.mean(np.array(surr_data) >= val)

        tipo_esperado = expected_dict.get(nombre, 'Exito_SIG')
        status, color_text = obtener_etiqueta_estado(p_val, tipo_esperado)

        max_surr, min_surr = np.max(surr_data), np.min(surr_data)
        rango = max_surr - min_surr
        line_color = next((c for k, c in colores_linea.items() if k in nombre), 'black')

        if val > max_surr + 2 * rango:
            ax.set_xlim(min_surr - 0.5 * rango, max_surr + 1.5 * rango)
            ax.annotate(f'Obs: {val:.3f} \u2192', xy=(0.95, 0.4), xycoords='axes fraction',
                        color=line_color, fontweight='bold', fontsize=13, ha='right', va='center')
        else:
            ax_max = max(max_surr, val)
            ax.set_xlim(min_surr - 0.5 * rango, ax_max + 0.5 * rango)
            ax.axvline(val, color=line_color, linestyle='-', linewidth=3, label=f'Obs: {val:.3f}')
            ax.legend(loc='upper left', fontsize=13)

        ax.set_title(nombre, fontweight='bold')
        ax.text(0.95, 0.95, f"p = {p_val:.4f}\n{status}", transform=ax.transAxes,
                ha='right', va='top', fontweight='bold', color=color_text, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2.0))

        ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 2. FUNCIONES DE LORENZ-96 (Caso 7)
# ------------------------------------------------------------------------------

def lorenz96(x, t, F):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def fast_te(x_src, x_tgt, bins=4):
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

def render_validacion_local_lorenz96(t_axis, x2, x3, x4, res_der, res_izq, res_x1, filename):
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'legend.fontsize': 11, 'font.family': 'serif'})
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35)
    fig.suptitle("Lorenz-96: Dinámica local en el vecindario y test PCMCI", fontsize=16, fontweight='bold', y=0.98)

    ax_time = fig.add_subplot(gs[0, :])
    zoom = slice(0, 150)

    ax_time.plot(t_axis[zoom], standardize(x2)[zoom], 'o-', label='$X_2$ (Oeste)', color='#377eb8', markersize=4, linewidth=1.5)
    ax_time.plot(t_axis[zoom], standardize(x3)[zoom], 's-', label='$X_3$ (Centro)', color='#ff7f00', markersize=4, linewidth=2.5)
    ax_time.plot(t_axis[zoom], standardize(x4)[zoom], '^-', label='$X_4$ (Este)', color='#4daf4a', markersize=4, linewidth=1.5)

    ax_time.set_xlabel('Paso de Tiempo ($t$)')
    ax_time.set_ylabel('Amplitud (std)')
    ax_time.legend(loc='upper right', ncol=3, frameon=True, bbox_to_anchor=(1.0, 1.15))
    ax_time.grid(True, alpha=0.3, linestyle='--')
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    def plot_hist(ax, obs_val, surr_data, p_val, title, color_line, is_sig_expected, extra_text=""):
        ax.hist(surr_data, bins=30, color='#d9d9d9', edgecolor='white', density=True, alpha=0.9)
        max_surr, min_surr = np.max(surr_data), np.min(surr_data)
        ax_max = max(max_surr, obs_val)
        rango = ax_max - min_surr
        ax.set_xlim(min_surr - 0.2 * rango, ax_max + 0.2 * rango)
        ax.axvline(obs_val, color=color_line, linestyle='-', linewidth=2.5, label=f'Obs: {obs_val:.3f}')

        status = "SIG.\n(Éxito)" if p_val < 0.05 else "NO SIG.\n(Falso Neg.)" if is_sig_expected else "NO SIG.\n(Éxito)"
        if is_sig_expected == (p_val < 0.05):
            color_text = '#2ca02c'
        else:
            status = "SIG.\n(Falso Pos.)" if p_val < 0.05 else "NO SIG.\n(Fallo)"
            color_text = '#d62728'

        texto_caja = f"PCMCI p = {p_val:.4f}\n{status}"
        if extra_text: texto_caja += f"\n{extra_text}"

        ax.text(0.05, 0.95, texto_caja, transform=ax.transAxes, ha='left', va='top', fontweight='bold', color=color_text, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#cccccc', pad=2.0))
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False)
        ax.set_yticks([])
        for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)

    plot_hist(fig.add_subplot(gs[1, 0]), res_der['obs'], res_der['surr'], res_der['p'], "Advección Derecha ($X_2 \\rightarrow X_3 | X_4$)", '#2ca02c', True)
    plot_hist(fig.add_subplot(gs[1, 1]), res_izq['obs'], res_izq['surr'], res_izq['p'], "Advección Izquierda ($X_4 \\rightarrow X_3 | X_2$)", '#2ca02c', True)
    plot_hist(fig.add_subplot(gs[1, 2]), res_x1['obs'], res_x1['surr'], res_x1['p'], "Causalidad Indirecta ($X_1 \\rightarrow X_4 | X_3$)", '#1f77b4', False, f"Corr. Pearson: {res_x1['corr']:.2f}")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 3. EJECUCIÓN DE CASOS (1 al 6)
# ------------------------------------------------------------------------------

def ejecutar_caso_1():
    print("\nProcesando Caso 1 (Ruido)...")
    np.random.seed(42)
    n_samples, lag, bins, n_surr = 250, 1, 5, 500
    x, y, z = np.random.randn(n_samples), np.random.randn(n_samples), np.random.randn(n_samples)

    obs = {'Pearson': manual_pearson_corr(x, y), 'Info. Mutua': manual_mi(x, y, bins),
           'Transfer Entropy': manual_te_test(y, x, bins, lag),
           'Granger': manual_granger_test(y, x, lag)[0], 'PCMCI': manual_cmi(x, y, z, bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx, sy, sz = create_surrogate(x), create_surrogate(y), create_surrogate(z)
        surr['Pearson'].append(manual_pearson_corr(sx, sy)); surr['Info. Mutua'].append(manual_mi(sx, sy, bins))
        surr['Transfer Entropy'].append(manual_te_test(sy, sx, bins, lag))
        surr['Granger'].append(manual_granger_test(sy, sx, lag)[0]); surr['PCMCI'].append(manual_cmi(sx, sy, sz, bins))

    expected = {k: 'Exito_NO_SIG' for k in obs.keys()}
    series = {'$X_t$ (Ruido)': x, '$Y_t$ (Ruido)': y}

    render_tfg_dashboard(series, obs, surr, expected, "Caso 1: Baseline (Ruido Blanco Independiente)", "Anexo_Caso1.png")
    render_tfg_highlight(series, obs, surr, expected, ['Pearson', 'Info. Mutua', 'Transfer Entropy'], "Caso 1: Baseline (Ruido Blanco Independiente)", "Destacado_Caso1.png")

def ejecutar_caso_2():
    print("Procesando Caso 2 (Causa Común)...")
    np.random.seed(42)
    n_samples, lag, bins, n_surr = 300, 1, 5, 500
    u = 2.0 * np.random.randn(n_samples)
    x, y = u + 0.8 * np.random.randn(n_samples), u + 0.8 * np.random.randn(n_samples)

    obs = {'Pearson': manual_pearson_corr(x, y), 'Info. Mutua': manual_mi(x, y, bins),
           'Transfer Entropy': manual_te_test(y, x, bins, lag),
           'Granger': manual_granger_test(y, x, lag)[0], 'PCMCI': manual_cmi(x, y, u, bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx, sy, su = create_surrogate(x), create_surrogate(y), create_surrogate(u)
        surr['Pearson'].append(manual_pearson_corr(sx, y)); surr['Info. Mutua'].append(manual_mi(sx, y, bins))
        surr['Transfer Entropy'].append(manual_te_test(sy, sx, bins, lag))
        surr['Granger'].append(manual_granger_test(sy, sx, lag)[0]); surr['PCMCI'].append(manual_cmi(sx, y, su, bins))

    expected = {
        'Pearson': 'Falso_Positivo',
        'Info. Mutua': 'Falso_Positivo',
        'Transfer Entropy': 'Exito_NO_SIG',
        'Granger': 'Exito_NO_SIG',
        'PCMCI': 'Exito_NO_SIG'
    }
    series = {'$U_t$ (Causa Común)': u, '$X_t$': x, '$Y_t$': y}

    render_tfg_dashboard(series, obs, surr, expected, "Caso 2: La trampa de la causa común", "Anexo_Caso2.png")
    render_tfg_highlight(series, obs, surr, expected, ['Pearson', 'Transfer Entropy', 'PCMCI'], "Caso 2: La trampa de la causa común", "Destacado_Caso2.png")

def ejecutar_caso_3():
    print("Procesando Caso 3 (Memoria)...")
    np.random.seed(42)
    burn_in = 200
    n_samples, lag, bins, n_surr = 400, 2, 4, 500
    n_total = n_samples + burn_in

    x, y, z = np.zeros(n_total), np.zeros(n_total), np.random.randn(n_total)
    for t in range(1, n_total): x[t] = 0.7 * x[t-1] + np.random.randn()
    for t in range(2, n_total): y[t] = 0.5 * y[t-1] + 0.8 * x[t-2] + np.random.randn()
    # Descartamos el transitorio inicial para trabajar en régimen estacionario
    x, y, z = x[burn_in:], y[burn_in:], z[burn_in:]

    x_lagged, y_target, z_cond = x[:-lag], y[lag:], z[:-lag]

    obs = {'Pearson': manual_pearson_corr(x_lagged, y_target), 'Info. Mutua': manual_mi(x_lagged, y_target, bins),
           'Transfer Entropy': manual_te_test(y, x, bins, lag),
           'Granger': manual_granger_test(y, x, lag)[0], 'PCMCI': manual_cmi(x_lagged, y_target, z_cond, bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx, sz = create_surrogate(x), create_surrogate(z)
        surr['Pearson'].append(manual_pearson_corr(sx[:-lag], y_target)); surr['Info. Mutua'].append(manual_mi(sx[:-lag], y_target, bins))
        surr['Transfer Entropy'].append(manual_te_test(y, sx, bins, lag))
        surr['Granger'].append(manual_granger_test(y, sx, lag)[0]); surr['PCMCI'].append(manual_cmi(sx[:-lag], y_target, sz[:-lag], bins))

    expected = {
        'Pearson': 'Fallo_Ambiguedad',
        'Info. Mutua': 'Fallo_Ambiguedad',
        'Transfer Entropy': 'Exito_SIG',
        'Granger': 'Exito_SIG',
        'PCMCI': 'Exito_SIG'
    }
    series = {'$X_{t-2}$': x, '$Y_t$': y}

    render_tfg_dashboard(series, obs, surr, expected, "Caso 3: Causalidad Lineal con Memoria", "Anexo_Caso3.png")
    render_tfg_highlight(series, obs, surr, expected, ['Pearson', 'Transfer Entropy', 'Granger'], "Caso 3: Causalidad lineal con memoria", "Destacado_Caso3.png")

def ejecutar_caso_4():
    print("Procesando Caso 4 (Cuadrática)...")
    np.random.seed(123)
    burn_in = 200
    n_samples, lag, bins, n_surr = 600, 2, 5, 500
    n_total = n_samples + burn_in

    x, y, z = np.zeros(n_total), np.zeros(n_total), np.random.randn(n_total)
    for t in range(lag, n_total):
        x[t] = 0.8 * x[t-1] + 0.5 * np.random.randn()
        y[t] = (x[t-lag])**2 + 0.5 * np.random.randn()
    x, y, z = x[burn_in:], y[burn_in:], z[burn_in:]

    x_lagged, y_target, z_cond = x[:-lag], y[lag:], z[:-lag]

    obs = {'Pearson': manual_pearson_corr(x_lagged, y_target), 'Info. Mutua': manual_mi(x_lagged, y_target, bins),
           'Transfer Entropy': manual_te_test(y, x, bins, lag),
           'Granger': manual_granger_test(y, x, lag)[0], 'PCMCI': manual_cmi(x_lagged, y_target, z_cond, bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx, sz = create_surrogate(x), create_surrogate(z)
        surr['Pearson'].append(manual_pearson_corr(sx[:-lag], y_target)); surr['Info. Mutua'].append(manual_mi(sx[:-lag], y_target, bins))
        surr['Transfer Entropy'].append(manual_te_test(y, sx, bins, lag))
        surr['Granger'].append(manual_granger_test(y, sx, lag)[0]); surr['PCMCI'].append(manual_cmi(sx[:-lag], y_target, sz[:-lag], bins))

    expected = {
        'Pearson': 'Fallo_Ceguera',
        'Info. Mutua': 'Exito_SIG',
        'Transfer Entropy': 'Exito_SIG',
        'Granger': 'Fallo_Ceguera',
        'PCMCI': 'Exito_SIG'
    }
    series = {'$X_{t-2}$': x, '$Y_t \propto X_{t-2}^2$': y}

    render_tfg_dashboard(series, obs, surr, expected, "Caso 4: Ceguera lineal ante causalidad cuadrática", "Anexo_Caso4.png")
    render_tfg_highlight(series, obs, surr, expected, ['Pearson', 'Granger', 'Transfer Entropy'], "Caso 4: Ceguera lineal ante causalidad cuadrática", "Destacado_Caso4.png")

def ejecutar_caso_5():
    print("Procesando Caso 5 (Cadena Transitiva)...")
    np.random.seed(42)
    burn_in = 200
    n_samples, lag, bins, n_surr = 500, 1, 4, 500
    n_total = n_samples + burn_in

    x, y, z = np.zeros(n_total), np.zeros(n_total), np.zeros(n_total)
    for t in range(lag, n_total):
        x[t] = 0.8 * x[t-1] + 0.5 * np.random.randn()
        y[t] = 0.5 * y[t-1] + 0.4 * x[t-lag] + 0.5 * np.random.randn()
        z[t] = 0.7 * z[t-1] + 0.3 * y[t-lag] + 0.5 * np.random.randn()
    x, y, z = x[burn_in:], y[burn_in:], z[burn_in:]

    x_source, y_mediator, z_target = x[:-lag], y[:-lag], z[lag:]

    obs = {'Pearson': manual_pearson_corr(x_source, z_target), 'Info. Mutua': manual_mi(x_source, z_target, bins),
           'Transfer Entropy': manual_te_test(z, x, bins, lag),
           'Granger': manual_granger_test(z, x, lag)[0], 'PCMCI': manual_cmi(x_source, z_target, y_mediator, bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx_full, sx_source = create_surrogate(x), create_surrogate(x_source)
        surr['Pearson'].append(manual_pearson_corr(sx_source, z_target)); surr['Info. Mutua'].append(manual_mi(sx_source, z_target, bins))
        surr['Transfer Entropy'].append(manual_te_test(z, sx_full, bins, lag))
        surr['Granger'].append(manual_granger_test(z, sx_full, max_lag=lag)[0]); surr['PCMCI'].append(manual_cmi(sx_source, z_target, y_mediator, bins))

    expected = {
        'Pearson': 'Falso_Positivo',
        'Info. Mutua': 'Falso_Positivo',
        # Con el bug de direccionalidad corregido, la TE bivariada mide ahora
        # correctamente TE_{X->Z} (antes medía por error TE_{Z->X}, una
        # direccion sin vinculo causal real, que por eso "resistia" la
        # trampa de la transitividad). Al medir la direccion correcta, la TE
        # cae en la misma trampa que Pearson/MI/Granger: es un falso
        # positivo bivariado, no una degradacion de la señal por el ruido
        # del mediador. Ver discusión en el TFG (Sección 3.5 / Tabla 1).
        'Transfer Entropy': 'Falso_Positivo',
        'Granger': 'Falso_Positivo',
        'PCMCI': 'Exito_NO_SIG'
    }
    series = {'$X_{t-2}$ (Raíz)': x, '$Y_{t-1}$ (Mediador)': y, '$Z_t$ (Efecto Final)': z}

    render_tfg_dashboard(series, obs, surr, expected, "Caso 5: Propagación transitiva en cadena (X -> Z)", "Anexo_Caso5.png")
    render_tfg_highlight(series, obs, surr, expected, ['Info. Mutua', 'Transfer Entropy', 'PCMCI'], "Caso 5: Propagación transitiva y transitividad bivariada", "Destacado_Caso5.png")

def ejecutar_caso_6():
    print("Procesando Caso 6 (Lorenz 63)...")
    np.random.seed(42)
    n_samples, lag, bins, n_surr = 400, 2, 6, 500
    dt, pasos = 0.01, 3000
    xs, ys, zs = np.zeros(pasos), np.zeros(pasos), np.zeros(pasos)
    xs[0], ys[0], zs[0] = 0.1, 0.1, 0.1
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    for t in range(pasos - 1):
        xs[t+1] = xs[t] + (sigma * (ys[t] - xs[t])) * dt
        ys[t+1] = ys[t] + (xs[t] * (rho - zs[t]) - ys[t]) * dt
        zs[t+1] = zs[t] + (xs[t] * ys[t] - beta * zs[t]) * dt

    skip = 5
    x, y, z = xs[1000:1000 + n_samples*skip:skip], ys[1000:1000 + n_samples*skip:skip], zs[1000:1000 + n_samples*skip:skip]
    x_lagged, z_target = x[:-lag], z[lag:]

    obs = {'Pearson': manual_pearson_corr(x, z), 'Info. Mutua': manual_mi(x, z, bins),
           'Transfer Entropy': manual_te_test(z, x, bins, lag),
           'Granger': manual_granger_test(z, x, lag)[0], 'PCMCI': manual_cmi(x_lagged, z_target, y[:-lag], bins)}
    surr = {k: [] for k in obs}
    for _ in range(n_surr):
        sx, sz = create_surrogate(x), create_surrogate(z)
        surr['Pearson'].append(manual_pearson_corr(sx, z)); surr['Info. Mutua'].append(manual_mi(sx, z, bins))
        surr['Transfer Entropy'].append(manual_te_test(z, sx, bins, lag))
        surr['Granger'].append(manual_granger_test(z, sx, lag)[0]); surr['PCMCI'].append(manual_cmi(sx[:-lag], z_target, y[:-lag], bins))

    expected = {
        'Pearson': 'Fallo_Ceguera',
        'Info. Mutua': 'Exito_SIG',
        'Transfer Entropy': 'Exito_SIG',
        'Granger': 'Fallo_Ceguera',
        'PCMCI': 'Exito_SIG'
    }
    series = {'$X_t$': x, '$Y_t$': y, '$Z_t$': z}

    render_tfg_dashboard(series, obs, surr, expected, "Lorenz 63: Ceguera Lineal en la relación X <-> Z", "Anexo_Caso6.png")
    render_tfg_highlight(series, obs, surr, expected, ['Pearson', 'Info. Mutua', 'Transfer Entropy'], "Lorenz 63: Ceguera Lineal en la relación X <-> Z", "Destacado_Caso6.png")

def ejecutar_caso_4_robustez_multisemilla(semillas=(1, 7, 21, 42, 99, 123, 2024)):
    """
    Repite el experimento del Caso 4 (Y = X_{t-2}^2 + ruido) con varias
    semillas distintas para comprobar si el resultado "Inesperado" de
    Pearson/Granger (SIG cuando en teoría deberían ser NO SIG) es una
    casualidad de la semilla 42 o un sesgo sistemático del pipeline.

    No cambia ningún resultado: solo informa. Si la fracción de semillas
    con p<0.05 para Pearson/Granger ronda el ~5%, es el comportamiento
    esperado de cualquier test de hipótesis (falsos positivos naturales al
    nivel de significancia elegido) y no hace falta tocar nada más. Si esa
    fracción es mucho mayor (p.ej. >20-30%), hay un sesgo sistemático que
    merece investigarse (candidatos: tamaño de muestra insuficiente para
    que se cancele el momento de tercer orden, o algún artefacto de
    alineación de índices).
    """
    burn_in = 200
    n_samples, lag, bins = 600, 2, 5
    n_total = n_samples + burn_in

    resultados = {'Pearson': [], 'Granger': [], 'Transfer Entropy': []}

    for s in semillas:
        np.random.seed(s)
        x, y = np.zeros(n_total), np.zeros(n_total)
        for t in range(lag, n_total):
            x[t] = 0.8 * x[t-1] + 0.5 * np.random.randn()
            y[t] = (x[t-lag])**2 + 0.5 * np.random.randn()
        x, y = x[burn_in:], y[burn_in:]
        x_lagged, y_target = x[:-lag], y[lag:]

        p_obs = manual_pearson_corr(x_lagged, y_target)
        g_obs = manual_granger_test(y, x, lag)[0]
        te_obs = manual_te_test(y, x, bins, lag)

        n_surr_local = 200  # reducido para que la comprobación sea rápida
        p_surr, g_surr, te_surr = [], [], []
        for _ in range(n_surr_local):
            sx = create_surrogate(x)
            p_surr.append(manual_pearson_corr(sx[:-lag], y_target))
            g_surr.append(manual_granger_test(y, sx, lag)[0])
            te_surr.append(manual_te_test(y, sx, bins, lag))

        p_pval = np.mean(np.abs(p_surr) >= np.abs(p_obs))
        g_pval = np.mean(np.array(g_surr) >= g_obs)
        te_pval = np.mean(np.array(te_surr) >= te_obs)

        resultados['Pearson'].append(p_pval)
        resultados['Granger'].append(g_pval)
        resultados['Transfer Entropy'].append(te_pval)

        print(f"  semilla={s:>5} | Pearson p={p_pval:.3f} | Granger p={g_pval:.3f} | TE p={te_pval:.3f}")

    print("\nResumen Caso 4 (fracción de semillas con p<0.05, se espera bajo/cero para Pearson y Granger):")
    for k, vals in resultados.items():
        frac_sig = np.mean(np.array(vals) < 0.05)
        print(f"  {k}: {frac_sig*100:.0f}% de {len(semillas)} semillas dieron SIG (p<0.05)")

    return resultados

# ------------------------------------------------------------------------------
# 4. EJECUCIÓN LORENZ-96 (Caso 7)
# ------------------------------------------------------------------------------

def ejecutar_lorenz96():
    print("\nProcesando Caso 7 (Lorenz 96)...")
    N = 40
    F = 8.0

    print("-> Integrando dinámica espacial de Lorenz-96...")
    x0 = np.full(N, F)
    x0[19] += 0.01  # Perturbación
    t_raw = np.arange(0.0, 50.0, 0.01)
    states_raw = odeint(lorenz96, x0, t_raw, args=(F,))

    states_clean = states_raw[500::5]
    t_clean = t_raw[500::5]

    print("-> Generando gráfica Espaciotemporal estática...")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    c1 = ax1.contourf(t_raw[0:1500], np.arange(N), states_raw[0:1500].T, levels=20, cmap='RdBu_r')
    ax1.set_title("Evolución Espaciotemporal y Propagación de Ondas (Lorenz-96)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Tiempo", fontsize=12)
    ax1.set_ylabel("Índice de Variable ($X_0$ a $X_{39}$)", fontsize=12)
    fig1.colorbar(c1, ax=ax1, label="Amplitud de la Variable")
    plt.tight_layout()
    fig1.savefig("Lorenz96_Espaciotemporal.png", dpi=300)
    plt.close(fig1)

    print("-> Test PCMCI Local y Validación de falsa correlación (X1 -> X4)...")
    x1 = states_clean[:, 1]
    x2 = states_clean[:, 2]
    x3 = states_clean[:, 3]
    x4 = states_clean[:, 4]

    N_SURR = 250
    def calc_pcmci(src, tgt, cond):
        obs = manual_cmi(src, tgt, cond, bins=5)
        surr = [manual_cmi(create_surrogate(src), tgt, cond, bins=5) for _ in range(N_SURR)]
        return {'obs': obs, 'surr': surr, 'p': np.mean(np.array(surr) >= obs)}

    res_der = calc_pcmci(x2[:-1], x3[1:], x4[:-1])
    res_izq = calc_pcmci(x4[:-1], x3[1:], x2[:-1])

    res_x1 = calc_pcmci(x1[:-1], x4[1:], x3[:-1])
    res_x1['corr'] = np.corrcoef(x1[:-1], x4[1:])[0, 1]

    render_validacion_local_lorenz96(
        t_clean, x2, x3, x4,
        res_der, res_izq, res_x1,
        "Lorenz96_Local.png"
    )

    print("-> Generando Matriz 40x40 Animada (GIF)... (tardará unos 30s)")
    WINDOW_SIZE = 150
    STEP = 15
    N_FRAMES = min((len(states_clean) - WINDOW_SIZE) // STEP, 60)

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

    fig2, (ax_evol, ax_mat) = plt.subplots(1, 2, figsize=(15, 6))
    evol_plot = ax_evol.contourf(np.arange(len(states_clean)), np.arange(N), states_clean.T, levels=20, cmap='RdBu_r')
    ax_evol.set_title("Dinámica Espaciotemporal", fontsize=14, fontweight='bold')
    ax_evol.set_ylabel("Variables", fontsize=12)
    ax_mat.set_title("Matriz Causal 40x40", fontsize=14, fontweight='bold')

    cmap_mat = plt.cm.magma
    norm_mat = mcolors.Normalize(vmin=0.0, vmax=np.max(matrices))
    mat_im = ax_mat.imshow(matrices[0], cmap=cmap_mat, norm=norm_mat, origin='lower')
    fig2.colorbar(mat_im, ax=ax_mat, shrink=0.8).set_label('CMI (Bits)')

    rect = patches.Rectangle((0, 0), WINDOW_SIZE, N-1, linewidth=3, edgecolor='lime', facecolor='none')
    ax_evol.add_patch(rect)

    def update(frame):
        start_idx = frame * STEP
        rect.set_xy((start_idx, 0))
        mat_im.set_array(matrices[frame])
        ax_mat.set_title(f"Matriz Causal\nVentana: {start_idx} a {start_idx + WINDOW_SIZE}", fontsize=12)
        return [rect, mat_im]

    ani = FuncAnimation(fig2, update, frames=N_FRAMES, interval=150)
    ani.save('Lorenz96_Matriz.gif', writer=PillowWriter(fps=6))
    plt.close(fig2)

# ------------------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("INICIANDO GENERACIÓN COMPLETA (TFG MASTER SCRIPT)...")
    ejecutar_caso_1()
    ejecutar_caso_2()
    ejecutar_caso_3()
    ejecutar_caso_4()
    print("\n-> Comprobación de robustez multi-semilla del Caso 4 (diagnóstico, no genera figuras)...")
    ejecutar_caso_4_robustez_multisemilla()
    ejecutar_caso_5()
    ejecutar_caso_6()
    ejecutar_lorenz96()
    print("\n¡PROCESO FINALIZADO CON ÉXITO!")