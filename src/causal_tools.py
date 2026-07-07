# ==============================================================================
# Archivo: causal_tools.py
# Descripción: Funciones matemáticas y estadísticas centrales para la inferencia causal.
# Autor: Daniel Ibáñez Caba (Trabajo de Fin de Grado, Universidad de Zaragoza)
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import f

# ------------------------------------------------------------------------------
# 1. UTILIDADES Y VALIDACIÓN ESTADÍSTICA
# ------------------------------------------------------------------------------

def create_surrogate(ts):
    """
    Genera una serie temporal sustituta (surrogate) mediante la aleatorización
    de fases en el dominio de la frecuencia (Transformada de Fourier).
    Conserva las amplitudes originales y el espectro de potencia de la señal.
    """
    f_coeffs = np.fft.rfft(ts)
    amplitudes = np.abs(f_coeffs)
    random_phases = np.random.uniform(0, 2 * np.pi, len(f_coeffs))
    new_f_coeffs = amplitudes * np.exp(1j * random_phases)
    new_f_coeffs[0] = f_coeffs[0]
    if len(ts) % 2 == 0:
        new_f_coeffs[-1] = f_coeffs[-1]
    return np.fft.irfft(new_f_coeffs, n=len(ts))

def standardize(ts):
    """
    Estandariza una serie temporal (normalización Z-score).
    """
    return (ts - np.mean(ts)) / np.std(ts)

# ------------------------------------------------------------------------------
# 2. MEDIDAS DE ASOCIACIÓN (NO DIRIGIDAS)
# ------------------------------------------------------------------------------

def manual_pearson_corr(x, y):
    """
    Calcula el coeficiente de correlación de Pearson manualmente.
    """
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    covariance_numerator = np.sum(x_centered * y_centered)
    x_std_dev_part = np.sqrt(np.sum(x_centered**2))
    y_std_dev_part = np.sqrt(np.sum(y_centered**2))
    denominator = x_std_dev_part * y_std_dev_part

    if denominator == 0:
        return 0.0
    return covariance_numerator / denominator

def manual_mi(x, y, bins):
    """
    Calcula la Información Mutua bivariada I(X; Y) en bits.
    """
    joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    N = len(x)
    if N == 0: return 0.0

    p_xy = joint_hist / N
    p_x = np.histogram(x, bins=x_edges)[0] / N
    p_y = np.histogram(y, bins=y_edges)[0] / N

    mi = 0.0
    epsilon = 1e-12

    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > epsilon and p_x[i] > epsilon and p_y[j] > epsilon:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi / np.log(2)

def manual_lagged_mutual_information(x, y, bins, lag):
    """
    Mide la Información Mutua entre X(t-lag) y Y(t).
    """
    y_target = y[lag:]
    x_source_lagged = x[:-lag]
    return manual_mi(x_source_lagged, y_target, bins)

# ------------------------------------------------------------------------------
# 3. MEDIDAS DE CAUSALIDAD (DIRIGIDAS Y CONDICIONALES)
# ------------------------------------------------------------------------------

def create_lagged_data(data, max_lag):
    """
    Función auxiliar para crear columnas con retardos temporales 
    para la Causalidad de Granger.
    """
    if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data)
    lags = []
    for i in range(1, max_lag + 1):
        lagged_series = data.shift(i)
        lagged_series.columns = [f'{col}_lag_{i}' for col in data.columns]
        lags.append(lagged_series)
    return pd.concat(lags, axis=1)

def manual_granger_test(y, x, max_lag):
    """
    Implementación manual del Test de Causalidad de Granger (X -> Y) mediante
    Mínimos Cuadrados Ordinarios (OLS).

    Convención de argumentos: el primero (y) es la variable OBJETIVO (efecto), 
    el segundo (x) es la variable ORIGEN (causa).
    """
    y_df = pd.DataFrame(y, columns=['Y'])
    x_df = pd.DataFrame(x, columns=['X'])

    y_lagged_df = create_lagged_data(y_df, max_lag)
    x_lagged_df = create_lagged_data(x_df, max_lag)

    data = pd.concat([y_df.rename(columns={'Y': 'Y_target'}), y_lagged_df, x_lagged_df], axis=1)
    data_aligned = data.iloc[max_lag:].copy()

    y_target = data_aligned['Y_target'].values
    y_lags_np = data_aligned.filter(like='Y_lag_').values
    x_lags_np = data_aligned.filter(like='X_lag_').values

    n_obs = len(y_target)
    intercept = np.ones((n_obs, 1))

    # Modelo Restringido (solo con retardos de Y)
    design_restricted = np.hstack([intercept, y_lags_np])
    _, res_restricted_sum, _, _ = np.linalg.lstsq(design_restricted, y_target, rcond=None)
    ssr_restricted = np.sum(res_restricted_sum)
    df_restricted = n_obs - (max_lag + 1)

    # Modelo No Restringido (retardos de Y y retardos de X)
    design_unrestricted = np.hstack([intercept, y_lags_np, x_lags_np])
    _, res_unrestricted_sum, _, _ = np.linalg.lstsq(design_unrestricted, y_target, rcond=None)
    ssr_unrestricted = np.sum(res_unrestricted_sum)
    df_unrestricted = n_obs - (max_lag + max_lag + 1)

    k_restrictions = max_lag
    F_numerator = (ssr_restricted - ssr_unrestricted) / k_restrictions
    F_denominator = ssr_unrestricted / df_unrestricted

    if F_denominator <= 1e-12: return 0.0, 1.0
    F_stat = F_numerator / F_denominator
    p_value = f.sf(F_stat, k_restrictions, df_unrestricted)

    return F_stat, p_value

def get_best_granger_pvalue(results, max_lag):
    """
    Extrae el p-valor mínimo de una estructura de resultados 
    de grangercausalitytests (statsmodels).
    """
    p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
    return np.min(p_values)

# ------------------------------------------------------------------------------
# 4. CAUSALIDAD AVANZADA (TE Y CMI)
# ------------------------------------------------------------------------------

def manual_transfer_entropy_lagged(y_source, x_target, bins, k_lag):
    """
    Estima la Entropía de Transferencia de Y (origen) a X (objetivo) 
    con un retardo temporal específico k_lag.

    Fórmula:
    T E_{Y->X} = sum p(x_t, x_{t-k}, y_{t-k}) *
                 log2[ (p(x_t, x_{t-k}, y_{t-k}) * p(x_{t-k})) /
                       (p(x_{t-k}, y_{t-k}) * p(x_t, x_{t-k})) ]
    """
    x_t_target = x_target[k_lag:]
    x_t_minus_k = x_target[:-k_lag]
    y_t_minus_k = y_source[:-k_lag]
    N = len(x_t_target)
    if N == 0:
        return 0.0

    joint_data = np.stack([x_t_target, x_t_minus_k, y_t_minus_k], axis=1)
    p_ijk_counts, edges = np.histogramdd(joint_data, bins=bins)
    p_ijk = p_ijk_counts / N

    x_t_edges, x_tk_edges, y_tk_edges = edges

    p_jk_counts, _, _ = np.histogram2d(x_t_minus_k, y_t_minus_k,
                                        bins=[x_tk_edges, y_tk_edges])
    p_jk = p_jk_counts / N

    p_ij_counts, _, _ = np.histogram2d(x_t_target, x_t_minus_k,
                                        bins=[x_t_edges, x_tk_edges])
    p_ij = p_ij_counts / N

    p_j_counts, _ = np.histogram(x_t_minus_k, bins=x_tk_edges)
    p_j = p_j_counts / N

    te = 0.0
    epsilon = 1e-12
    for i in range(bins):
        for j in range(bins):
            if p_j[j] <= epsilon:
                continue
            for k in range(bins):
                p_ijk_val = p_ijk[i, j, k]
                if p_ijk_val <= epsilon:
                    continue
                p_jk_val = p_jk[j, k]
                p_ij_val = p_ij[i, j]
                if p_jk_val > epsilon and p_ij_val > epsilon:
                    log_term = np.log((p_ijk_val * p_j[j]) / (p_jk_val * p_ij_val))
                    te += p_ijk_val * log_term

    return te / np.log(2)

def manual_te_test(y_target, x_source, bins, k_lag):
    """
    Entropía de Transferencia TE_{X -> Y}: Evalúa si el conocimiento del pasado 
    de X mejora la predicción del futuro de Y, superando la información que 
    ya proporciona el propio pasado de Y.

    El orden de los argumentos es (objetivo, origen, bins, lag) para 
    mantener la misma convención y compatibilidad que el test de Granger.
    """
    return manual_transfer_entropy_lagged(x_source, y_target, bins, k_lag)

def manual_cmi(x, y, z, bins):
    """
    Calcula la Información Mutua Condicional I(X; Y | Z) en bits.
    Es la métrica central para el algoritmo PCMCI y la resolución de cadenas causales.
    """
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

    return cmi / np.log(2)  # Devuelve el valor en bits