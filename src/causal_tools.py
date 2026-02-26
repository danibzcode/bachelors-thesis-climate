# ==============================================================================
# File: causal_tools.py
# Description: Core mathematical and statistical functions for causal inference.
# Author: Daniel Ibáñez (Bachelor's Thesis, Universidad de Zaragoza)
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import f

# ------------------------------------------------------------------------------
# 1. UTILITIES & STATISTICAL VALIDATION
# ------------------------------------------------------------------------------

def create_surrogate(ts):
    """
    Generates a linear surrogate of a time series using IAAFT.
    
    This function randomizes the phases of the Fourier transform while preserving 
    the power spectrum (amplitudes) and the distribution of the original signal.

    Parameters
    ----------
    ts : numpy.ndarray or list
        Original time series data (1D array).

    Returns
    -------
    numpy.ndarray
        Phase-randomized surrogate time series.
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
    Standardizes a time series (Z-score normalization).

    Parameters
    ----------
    ts : numpy.ndarray
        Time series data.

    Returns
    -------
    numpy.ndarray
        Standardized time series with mean 0 and standard deviation 1.
    """
    return (ts - np.mean(ts)) / np.std(ts)

# ------------------------------------------------------------------------------
# 2. ASSOCIATION MEASURES (UNDIRECTED)
# ------------------------------------------------------------------------------

def manual_pearson_corr(x, y):
    """
    Calculates the Pearson correlation coefficient manually.

    Parameters
    ----------
    x, y : numpy.ndarray
        Input time series arrays of the same length.

    Returns
    -------
    float
        Pearson correlation coefficient between -1.0 and 1.0.
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
    Calculates the bivariate Mutual Information I(X; Y) in bits.

    Parameters
    ----------
    x, y : numpy.ndarray
        Input time series arrays.
    bins : int
        Number of bins for the 2D histogram density estimation.

    Returns
    -------
    float
        Mutual Information in bits.
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
    Measures the Mutual Information between X(t-lag) and Y(t).

    Parameters
    ----------
    x : numpy.ndarray
        Source time series.
    y : numpy.ndarray
        Target time series.
    bins : int
        Number of bins for density estimation.
    lag : int
        Time delay to apply to the source variable X.

    Returns
    -------
    float
        Lagged Mutual Information value.
    """
    y_target = y[lag:]
    x_source_lagged = x[:-lag]
    return manual_mi(x_source_lagged, y_target, bins)

# ------------------------------------------------------------------------------
# 3. CAUSALITY MEASURES (DIRECTED & CONDITIONAL)
# ------------------------------------------------------------------------------

def manual_transfer_entropy_lagged(y_source, x_target, bins, k_lag):
    """
    Estimates the Transfer Entropy from Y to X with a specific lag.

    Parameters
    ----------
    y_source : numpy.ndarray
        Source time series.
    x_target : numpy.ndarray
        Target time series.
    bins : int
        Number of bins for the 3D histogram density estimation.
    k_lag : int
        Time lag for the historical dependencies.

    Returns
    -------
    float
        Transfer Entropy value.
    """
    x_t_target = x_target[k_lag:]
    x_t_minus_k = x_target[:-k_lag]
    y_t_minus_k = y_source[:-k_lag]
    N = len(x_t_target)
    
    all_data = np.concatenate([x_t_target, x_t_minus_k, y_t_minus_k])
    min_val, max_val = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    x_t_d = np.digitize(x_t_target, bin_edges[1:-1])
    x_tk_d = np.digitize(x_t_minus_k, bin_edges[1:-1])
    y_tk_d = np.digitize(y_t_minus_k, bin_edges[1:-1])
    
    p_ijk_counts = np.histogramdd([x_t_d, x_tk_d, y_tk_d], bins=bins, range=[(0, bins), (0, bins), (0, bins)])[0]
    p_ijk = p_ijk_counts / N
    p_jk_counts = np.histogram2d(x_tk_d, y_tk_d, bins=bins, range=[(0, bins), (0, bins)])[0]
    p_jk = p_jk_counts / N
    p_ij_counts = np.histogram2d(x_t_d, x_tk_d, bins=bins, range=[(0, bins), (0, bins)])[0]
    p_ij = p_ij_counts / N
    p_j_counts = np.histogram(x_tk_d, bins=bins, range=(0, bins))[0]
    p_j = p_j_counts / N
    
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if p_ijk[i, j, k] > 1e-12 and p_j[j] > 1e-12 and p_jk[j, k] > 1e-12 and p_ij[i, j] > 1e-12:
                    log_term = np.log((p_ijk[i, j, k] * p_j[j]) / (p_jk[j, k] * p_ij[i, j]))
                    te += p_ijk[i, j, k] * log_term
    return te

def manual_cmi(x, y, z, bins):
    """
    Calculates the Conditional Mutual Information I(X; Y | Z) in bits.

    Parameters
    ----------
    x, y : numpy.ndarray
        Target variables to test for independence.
    z : numpy.ndarray
        Conditioning variable (the confounding factor or intermediate).
    bins : int
        Number of bins for density estimation.

    Returns
    -------
    float
        Conditional Mutual Information in bits.
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
                            
    return cmi / np.log(2)

def create_lagged_data(data, max_lag):
    """Helper function to create lagged columns for Granger Causality."""
    if not isinstance(data, pd.DataFrame): data = pd.DataFrame(data)
    lags = []
    for i in range(1, max_lag + 1):
        lagged_series = data.shift(i)
        lagged_series.columns = [f'{col}_lag_{i}' for col in data.columns]
        lags.append(lagged_series)
    return pd.concat(lags, axis=1)

def manual_granger_test(y, x, max_lag):
    """
    Manual implementation of Granger Causality Test (X -> Y) using OLS.

    Parameters
    ----------
    y, x : numpy.ndarray or pandas.Series
        Target (Y) and Source (X) time series.
    max_lag : int
        Maximum number of lags to include in the models.

    Returns
    -------
    F_stat : float
        F-statistic of the Granger causality test.
    p_value : float
        P-value associated with the F-statistic.
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

    # Restricted Model
    design_restricted = np.hstack([intercept, y_lags_np])
    _, res_restricted_sum, _, _ = np.linalg.lstsq(design_restricted, y_target, rcond=None)
    ssr_restricted = np.sum(res_restricted_sum)
    df_restricted = n_obs - (max_lag + 1)

    # Unrestricted Model
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
    """Extracts the minimum p-value from statsmodels grangercausalitytests."""
    p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
    return np.min(p_values)

# ------------------------------------------------------------------------------
# 4. ADVANCED CAUSALITY (TE & CMI) - FIXES "NOT DEFINED" ERROR
# ------------------------------------------------------------------------------

def manual_transfer_entropy_lagged(y_source, x_target, bins, k_lag):
    """
    Estimates the Transfer Entropy from Y to X with a specific lag.
    Essential for Case 4 (Non-linear causality). [cite: 189, 190, 191]
    """
    # Aligning data for T, T-k
    x_t_target = x_target[k_lag:]
    x_t_minus_k = x_target[:-k_lag]
    y_t_minus_k = y_source[:-k_lag]
    N = len(x_t_target)
    
    # Normalizing range for binning
    all_data = np.concatenate([x_t_target, x_t_minus_k, y_t_minus_k])
    min_val, max_val = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Digitizing (Binning)
    x_t_d = np.digitize(x_t_target, bin_edges[1:-1])
    x_tk_d = np.digitize(x_t_minus_k, bin_edges[1:-1])
    y_tk_d = np.digitize(y_t_minus_k, bin_edges[1:-1])
    
    # Probabilities calculation
    p_ijk_counts = np.histogramdd([x_t_d, x_tk_d, y_tk_d], bins=bins, range=[(0, bins), (0, bins), (0, bins)])[0]
    p_ijk = p_ijk_counts / N
    p_jk_counts = np.histogram2d(x_tk_d, y_tk_d, bins=bins, range=[(0, bins), (0, bins)])[0]
    p_jk = p_jk_counts / N
    p_ij_counts = np.histogram2d(x_t_d, x_tk_d, bins=bins, range=[(0, bins), (0, bins)])[0]
    p_ij = p_ij_counts / N
    p_j_counts = np.histogram(x_tk_d, bins=bins, range=(0, bins))[0]
    p_j = p_j_counts / N
    
    te = 0.0
    epsilon = 1e-12
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                if p_ijk[i, j, k] > epsilon and p_j[j] > epsilon and p_jk[j, k] > epsilon and p_ij[i, j] > epsilon:
                    log_term = np.log((p_ijk[i, j, k] * p_j[j]) / (p_jk[j, k] * p_ij[i, j]))
                    te += p_ijk[i, j, k] * log_term
    return te / np.log(2) # Return in bits [cite: 151]

def manual_cmi(x, y, z, bins):
    """
    Calculates the Conditional Mutual Information I(X; Y | Z) in bits.
    Core logic for PCMCI and Case 5 (Causal Chains). [cite: 201, 209, 212]
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
                            
    return cmi / np.log(2) # Return in bits [cite: 151]