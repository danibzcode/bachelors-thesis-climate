# ==============================================================================
# File: causal_tools.py
# Description: Core mathematical and statistical functions for causal inference.
# Author: Daniel Ibáñez (Bachelor's Thesis, Universidad de Zaragoza)
#
# CHANGELOG:
#   - [Previous fix] manual_transfer_entropy_lagged: switched from a single
#     shared bin range (concatenation of X_t, X_(t-k), Y_(t-k)) to
#     np.histogramdd, which bins each dimension independently -- consistent
#     with manual_mi / manual_cmi. Needed because variables on very
#     different scales (e.g. Case 4, Y = X^2) were losing resolution under
#     a shared range.
#
#   - [NEW] Added manual_te_test(y_target, x_source, bins, k_lag): a thin
#     wrapper around manual_transfer_entropy_lagged that uses the SAME
#     (target, source) argument order as manual_granger_test(y, x, max_lag).
#
#     Root cause this fixes: manual_transfer_entropy_lagged has the
#     parameter order (y_source, x_target, ...) -- i.e. FIRST argument is
#     the cause/source, SECOND is the effect/target. This is the OPPOSITE
#     convention to manual_granger_test(y, x, max_lag), where the FIRST
#     argument is the effect (target) and the SECOND is the candidate
#     cause (source).
#
#     Because both functions were called with the same (y, x) argument
#     pattern throughout the master script (to mirror the Granger calls),
#     every TE call was silently testing the REVERSE causal direction:
#       - Case 3 (X -> Y):      called TE(y, x) -> actually computed TE_{Y->X}
#       - Case 4 (X -> Y, X^2): called TE(y, x) -> actually computed TE_{Y->X}
#       - Case 5 (X -> Y -> Z): called TE(z, x) -> actually computed TE_{Z->X}
#     In each case the "wrong" direction carries little to no real
#     information (X does not depend on Y; Z does not directly cause X),
#     which explains the unexpected non-significant / false-negative TE
#     results reported in those figures.
#
#     manual_transfer_entropy_lagged() itself is untouched -- it is
#     internally self-consistent and correctly documented. The fix lives
#     in the calling convention: use manual_te_test(y, x, bins, lag) from
#     now on (same argument order as manual_granger_test) instead of
#     calling manual_transfer_entropy_lagged(y, x, bins, lag) directly.
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import f

# ------------------------------------------------------------------------------
# 1. UTILITIES & STATISTICAL VALIDATION
# ------------------------------------------------------------------------------

def create_surrogate(ts):
    """
    Generates a phase-randomized (FT) surrogate of a time series.

    NOTE ON METHODOLOGY: this implements a single-pass Fourier-Transform
    surrogate (random phases, original amplitudes preserved), NOT the full
    iterative IAAFT algorithm (amplitude-adjusted + rank-matched to the
    exact marginal distribution of the original series) described in the
    thesis text. For signals that are already close to Gaussian (e.g. the
    AR(1)-driven synthetic cases in this work), the two are very similar
    in practice, since FT surrogates are the exact null model for linear
    Gaussian processes. However, if this function is ever applied to a
    series with a strongly non-Gaussian marginal distribution (e.g. highly
    skewed real climate data), the missing iterative rank-matching step
    could matter and the docstring/thesis text should either be updated to
    describe this simpler method, or the iterative IAAFT loop (rank
    matching in the time domain + amplitude restoration in the frequency
    domain, iterated to convergence) should be implemented to match what
    is written in Section 3 of the report.

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
    """
    return (ts - np.mean(ts)) / np.std(ts)

# ------------------------------------------------------------------------------
# 2. ASSOCIATION MEASURES (UNDIRECTED)
# ------------------------------------------------------------------------------

def manual_pearson_corr(x, y):
    """
    Calculates the Pearson correlation coefficient manually.
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
    """
    y_target = y[lag:]
    x_source_lagged = x[:-lag]
    return manual_mi(x_source_lagged, y_target, bins)

# ------------------------------------------------------------------------------
# 3. CAUSALITY MEASURES (DIRECTED & CONDITIONAL)
# ------------------------------------------------------------------------------

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

    Convention: first argument (y) is the TARGET/effect, second (x) is the
    candidate SOURCE/cause. This matches manual_te_test() below.
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
# 4. ADVANCED CAUSALITY (TE & CMI)
# ------------------------------------------------------------------------------

def manual_transfer_entropy_lagged(y_source, x_target, bins, k_lag):
    """
    Estimates the Transfer Entropy from Y (source) to X (target) with a
    specific lag k_lag.

    T E_{Y->X} = sum p(x_t, x_{t-k}, y_{t-k}) *
                 log2[ p(x_t, x_{t-k}, y_{t-k}) * p(x_{t-k}) /
                       ( p(x_{t-k}, y_{t-k}) * p(x_t, x_{t-k}) ) ]

    IMPORTANT: this function's parameter order is (source, target) --
    OPPOSITE of manual_granger_test(target, source, max_lag). Prefer
    calling manual_te_test(y_target, x_source, bins, k_lag) instead,
    which uses the same (target, source) order as Granger and delegates
    here internally. This avoids the source/target mix-up that previously
    caused several test cases to silently measure the reverse causal
    direction.
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
    Transfer Entropy TE_{X -> Y}: does the past of X improve the prediction
    of Y's future, beyond what Y's own past already provides?

    Argument order is (target, source, bins, lag) -- IDENTICAL to
    manual_granger_test(y, x, max_lag). Use this (not
    manual_transfer_entropy_lagged directly) everywhere you would write
    manual_granger_test(y, x, lag), so both directional tests are always
    evaluated on the same causal hypothesis "does X -> Y".

    Internally this just calls
    manual_transfer_entropy_lagged(x_source, y_target, bins, k_lag),
    i.e. swaps the arguments into that function's (source, target) order.
    """
    return manual_transfer_entropy_lagged(x_source, y_target, bins, k_lag)

def manual_cmi(x, y, z, bins):
    """
    Calculates the Conditional Mutual Information I(X; Y | Z) in bits.
    Core logic for PCMCI and Case 5 (Causal Chains).
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

    return cmi / np.log(2)  # Return in bits
