# ==============================================================================
# Case 1: Independent White Noise (Baseline)
# Description: Testing all algorithms against independent stochastic processes.
#              This validates the IAAFT surrogate framework for the TFG.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Import custom functions from our modular library
from causal_tools import (
    create_surrogate, 
    manual_pearson_corr, 
    manual_mi, 
    manual_granger_test,
    manual_transfer_entropy_lagged,
    standardize
)

# ---------------------------------------------------------
# 1. DATA GENERATION: Independent Noise
# ---------------------------------------------------------
# Equation: X(t) = sigma_x * epsilon_x
#           Y(t) = sigma_y * epsilon_y
n_samples = 250
np.random.seed(123)

x = np.random.randn(n_samples) 
y = np.random.randn(n_samples)

print("Case 1: Independent White Noise (Baseline)")
print(f"Mathematical Model: X_t = noise, Y_t = noise (No coupling)")
print("-" * 65)

# ---------------------------------------------------------
# 2. ANALYSIS PARAMETERS
# ---------------------------------------------------------
N_BINS = 5           
TEST_LAG = 1         
n_surrogates = 500   

# ---------------------------------------------------------
# 3. STATISTICAL CALCULATIONS
# ---------------------------------------------------------
# A. Calculate Observed Metrics
obs_pearson = manual_pearson_corr(x, y)
obs_mi = manual_mi(x, y, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(x, y, bins=N_BINS, k_lag=TEST_LAG)
_, p_granger_parametric = manual_granger_test(y, x, max_lag=TEST_LAG)

# B. Generate Surrogate Distributions (IAAFT)
print(f"Generating {n_surrogates} surrogates...")
surr_pearson, surr_mi, surr_te = [], [], []

for _ in range(n_surrogates):
    sx = create_surrogate(x)
    sy = create_surrogate(y)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=TEST_LAG))

# C. Calculate Empirical p-values
p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 4. RESULTS DISPLAY
# ---------------------------------------------------------
print("\n" + "="*70)
print("BENCHMARK RESULTS: NOISE ANALYSIS")
print("="*70)
def show_res(label, p):
    res = "CORRECT (Not Significant)" if p >= 0.05 else "FAIL (False Positive)"
    print(f"{label:<25} | p-value: {p:7.4f} | {res}")

show_res("Pearson Correlation", p_pearson)
show_res("Mutual Information", p_mi)
show_res("Granger Causality", p_granger_parametric)
show_res("Transfer Entropy", p_te)
print("="*70)

# ---------------------------------------------------------
# 5. VISUALIZATION 1: TIME SERIES
# ---------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(standardize(x), label='Signal X (Noise)', color='#1f77b4', linewidth=1.2)
plt.plot(standardize(y), label='Signal Y (Noise)', color='#ff7f0e', linewidth=1.2)
plt.title('Case 1: Visual Inspection of Independent Signals', fontsize=14)
plt.xlabel('Time Step')
plt.ylabel('Standardized Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# ---------------------------------------------------------
# 6. VISUALIZATION 2: SIGNIFICANCE (With Frames/Edgecolor)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Statistical Significance Distributions (Case 1)', fontsize=16)

def plot_signif(ax, surr_data, obs_val, p_val, title):
    # Added 'edgecolor' to highlight individual bars
    ax.hist(surr_data, bins=30, color='silver', edgecolor='black', linewidth=0.5, 
            alpha=0.6, label='Null Dist (Surrogates)', density=True)
    ax.axvline(obs_val, color='red', linestyle='--', linewidth=2, label=f'Observed ({obs_val:.3f})')
    ax.set_title(f'{title}\np-value = {p_val:.4f}')
    ax.legend(fontsize='small')
    ax.grid(axis='y', alpha=0.3)

plot_signif(axes[0, 0], surr_pearson, obs_pearson, p_pearson, 'Pearson Correlation')
plot_signif(axes[0, 1], surr_mi, obs_mi, p_mi, 'Mutual Information')
plot_signif(axes[1, 0], np.random.f(1, n_samples-3, 1000), 1.0, p_granger_parametric, 'Granger F-Test (Theoretical)')
plot_signif(axes[1, 1], surr_te, obs_te, p_te, 'Transfer Entropy')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()