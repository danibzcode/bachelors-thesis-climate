# ==============================================================================
# Case 2: Common Cause (Spurious Correlation)
# Description: X and Y are both driven by a third variable U. 
#              Testing if algorithms can distinguish between U->X, U->Y and X-Y.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import from our custom library
from causal_tools import (
    create_surrogate, 
    manual_pearson_corr, 
    manual_mi, 
    manual_granger_test,
    manual_transfer_entropy_lagged,
    standardize
)

# ---------------------------------------------------------
# 1. DATA GENERATION: Common Cause Model
# ---------------------------------------------------------
n_samples = 300
np.random.seed(10)

# U: The hidden confounder (Common Cause)
u = 2.0 * np.random.randn(n_samples)

# X and Y depend on U, but NOT on each other
x = u + 1.0 * np.random.randn(n_samples)
y = u + 1.0 * np.random.randn(n_samples)

print("Case 2: Common Cause Scenario")
print(f"Model: X_t = U_t + ex, Y_t = U_t + ey")
print("-" * 60)

# ---------------------------------------------------------
# 2. ANALYSIS PARAMETERS
# ---------------------------------------------------------
N_BINS = 5           
TEST_LAG = 1         
n_surrogates = 500   

# ---------------------------------------------------------
# 3. STATISTICAL CALCULATIONS
# ---------------------------------------------------------
obs_pearson = manual_pearson_corr(x, y)
obs_mi = manual_mi(x, y, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(x, y, bins=N_BINS, k_lag=TEST_LAG)
_, p_granger_parametric = manual_granger_test(y, x, max_lag=TEST_LAG)

print(f"Calculating {n_surrogates} surrogates...")
surr_pearson, surr_mi, surr_te = [], [], []

for _ in range(n_surrogates):
    sx = create_surrogate(x); sy = create_surrogate(y)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=TEST_LAG))

p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 4. RESULTS DISPLAY
# ---------------------------------------------------------
print("\n" + "="*70)
print("RESULTS: THE COMMON CAUSE TRAP")
print("Expected: Pearson/MI fail (Significant), Granger/TE succeed (Not Sig)")
print("="*70)

def display_res(name, p_val):
    # Success here means NOT falling for the spurious correlation
    # For Pearson/MI, p < 0.05 is a FAIL (False Positive)
    # For Granger/TE, p >= 0.05 is CORRECT (True Negative)
    status = "FAIL (False Positive)" if p_val < 0.05 else "CORRECT (True Negative)"
    print(f"{name:<25} | p-value: {p_val:7.4f} | {status}")

display_res("Pearson Correlation", p_pearson)
display_res("Mutual Information", p_mi)
display_res("Granger Causality", p_granger_parametric)
display_res("Transfer Entropy", p_te)
print("="*70)

# ---------------------------------------------------------
# 5. VISUALIZATION: TIME SERIES & SIGNIFICANCE
# ---------------------------------------------------------
# Plot 1: Signals
plt.figure(figsize=(12, 4))
plt.plot(standardize(x), label='Effect X (Ice Cream)', color='red', alpha=0.7)
plt.plot(standardize(y), label='Effect Y (A/C Usage)', color='blue', alpha=0.7)
plt.plot(standardize(u), label='Common Cause U (Temp)', color='black', linestyle='--', alpha=0.4)
plt.title('Case 2: Spurious Synchrony due to Common Cause', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Plot 2: Distributions (with frames)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Significance Analysis: Common Cause Confounding', fontsize=16)

def plot_signif(ax, surr_data, obs_val, p_val, title):
    ax.hist(surr_data, bins=30, color='silver', edgecolor='black', linewidth=0.5, alpha=0.6, density=True)
    ax.axvline(obs_val, color='red', linestyle='--', linewidth=2, label=f'Observed ({obs_val:.3f})')
    ax.set_title(f'{title}\np-value = {p_val:.4f}')
    ax.legend(fontsize='small')
    ax.grid(axis='y', alpha=0.3)

plot_signif(axes[0, 0], surr_pearson, obs_pearson, p_pearson, 'Pearson Correlation')
plot_signif(axes[0, 1], surr_mi, obs_mi, p_mi, 'Mutual Information')
plot_signif(axes[1, 0], np.random.f(1, n_samples-3, 1000), 1.0, p_granger_parametric, 'Granger F-Test')
plot_signif(axes[1, 1], surr_te, obs_te, p_te, 'Transfer Entropy')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
