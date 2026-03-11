# ==============================================================================
# Case 6: Lorenz Attractor (Non-linear Deterministic System)
# Description: Contrasting Granger Causality vs Transfer Entropy.
#              Validating non-linear causality with IAAFT surrogates.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from statsmodels.tsa.stattools import grangercausalitytests

# Import custom functions from our modular library
from causal_tools import (
    create_surrogate, 
    get_best_granger_pvalue, 
    manual_transfer_entropy_lagged,
    standardize
)

# ---------------------------------------------------------
# 1. DATA GENERATION: Lorenz Attractor
# ---------------------------------------------------------
def lorenz_deriv(state, t):
    x, y, z = state
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

print("Case 6: Lorenz Attractor (Chaotic, Non-linear, Deterministic)")
print("-" * 75)

# Generate data
np.random.seed(42)
t_raw = np.arange(0.0, 50.0, 0.01) # 5000 points
state0 = [1.0, 1.0, 1.0]
states_raw = odeint(lorenz_deriv, state0, t_raw)

# CRITICAL STEPS FOR CHAOTIC SYSTEMS:
# a) Remove transient (first 1000 steps)
# b) Downsample (take 1 every 10 steps) to reduce local autocorrelation
states_clean = states_raw[1000::10]
t_clean = t_raw[1000::10]

x = states_clean[:, 0]
y = states_clean[:, 1]
# z = states_clean[:, 2] # Z is not used in this specific bivariate test

# ---------------------------------------------------------
# 2. ANALYSIS PARAMETERS
# ---------------------------------------------------------
N_BINS = 5           
TEST_LAG = 1         
n_surrogates = 100  # 100 is enough for a fast demo (use 500 for the final TFG document)

# ---------------------------------------------------------
# 3. STATISTICAL CALCULATIONS
# ---------------------------------------------------------
print("Calculating observed metrics (Direction: X -> Y)...")

# A. Granger Causality (Linear - Expected to fail/be inconsistent)
data_g = pd.DataFrame({'Target_Y': y, 'Source_X': x})
res_xy = grangercausalitytests(data_g[['Target_Y', 'Source_X']], maxlag=TEST_LAG, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, TEST_LAG)

# B. Transfer Entropy (Non-linear - Expected to succeed)
obs_te = manual_transfer_entropy_lagged(x, y, bins=N_BINS, k_lag=TEST_LAG)

# C. Generate Surrogate Distributions (IAAFT)
print(f"Generating {n_surrogates} IAAFT surrogates for Transfer Entropy validation...")
surr_te = []

for _ in range(n_surrogates):
    # We only need to surrogate the source (X) to break the coupling
    sx = create_surrogate(x)
    surr_te.append(manual_transfer_entropy_lagged(sx, y, bins=N_BINS, k_lag=TEST_LAG))

# D. Calculate Empirical p-value for TE
p_te = np.mean(np.array(surr_te) >= obs_te)

# ---------------------------------------------------------
# 4. RESULTS DISPLAY
# ---------------------------------------------------------
print("\n" + "="*75)
print("RESULTS: LORENZ SYSTEM CAUSALITY (X -> Y)")
print("="*75)

def show_res(label, p, expected_sig):
    sig_label = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
    status = "CORRECT" if (p < 0.05) == expected_sig else "FAIL (Linearity limitation)"
    print(f"{label:<25} | p-value: {p:7.4f} | {sig_label:<15} | {status}")

# For Lorenz, Granger often fails to reject null cleanly. TE detects the coupling perfectly.
show_res("Granger (X -> Y)", p_granger, False) 
show_res("Transfer Ent (X -> Y)", p_te, True)  
print("="*75)

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------
fig = plt.figure(figsize=(14, 5))

# Plot 1: Time Series (Standardized)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t_clean[:150], standardize(x)[:150], label='Signal X', color='#1f77b4', linewidth=1.5)
ax1.plot(t_clean[:150], standardize(y)[:150], label='Signal Y', color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax1.set_title('Lorenz Time Series (Subset after downsampling)', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Standardized Value')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2: TE Surrogate Distribution vs Observed
ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(surr_te, bins=20, color='silver', edgecolor='black', linewidth=0.8, 
         alpha=0.7, label='Null Dist (IAAFT Surrogates)')
ax2.axvline(obs_te, color='red', linestyle='--', linewidth=2.5, 
            label=f'Observed TE ({obs_te:.3f} bits)')
ax2.set_title(f'Transfer Entropy Significance\np-value = {p_te:.4f}', fontsize=12)
ax2.set_xlabel('Transfer Entropy (bits)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
