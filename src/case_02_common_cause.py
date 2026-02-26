import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from causal_tools import (create_surrogate, manual_pearson_corr, manual_mi, 
                          get_best_granger_pvalue, manual_transfer_entropy_lagged, standardize)

# 1. DATA GENERATION: Temperature driving Ice Cream and A/C
n_samples = 300
np.random.seed(10)
temp = 2.0 * np.random.randn(n_samples) # Common Cause (U)
ice_cream = temp + 1.0 * np.random.randn(n_samples) # Effect X
ac_usage = temp + 1.0 * np.random.randn(n_samples)  # Effect Y

# 2. ANALYSIS
N_BINS, TEST_LAG, n_surrogates = 5, 1, 500
obs_pearson = manual_pearson_corr(ice_cream, ac_usage)
obs_mi = manual_mi(ice_cream, ac_usage, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(ice_cream, ac_usage, bins=N_BINS, k_lag=TEST_LAG)
data_g = pd.DataFrame({'AC': ac_usage, 'IceCream': ice_cream})
res_xy = grangercausalitytests(data_g[['AC', 'IceCream']], maxlag=1, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, 1)

surr_pearson, surr_mi, surr_te = [], [], []
for _ in range(n_surrogates):
    sx, sy = create_surrogate(ice_cream), create_surrogate(ac_usage)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=TEST_LAG))

p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

# 3. CONSOLE OUTPUT
print("\n" + "="*75)
print("RESULTS: CASE 2 - COMMON CAUSE (Temp -> Ice Cream & A/C)")
print("="*75)
def show_res(label, p, expected_sig):
    sig_label = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
    status = "CORRECT" if (p < 0.05) == expected_sig else "FAIL (Unexpected)"
    print(f"{label:<25} | p-value: {p:7.4f} | {sig_label:<15} | {status}")

show_res("Pearson (Ice-AC)", p_pearson, False) # Expected False Positive
show_res("Mutual Info (Ice-AC)", p_mi, False)     # Expected False Positive
show_res("Granger (Ice->AC)", p_granger, False)
show_res("Transfer Ent (Ice->AC)", p_te, False)
print("="*75)

# 4. PLOTS
plt.figure(figsize=(12, 4))
plt.plot(standardize(ice_cream), label='Ice Cream Sales', color='red')
plt.plot(standardize(ac_usage), label='A/C Usage', color='blue')
plt.title('Case 2: Spurious Correlation (Driven by Temperature)'); plt.xlabel('Days'); plt.ylabel('Z-score'); plt.legend(); plt.show()