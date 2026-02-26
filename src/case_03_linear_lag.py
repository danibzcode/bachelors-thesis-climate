import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from causal_tools import (create_surrogate, manual_pearson_corr, manual_mi, 
                          get_best_granger_pvalue, manual_transfer_entropy_lagged, standardize)

def show_res(label, p, expected_sig):
    sig_label = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
    status = "CORRECT" if (p < 0.05) == expected_sig else "FAIL (Unexpected)"
    print(f"{label:<25} | p-value: {p:7.4f} | {sig_label:<15} | {status}")

# 1. DATA GENERATION
n_samples, t_lag = 400, 2
np.random.seed(42)
rates = np.zeros(n_samples); unemp = np.zeros(n_samples)
for t in range(t_lag, n_samples):
    rates[t] = 0.8 * rates[t-1] + 0.5 * np.random.randn()
    unemp[t] = 0.7 * unemp[t-1] + 0.4 * rates[t-t_lag] + 0.5 * np.random.randn()

# 2. ANALYSIS
N_BINS, n_surrogates = 5, 500
obs_pearson = manual_pearson_corr(rates, unemp)
obs_mi = manual_mi(rates, unemp, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(rates, unemp, bins=N_BINS, k_lag=t_lag)
data_g = pd.DataFrame({'Unemp': unemp, 'Rates': rates})
res_xy = grangercausalitytests(data_g[['Unemp', 'Rates']], maxlag=4, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, 4)

surr_pearson, surr_mi, surr_te = [], [], []
for _ in range(n_surrogates):
    sx, sy = create_surrogate(rates), create_surrogate(unemp)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=t_lag))

p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

print("\n" + "="*75)
print("RESULTS: CASE 3 - LINEAR LAG (Interest Rates -> Unemployment)")
print("="*75)
show_res("Pearson (Rates-Unemp)", p_pearson, True)
show_res("Mutual Info (Rates-Unemp)", p_mi, True)
show_res("Granger (Rates->Unemp)", p_granger, True)
show_res("Transfer Ent (Rates->Unemp)", p_te, True)
print("="*75)