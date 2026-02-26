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
n_samples, t_lag = 500, 2
np.random.seed(42)
fertilizer = np.zeros(n_samples); yield_crop = np.zeros(n_samples)
for t in range(t_lag, n_samples):
    fertilizer[t] = 0.8 * fertilizer[t-1] + 0.5 * np.random.randn()
    yield_crop[t] = (fertilizer[t-t_lag])**2 + 0.5 * np.random.randn()

# 2. ANALYSIS
N_BINS, n_surrogates = 6, 500
obs_pearson = manual_pearson_corr(fertilizer, yield_crop)
obs_mi = manual_mi(fertilizer, yield_crop, bins=N_BINS)
obs_te = manual_transfer_entropy_lagged(fertilizer, yield_crop, bins=N_BINS, k_lag=t_lag)
data_g = pd.DataFrame({'Yield': yield_crop, 'Fertilizer': fertilizer})
res_xy = grangercausalitytests(data_g[['Yield', 'Fertilizer']], maxlag=4, verbose=False)
p_granger = get_best_granger_pvalue(res_xy, 4)

surr_pearson, surr_mi, surr_te = [], [], []
for _ in range(n_surrogates):
    sx, sy = create_surrogate(fertilizer), create_surrogate(yield_crop)
    surr_pearson.append(manual_pearson_corr(sx, sy))
    surr_mi.append(manual_mi(sx, sy, bins=N_BINS))
    surr_te.append(manual_transfer_entropy_lagged(sx, sy, bins=N_BINS, k_lag=t_lag))

p_pearson = np.mean(np.abs(surr_pearson) >= np.abs(obs_pearson))
p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_te = np.mean(np.array(surr_te) >= obs_te)

print("\n" + "="*75)
print("RESULTS: CASE 4 - NON-LINEAR (Fertilizer -> Crop Yield)")
print("="*75)
show_res("Pearson (Fert-Yield)", p_pearson, False) 
show_res("Mutual Info (Fert-Yield)", p_mi, True)    
show_res("Granger (Fert->Yield)", p_granger, False) 
show_res("Transfer Ent (Fert->Yield)", p_te, True)  
print("="*75)