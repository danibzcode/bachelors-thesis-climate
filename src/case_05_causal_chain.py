import numpy as np
import matplotlib.pyplot as plt
from causal_tools import (create_surrogate, manual_mi, manual_cmi, standardize)

# Definimos la función de ayuda que faltaba en los otros scripts
def show_res(label, p, expected_sig):
    sig_label = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
    status = "CORRECT" if (p < 0.05) == expected_sig else "FAIL (Unexpected)"
    print(f"{label:<25} | p-value: {p:7.4f} | {sig_label:<15} | {status}")

# 1. DATA GENERATION: Causal Chain (X -> Y -> Z)
# Model from PDF Page 81: Oil -> Gas -> Transport
n_samples, t_lag = 300, 1
np.random.seed(99)
oil = np.zeros(n_samples); gas = np.zeros(n_samples); trans = np.zeros(n_samples)

for t in range(t_lag, n_samples):
    oil[t] = 0.8 * oil[t-1] + 0.5 * np.random.randn()
    gas[t] = 0.5 * gas[t-1] + 0.4 * oil[t-t_lag] + 0.5 * np.random.randn()
    trans[t] = 0.7 * trans[t-1] + 0.3 * gas[t-t_lag] + 0.5 * np.random.randn()

# 2. ALIGNMENT FOR CAUSAL TESTING
# We test if Oil(t-1) causes Transport(t)
z_target = trans[t_lag:]
x_source = oil[:-t_lag]
y_mediator = gas[:-t_lag] # The conditioning variable

# 3. ANALYSIS: Bivariate MI vs Conditional MI (CMI)
N_BINS, n_surrogates = 5, 500
obs_mi = manual_mi(x_source, z_target, bins=N_BINS)
obs_cmi = manual_cmi(x_source, z_target, y_mediator, bins=N_BINS)

print(f"Case 5: Causal Chain (Oil -> Gas -> Transport)")
print(f"Calculating {n_surrogates} surrogates for CMI validation...")

surr_mi, surr_cmi = [], []
for _ in range(n_surrogates):
    sx = create_surrogate(oil); sy = create_surrogate(gas); sz = create_surrogate(trans)
    surr_mi.append(manual_mi(sx[:-t_lag], sz[t_lag:], bins=N_BINS))
    surr_cmi.append(manual_cmi(sx[:-t_lag], sz[t_lag:], sy[:-t_lag], bins=N_BINS))

p_mi = np.mean(np.array(surr_mi) >= obs_mi)
p_cmi = np.mean(np.array(surr_cmi) >= obs_cmi)

# 4. CONSOLE OUTPUT
print("\n" + "="*75)
print("RESULTS: CASE 5 - CAUSAL CHAIN & CONDITIONAL INDEPENDENCE")
print("="*75)
# Bivariate MI should be significant (Oil and Transport move together)
show_res("MI (Oil -> Transport)", p_mi, True)   
# CMI should be NOT significant (Oil doesn't add info if Gas is known)
show_res("CMI (Oil->Trans | Gas)", p_cmi, False) 
print("="*75)

# 5. VISUALIZATION
plt.figure(figsize=(12, 5))
plt.plot(standardize(oil), label='Oil Price (X)', color='black', alpha=0.8)
plt.plot(standardize(gas), label='Gas Price (Y)', color='green', alpha=0.6)
plt.plot(standardize(trans), label='Transport Cost (Z)', color='blue', alpha=0.6)
plt.title('Case 5: Causal Chain (Information Flow: X -> Y -> Z)', fontsize=14)
plt.xlabel('Time Step'); plt.ylabel('Z-score'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
