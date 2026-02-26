# 🌍Discovery of Causal Relationships in the Climate System Using Causal Networks

## Abstract
The Earth's climate is a complex system where regional changes (e.g., El Niño, Asian monsoon) have global repercussions. Anticipating critical transitions in the context of climate change requires understanding how these interactions might be modified. Traditional correlation-based methods detect dependency patterns but fail to capture directional or causal relationships. 

Therefore, this project explores modern causal inference methodologies based on information theory. These tools are initially validated on a controlled reference system of coupled non-linear equations. Subsequently, they are applied to real climate indices to characterize relevant mechanisms in global climate dynamics through causal networks.

##  Algorithms & Test Cases

This project evaluates different association and causal inference methods:
* **Undirected Association:** Pearson Correlation Coefficient and Mutual Information (MI).
* **Directed Causal Inference:** Granger Causality (GC) and Transfer Entropy (TE).
* **Causal Network Discovery:** PCMCI algorithm (used to isolate direct causality and discard spurious correlations).

### 📈 Statistical Validation Framework (Surrogate Testing)

A core component of this project is the rigorous statistical validation of the inferred causal links. A non-zero metric (like Mutual Information or Transfer Entropy) can arise purely from stochastic noise in finite time series. To address this, the project implements hypothesis testing using surrogate data:

* **IAAFT Surrogates:** We generate null-hypothesis distributions using Iterative Amplitude Adjusted Fourier Transform (IAAFT) techniques. This destroys any cross-coupling between variables while preserving the power spectrum (autocorrelation) and amplitude distribution of the individual original signals.
* **Empirical p-values:** The causal algorithms are evaluated against hundreds of these surrogate pairs. A causal link is only considered valid if the observed metric significantly exceeds the null distribution (typically using a confidence level of $p < 0.05$).

To rigorously test these tools before applying them to real climate data, they are validated against five synthetic scenarios:
1. **Case 1:** Independent Noise (Baseline) 
2. **Case 2:** Common Cause (Confounding variables) 
3. **Case 3:** Linear Causality with Memory 
4. **Case 4:** Non-Linear Causality and Complex Dynamics 
5. **Case 5:** Causal Chains and Transitivity 

---

## How to Run the Scripts

This project is developed in **Python**. To run the scripts and reproduce the causal networks, you need to install the following libraries:

* `numpy` and `pandas`: For data manipulation and matrix operations.
* `matplotlib`: For generating temporal evolution plots and scatter graphs.
* `scipy`: For statistical functions and F-tests in manual implementations.
* `statsmodels`: For standard Granger Causality tests.
* `tigramite`: For the PCMCI algorithm and Conditional Mutual Information (CMI) tests.

You can install all dependencies via your terminal using:
```bash
pip install numpy pandas matplotlib scipy statsmodels tigramite
