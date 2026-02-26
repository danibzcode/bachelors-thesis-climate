# 🌍Discovery of Causal Relationships in the Climate System Using Causal Networks

## Abstract
The Earth's climate is a complex system where regional changes, such as El Niño or the Asian monsoon, have global repercussions. Anticipating critical transitions in the context of climate change requires understanding how these interactions might be modified. Traditional correlation-based methods detect dependency patterns but fail to capture directional or causal relationships.

Therefore, this project explores modern causal inference methodologies based on information theory. These tools are initially validated on a controlled reference system of coupled non-linear equations. Subsequently, they are applied to real climate indices to characterize relevant mechanisms in global climate dynamics through causal networks.

## Algorithms and Test Cases

This project evaluates different association and causal inference methods:
* **Undirected Association:** Pearson Correlation Coefficient ($r_{XY}$) and Mutual Information ($I(X;Y)$).
* **Directed Causal Inference:** Granger Causality (GC) and Transfer Entropy (TE).
* **Causal Network Discovery:** PCMCI algorithm, used to isolate direct causality and discard spurious correlations.

### 📈Statistical Validation Framework (Surrogate Testing)

A core component of this project is the rigorous statistical validation using surrogate data. Because a non-zero metric can arise purely from stochastic noise in finite time series, we implement:

* **IAAFT Surrogates:** Iterative Amplitude Adjusted Fourier Transform techniques generate null-hypothesis distributions that preserve the power spectrum (autocorrelation) and amplitude distribution of the original signals while destroying cross-dependence.
* **Empirical p-values:** Causal links are only considered valid if the observed metric significantly exceeds the null distribution, typically using a confidence level of $p < 0.05$.

---

## Validation Cases and Implementation

Each case corresponds to a script in the `src/` directory, testing specific properties of the algorithms.

### 1. Case 1: Independent Noise (Baseline)
**Objective:** Verify that no patterns are detected in purely stochastic data.

**Mathematical Model:**

$$X(t) = \sigma \epsilon_t^X$$

$$Y(t) = \sigma \epsilon_t^Y$$

Where $\epsilon$ is independent Gaussian white noise.

* **Script:** `case_01_baseline.py`
* **Result:** All p-values $> 0.05$. Observed values fall within the null distribution center.



### 2. Case 2: Common Cause (Confounding)
**Objective:** Distinguish between direct links and shared external drivers ($U$).

**Mathematical Model:**

$$U_t = \sigma \epsilon_t^u \quad \text{(Common Driver)}$$

$$X_t = U_t + \sigma \epsilon_t^x$$

$$Y_t = U_t + \sigma \epsilon_t^y$$

* **Script:** `case_02_common_cause.py`
* **Result:** Pearson and Mutual Information fail by reporting spurious significance, while Granger and Transfer Entropy correctly identify no direct causality.



### 3. Case 3: Linear Causality with Memory
**Objective:** Detect directed relationships with a specific time delay ($\tau$).

**Mathematical Model:**

$$X_t = a X_{t-1} + \sigma \epsilon_t^x$$

$$Y_t = b Y_{t-1} + c X_{t-\tau} + \sigma \epsilon_t^y$$

* **Script:** `case_03_linear_lag.py`
* **Result:** All methods detect the link ($p < 0.05$), with directed methods successfully identifying the $X \rightarrow Y$ flow.



### 4. Case 4: Non-Linear Causality
**Objective:** Demonstrate that Information Theory handles non-linearity where linear methods fail.

**Mathematical Model:**

$$X_t = a X_{t-1} + \sigma \epsilon_t^x$$

$$Y_t = (X_{t-\tau})^2 + \sigma \epsilon_t^y$$

* **Script:** `case_04_nonlinear_te.py`
* **Result:** Pearson and Granger report non-significance ($p > 0.05$), while Mutual Information and Transfer Entropy successfully detect the quadratic coupling.



### 5. Case 5: Causal Chains and Transitivity
**Objective:** Isolate direct links from indirect paths using Conditional Mutual Information (CMI).

**Mathematical Model:**

$$X_t = aX_{t-1} + \sigma\epsilon_t^X$$

$$Y_t = bY_{t-1} + cX_{t-1} + \sigma\epsilon_t^Y$$

$$Z_t = dZ_{t-1} + eY_{t-1} + \sigma\epsilon_t^Z$$

* **Script:** `case_05_causal_chain.py`
* **Result:** Bivariate MI $(X,Z)$ is significant but spurious; CMI $(X,Z | Y)$ vanishes, proving $Y$ mediates the information flow.



---

## Installation and Usage

To reproduce these results, ensure you have a Python environment with the following dependencies:

```bash
pip install numpy pandas matplotlib scipy statsmodels tigramite



---

## Installation and Usage

To reproduce these results, ensure you have a Python environment with the following dependencies:

```bash
pip install numpy pandas matplotlib scipy statsmodels tigramite
