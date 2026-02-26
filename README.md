# 🌍Discovery of Causal Relationships in the Climate System Using Causal Networks

## Abstract
The Earth's climate is a complex system where regional changes (e.g., El Niño, Asian monsoon) have global repercussions. Anticipating critical transitions in the context of climate change requires understanding how these interactions might be modified. Traditional correlation-based methods detect dependency patterns but fail to capture directional or causal relationships. 

Therefore, this project explores modern causal inference methodologies based on information theory. These tools are initially validated on a controlled reference system of coupled non-linear equations. Subsequently, they are applied to real climate indices to characterize relevant mechanisms in global climate dynamics through causal networks.

##  Algorithms & Test Cases

This project evaluates different association and causal inference methods:
* **Undirected Association:** Pearson Correlation Coefficient and Mutual Information (MI).
* **Directed Causal Inference:** Granger Causality (GC) and Transfer Entropy (TE).
* **Causal Network Discovery:** PCMCI algorithm (used to isolate direct causality and discard spurious correlations).

To rigorously test these tools before applying them to real climate data, they are validated against five synthetic scenarios:
1. **Case 1:** Independent Noise (Baseline) 
2. **Case 2:** Common Cause (Confounding variables) 
3. **Case 3:** Linear Causality with Memory 
4. **Case 4:** Non-Linear Causality and Complex Dynamics 
5. **Case 5:** Causal Chains and Transitivity 

---

## How to Run the Scripts

The code is developed in **Python** and is designed to be easily executed in IDEs like Spyder.

### 1. Prerequisites
Make sure you have installed the required libraries. You can install the basic data science stack via your terminal:
```bash
pip install numpy pandas matplotlib seaborn scipy
