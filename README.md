# PC-OAAD: Physics-Constrained Online Adaptive Anomaly Detection

Reference implementation of the framework described in:

> **A Physics-Constrained Online Adaptive Anomaly Detection Framework for Geological Drilling Time Series**
> Xuehao Sun et al., submitted to *Computers & Geosciences*.

This repository contains a minimal, runnable implementation of **Algorithm 1** in the manuscript. It is intended to support reproducibility and to allow readers to verify the core detection behavior on a synthetic test case.

## Method summary

The framework combines:

1. **Multivariate time-delay embedding** — projects drilling parameters (ROP, WOB, torque, RPM) into a state-space with explicit time-lag couplings. The delay parameter `tau` is determined automatically via the multivariate mutual information method (Eqs. 8–9).
2. **Online MSET reconstruction** — a dictionary of normal-state embedding vectors is maintained online; each new observation is reconstructed using cosine-similarity-based MSET (Eqs. 13–14).
3. **Physics-constrained dynamic threshold** — the Mechanical Specific Energy (MSE) model is used to modulate the reconstruction-error tolerance (Eqs. 15–19).
4. **Dual-criterion decision** — an alarm is raised only when both the reconstruction error exceeds the dynamic tolerance and the MSE deviation exceeds a threshold (Eq. 20). Normal samples update the dictionary (Eq. 21); anomalous samples do not.

## Requirements

- Python 3.8+
- NumPy >= 1.20
- Matplotlib >= 3.4 (only for the example plot)

Install with:

```bash
pip install -r requirements.txt
