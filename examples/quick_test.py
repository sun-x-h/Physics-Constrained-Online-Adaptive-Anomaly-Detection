"""End-to-end quick test of the PC-OAAD detector on synthetic data."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make both the repository root (for `src`) and this directory importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from src.detector import PCOAADDetector          # noqa: E402
from generate_synthetic import generate_synthetic  # noqa: E402


def main():
    # --- Synthetic data ---------------------------------------------------
    n_normal = 1500
    n_anomaly = 300
    n_after = 500
    X, labels = generate_synthetic(
        n_normal=n_normal, n_anomaly=n_anomaly, n_after=n_after,
        seed=42, anomaly_type="progressive",
    )
    T, N = X.shape
    print(f"Synthetic data: {T} samples, {N} parameters")
    print(f"Labeled anomaly interval: [{n_normal}, {n_normal + n_anomaly})")

    # --- Initialization ---------------------------------------------------
    n_init = 800
    detector = PCOAADDetector(
        m=3, L=100, eta=0.05, delta0=0.03, alpha=0.15,
        theta=2.5, A_b=0.0366,
    )
    tau = detector.initialize(X[:n_init])
    print(f"Estimated delay parameter tau = {tau}")

    # --- Online detection -------------------------------------------------
    results = []
    for t in range(n_init, T):
        results.append(detector.step(X[t]))

    alarms = np.array([r["alarm"] for r in results], dtype=int)
    first_idx = int(np.argmax(alarms)) if alarms.any() else -1
    if first_idx >= 0:
        alarm_time = n_init + first_idx
        lead = n_normal - alarm_time
        print(f"First alarm at t = {alarm_time}")
        print(f"Lead time relative to labeled onset: {lead} samples")
    else:
        print("No alarm detected.")
    print(f"Total alarm steps: {int(alarms.sum())}")

    # --- Plot -------------------------------------------------------------
    t_axis = np.arange(n_init, T)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_axis, X[n_init:, 0], label="ROP (m/h)", color="C0")
    axes[0].plot(t_axis, X[n_init:, 1] / 50.0, label="WOB/50", color="C1")
    axes[0].plot(t_axis, X[n_init:, 2], label="Torque (kNm)", color="C2")
    axes[0].plot(t_axis, X[n_init:, 3] / 20.0, label="RPM/20", color="C3")
    axes[0].axvspan(n_normal, n_normal + n_anomaly,
                    color="red", alpha=0.15, label="Labeled anomaly")
    axes[0].set_ylabel("Parameters (scaled)")
    axes[0].legend(loc="upper right", fontsize=7)

    E = np.array([r["E"] for r in results])
    delta = np.array([r["delta"] for r in results])
    axes[1].plot(t_axis, E, label="Reconstruction error E(t)", color="C0")
    axes[1].plot(t_axis, delta, label="Dynamic threshold delta(t)", color="C3")
    axes[1].axvspan(n_normal, n_normal + n_anomaly, color="red", alpha=0.15)
    axes[1].set_ylabel("E / delta")
    axes[1].legend(loc="upper right", fontsize=8)

    dMSE = np.array([r["delta_MSE"] for r in results])
    axes[2].plot(t_axis, dMSE, label="|MSE - mu| / sigma", color="C0")
    axes[2].axhline(detector.theta, color="red", linestyle="--",
                    label=f"theta = {detector.theta}")
    axes[2].axvspan(n_normal, n_normal + n_anomaly, color="red", alpha=0.15)
    axes[2].set_xlabel("Time (samples)")
    axes[2].set_ylabel("MSE deviation")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(_HERE, "quick_test_result.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
