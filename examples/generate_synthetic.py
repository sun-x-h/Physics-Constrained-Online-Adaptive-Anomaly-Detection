"""Synthetic multivariate drilling data with a controlled anomaly."""

import numpy as np


def generate_synthetic(n_normal=1500, n_anomaly=300, n_after=500,
                       seed=42, anomaly_type="progressive"):
    """Generate a 4-channel synthetic drilling time series.

    Parameters
    ----------
    n_normal : int
        Number of normal samples before the labeled anomaly onset.
    n_anomaly : int
        Number of samples in the labeled anomaly interval.
    n_after : int
        Number of recovery samples after the anomaly interval.
    seed : int
        Random seed.
    anomaly_type : {"progressive", "sudden"}
        Progressive: anomaly intensity ramps up from t = n_normal - 100 to
        the middle of the anomaly interval, then stays strong.
        Sudden: step change at t = n_normal.

    Returns
    -------
    X : ndarray, shape (T, 4)
        Columns ordered as [ROP, WOB, Torque, RPM].
    labels : ndarray, shape (T,)
        1 inside the labeled anomaly interval, 0 elsewhere.
    """
    rng = np.random.default_rng(seed)
    T = n_normal + n_anomaly + n_after

    ROP = 0.20 + 0.02 * rng.standard_normal(T)
    WOB = 15.0 + 0.8 * rng.standard_normal(T)
    Torque = 4.0 + 0.20 * rng.standard_normal(T)
    RPM = 60.0 + 0.4 * rng.standard_normal(T)

    if anomaly_type == "progressive":
        # Ramp starts slightly before the labeled onset and saturates midway
        # through the anomaly interval. This mimics a gradually developing
        # downhole condition and gives the detector a chance to alarm early.
        ramp_start = max(0, n_normal - 100)
        ramp_peak = n_normal + n_anomaly // 2
        ramp = np.clip(
            (np.arange(T) - ramp_start) / max(ramp_peak - ramp_start, 1),
            0.0, 1.0,
        )
        ROP = ROP - 0.08 * ramp
        WOB = WOB + 8.0 * ramp
        Torque = Torque + 2.5 * ramp
        RPM = RPM + 1.0 * ramp
    elif anomaly_type == "sudden":
        a0 = n_normal
        a1 = n_normal + n_anomaly
        ROP[a0:a1] -= 0.08
        WOB[a0:a1] += 8.0
        Torque[a0:a1] += 2.5
    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    ROP = np.clip(ROP, 0.01, None)

    X = np.column_stack([ROP, WOB, Torque, RPM])
    labels = np.zeros(T, dtype=int)
    labels[n_normal:n_normal + n_anomaly] = 1
    return X, labels
