"""Physics-constrained online adaptive anomaly detector (Algorithm 1)."""

import numpy as np

from .embedding import estimate_delay, embed_multivariate
from .mset import MSETReconstructor
from .mse import compute_mse


class PCOAADDetector:
    """Online detector implementing Algorithm 1 of the manuscript.

    Parameters
    ----------
    m : int
        Embedding dimension.
    L : int
        Dictionary size (number of embedded vectors retained).
    eta : float
        Dictionary learning rate.
    delta0 : float
        Initial relative tolerance.
    alpha : float
        Decay coefficient for MSE-driven tolerance adjustment.
    theta : float
        MSE deviation threshold for the dual-criterion decision.
    A_b : float
        Bit cross-sectional area (m^2).
    max_tau : int
        Maximum delay searched by the mutual information method.
    reg : float
        Tikhonov regularization for the MSET Gram matrix.

    Notes
    -----
    Input vectors x_t are expected as [ROP, WOB, Torque, RPM] in the units
    used by Eq. (1): ROP m/h, WOB kN, Torque kNm, RPM rpm.
    """

    def __init__(self, m=3, L=100, eta=0.05, delta0=0.03, alpha=0.15,
                 theta=2.5, A_b=0.0366, max_tau=50, reg=1e-6):
        self.m = m
        self.L = L
        self.eta = eta
        self.delta0 = delta0
        self.alpha = alpha
        self.theta = theta
        self.A_b = A_b
        self.max_tau = max_tau
        self.reg = reg

        self.tau = None
        self.D = None
        self._mset = MSETReconstructor(reg=reg)
        self._mse_window = []
        self._buffer = []
        self._ready = False

    def initialize(self, X_init):
        """Warm-up stage: determine tau, build the dictionary, seed the MSE window.

        Parameters
        ----------
        X_init : ndarray, shape (T_init, 4)
            Initial observations. Must satisfy T_init >= (m-1)*tau + L for the
            estimated tau; otherwise a ValueError is raised.

        Returns
        -------
        int
            Estimated delay parameter tau.
        """
        X_init = np.asarray(X_init, dtype=float)
        self.tau = estimate_delay(X_init, max_tau=self.max_tau)
        X_emb = embed_multivariate(X_init, self.m, self.tau)

        if X_emb.shape[0] < self.L:
            raise ValueError(
                f"Need at least L={self.L} embedded vectors, got {X_emb.shape[0]}. "
                "Increase T_init or reduce L."
            )

        # Use the most recent L embedded vectors so that the dictionary is
        # contiguous with the online phase.
        self.D = X_emb[-self.L:].T.copy()
        self._mset.fit(self.D)

        # Seed the MSE sliding window from the tail of the initialization data.
        mse_init = compute_mse(
            WOB=X_init[:, 1], RPM=X_init[:, 3],
            T=X_init[:, 2], ROP=X_init[:, 0], A_b=self.A_b,
        )
        mse_init = mse_init[np.isfinite(mse_init)]
        if mse_init.size == 0:
            raise ValueError("No finite MSE values in initialization data.")
        if mse_init.size < self.L:
            pad = np.full(self.L - mse_init.size, mse_init[0])
            mse_init = np.concatenate([pad, mse_init])
        self._mse_window = list(mse_init[-self.L:])

        # Pre-fill the raw-data buffer so the first online step yields a
        # complete embedded vector.
        n_buffer = (self.m - 1) * self.tau + 1
        tail = X_init[-n_buffer:]
        self._buffer = [row.copy() for row in tail]

        self._ready = True
        return self.tau

    def step(self, x_t):
        """Process one online sample.

        Parameters
        ----------
        x_t : ndarray, shape (4,)
            [ROP, WOB, Torque, RPM].

        Returns
        -------
        dict
            Keys: 'alarm' (0/1), 'E' (reconstruction error), 'delta'
            (dynamic tolerance), 'MSE', 'delta_MSE'.
        """
        if not self._ready:
            raise RuntimeError("Call initialize() before step().")

        x_t = np.asarray(x_t, dtype=float).reshape(-1)
        self._buffer.append(x_t.copy())
        if len(self._buffer) > (self.m - 1) * self.tau + 1:
            self._buffer.pop(0)

        # Build the latest embedded vector.
        y = self._embed_single().reshape(-1, 1)

        # Reconstruction and reconstruction error (Eqs. 13-14).
        y_hat = self._mset.reconstruct(y).reshape(-1, 1)
        E = float(np.linalg.norm(y - y_hat) / (np.linalg.norm(y) + 1e-12))

        # MSE and sliding-window statistics (Eqs. 1, 15-16).
        ROP, WOB, Torque, RPM = x_t
        MSE_t = compute_mse(WOB, RPM, Torque, ROP, self.A_b)

        window = np.asarray(self._mse_window, dtype=float)
        mu = float(np.mean(window))
        sigma = float(np.std(window) + 1e-12)
        delta_MSE = float(abs(MSE_t - mu) / sigma)

        # Dynamic tolerance (Eq. 19).
        delta = float(self.delta0 * np.exp(-self.alpha * delta_MSE))

        # Dual-criterion decision (Eq. 20).
        if E > delta and delta_MSE > self.theta:
            alarm = 1
        else:
            alarm = 0
            # Dictionary update on normal samples (Eq. 21).
            self.D = (1.0 - self.eta) * self.D + self.eta * y
            self._mset.fit(self.D)
            # Update MSE window only on normal samples to avoid contamination.
            self._mse_window.append(MSE_t)
            if len(self._mse_window) > self.L:
                self._mse_window.pop(0)

        return {
            "alarm": alarm,
            "E": E,
            "delta": delta,
            "MSE": float(MSE_t),
            "delta_MSE": delta_MSE,
        }

    def _embed_single(self):
        """Build one embedded vector from the buffer, matching embed_multivariate."""
        n_needed = (self.m - 1) * self.tau + 1
        buf = np.asarray(self._buffer[-n_needed:], dtype=float)
        T_buf, N = buf.shape
        cols = []
        for n in range(N):
            for p in range(self.m):
                cols.append(buf[T_buf - 1 - p * self.tau, n])
        return np.asarray(cols, dtype=float)
