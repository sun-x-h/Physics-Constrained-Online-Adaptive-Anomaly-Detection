"""Mutual-information delay estimation and multivariate time-delay embedding."""

import numpy as np


def mutual_information(x, y, n_bins=20):
    """Estimate mutual information between two scalar series via 2D histogram.

    Parameters
    ----------
    x, y : ndarray, shape (n,)
        Scalar series of equal length.
    n_bins : int
        Number of histogram bins per dimension.

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_2d / max(np.sum(hist_2d), 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    if not np.any(nz):
        return 0.0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def estimate_delay(X, max_tau=50, n_bins=20):
    """Determine the delay parameter tau via the multivariate mutual information method.

    Implements Eqs. (8)-(9) of the manuscript: the average pairwise mutual
    information Gamma(tau) is computed for tau = 1..max_tau, and the first
    local minimum of Gamma(tau) is returned as the delay parameter.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Multivariate time series.
    max_tau : int
        Maximum delay to search.
    n_bins : int
        Number of histogram bins for mutual-information estimation.

    Returns
    -------
    int
        Estimated delay parameter tau >= 1.
    """
    T, N = X.shape
    if T <= max_tau + 5:
        raise ValueError("Sequence too short for the requested max_tau.")

    gammas = []
    taus = list(range(1, max_tau + 1))
    for tau in taus:
        total = 0.0
        count = 0
        for i in range(N):
            for j in range(N):
                xi = X[: T - tau, i]
                xj = X[tau:, j]
                total += mutual_information(xi, xj, n_bins=n_bins)
                count += 1
        gammas.append(total / (N * N))

    gammas = np.asarray(gammas)
    for k in range(1, len(gammas) - 1):
        if gammas[k] <= gammas[k - 1] and gammas[k] <= gammas[k + 1]:
            return taus[k]
    # Fallback: global minimum
    return taus[int(np.argmin(gammas))]


def embed_multivariate(X, m, tau):
    """Construct multivariate time-delay embedding vectors.

    Implements Eqs. (10)-(11): for each parameter n and each lag p,
    the column is x_n(t - p*tau). Columns are ordered parameter-first,
    then lag, matching Eq. (11).

    Parameters
    ----------
    X : ndarray, shape (T, N)
    m : int
        Embedding dimension.
    tau : int
        Delay parameter.

    Returns
    -------
    ndarray, shape (T - (m-1)*tau, N*m)
    """
    T, N = X.shape
    n_out = T - (m - 1) * tau
    if n_out <= 0:
        raise ValueError("Embedding dimension too large for sequence length.")
    cols = []
    for n in range(N):
        for p in range(m):
            cols.append(X[p * tau: p * tau + n_out, n])
    return np.column_stack(cols)
