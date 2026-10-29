"""Cosine-similarity-based MSET reconstruction (Eqs. 6-7, 13)."""

import numpy as np


def cosine_similarity_matrix(A, B):
    """Column-wise cosine similarity.

    Parameters
    ----------
    A : ndarray, shape (d, n)
    B : ndarray, shape (d, m)

    Returns
    -------
    ndarray, shape (n, m)
        Element (i, j) is cos(A[:, i], B[:, j]).
    """
    A_norm = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return A_norm.T @ B_norm


class MSETReconstructor:
    """MSET reconstruction with cosine similarity as the inner-product operator.

    Given a dictionary D (d, L), the reconstruction of an observation y (d, 1)
    follows Eq. (6) with the standard inner product replaced by cosine
    similarity:

        w = (S_DD)^{-1} s_Dy
        y_hat = D w

    where S_DD[i, j] = cos(D[:, i], D[:, j]) and s_Dy[i] = cos(D[:, i], y).
    A small Tikhonov regularization is added to S_DD for numerical stability.
    """

    def __init__(self, reg=1e-6):
        self.reg = reg
        self.D = None
        self._S_DD_inv = None

    def fit(self, D):
        """Set the dictionary and precompute the inverse Gram matrix.

        Parameters
        ----------
        D : ndarray, shape (d, L)
        """
        self.D = np.asarray(D, dtype=float).copy()
        L = self.D.shape[1]
        S_DD = cosine_similarity_matrix(self.D, self.D)
        S_DD_reg = S_DD + self.reg * np.eye(L)
        self._S_DD_inv = np.linalg.pinv(S_DD_reg)

    def reconstruct(self, y):
        """Reconstruct a single observation.

        Parameters
        ----------
        y : ndarray, shape (d,) or (d, 1)

        Returns
        -------
        ndarray, shape (d,)
        """
        if self.D is None:
            raise RuntimeError("Call fit() before reconstruct().")
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        s_Dy = cosine_similarity_matrix(self.D, y)  # (L, 1)
        w = self._S_DD_inv @ s_Dy                     # (L, 1)
        y_hat = self.D @ w                            # (d, 1)
        return y_hat.flatten()
