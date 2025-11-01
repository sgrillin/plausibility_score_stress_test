import numpy as np

try:
    from scipy.stats import norm, t as student_t
except Exception as e:
    raise ImportError(
        "This module requires SciPy (scipy.stats.norm and scipy.stats.t). "
        "Install with: pip install scipy"
    )

class PlausibilityStress:
    """
    Implements the 'extremely (un)likely' plausibility approach to stress testing.

    References:
    - Closed-form optimal scenario for elliptical distributions (Gaussian / Student-t):
      S* = -q_alpha * (Sigma @ P) / sqrt(P' Sigma P),  with P' S* = - q_alpha * sqrt(P' Sigma P)
    - Meta-elliptical (meta-t) proxy via per-marginal t CDF mapping.

    Parameters
    ----------
    Sigma : (n,n) array_like
        Positive-definite covariance (or scatter) matrix for risk-factor returns.
    dist : {"gaussian","student"}
        Elliptical family in which the closed-form is computed.
    df : int or float, optional
        Degrees of freedom if dist == "student".
    mu : (n,) array_like, optional
        Mean vector. Defaults to zeros; the method uses centered returns.

    Notes
    -----
    - Elliptical (Gaussian/t) optimal scenario implements the “most plausible among worst-loss”
      at a fixed loss quantile q (implemented here via alpha -> q_alpha closed-form).
    - Meta-t proxy follows the paper’s 4-step mapping: choose per-marginal dfs (nu_i) and a copula df (nu_g),
      solve in t-elliptical space with df=nu_g, then inverse-map to each marginal.
    """

    def __init__(self, Sigma, dist="gaussian", df=None, mu=None):
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be a square (n x n) matrix.")
        self.Sigma = Sigma
        self.n = Sigma.shape[0]
        self.dist = dist.lower()
        self.df = df
        if self.dist not in ("gaussian", "student"):
            raise ValueError("dist must be 'gaussian' or 'student'.")
        if self.dist == "student" and (df is None or df <= 2):
            # df>2 ensures finite covariance for classical multivariate t with scatter Sigma
            raise ValueError("For Student-t, supply df > 2.")
        self.mu = np.zeros(self.n) if mu is None else np.asarray(mu, dtype=float)
        if self.mu.shape != (self.n,):
            raise ValueError("mu must have shape (n,)")

        # Precompute Cholesky for Mahalanobis and checks
        self._chol = np.linalg.cholesky(self.Sigma)

    # ---------- Utilities ----------

    def _loss_scale(self, P):
        """Return sqrt(P' Sigma P)."""
        P = np.asarray(P, dtype=float).reshape(-1)
        if P.shape != (self.n,):
            raise ValueError("P must have shape (n,)")
        s2 = float(P @ self.Sigma @ P)
        if s2 <= 0:
            raise ValueError("P' Sigma P must be positive.")
        return np.sqrt(s2)

    @staticmethod
    def _z_alpha(alpha, dist="gaussian", df=None):
        """Return positive q_alpha = Phi^{-1}(alpha) or t_df^{-1}(alpha)."""
        if not (0.5 < alpha < 1.0):
            raise ValueError("alpha must be in (0.5, 1)")
        if dist == "gaussian":
            return float(norm.ppf(alpha))
        elif dist == "student":
            if df is None:
                raise ValueError("Provide df for Student-t.")
            return float(student_t.ppf(alpha, df))
        else:
            raise ValueError("dist must be 'gaussian' or 'student'.")

    def mahalanobis(self, S):
        """Mahalanobis distance sqrt( (S-mu)' Sigma^{-1} (S-mu) )."""
        S = np.asarray(S, dtype=float).reshape(-1)
        d = S - self.mu
        # Solve via Cholesky: ||L^{-1} d||_2
        y = np.linalg.solve(self._chol, d)
        return float(np.linalg.norm(y))

    # ---------- Core: Elliptical (closed-form) ----------

    def optimal_scenario(self, P, alpha):
        """
        Closed-form optimal scenario in Gaussian or Student-t elliptical model.

        Parameters
        ----------
        P : (n,) array_like
            Portfolio weights (P&L sensitivities). P'S is the portfolio return/PNL.
        alpha : float in (0.5, 1)
            One-sided loss quantile level (e.g., 0.999).

        Returns
        -------
        S_star : (n,) ndarray
            Optimal stress scenario (risk-factor returns).
        loss_quantile : float
            Portfolio loss (negative P' S*) at the chosen alpha (positive number).
        """
        P = np.asarray(P, dtype=float).reshape(-1)
        if P.shape != (self.n,):
            raise ValueError("P must have shape (n,)")

        z = self._z_alpha(alpha, dist=self.dist, df=self.df)
        scale = self._loss_scale(P)  # sqrt(P' Sigma P)
        # loss quantile q = z * scale  (positive). Portfolio loss is - P'S*, so P'S* = -q.
        q = z * scale

        direction = self.Sigma @ P
        denom = float(P @ direction)  # = P' Sigma P = scale^2
        S_star = - q * direction / denom  # closed-form
        return S_star, q

    # ---------- Meta-t proxy ----------

    def meta_t_optimal_scenario(self, P, alpha, dfs_marginal, df_copula, scales=None):
        """
        Meta-elliptical (meta-t) proxy:
        1) Solve in t-elliptical space with df=df_copula -> X*
        2) Map back S*_i = F^{-1}_{t(nu_i, scale=scale_i)}( F_{t(df_copula)}( X*_i ) )

        Parameters
        ----------
        P : (n,) array_like
            Portfolio weights.
        alpha : float in (0.5,1)
            One-sided loss quantile.
        dfs_marginal : (n,) array_like
            Per-marginal degrees of freedom (nu_i > 2 recommended).
        df_copula : float
            Copula (elliptical) df used in step (1) (nu_g in the paper).
        scales : (n,) array_like, optional
            Per-marginal scale (e.g., vol). Defaults to 1 for all.

        Returns
        -------
        S_star_meta : (n,) ndarray
            Meta-t proxy optimal scenario in original factor space.
        q_meta : float
            Portfolio loss quantile implied by the t-elliptical step (same as step 1).
        """
        dfs_marginal = np.asarray(dfs_marginal, dtype=float).reshape(-1)
        if dfs_marginal.shape != (self.n,):
            raise ValueError("dfs_marginal must have shape (n,)")
        if np.any(dfs_marginal <= 2):
            raise ValueError("All marginal dfs should be > 2 for finite variance.")

        if df_copula <= 2:
            raise ValueError("df_copula should be > 2.")

        if scales is None:
            scales = np.ones(self.n, dtype=float)
        else:
            scales = np.asarray(scales, dtype=float).reshape(-1)
            if scales.shape != (self.n,):
                raise ValueError("scales must have shape (n,)")

        # Step (1): solve as a Student-t elliptical with df = df_copula
        saved = (self.dist, self.df)
        self.dist, self.df = "student", df_copula
        X_star, q_meta = self.optimal_scenario(P, alpha)
        self.dist, self.df = saved  # restore

        # Step (2): component-wise CDF mapping t_{df_copula} -> t_{nu_i} with per-marginal scale.
        u = student_t.cdf(X_star, df_copula)  # (n,)
        # Clamp numerically
        u = np.clip(u, 1e-12, 1 - 1e-12)

        S_star_meta = np.empty(self.n)
        for i in range(self.n):
            # Inverse CDF of t_{nu_i} then scale
            S_star_meta[i] = scales[i] * student_t.ppf(u[i], dfs_marginal[i])

        return S_star_meta, q_meta

