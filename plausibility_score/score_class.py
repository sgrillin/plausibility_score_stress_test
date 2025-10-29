from __future__ import annotations
import numpy as np

try:
    from scipy.stats import chi2, norm, t
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

class StressScenarioEngine:
    """
    Implements the two optimization frameworks from Mouy–Archer–Selmi (Risk, Aug 2017),
    with an option to use P&L in the Mahalanobis measure.

    - worst_loss_under_plausibility:   min P^T S  subject to plausibility constraint
    - most_plausible_given_loss:       min Mahalanobis subject to P^T S >= q

    Parameters
    ----------
    Sigma : (n,n) ndarray
        Covariance matrix of risk-factor returns.
    dist : {'normal','t'}
        Elliptical family for closed-form mapping.
    dof : int or None
        Degrees of freedom if dist='t'.

    Notes
    -----
    All formulas assume mean 0. Numerical stability: we auto-symmetrize Sigma.
    """

    def __init__(self, Sigma: np.ndarray, dist: str = "normal", dof: int | None = None):
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be square.")
        # Symmetrize for numerical stability
        self.Sigma = 0.5 * (Sigma + Sigma.T)
        self.n = Sigma.shape[0]
        self.dist = dist.lower()
        self.dof = dof
        if self.dist not in {"normal", "t"}:
            raise ValueError("dist must be 'normal' or 't'.")
        if self.dist == "t" and (self.dof is None or self.dof <= 2):
            # dof>2 to ensure finite variance; adjust if you handle low-dof explicitly
            raise ValueError("For Student-t, provide dof>2.")

        if not _HAVE_SCIPY:
            if self.dist == "t":
                raise RuntimeError("scipy is required for Student-t quantiles.")
            # For normal-only usage without SciPy, we can approximate with a Beasley-Springer/Moro if desired.
            # To keep code compact, require SciPy for now.
            raise RuntimeError("scipy is required for quantiles. Please install scipy.")

    # ---------- helpers (quantiles & radii) ----------
    def _p_radius_returns_mahalanobis(self, alpha: float) -> float:
        """
        Radius p in returns-space Mahalanobis: p^2 = chi2_{n, alpha}.
        """
        return np.sqrt(chi2.ppf(alpha, df=self.n))

    def _p_radius_pnl_mahalanobis(self, alpha: float) -> float:
        """
        Radius p in P&L Mahalanobis: |P^T S| / sqrt(P^T Σ P) <= p, with p=z_{alpha} for Normal
        or p = t_{alpha, dof} for Student-t if you use t on the *scalar* P&L.
        We implement it consistent with the chosen 'dist'.
        """
        if self.dist == "normal":
            return norm.ppf(alpha)
        else:
            return t.ppf(alpha, df=self.dof)

    def _loss_quantile(self, alpha: float, v: float) -> float:
        """
        q = loss quantile for P^T S ~ N(0, v) or t(0, v * dof/(dof-2)) depending on 'dist'.
        We return q as a negative number for a 'loss to the left'.
        """
        if self.dist == "normal":
            z = norm.ppf(alpha)
            return -z * np.sqrt(v)
        else:
            # For a standardized t with dof ν, Var = ν/(ν-2); scale so variance = v
            # Let Z ~ t_ν (std), so Var(Z) = ν/(ν-2); to get variance 1, divide by sqrt(ν/(ν-2))
            # Equivalently, quantile for mean 0 variance v is: q = - t_{α,ν} * sqrt(v * (ν-2)/ν)
            ta = t.ppf(alpha, df=self.dof)
            scale = np.sqrt(v * (self.dof - 2) / self.dof)
            return -ta * scale

    # ---------- core closed-form solutions ----------
    def worst_loss_under_plausibility(
        self,
        P: np.ndarray,
        alpha: float,
        plausibility_metric: str = "returns",
    ) -> tuple[np.ndarray, float]:
        """
        Problem (1): min P^T S  subject to a plausibility cap.
        Two plausibility choices:
          - 'returns': S^T Σ^{-1} S <= p^2 with p = sqrt(chi2_{n,alpha})
          - 'pnl'    : |P^T S| / sqrt(P^T Σ P) <= p  with p = z_{alpha} (or t-quantile)
        Returns
        -------
        S_star : ndarray (n,)
        loss   : float (negative)
        """
        P = np.asarray(P, dtype=float).reshape(-1)
        if P.shape[0] != self.n:
            raise ValueError("P length must match Sigma dimension.")
        v = float(P @ self.Sigma @ P)  # variance of P&L

        # Direction is ΣP in both cases (elliptical, linear objective)
        direction = self.Sigma @ P
        denom = np.sqrt(v) if v > 0 else 0.0
        if denom <= 0:
            # Zero variance portfolio -> any move doesn't change P&L; return zeros
            return np.zeros_like(P), 0.0

        plausibility_metric = plausibility_metric.lower()
        if plausibility_metric not in {"returns", "pnl"}:
            raise ValueError("plausibility_metric must be 'returns' or 'pnl'.")

        if plausibility_metric == "returns":
            # p is radius in returns-space ellipsoid
            p = self._p_radius_returns_mahalanobis(alpha)
            S_star = -p * direction / denom
            loss = P @ S_star  # = -p * sqrt(v)
        else:
            # P&L-based Mahalanobis radius (scalar), i.e., |P^T S| / sqrt(v) <= p
            p = self._p_radius_pnl_mahalanobis(alpha)
            # Minimizing P^T S => set P^T S = -p * sqrt(v). Among all S achieving that,
            # the most plausible in returns-space is again along ΣP:
            S_star = -(p * direction) / denom
            loss = P @ S_star  # = -p * sqrt(v)  (Normal or t scalar quantile)

        return S_star, float(loss)

    def most_plausible_given_loss(
        self,
        P: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, float]:
        """
        Problem (4)/(5): min S^T Σ^{-1} S  subject to P^T S >= q, with q the α-quantile
        of the P&L loss distribution (one-dimensional). Closed-form:
            S* = (q / (P^T Σ P)) * Σ P
        where q<0 for a left-tail loss.

        Returns
        -------
        S_star : ndarray (n,)
        loss   : float (should equal q)
        """
        P = np.asarray(P, dtype=float).reshape(-1)
        if P.shape[0] != self.n:
            raise ValueError("P length must match Sigma dimension.")

        v = float(P @ self.Sigma @ P)
        if v <= 0:
            return np.zeros_like(P), 0.0

        q = self._loss_quantile(alpha, v)  # negative
        S_star = (q / v) * (self.Sigma @ P)
        loss = float(P @ S_star)  # should equal q (up to rounding)
        return S_star, loss

    # ---------- meta-elliptical proxy (sketch) ----------
    def meta_elliptical_proxy(
        self,
        P: np.ndarray,
        alpha: float,
        Fi_ppf_list,   # list of callables s -> F_i^{-1}(s) for each marginal
        Fg_cdf,        # callable x -> F_g(x)   (common "generator" CDF, e.g., t_ν)
        Fg_ppf,        # callable s -> F_g^{-1}(s)
        mode: str = "most_plausible",
    ) -> tuple[np.ndarray, float]:
        """
        Proxy approach from the paper for meta-elliptical: map marginals to a common
        elliptical family, solve in X-space, map back.

        Parameters
        ----------
        Fi_ppf_list : list of quantile functions for each marginal S_i
        Fg_cdf/Fg_ppf: CDF/PPF of the common elliptical generator (e.g., Student-t_ν)
        mode : 'most_plausible' or 'worst_under_plausibility'

        Returns
        -------
        S_star, loss
        """
        # 1) Define a linearization around the center using the elliptical engine itself.
        # In practice, you would (a) transform an S to X via X_i = Fg^{-1}(F_i(S_i)),
        # (b) solve with the closed-form in X-space using the same Σ (interpreted there),
        # (c) map back via S_i = F_i^{-1}(Fg(X_i)).
        #
        # Here we only provide a minimal scaffold since full calibration & copula fitting
        # are application-specific.
        if mode == "most_plausible":
            S_star, loss = self.most_plausible_given_loss(P, alpha)
        else:
            S_star, loss = self.worst_loss_under_plausibility(P, alpha, plausibility_metric="returns")
        # Optional: push S_star through Fi/Fg mappings coordinate-wise for a first-order proxy.
        # For compactness we skip explicit transformations here.
        return S_star, loss


# ----------------------------- usage example -----------------------------
if __name__ == "__main__":
    # Example from the paper: two factors, 5% vol each, corr = 0.8, P=(2, -5)
    sigma = 0.05
    corr = 0.8
    Sigma = np.array([[sigma**2, corr*sigma**2],
                      [corr*sigma**2, sigma**2]])
    P = np.array([2.0, -5.0])
    alpha = 0.999  # 99.9%

    eng = StressScenarioEngine(Sigma, dist="normal")

    # (A) Max loss under plausibility cap with RETURNS-based Mahalanobis
    S_star_A, loss_A = eng.worst_loss_under_plausibility(P, alpha, plausibility_metric="returns")
    print("(A) returns-Mahalanobis: S* =", S_star_A, "loss =", loss_A)

    # (B) Max loss under plausibility cap with P&L-based Mahalanobis (your tweak)
    S_star_B, loss_B = eng.worst_loss_under_plausibility(P, alpha, plausibility_metric="pnl")
    print("(B) pnl-Mahalanobis:      S* =", S_star_B, "loss =", loss_B)

    # (C) Most plausible scenario for a target loss quantile (closed-form)
    S_star_C, loss_C = eng.most_plausible_given_loss(P, alpha)
    print("(C) most-plausible:       S* =", S_star_C, "loss =", loss_C)
