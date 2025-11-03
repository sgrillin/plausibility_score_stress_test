# Plausibility Stress Testing (Replication of *A Plausibility Metric for Stress Scenarios*)

## Overview

This repository provides a Python implementation of the methodology described in the paper:

> **Blum, Philipp; Papenbrock, Jochen; Schwendner, Peter; and Shevchenko, Pavel (2023)**  
> *"A Plausibility Metric for Stress Scenarios."*  
> _The Journal of Risk Model Validation, Volume 17, Number 4 (2023), pp. 37–68._

The paper introduces a **plausibility-based approach to stress testing**, defining stress scenarios as the *most likely* configuration of joint factor movements that produce a given portfolio loss.  
This implementation replicates the paper’s mathematical framework and reproduces its numerical examples (e.g., the 2-factor CAC–AEX case).

---

## Core Idea

Traditional stress testing solves a **“worst-loss-for-given-distance”** optimization:

\[
\max_{S} P^\top S \quad \text{s.t. } S^\top \Sigma^{-1} S \leq r^2,
\]

where \(r^2\) corresponds to a fixed Mahalanobis radius (e.g., the 99.9% χ² quantile).  
This approach is unstable, because \(r^2\) depends on the dimension of Σ — adding uncorrelated factors changes the scenario severity.

The *Plausibility Metric* paper instead proposes the **dual problem**:

\[
\min_{S} S^\top \Sigma^{-1} S \quad \text{s.t. } P^\top S = -q,
\]

which asks:

> *Among all scenarios that produce a portfolio loss \(q\), which one is most plausible (i.e. closest to the mean under Σ)?*

This yields an **analytical solution** for elliptical distributions:

\[
S^\* = -q_\alpha \frac{\Sigma P}{P^\top \Sigma P}, \qquad
P^\top S^\* = -q_\alpha \sqrt{P^\top \Sigma P}.
\]

where:
- \(S^\*\): most plausible stress scenario  
- \(P\): vector of portfolio sensitivities or exposures  
- \(\Sigma\): covariance matrix of factor returns  
- \(q_\alpha\): univariate quantile (Gaussian or Student-t)

---

## Features

| Function | Description |
|-----------|-------------|
| `optimal_scenario(P, alpha)` | Computes the closed-form optimal stress scenario under a Gaussian or Student-t model. |
| `meta_t_optimal_scenario(P, alpha, dfs_marginal, df_copula, scales)` | Implements the paper’s **meta-t proxy** (heterogeneous marginal tails). |
| `mahalanobis(S)` | Computes Mahalanobis distance \(r = \sqrt{S^\top \Sigma^{-1} S}\). |
| `implied_alpha_portfolio(S, P)` | Estimates the α-quantile implied by a given portfolio loss. |
| `implied_alpha_joint(S)` | Estimates the joint plausibility α of any vector of shocks. |
| `plausibility_summary(S)` | Provides log-likelihood and tail probability summary for any scenario. |

---

## Mathematical Models

### 1. Elliptical (Gaussian / Student-t)

For an elliptical distribution with mean 0 and covariance Σ:

\[
S^\* = - q_\alpha \frac{\Sigma P}{\sqrt{P^\top \Sigma P}},
\]

with:
- \(q_\alpha = \Phi^{-1}(\alpha)\) for Gaussian,  
- \(q_\alpha = t_{\nu}^{-1}(\alpha)\) for Student-t(ν).

The corresponding portfolio loss is \(P^\top S^\* = -q_\alpha \sqrt{P^\top \Sigma P}\).  
This represents the *most plausible configuration* of factor moves for a fixed α-level portfolio loss.

---

### 2. Meta-t Proxy (Section 3.3 of the Paper)

To allow for different marginal tail thicknesses, the paper defines a **meta-t (meta-elliptical)** construction:

1. Solve in t-elliptical space with copula df = ν_g (the dependence structure).  
2. Transform each component back to its marginal distribution:  
   \[
   S_i^\* = F^{-1}_{t(\nu_i)}\!\big(F_{t(\nu_g)}(X_i^\*)\big),
   \]
   where \(X_i^\*\) are the t-elliptical optimal factors.

This proxy avoids full numerical optimization and remains accurate up to extreme quantiles (≈ 99.9%).

---

## Example Usage

```python
import numpy as np
import pandas as pd
import yfinance as yf
import score_class as sc

# Download market data
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2020-01-01', end='2025-10-31',
                   progress=False, auto_adjust=False)['Adj Close']
returns = np.log(data / data.shift(1)).dropna()
Sigma = returns.cov().values
P = np.array([2.0, -5.0, 10.0])

# Student-t elliptical model
ps = sc.PlausibilityStress(Sigma, dist="student", df=5)
S_star, q = ps.optimal_scenario(P, alpha=0.999)

print("Optimal scenario (Student-t df=5):")
print(pd.Series(100*S_star, index=tickers))
print("Portfolio 99.9% loss quantile:", q)

# Evaluate arbitrary shock scenario
s = np.array([-0.10, -0.08, -0.12])
alpha_port = ps.implied_alpha_portfolio(s, P)
alpha_joint = ps.implied_alpha_joint(s)

print("Implied portfolio alpha:", alpha_port)
print("Implied joint alpha:", alpha_joint)
