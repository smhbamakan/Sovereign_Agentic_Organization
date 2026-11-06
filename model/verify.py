"""Monte Carlo verification of the closed-form results in the formal model.
Each check compares an analytical expression with a simulation estimate."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import numpy as np, json
rng = np.random.default_rng(20261031)
out = {}

# ---------- Module A: telic revision ----------
def telic_sim(alpha, gamma, mu, se, su, theta0=0.0, delta=1.0, tau=200, T=6000, R=400):
    Tt = np.full(R, theta0 + gamma * mu / (1 - gamma) if gamma > 0 else theta0)
    M = np.zeros((T, R))
    for t in range(T):
        theta = theta0 + (delta if t >= tau else 0.0)
        y = Tt + mu + su * rng.standard_normal(R)
        s = (1 - gamma) * (theta + se * rng.standard_normal(R)) + gamma * y
        M[t] = Tt - theta
        Tt = Tt + alpha * (s - Tt)
    return M

def steady(alpha, gamma, mu, se, su):
    k = alpha * (1 - gamma)
    bias = gamma * mu / (1 - gamma)
    var = alpha**2 * ((1 - gamma)**2 * se**2 + gamma**2 * su**2) / (k * (2 - k))
    return bias, var

checks = []
for alpha, gamma, mu, se, su in [(0.3, 0.0, 0.5, 0.4, 0.4), (0.3, 0.4, 0.5, 0.4, 0.4), (0.6, 0.7, 0.3, 0.2, 0.5)]:
    M = telic_sim(alpha, gamma, mu, se, su)
    tail = M[3000:].ravel()
    b, v = steady(alpha, gamma, mu, se, su)
    checks.append(dict(alpha=alpha, gamma=gamma, mu=mu, bias_theory=b, bias_sim=float(tail.mean()),
                       var_theory=v, var_sim=float(tail.var())))
out['A_steady_state'] = checks

# half-life of mean misalignment after shock
alpha, gamma, mu = 0.3, 0.4, 0.0
M = telic_sim(alpha, gamma, mu, 0.0, 0.0, R=1, T=400)
m = -M[200:, 0]  # remaining gap to new theta (delta=1)
k = alpha * (1 - gamma)
hl_theory = np.log(2) / -np.log(1 - k)
hl_sim = int(np.argmax(m <= 0.5))
out['A_half_life'] = dict(theory=hl_theory, sim_first_period_at_or_below_half=hl_sim)

# Theorem 2: threshold comparison fixed vs adaptive
def adaptive_loss(alpha, gamma, mu, se, su):
    b, v = steady(alpha, gamma, mu, se, su)
    return b**2 + v
out['T2_example'] = dict(L_fixed_delta1=1.0,
                         L_sep=adaptive_loss(0.3, 0.0, 0.5, 0.4, 0.4),
                         L_coupled_g05=adaptive_loss(0.3, 0.5, 0.5, 0.4, 0.4),
                         L_coupled_g07=adaptive_loss(0.3, 0.7, 0.5, 0.4, 0.4))
# critical coupling gamma* for delta=1, alpha=.3, mu=.5, se=su=.4
gs = np.linspace(0, 0.99, 9901)
L = np.array([adaptive_loss(0.3, g, 0.5, 0.4, 0.4) for g in gs])
out['T2_gamma_star'] = float(gs[np.argmax(L > 1.0)])

# ---------- Module B: aggregation ----------
def agg_sim(K, m, rho_w, rho_b, R=200000):
    n = K * m
    zb = rng.standard_normal(R)
    zf = rng.standard_normal((R, K))
    zi = rng.standard_normal((R, n))
    fam = np.repeat(np.arange(K), m)
    e = np.sqrt(rho_b) * zb[:, None] + np.sqrt(rho_w - rho_b) * zf[:, fam] + np.sqrt(1 - rho_w) * zi
    return float(e.mean(axis=1).var())
def agg_theory(K, m, rho_w, rho_b):
    n = K * m
    return (1 + (m - 1) * rho_w + (n - m) * rho_b) / n
out['B_variance'] = [dict(K=K, m=m, theory=agg_theory(K, m, .6, .15), sim=agg_sim(K, m, .6, .15))
                     for K, m in [(1, 3), (3, 1), (1, 9), (3, 3), (9, 1)]]
out['B_floor'] = {K: .15 + (.6 - .15) / K for K in [1, 2, 3, 5]}

# critic detection with correlated binary errors
def critic_sim(q, phi, R=400000):
    # bivariate Bernoulli with marginal q and correlation phi
    p11 = q * q + phi * q * (1 - q)
    p10 = q - p11
    u = rng.random(R)
    ep = u < q
    ec = np.where(ep, rng.random(R) < p11 / q, rng.random(R) < p10 / (1 - q))
    return float(ec[ep].mean()), float((ep & ec).mean())
cr = []
for q, phi in [(.2, 0), (.2, .5), (.2, .9)]:
    miss_sim, joint_sim = critic_sim(q, phi)
    cr.append(dict(q=q, phi=phi, miss_theory=q + phi * (1 - q), miss_sim=miss_sim,
                   residual_theory=q * (q + phi * (1 - q)), residual_sim=joint_sim))
out['B_critic'] = cr

# ---------- Module D: oversight capacity ----------
def oversight(e, N=100, c=20, q0=.8, p=.1):
    reviewed = e * N
    q = q0 * min(1, c / reviewed) if reviewed > 0 else 0
    caught = reviewed * p * q
    rH = e
    h = q0 * min(e, c / N)
    return caught, rH, h, rH - h
out['D_oversight'] = [dict(e=e, caught=oversight(e)[0], Psi=oversight(e)[3]) for e in [.05, .1, .2, .4, .8]]

print(json.dumps(out, indent=1, default=float))
json.dump(out, open(ROOT/'outputs'/'model'/'verify_results.json', 'w'), indent=1, default=float)
