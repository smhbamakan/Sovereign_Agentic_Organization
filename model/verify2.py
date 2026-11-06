"""Verification of the v0.4 formal results. Every check reports the analytical value,
the Monte Carlo estimate and its 95% CI from the across-replication standard error.
Master seed 20261102; per-check seeds derived with SeedSequence.spawn."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import numpy as np, json
SS = np.random.SeedSequence(20261102)
seeds = iter(SS.spawn(40))
R = lambda: np.random.default_rng(next(seeds))
def ci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    return dict(mc=float(m), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se), se=float(se))
out = {"meta": dict(master_seed=20261102, generator="numpy PCG64", note="CI = mean ± 1.96·SE across independent replications")}

# ---------- Theorem 1 (vector, simplex) ----------
d = 4
def zs_cov(s):            # isotropic covariance restricted to the zero-sum subspace
    P = np.eye(d) - np.ones((d, d)) / d
    return s**2 * P
def run_vec(alpha, gamma, mu, se, su, rho, reps=300, T=4000, burn=1000, rng=None):
    P = np.eye(d) - np.ones((d, d)) / d
    theta = np.full(d, .25)
    Tt = np.tile(theta + gamma * mu / (1 - gamma), (reps, 1))
    acc = np.zeros((reps, d)); acc2 = np.zeros(reps); n = 0; minw = 1.0
    for t in range(T):
        z1 = rng.standard_normal((reps, d)) @ P; z2 = rng.standard_normal((reps, d)) @ P
        eps = se * z1; u = su * (rho * z1 + np.sqrt(1 - rho**2) * z2)
        y = Tt + mu + u
        s = (1 - gamma) * (theta + eps) + gamma * y
        Tt = (1 - alpha) * Tt + alpha * s
        if t >= burn:
            M = Tt - theta; acc += M; acc2 += (M * M).sum(1); n += 1; minw = min(minw, Tt.min())
    return acc / n, acc2 / n, np.array([minw])

def theory_vec(alpha, gamma, mu, se, su, rho):
    k = alpha * (1 - gamma)
    tr_e, tr_u, tr_eu = se**2 * (d - 1), su**2 * (d - 1), rho * se * su * (d - 1)
    trS = alpha**2 * ((1 - gamma)**2 * tr_e + gamma**2 * tr_u + 2 * gamma * (1 - gamma) * tr_eu)
    D = gamma * mu / (1 - gamma)
    return D, float(D @ D + trS / (k * (2 - k)))

mu = np.array([.06, -.02, -.02, -.02])
rng = R(); res = []
for (a, g, rho) in [(.3, 0., 0.), (.3, .4, 0.), (.3, .4, .5), (.6, .7, -.3)]:
    m, q, ins = run_vec(a, g, mu, .02, .02, rho, rng=rng)
    D, mse = theory_vec(a, g, mu, .02, .02, rho)
    res.append(dict(alpha=a, gamma=g, rho_eu=rho, D1_theory=float(D[0]), D1=ci(m[:, 0]), mse_theory=mse, mse=ci(q),
                    min_weight_observed=float(ins.min())))
out["T1_vector"] = res

# half-life incl. boundary
def half(alpha, gamma):
    k = alpha * (1 - gamma)
    if k >= 1: return 1.0
    return float(np.log(2) / -np.log(1 - k))
out["T1_half_life"] = dict(k018=half(.3, .4), ceil=int(np.ceil(half(.3, .4))), k1=half(1, 0))

# ---------- Theorem 2: loss, admissible set, optimal coupling ----------
def L(g, a=.3, mu=.5, se=.4, su=.4, c=0.):
    k = a * (1 - g)
    return (g * mu / (1 - g))**2 + a**2 * ((1 - g)**2 * se**2 + g**2 * su**2 + 2 * g * (1 - g) * c) / (k * (2 - k))
def dL0(a=.3, se=.4, c=0.):
    return 2 * a * (c * (2 - a) - se**2) / (2 - a)**2
gs = np.linspace(0, .99, 99001)
t2 = []
for mu_, c_ in [(.5, 0.), (.25, 0.), (.5, .1), (.5, .2)]:
    Ls = L(gs, mu=mu_, c=c_); i = int(Ls.argmin())
    num_d = (L(1e-6, mu=mu_, c=c_) - L(0, mu=mu_, c=c_)) / 1e-6
    adm = gs[Ls < 1.0]
    t2.append(dict(mu=mu_, cov_eu=c_, L0=float(L(0, mu=mu_, c=c_)), gamma_opt=float(gs[i]), L_opt=float(Ls[i]),
                   dL0_theory=dL0(c=c_), dL0_numeric=num_d, cov_threshold=.4**2 / (2 - .3),
                   admissible_max_delta1=float(adm.max()), bound_delta1=1 / (1 + mu_)))
out["T2"] = t2
# Monte Carlo check of L(gamma) at the optimum and at 0 (scalar)
def run_scalar(alpha, gamma, mu, se, su, reps=400, T=5000, burn=1000, rng=None, gate=None):
    M = np.zeros(reps) + gamma * mu / (1 - gamma); acc = np.zeros(reps); acc2 = np.zeros(reps); n = 0
    for t in range(T):
        e = se * rng.standard_normal(reps); u = su * rng.standard_normal(reps)
        warr = alpha * (1 - gamma) * (-M + e); selfr = alpha * gamma * (mu + u)
        if gate is None:
            M = M + warr + selfr
        elif gate[0] == 'whole':
            M = M + (rng.random(reps) < gate[1]) * (warr + selfr)
        else:
            r, f = gate
            M = M + (rng.random(reps) < r) * warr + (rng.random(reps) < f) * selfr
        if t >= burn:
            acc += M; acc2 += M**2; n += 1
    return acc / n, acc2 / n
rng = R()
_, q0 = run_scalar(.3, 0., .5, .4, .4, rng=rng)
go = [x for x in t2 if x["mu"] == .5 and x["cov_eu"] == 0][0]["gamma_opt"]
_, qo = run_scalar(.3, go, .5, .4, .4, rng=rng)
out["T2_mc"] = dict(L0_theory=float(L(0)), L0=ci(q0), Lopt_theory=float(L(go)), Lopt=ci(qo), gamma_opt=go)

# finite-horizon regret after one shift (scalar), H quarters
def regret(H, a, g, mu, se, su, Delta):
    k = a * (1 - g); D = g * mu / (1 - g); V = L(g, a, mu, se, su) - D**2
    S1 = (1 - (1 - k)**H) / k; S2 = (1 - (1 - k)**(2 * H)) / (k * (2 - k))
    return H * (D**2 + V) - 2 * D * Delta * S1 + Delta**2 * S2
def regret_mc(H, a, g, mu, se, su, Delta, reps=40000, rng=None):
    D = g * mu / (1 - g); V = L(g, a, mu, se, su) - D**2
    M = D + np.sqrt(V) * rng.standard_normal(reps) - Delta; tot = np.zeros(reps)
    for t in range(H):
        tot += M**2
        e = se * rng.standard_normal(reps); u = su * rng.standard_normal(reps)
        M = M + a * (1 - g) * (-M + e) + a * g * (mu + u)
    return tot
rng = R(); rg = []
for g, De in [(0., 1.), (.4, 1.), (.4, -1.)]:
    th = regret(16, .3, g, .5, .4, .4, De)
    rg.append(dict(gamma=g, Delta=De, H=16, theory=float(th), mc=ci(regret_mc(16, .3, g, .5, .4, .4, De, rng=rng)), bounded=16 * De**2))
out["P_regret"] = rg

# Proposition: drifting telos vs periodic human charter revision
def LS(g, a, mu, se, su, om):
    k = a * (1 - g); D = g * mu / (1 - g)
    return D**2 + (a**2 * ((1 - g)**2 * se**2 + g**2 * su**2) + om**2) / (k * (2 - k))
def LH(P, ell, sh, om, cP):
    return sh**2 + om**2 * (ell + (P - 1) / 2) + cP / P
def sim_periodic(P, ell, sh, om, reps=2000, T=4000, burn=400, rng=None):
    th = np.zeros(reps); hist = [th.copy()]; Tt = np.zeros(reps); acc = np.zeros(reps); n = 0
    for t in range(T):
        if t % P == 0:
            past = hist[max(0, len(hist) - 1 - ell)]
            Tt = past + sh * rng.standard_normal(reps)
        if t >= burn:
            acc += (Tt - th)**2; n += 1
        th = th + om * rng.standard_normal(reps); hist.append(th.copy())
        if len(hist) > ell + 2: hist.pop(0)
    return acc / n
def sim_sov_drift(a, g, mu, se, su, om, reps=2000, T=4000, burn=1000, rng=None):
    M = np.zeros(reps) + g * mu / (1 - g); acc = np.zeros(reps); n = 0
    for t in range(T):
        e = se * rng.standard_normal(reps); u = su * rng.standard_normal(reps)
        M = M + a * (1 - g) * (-M + e) + a * g * (mu + u) - om * rng.standard_normal(reps)
        if t >= burn:
            acc += M**2; n += 1
    return acc / n
rng = R()
out["P_periodic"] = dict(
    human=dict(P=4, ell=1, sigma_h=.1, omega=.1, theory=LH(4, 1, .1, .1, 0), mc=ci(sim_periodic(4, 1, .1, .1, rng=rng))),
    sovereign=dict(alpha=.3, gamma=.1, theory=LS(.1, .3, .5, .4, .4, .1), mc=ci(sim_sov_drift(.3, .1, .5, .4, .4, .1, rng=rng))))

# ---------- Proposition: stochastic ratification ----------
def rat_mean(g, mu, r, f): return (f / r) * g * mu / (1 - g)
def rat_var(a, g, mu, se, su, r, f, c=0.):
    k = a * (1 - g); A = a * (1 - g); Bb = a * g; m = rat_mean(g, mu, r, f)
    S = (r * A**2 * se**2 + f * Bb**2 * (mu**2 + su**2) + 2 * m * (1 - r * k) * f * Bb * mu + 2 * r * f * A * Bb * c) / (r * k * (2 - k))
    return S - m**2
rng = R(); rr = []
for g, r, f in [(.4, .5, .5), (.4, .8, .2), (.4, .8, .05), (0., .5, .5)]:
    mm, q = run_scalar(.3, g, .5, .4, .4, rng=rng, gate=(r, f))
    rr.append(dict(gamma=g, r=r, f=f, mean_theory=rat_mean(g, .5, r, f), mean=ci(mm),
                   var_theory=rat_var(.3, g, .5, .4, .4, r, f), var=ci(q - mm**2),
                   var_ungated=L(g) - (g * .5 / (1 - g))**2))
mm, q = run_scalar(.3, .4, .5, .4, .4, rng=rng, gate=('whole', .5))
out["P_ratification_whole_blind"] = dict(gamma=.4, r=.5, mean_theory=.4 * .5 / .6, mean=ci(mm), var_theory=L(.4) - (.4 * .5 / .6)**2, var=ci(q - mm**2))
out["P_ratification"] = rr

# ---------- Theorem 3: council, PSD eigenvalues ----------
def eig_block(K, m, rw, rb):
    n = K * m; C = np.full((n, n), rb); fam = np.repeat(np.arange(K), m)
    C[fam[:, None] == fam[None, :]] = rw; np.fill_diagonal(C, 1)
    return np.sort(np.linalg.eigvalsh(C))
def eig_theory(K, m, rw, rb):
    return sorted([1 - rw] * (K * m - K) + [1 + (m - 1) * rw - m * rb] * (K - 1) + [1 + (m - 1) * rw + (K * m - m) * rb])
out["T3_eigen"] = dict(numeric=[float(x) for x in eig_block(3, 3, .6, .15)], theory=[float(x) for x in eig_theory(3, 3, .6, .15)])
rng = R(); t3 = []
for K, m in [(1, 3), (3, 1), (3, 3), (9, 1)]:
    n = K * m; reps = 200
    vals = []
    for _ in range(reps):
        Rn = 2000
        zb = rng.standard_normal(Rn); zf = rng.standard_normal((Rn, K)); zi = rng.standard_normal((Rn, n)); fam = np.repeat(np.arange(K), m)
        e = np.sqrt(.15) * zb[:, None] + np.sqrt(.45) * zf[:, fam] + np.sqrt(.4) * zi
        vals.append(e.mean(1).var())
    t3.append(dict(K=K, m=m, theory=(1 + (m - 1) * .6 + (n - m) * .15) / n, mc=ci(vals)))
out["T3"] = t3

# ---------- Lemma 1 with explicit veto rule ----------
rng = R(); l1 = []
for q, phi in [(.2, 0.), (.2, .5), (.2, .9)]:
    miss, fv, resid = [], [], []
    for _ in range(100):
        Rn = 20000
        p11 = q * q + phi * q * (1 - q); u = rng.random(Rn); ep = u < q
        ec = np.where(ep, rng.random(Rn) < p11 / q, rng.random(Rn) < (q - p11) / (1 - q))
        miss.append(ec[ep].mean()); fv.append((~ep & ec).mean()); resid.append((ep & ec).mean())
    l1.append(dict(q=q, phi=phi, miss_theory=q + phi * (1 - q), miss=ci(miss), false_veto_theory=(1 - q) * q * (1 - phi), false_veto=ci(fv),
                   residual_theory=q * (q + phi * (1 - q)), residual=ci(resid)))
out["L1"] = l1

# ---------- Theorem 5 generalized ----------
def h_uniform(e, N=100, c=20, q0=.8): return q0 * np.minimum(e, c / N)
def h_overload(e, N=100, c=20, q0=.8, k=1.5):   # detection falls faster than 1/e beyond capacity
    return np.where(e * N <= c, q0 * e, q0 * (c / N) * (c / (e * N))**(k - 1))
es = np.array([.1, .2, .3, .5, .8])
out["T5"] = dict(e=es.tolist(), psi_uniform=(es - h_uniform(es)).tolist(), psi_overload=(es - h_overload(es)).tolist())
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(ROOT/'outputs'/'model'/'verify2_results.json', 'w'), indent=1, default=float)
