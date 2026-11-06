"""Environment-defined warranted telos (replaces the expert panel).

Criterion. One regime-invariant criterion W is fixed in advance: the long-run expected value of the
firm (mean expected quarterly profit net of expected fines and turnover costs over quarters
H0..H of a regime), subject to the constitutive commitments (no shortcut; quality >= 0.2).

Induced policy. A telos w is pursued quarter by quarter: each quarter the firm chooses the decision
that maximizes w . E[y_t] given the current state (y_t is the current-quarter outcome profile the
agents observe). Profit is therefore measured myopically, while reputation, morale and risk are
stocks whose value to W is delayed; weighting them corrects myopia.

Warranted telos. theta(regime) is the centroid of the warranted set
Theta_eps = { w on the simplex : W(policy induced by w) >= (1 - eps) * max_w W }.
Because W never changes, theta moves only because the environment's parameters change.
Agents never see W, theta or Theta.

Because the map from telos to induced policy is many-to-one (policies often sit at corners of the
decision space), Theta is a set rather than a point. The decision-relevant outcome is therefore the
VALUE REGRET of a telos, W* - W(policy induced by T), which needs no point target; distance to the set
Theta and distance to its centroid theta (the formal model's target) are reported alongside.
"""
import itertools, json, sys, numpy as np
from env import BASE_PARAMS, SHIFTS, expected_step

def regret_lookup(T, regime, table):
    """Value regret of telos T in a regime, from the stored grid (nearest simplex point, step 1/20)."""
    g = np.rint(np.asarray(T) * 20).astype(int); g[np.argmax(np.asarray(T))] += 20 - g.sum()
    r = table[regime]; return r["W_star"] - r["grid_values"][r["index"][tuple(g)]]

def load_table(path=None):
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "warranted_telos.json")
    W = json.load(open(path)); Wg = np.rint(simplex() * 20).astype(int)
    for r in W:
        r["index"] = {tuple(x): i for i, x in enumerate(Wg)}
    return W

LEVELS = np.round(np.linspace(0, 1, 9), 4)                 # decision grid, step 0.125
H0, H = 12, 40                                             # evaluation window (quarters into a regime)
S_INIT = np.array([0.6, 0.6, 0.1])                         # reputation, morale, risk at founding

def regime_params():
    p = dict(BASE_PARAMS); out = [dict(p)]
    for s in SHIFTS:
        p = dict(p, **s.effects); out.append(dict(p))
    return out

def decisions():
    return np.array([d for d in itertools.product(LEVELS, repeat=4) if d[1] >= 0.2])

def simplex(step=20):
    return np.array([(a, b, c, step - a - b - c) for a in range(step + 1) for b in range(step + 1 - a)
                     for c in range(step + 1 - a - b)], float) / step

def value_of(Wt, p, D, s0=S_INIT, batch=150):
    """Long-run value W of the myopic policy induced by each telos in Wt (n,4)."""
    vals = np.empty(len(Wt)); traj_d = np.empty((len(Wt), 4))
    for i in range(0, len(Wt), batch):
        w = Wt[i:i + batch]; S = np.tile(s0, (len(w), 1)); acc = np.zeros(len(w))
        for t in range(H):
            S_all, Y, Pr = expected_step(D[None, :, :], S[:, None, :], p)   # (b, nD, ...)
            k = np.argmax((Y * w[:, None, :]).sum(-1), axis=1)
            r = np.arange(len(w))
            if t >= H0:
                acc += Pr[r, k]
            S = S_all[r, k]
        vals[i:i + batch] = acc / (H - H0); traj_d[i:i + batch] = D[k]
    return vals, traj_d

def warranted(p, eps=0.002, s0=S_INIT):
    D = decisions(); Wg = simplex()
    v, dec = value_of(Wg, p, D, s0)
    best = v.max(); ok = v >= best - eps * abs(best)
    S = Wg[ok]; th = S.mean(0)
    v_th, _ = value_of(th[None], p, D, s0)
    prof_only = v[np.argmax(Wg[:, 0])]
    return dict(theta=th.tolist(), set=S.tolist(), grid_values=v.tolist(), eps=eps, set_size=int(ok.sum()), W_star=float(best), W_theta=float(v_th[0]),
                W_profit_only=float(prof_only), W_equal=float(value_of(np.full((1, 4), .25), p, D, s0)[0][0]),
                spread=float(np.sqrt(((S - th) ** 2).sum(1)).mean()), best_policy=dec[np.argmax(v)].tolist())

if __name__ == "__main__":
    res = []
    for k, p in enumerate(regime_params()):
        r = warranted(p); r["regime"] = k; r["params"] = p; res.append(r)
        print(k, "theta", np.round(r["theta"], 3), "|Theta|", r["set_size"], "spread", round(r["spread"], 3),
              "W*", round(r["W_star"], 2), "W(theta)", round(r["W_theta"], 2), "W(profit-only)", round(r["W_profit_only"], 2),
              "W(equal)", round(r["W_equal"], 2), "policy", r["best_policy"], flush=True)
    json.dump(res, open("warranted_telos.json", "w"), indent=1)
