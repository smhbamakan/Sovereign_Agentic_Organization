"""Environment-defined warranted telos for TelicFirm-B (same principle as Layer A).
W = mean expected quarterly free cash flow over quarters 8-40 of a regime, minus a liquidity penalty
(0.5 x mean shortfall of cash below zero), starting from the founding state. Each telos w on the
5-dimensional simplex (step 0.1; 1,001 vectors) induces a myopic management policy: each quarter the
firm adjusts last quarter's decisions by coordinate ascent on w . E[y_t] (two sweeps over eleven decisions,
nine levels each; borrowing fixed at zero). The procedure is deterministic; robustness is checked with
thirteen levels. Theta = teloi attaining >= 99% of W*."""
import json, sys, numpy as np
from envb import expected_step, BOUNDS, BASE_DEC, S0, regime_params_b

H0, H, NC = 8, 80, 3000
TAU = 0.005                                                # response temperature in outcome-profile units

def candidates(seed=20261103):
    rng = np.random.default_rng(seed); n = NC - 1
    u = (np.argsort(rng.random((n, 12)), 0) + rng.random((n, 12))) / n          # Latin hypercube
    C = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0]); C[:, 11] = 0.0
    B = BASE_DEC.copy(); B[11] = 0.0
    return np.vstack([B, C])

def simplex5(step=10):
    out = []
    for a in range(step + 1):
        for b in range(step + 1 - a):
            for c in range(step + 1 - a - b):
                for d in range(step + 1 - a - b - c):
                    out.append((a, b, c, d, step - a - b - c - d))
    return np.array(out, float) / step

def induced_decision(S, w, p, D0, sweeps=2, levels=9):
    """Myopic management heuristic: coordinate-wise quantal (softmax, temperature TAU) response to w . E[y_t] over the decisions (borrowing fixed),
    warm-started from last quarter's decision D0. Vectorized over a batch of teloi. Deterministic."""
    D = D0.copy(); n = len(D); grid = np.linspace(0, 1, levels)
    for _ in range(sweeps):
        for j in range(11):
            Cn = np.repeat(D[:, None, :], levels, 1)
            Cn[:, :, j] = BOUNDS[j, 0] + grid[None, :] * (BOUNDS[j, 1] - BOUNDS[j, 0])
            _, Y, _ = expected_step(Cn, S[:, None, :], p)
            sc = (Y * w[:, None, :]).sum(-1); z = np.exp((sc - sc.max(1, keepdims=True)) / TAU); z /= z.sum(1, keepdims=True)
            D = D.copy(); D[:, j] = (z * Cn[:, :, j]).sum(1)          # quantal (softmax) response: smooth, no tie artefacts
    return D

def value_of(Wt, p, C=None, batch=1001, levels=9):
    vals = np.empty(len(Wt)); last = np.empty((len(Wt), 12))
    for i in range(0, len(Wt), batch):
        w = Wt[i:i + batch]; S = np.tile(S0, (len(w), 1)); acc = np.zeros(len(w)); short = np.zeros(len(w))
        D = np.tile(BASE_DEC, (len(w), 1)); D[:, 11] = 0.0
        for t in range(H):
            D = induced_decision(S, w, p, D, levels=levels)
            S, _, info = expected_step(D, S, p)
            if t >= H0:
                # constitutive commitments as constraints: EUR 2m penalty per quarter in breach (BU1 quality floor, liquidity)
                breach = (S[:, 8] < 0.3) | (S[:, 0] < 0)
                acc += info["fcf"] - 2.0 * breach; short += np.maximum(0, -S[:, 0])
        vals[i:i + batch] = (acc - 0.5 * short) / (H - H0); last[i:i + batch] = D
    return vals, last

def warranted(p, C, Wg, eps=0.01, levels=9):
    v, last = value_of(Wg, p, levels=levels)
    best = v.max(); ok = v >= best - eps * abs(best); S = Wg[ok]; th = S.mean(0)
    one = lambda w: float(value_of(np.array([w], float), p, levels=levels)[0][0])
    return dict(theta=th.tolist(), set=S.tolist(), grid_values=v.tolist(), set_size=int(ok.sum()), W_star=float(best),
                W_theta=one(th), W_financial_only=one([1, 0, 0, 0, 0]), W_equal=one([.2] * 5),
                spread=float(np.sqrt(((S - th) ** 2).sum(1)).mean()), best_policy=last[np.argmax(v)].round(3).tolist())

def regret_lookup_b(T, regime, table):
    g = np.rint(np.asarray(T) * 10).astype(int); g[np.argmax(np.asarray(T))] += 10 - g.sum()
    r = table[regime]; return r["W_star"] - r["grid_values"][r["index"][tuple(g)]]

def load_table_b(path=None):
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "warranted_telos_b.json")
    W = json.load(open(path)); Wg = np.rint(simplex5() * 10).astype(int)
    for r in W: r["index"] = {tuple(x): i for i, x in enumerate(Wg)}
    return W

if __name__ == "__main__":
    levels = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    outp = sys.argv[2] if len(sys.argv) > 2 else "warranted_telos_b.json"
    C = None; Wg = simplex5(); res = []
    for k, p in enumerate(regime_params_b()):
        r = warranted(p, C, Wg, levels=levels); r["regime"] = k; r["params"] = p; res.append(r)
        print(k, "theta", np.round(r["theta"], 3), "|Theta|", r["set_size"], "spread", round(r["spread"], 3), "W*", round(r["W_star"], 2),
              "W(theta)", round(r["W_theta"], 2), "W(fin-only)", round(r["W_financial_only"], 2), "W(equal)", round(r["W_equal"], 2), flush=True)
    json.dump(res, open(outp, "w"), indent=1)
