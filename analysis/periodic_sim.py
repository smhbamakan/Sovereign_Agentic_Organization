"""Offline re-simulation of periodic human revision under less informed founders (Case A).
Value regret depends only on the telos trajectory, so periodic revision can be re-simulated exactly
with other founder noise levels, review periods and lags, using the logged shift schedules and priors."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))
import json, sys, numpy as np
sys.path[:0] = [str(ROOT/'runner'/'core')]
from warrant import load_table, regret_lookup
from analysis import windows
TAB = load_table()
R = DATA + "/results"; man = json.load(open(f"{R}/manifest.json"))
eps = [e for e, v in man["episodes"].items() if v["spec"]["study"] == "mainA" and v["spec"]["cfg"]["regime"] == "periodic"]
logs = {e: [json.loads(l) for l in open(f"{R}/episodes/{e}.jsonl", encoding="utf-8") if '"ev": "step"' in l] for e in eps}
def norm(v):
    v = np.clip(v, 1e-6, None); return v / v.sum()
def regret_episode(steps, telos):
    out = []
    for a, b, c in windows(steps):
        out.append(np.mean([regret_lookup(telos[i], steps[i]["regime"], TAB) for i in range(c, b)]))
    return float(np.mean(out))
def simulate(steps, sigma, P=8, lag=2, rng=None, uninformed=False):
    th = [np.array(s["theta"]) for s in steps]; T = np.array(steps[0]["prior"]); traj = []
    for t, s in enumerate(steps):
        if t % P == 0 and t >= lag:
            T = norm(rng.dirichlet(np.ones(4))) if uninformed else norm(th[t - lag] + rng.normal(0, sigma, 4))
        traj.append(T.copy())
    return regret_episode(steps, traj)
observed = float(np.mean([regret_episode(st, [np.array(s["telos_after"]) for s in st]) for st in logs.values()]))
rng = np.random.default_rng(5); res = dict(observed_periodic=observed, grid=[])
for P, lag in [(8, 2), (8, 4), (12, 2), (12, 4)]:
    for sigma in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        v = [simulate(st, sigma, P, lag, rng) for st in logs.values() for _ in range(100)]
        res["grid"].append(dict(P=P, lag=lag, sigma=sigma, regret=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v)))))
        print(P, lag, sigma, round(np.mean(v), 2), flush=True)
v = [simulate(st, 0, 8, 2, rng, uninformed=True) for st in logs.values() for _ in range(100)]
res["uninformed_P8"] = dict(regret=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v))))
print("observed", observed, "uninformed", res["uninformed_P8"])
json.dump(res, open("periodic_sim.json", "w"), indent=1)
