"""Robustness of the warranted-telos benchmark (Case A).

1. Re-solves the shareholder-value warranted telos with an independent optimizer (finer simplex grid,
   step 1/30, plus 5,000 Dirichlet random-search points) and compares it with the stored theta.
2. Recomputes the warranted telos under two alternative, stakeholder-oriented criteria and re-runs the
   main analyses against them: revision dynamics (unrestricted specification), ratifier discrimination,
   end-of-episode misalignment by dimension and the ordering of governance arrangements.
Writes alt_criteria.json in the working directory."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))
import json, sys, time, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path[:0] = [str(ROOT/'runner'/'core')]
import warrant as WR
from env import expected_step
from analysis import episode_measures

CRIT = {  # per-quarter criterion added to expected profit, evaluated on next-quarter stocks
 "shareholder": lambda pr, S: pr,
 "stakeholder": lambda pr, S: pr + 10*((S[...,0]-.5) + (S[...,1]-.5) + (.5-S[...,2])),
 "sustainability": lambda pr, S: pr + 15*(S[...,1]-.5) + 15*(.5-S[...,2]),
}
D = WR.decisions(); P = WR.regime_params()

def value_of(Wt, p, crit, batch=150):
    vals = np.empty(len(Wt))
    for i in range(0, len(Wt), batch):
        w = Wt[i:i+batch]; S = np.tile(WR.S_INIT, (len(w), 1)); acc = np.zeros(len(w)); r = np.arange(len(w))
        for t in range(WR.H):
            S_all, Y, Pr = expected_step(D[None], S[:, None, :], p)
            k = np.argmax((Y * w[:, None, :]).sum(-1), axis=1)
            if t >= WR.H0: acc += CRIT[crit](Pr[r, k], S_all[r, k])
            S = S_all[r, k]
        vals[i:i+batch] = acc / (WR.H - WR.H0)
    return vals

def solve(Wg, p, crit, eps=0.002):
    v = value_of(Wg, p, crit); best = v.max(); ok = v >= best - eps*abs(best)
    return Wg[ok].mean(0), v, best

OUT = {"verification": [], "theta": {}}
G20 = WR.simplex(20); G40 = WR.simplex(30); rng = np.random.default_rng(11); RS = rng.dirichlet(np.ones(4), 5000)
stored = json.load(open(ROOT/'runner'/'core'/'warranted_telos.json'))
tables = {}
for crit in CRIT:
    th, tab = [], []
    for k, p in enumerate(P):
        t0 = time.time(); t20, v20, b20 = solve(G20, p, crit); th.append(t20)
        idx = {tuple(x): i for i, x in enumerate(np.rint(G20*20).astype(int))}
        tab.append(dict(W_star=b20, grid_values=v20, index=idx))
        if crit == "shareholder":
            t40, v40, b40 = solve(G40, p, crit); vr = value_of(RS, p, crit); brs = vr.max()
            OUT["verification"].append(dict(regime=k, theta_stored=stored[k]["theta"], theta_grid20=t20.tolist(), theta_grid30=t40.tolist(),
                dist_stored_vs_grid30=float(np.linalg.norm(np.array(stored[k]["theta"]) - t40)), W_star_grid20=float(b20), W_star_grid30=float(b40),
                W_star_random_search=float(brs), theta_random_search_best=RS[vr.argmax()].tolist()))
        print(crit, k, np.round(t20, 3), f"{time.time()-t0:.0f}s", flush=True)
    OUT["theta"][crit] = [t.tolist() for t in th]; tables[crit] = tab

def regret(T, k, crit):
    g = np.rint(np.asarray(T)*20).astype(int); g[np.argmax(np.asarray(T))] += 20 - g.sum()
    tb = tables[crit][k]; return tb["W_star"] - tb["grid_values"][tb["index"][tuple(g)]]

# ---------------- re-run analyses on the logged Case A episodes
R = DATA + "/results"; man = json.load(open(f"{R}/manifest.json"))
eps = {e: v for e, v in man["episodes"].items() if v["spec"]["study"] == "mainA" and v["status"] == "done"}
logs = {e: [json.loads(l) for l in open(f"{R}/episodes/{e}.jsonl", encoding="utf-8") if '"ev": "step"' in l] for e in eps}
res = {}
for crit in CRIT:
    TH = [np.array(t) for t in OUT["theta"][crit]]; sets = [t[None] for t in TH]
    W, rows, items = [], [], []
    for e, info in eps.items():
        c = info["spec"]["cfg"]; steps = []
        for s in logs[e]:
            s2 = dict(s); s2["theta"] = TH[s["regime"]].tolist(); steps.append(s2)
        m = episode_measures(steps, sets=sets, regret=lambda T, k: regret(T, k, crit))
        arr = c["regime"] if c["regime"] != "sovereign" else ("steward" if c["separation"] == "distinct" else "executive")
        for w in m["windows"]:
            W.append(dict(eid=e, arr=arr, sep=c["separation"], vis=c["visibility"], rat=c["ratifier"], council=c["apex"], regret=w["value_regret"], mse=w["mse"], signed=w["signed"]))
        if c["regime"] == "sovereign":
            for s in steps:
                for j in range(4):
                    rows.append(dict(eid=e, vis=c["visibility"], j=j, t=s["t"], dT=s["telos_after"][j]-s["telos_before"][j], T=s["telos_before"][j], th=s["theta"][j], y=s["y_norm"][j], T0=s["prior"][j]))
                for it in s.get("items") or []:
                    j = ["profit","customer","employee","compliance"].index(it["dimension"]); b = s["telos_before"][j]; t_ = s["theta"][j]
                    items.append(dict(eid=e, council=c["apex"], war=int(abs(float(it["proposed"])-t_) < abs(b-t_)), app=int(bool(it["approved"]))))
    Wd = pd.DataFrame(W); Dd = pd.DataFrame(rows); Id = pd.DataFrame(items)
    fit = lambda d, f: smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.eid)[0]})
    m = fit(Dd, "dT~th+y+T+T0+C(j)")
    r = dict(coupling={k: dict(b=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in ["th", "y", "T", "T0"]})
    Dd["shown"] = (Dd.vis == "shown").astype(float); mi = fit(Dd, "dT~th+y+T+T0+C(j)+y:shown")
    r["y_x_shown"] = dict(b=float(mi.params["y:shown"]), p=float(mi.pvalues["y:shown"]))
    ep = Wd.groupby(["arr", "eid"])[["regret", "mse"]].mean().groupby("arr").agg(["mean", "sem"])
    r["arrangements"] = {a: dict(regret=float(ep.loc[a, ("regret", "mean")]), regret_se=float(ep.loc[a, ("regret", "sem")]), mse=float(ep.loc[a, ("mse", "mean")])) for a in ep.index}
    rd = {}
    rng2 = np.random.default_rng(3)
    for co, g in Id[Id.eid.map(lambda e: eps[e]["spec"]["cfg"]["ratifier"] == "on")].groupby("council"):
        obs = g[g.war == 1].app.mean() - g[g.war == 0].app.mean(); gg = g.copy(); null = []
        for _ in range(2000):
            gg["p"] = gg.groupby("eid").app.transform(lambda x: rng2.permutation(x.values)); null.append(gg[gg.war == 1].p.mean() - gg[gg.war == 0].p.mean())
        rd[co] = dict(n=len(g), share_toward=float(g.war.mean()), app_toward=float(g[g.war == 1].app.mean()), app_away=float(g[g.war == 0].app.mean()), p_perm=float((np.abs(null) >= abs(obs)).mean()))
    r["ratifier"] = rd
    sov = Wd[Wd.arr.isin(["executive", "steward"])].copy()
    for f in ["sep", "vis", "rat", "council"]: sov[f + "_x"] = pd.factorize(sov[f], sort=True)[0] - .5
    mm = smf.mixedlm("signed ~ sep_x*vis_x + rat_x + council_x", sov, groups=sov.eid).fit()
    r["signed_vis"] = dict(b=float(mm.params["vis_x"]), p=float(mm.pvalues["vis_x"]))
    end = Dd[Dd.t == Dd.t.max()].groupby("j").apply(lambda g: float((g.th - g["T"]).mean())).to_dict()
    r["theta_minus_T_end"] = {["profit","customer","employee","compliance"][k]: v for k, v in end.items()}
    res[crit] = r; print(crit, json.dumps({k: r[k] for k in ["coupling", "y_x_shown", "signed_vis"]})[:600], flush=True)
OUT["results"] = res
json.dump(OUT, open("alt_criteria.json", "w"), indent=1, default=float)
