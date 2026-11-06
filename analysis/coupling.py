"""Step-level revision dynamics: dT_j = a (theta_j - T_j) + b (y_j - T_j) + e, by visibility and separation (Layer A main, sovereign).
Also the 'deficit' test: do revisers raise weights on dimensions where achieved performance is below the telos weight?"""
import json, glob, sys, numpy as np, pandas as pd
import statsmodels.formula.api as smf
R = sys.argv[1]; man = json.load(open(f"{R}/results/manifest.json")); rows = []
for eid, info in man["episodes"].items():
    c = info["spec"]["cfg"]
    if info["spec"]["study"] not in ("mainA", "frontA") or c["regime"] != "sovereign": continue
    for l in open(f"{R}/results/episodes/{eid}.jsonl", encoding="utf-8"):
        s = json.loads(l)
        if s["ev"] != "step": continue
        Tb, Ta, th, y = map(np.array, (s["telos_before"], s["telos_after"], s["theta"], s["y_norm"]))
        for j in range(4):
            rows.append(dict(eid=eid, study=info["spec"]["study"], vis=c["visibility"], sep=c["separation"], rat=c["ratifier"], council=c["apex"], t=s["t"], j=j,
                             dT=Ta[j] - Tb[j], gap_theta=th[j] - Tb[j], gap_y=y[j] - Tb[j]))
d = pd.DataFrame(rows); out = {}
for (st, vis), g in d.groupby(["study", "vis"]):
    m = smf.ols("dT ~ gap_theta + gap_y - 1", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.eid)[0]})
    out[f"{st}|{vis}"] = {k: dict(b=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in m.params.index} | {"n": int(len(g))}
g = d[d.study == "mainA"].copy(); g["shown"] = (g.vis == "shown").astype(float); g["same"] = (g.sep == "same").astype(float)
m = smf.ols("dT ~ gap_theta + gap_y + gap_y:shown + gap_y:shown:same + gap_y:same + gap_theta:shown - 1", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.eid)[0]})
out["mainA_interaction"] = {k: dict(b=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in m.params.index}
for sep in ["same", "distinct"]:
    for vis in ["shown", "hidden"]:
        h = g[(g.sep == sep) & (g.vis == vis)]
        mm = smf.ols("dT ~ gap_theta + gap_y - 1", h).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(h.eid)[0]})
        out[f"mainA|{sep}|{vis}"] = {k: dict(b=float(mm.params[k]), se=float(mm.bse[k]), p=float(mm.pvalues[k])) for k in mm.params.index}
json.dump(out, open("coupling.json", "w"), indent=1)
for k, v in out.items(): print(k, {a: (round(b["b"], 4), round(b["se"], 4), round(b["p"], 4)) for a, b in v.items() if isinstance(b, dict)})
