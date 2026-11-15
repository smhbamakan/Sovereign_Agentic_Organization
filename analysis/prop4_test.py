"""Test of Proposition 4a: non-discriminating ratification should slow adaptation without changing
stationary displacement. Compares ratification on/off within council in Case A sovereign episodes:
per-quarter revision activity, closing of the post-shift gap, and window-level displacement and misalignment."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))
import json, numpy as np, pandas as pd, statsmodels.formula.api as smf
R = DATA + "/results"; man = json.load(open(f"{R}/manifest.json")); rows, gaps = [], []
for e, v in man["episodes"].items():
    sp = v["spec"]; c = sp["cfg"]
    if sp["study"] != "mainA" or c["regime"] != "sovereign": continue
    st = [json.loads(l) for l in open(f"{R}/episodes/{e}.jsonl", encoding="utf-8") if '"ev": "step"' in l]
    T = np.array([s["telos_after"] for s in st]); Tb = np.array([s["telos_before"] for s in st]); th = np.array([s["theta"] for s in st])
    act = np.abs(T - Tb).sum(1).mean() / 2                            # mean total-variation change per quarter
    shifts = [s["t"] for s in st if s["shift"]]
    for a in shifts:
        if a + 6 < len(st):
            g0 = np.linalg.norm(T[a] - th[a]) if a == 0 else np.linalg.norm(T[a - 1] - th[a])   # gap when the regime changes
            g6 = np.linalg.norm(T[a + 6] - th[a + 6])
            gaps.append(dict(eid=e, rat=c["ratifier"], council=c["apex"], sep=c["separation"], vis=c["visibility"], closed=1 - g6 / g0))
    rows.append(dict(eid=e, rat=c["ratifier"], council=c["apex"], sep=c["separation"], vis=c["visibility"], activity=act))
A = pd.DataFrame(rows); G = pd.DataFrame(gaps); out = {}
for co in ["heterogeneous", "homogeneous"]:
    a = A[A.council == co]; g = G[G.council == co]
    m1 = smf.ols("activity ~ C(rat, Treatment('off')) + C(sep) + C(vis)", a).fit(cov_type="HC1")
    m2 = smf.ols("closed ~ C(rat, Treatment('off')) + C(sep) + C(vis)", g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.eid)[0]})
    k = "C(rat, Treatment('off'))[T.on]"
    out[co] = dict(activity_off=float(a[a.rat == "off"].activity.mean()), activity_on=float(a[a.rat == "on"].activity.mean()), activity_b=float(m1.params[k]), activity_p=float(m1.pvalues[k]),
                   closed_off=float(g[g.rat == "off"].closed.mean()), closed_on=float(g[g.rat == "on"].closed.mean()), closed_b=float(m2.params[k]), closed_p=float(m2.pvalues[k]))
    print(co, {x: round(y, 4) for x, y in out[co].items()})
json.dump(out, open("prop4_test.json", "w"), indent=1)
