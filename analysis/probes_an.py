import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json, sys, itertools, numpy as np, pandas as pd
R = sys.argv[1]; out = {}
items = {i["id"]: i for i in json.load(open(str(ROOT/'runner'/'core'/'probes_battery.json')))}
for council in ["heterogeneous", "frontier"]:
    rows = [json.loads(l) for l in open(f"{R}/results/probes/probes_{council}.jsonl", encoding="utf-8")]
    d = pd.DataFrame(rows)
    d = d.drop_duplicates(["family", "item"], keep="last")
    fams = sorted(d.family.unique()); res = {"models": {}}
    B = d[d.type == "binary"].copy(); B["err"] = (B.answer.astype(str).str.strip().str.upper().str[:1] != B.key).astype(float)
    N = d[d.type == "numeric"].copy(); N["e"] = N.estimate - N.truth
    for f in fams:
        b = B[B.family == f]; n = N[N.family == f].dropna(subset=["e"])
        res["models"][f] = dict(model=d[d.family == f].model.iloc[0], acc=float(1 - b.err.mean()), n_bin=int(len(b)),
                                missing_bin=int((~b.answer.astype(str).str.strip().str.upper().str[:1].isin(["A", "B"])).sum()),
                                mae=float(n.e.abs().mean()) if len(n) else None, n_num=int(len(n)), median_truth_abs=float(N.truth.abs().median()))
    pe = B.pivot(index="item", columns="family", values="err"); pn = N.pivot(index="item", columns="family", values="e")
    rng = np.random.default_rng(7); pairs = {}
    for a, b in itertools.combinations(fams, 2):
        x = pe[[a, b]].dropna(); y = pn[[a, b]].dropna()
        def phi(z): return float(np.corrcoef(z.iloc[:, 0], z.iloc[:, 1])[0, 1]) if z.iloc[:, 0].std() > 0 and z.iloc[:, 1].std() > 0 else np.nan
        boots = [phi(x.iloc[rng.integers(0, len(x), len(x))]) for _ in range(1000)]
        nb = [phi(y.iloc[rng.integers(0, len(y), len(y))]) for _ in range(1000)]
        pairs[f"{a}|{b}"] = dict(phi=phi(x), phi_lo=float(np.nanpercentile(boots, 2.5)), phi_hi=float(np.nanpercentile(boots, 97.5)), n_bin=len(x),
                                 both_wrong=float(((x.iloc[:, 0] == 1) & (x.iloc[:, 1] == 1)).mean()),
                                 rho_num=phi(y), rho_lo=float(np.nanpercentile(nb, 2.5)), rho_hi=float(np.nanpercentile(nb, 97.5)), n_num=len(y))
    res["pairs"] = pairs; out[council] = res
json.dump(out, open("probes.json", "w"), indent=1, default=float)
for c, r in out.items():
    print("==", c)
    for f, m in r["models"].items(): print(" ", f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    for p, v in r["pairs"].items(): print(" ", p, {k: round(v[k], 3) if isinstance(v[k], float) else v[k] for k in v})
