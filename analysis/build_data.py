"""Build window- and episode-level data frames from the main-study results."""
import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json, glob, os, sys, numpy as np, pandas as pd
R = sys.argv[1]
sys.path[:0] = [str(ROOT/'runner'/'core'), str(ROOT/'runner'/'core'/'layer_b')]
from analysis import episode_measures, exercised_authority
from warrant_b import load_table_b, regret_lookup_b
TB = load_table_b(); SB = [np.array(r["set"]) for r in TB]
man = json.load(open(f"{R}/results/manifest.json"))
W, EP, ITEMS = [], [], []
for eid, info in man["episodes"].items():
    if info["status"] != "done": continue
    if _os.environ.get("SAO_STUDIES") and info["spec"]["study"] not in _os.environ["SAO_STUDIES"].split(","): continue
    steps = [json.loads(l) for l in open(f"{R}/results/episodes/{eid}.jsonl", encoding="utf-8") if '"ev": "step"' in l]
    lay = info["spec"]["layer"]; c = info["spec"]["cfg"]
    m = episode_measures(steps) if lay == "A" else episode_measures(steps, sets=SB, regret=lambda T, k: regret_lookup_b(T, k, TB),
                                                                  dims=["financial", "customer", "employee", "compliance", "innovation"])
    study = info["spec"]["study"]
    base = dict(eid=eid, study=study, layer=lay, regime=c["regime"], sep=c["separation"], vis=c["visibility"], rat=c["ratifier"], council=c["apex"], seed=info["spec"]["seed"])
    vals = [s["out"]["financials"]["fcf"] if lay == "B" else s["out"]["profit"] for s in steps]
    viol = sum(bool(s["out"].get("constitutive_violation")) for s in steps)
    vt = {}
    for s in steps:
        for t in s["out"].get("violation_types", []) or []: vt[t] = vt.get(t, 0) + 1
    nchg = sum(np.abs(np.array(s["telos_after"]) - np.array(s["telos_before"])).sum() > 1e-9 for s in steps)
    EP.append(dict(base, gamma_hat=m.get("gamma_hat"), alpha_hat=m.get("alpha_hat"), r_hat=m.get("r_hat"), f_hat=m.get("f_hat"), n_items=m.get("n_items"),
                   realized=float(np.mean(vals)), violations=viol, vtypes=json.dumps(vt), proposed_viol=m.get("proposed_violations"), vetoes=m.get("critic_vetoes"),
                   telos_changes=int(nchg), prior=json.dumps(m["prior"]), telos_pre=json.dumps(m["telos_pre_shift"]), telos_end=json.dumps(m["telos_end"]),
                   verifiability=m.get("verifiability")))
    for k, w in enumerate(m["windows"]):
        W.append(dict(base, window=k + 1, **{x: w[x] for x in ("mse", "abs", "tv", "signed", "value_regret", "dist_to_set", "halflife_time", "halflife_event")}))
    for s in steps:
        for it in s.get("items", []) or []:
            ITEMS.append(dict(base, t=s["t"], regime_idx=s["regime"], **{k: it.get(k) for k in ("dimension", "proposed", "approved")},
                              before=s["telos_before"], theta=s["theta"]))
pd.DataFrame(W).to_csv("windows.csv", index=False); pd.DataFrame(EP).to_csv("episodes.csv", index=False)
pd.to_pickle(pd.DataFrame(ITEMS), "items.pkl")
print(len(W), "windows", len(EP), "episodes", len(ITEMS), "amendment items")
