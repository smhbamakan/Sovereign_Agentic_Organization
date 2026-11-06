import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json,numpy as np,pandas as pd,statsmodels.formula.api as smf
exec(open("review_an.py").read().split("# shock timing")[0].split("# ---------- step data")[1].replace("st=[]","st=[]",1)) if False else None
R=DATA+"/results"; man=json.load(open(f"{R}/manifest.json")); rows=[]
for eid,info in man["episodes"].items():
    sp=info["spec"]; c=sp["cfg"]
    if sp["study"]!="mainA" or c["regime"]!="sovereign": continue
    for l in open(f"{R}/episodes/{eid}.jsonl",encoding="utf-8"):
        s=json.loads(l)
        if s["ev"]!="step": continue
        for j in range(4): rows.append(dict(eid=eid,vis=c["visibility"],j=j,t=s["t"],dT=s["telos_after"][j]-s["telos_before"][j],T=s["telos_before"][j],th=s["theta"][j],y=s["y_norm"][j]))
D=pd.DataFrame(rows); D["gt"]=D.th-D["T"]; D["gy"]=D.y-D["T"]
D["ej"]=D.eid+D.j.astype(str)
f=lambda form,d=D: smf.ols(form,d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d.eid)[0]})
out={}
for name,form in [("restricted","dT~gt+gy-1"),("free_levels","dT~th+y+T"),("free_levels_dimFE","dT~th+y+T+C(j)"),("ep_dim_FE","dT~th+y+T+C(ej)")]:
    m=f(form); out[name]={k:(round(m.params[k],4),round(m.bse[k],4),round(m.pvalues[k],4)) for k in m.params.index if k in("gt","gy","th","y","T")}
# y deviation from its dimension-episode mean (transient) vs persistent component
D["y_bar"]=D.groupby(["eid","j"]).y.transform("mean"); D["y_dev"]=D.y-D.y_bar
m=f("dT~th+y_bar+y_dev+T"); out["persistent_vs_transient"]={k:(round(m.params[k],4),round(m.bse[k],4),round(m.pvalues[k],4)) for k in ["th","y_bar","y_dev","T"]}
# telos-level relation: where does telos end up? long-run regression of end telos on mean y and mean theta per (eid,j)
L=D.groupby(["eid","j"]).agg(Tend=("T","last"),ybar=("y","mean"),thbar=("th","mean"),T0=("T","first")).reset_index()
m=smf.ols("Tend~ybar+thbar+T0+C(j)",L).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(L.eid)[0]}); out["longrun_dimFE"]={k:(round(m.params[k],3),round(m.bse[k],3),round(m.pvalues[k],4)) for k in ["ybar","thbar","T0"]}
m=smf.ols("Tend~ybar+thbar+T0",L).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(L.eid)[0]}); out["longrun"]={k:(round(m.params[k],3),round(m.bse[k],3),round(m.pvalues[k],4)) for k in ["ybar","thbar","T0"]}
# mean achievability pull by dimension and displacement
out["mu_by_dim"]=D.groupby("j").apply(lambda g:(g.y-g["T"]).mean()).round(3).to_dict()
out["theta_minus_T_by_dim_end"]=D[D.t==39].groupby("j").apply(lambda g:(g.th-g["T"]).mean()).round(3).to_dict()
print(json.dumps(out,indent=0,default=float)); json.dump(out,open("spec_an.json","w"),default=float)
