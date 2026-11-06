import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json,numpy as np,pandas as pd,statsmodels.formula.api as smf
R=DATA+"/results"; man=json.load(open(f"{R}/manifest.json")); rows=[]
for eid,info in man["episodes"].items():
    sp=info["spec"]; c=sp["cfg"]
    if c["regime"]!="sovereign": continue
    for l in open(f"{R}/episodes/{eid}.jsonl",encoding="utf-8"):
        s=json.loads(l)
        if s["ev"]!="step": continue
        n=len(s["telos_before"])
        for j in range(n): rows.append(dict(eid=eid,study=sp["study"],council=sp["council"],sep=c["separation"],vis=c["visibility"],rat=c["ratifier"],j=j,t=s["t"],dT=s["telos_after"][j]-s["telos_before"][j],T=s["telos_before"][j],th=s["theta"][j],y=s["y_norm"][j],T0=s["prior"][j]))
D=pd.DataFrame(rows)
def fit(d,form="dT~th+y+T+T0+C(j)"):
    m=smf.ols(form,d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d.eid)[0]})
    return {k:dict(b=float(m.params[k]),se=float(m.bse[k]),p=float(m.pvalues[k])) for k in ["th","y","T","T0"] if k in m.params}|{"n_ep":int(d.eid.nunique()),"n":len(d)}
out={}
A=D[D.study=="mainA"]
out["A_all"]=fit(A); out["A_all_noT0"]=fit(A,"dT~th+y+T+C(j)")
for sep in ["same","distinct"]:
    for vis in ["hidden","shown"]: out[f"A|{sep}|{vis}"]=fit(A[(A.sep==sep)&(A.vis==vis)])
for vis in ["hidden","shown"]: out[f"front|{vis}"]=fit(D[(D.study=="frontA")&(D.vis==vis)])
B=D[D.study=="mainB"]
for sep in ["same","distinct"]:
    for vis in ["hidden","shown"]: 
        d=B[(B.sep==sep)&(B.vis==vis)]
        out[f"B|{sep}|{vis}"]=fit(d) if d.dT.abs().sum()>0 else None
A2=A.copy(); A2["shown"]=(A2.vis=="shown").astype(float); A2["same"]=(A2.sep=="same").astype(float)
m=smf.ols("dT~th+y+T+T0+C(j)+y:shown+y:same+th:shown+th:same",A2).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(A2.eid)[0]})
out["A_interactions"]={k:dict(b=float(m.params[k]),se=float(m.bse[k]),p=float(m.pvalues[k])) for k in ["y:shown","y:same","th:shown","th:same"]}
json.dump(out,open("coupling2.json","w"),indent=1)
for k,v in out.items():
    print(k, None if v is None else {a:(round(b["b"],4),round(b["se"],4),round(b["p"],3)) for a,b in v.items() if isinstance(b,dict)})
