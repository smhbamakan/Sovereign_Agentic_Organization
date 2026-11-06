import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json,numpy as np,pandas as pd
def load(R):
    man=json.load(open(f"{R}/manifest.json")); it=[]
    for eid,info in man["episodes"].items():
        sp=info["spec"]; c=sp["cfg"]
        if c["ratifier"]=="off": continue
        for l in open(f"{R}/episodes/{eid}.jsonl",encoding="utf-8"):
            s=json.loads(l)
            if s["ev"]!="step": continue
            n=len(s["telos_before"]); dims=["profit","customer","employee","compliance"] if n==4 else ["financial","customer","employee","compliance","innovation"]
            for x in s.get("items") or []:
                if x.get("dimension") in dims:
                    j=dims.index(x["dimension"]); b=s["telos_before"][j]; th=s["theta"][j]
                    it.append(dict(eid=eid,key=f'{sp["study"]}|{sp["council"]}|{c["ratifier"]}',war=int(abs(float(x["proposed"])-th)<abs(b-th)),app=int(bool(x["approved"]))))
    return pd.DataFrame(it)
I=load(DATA+"/results"); rng=np.random.default_rng(7); out={}
for k,g in I.groupby("key"):
    obs=g[g.war==1].app.mean()-g[g.war==0].app.mean()
    # permute approvals within episode (preserves each episode's approval rate)
    gg=g.copy(); null=[]
    for _ in range(4000):
        gg["p"]=gg.groupby("eid").app.transform(lambda x: rng.permutation(x.values))
        null.append(gg[gg.war==1].p.mean()-gg[gg.war==0].p.mean())
    null=np.array(null); p=float((np.abs(null)>=abs(obs)-1e-12).mean())
    out[k]=dict(n=len(g),eps=int(g.eid.nunique()),RD=float(obs),p_perm=p); print(k,out[k])
json.dump(out,open("perm.json","w"),indent=1)
