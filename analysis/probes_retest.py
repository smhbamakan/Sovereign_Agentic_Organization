import json,sys,numpy as np,pandas as pd
R=sys.argv[1]; out={}
L=lambda c: pd.DataFrame([json.loads(l) for l in open(f"{R}/results/probes/probes_{c}.jsonl",encoding="utf-8")]).drop_duplicates(["family","item"],keep="last")
H,F=L("heterogeneous"),L("frontier")
for fam in sorted(set(H.family)&set(F.family)):
    m=H[H.family==fam].merge(F[F.family==fam],on="item")
    b=m[m.type_x=="binary"]; ok=lambda a,k:(a.astype(str).str.strip().str.upper().str[:1]==k).astype(int)
    e1,e2=1-ok(b.answer_x,b.key_x),1-ok(b.answer_y,b.key_x); n=m[m.type_x=="numeric"]
    out[fam]=dict(n=len(b),phi_retest=float(np.corrcoef(e1,e2)[0,1]),rho_retest=float(np.corrcoef(n.estimate_x-n.truth_x,n.estimate_y-n.truth_x)[0,1]),agree=float((b.answer_x.astype(str).str[:1]==b.answer_y.astype(str).str[:1]).mean()))
json.dump(out,open("probes_retest.json","w"),indent=1); print(out)
