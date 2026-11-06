import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json, sys, itertools, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
R=DATA+"/results"; OUT={}
man=json.load(open(f"{R}/manifest.json"))
MAIN={"mainA","frontA","mainB"}; man["episodes"]={k:v for k,v in man["episodes"].items() if v["spec"]["study"] in MAIN}
# ---------- accounting
rows=[]
for eid,info in man["episodes"].items():
    sp=info["spec"]; rows.append(dict(eid=eid,study=sp["study"],layer=sp["layer"],regime=sp["cfg"]["regime"],council=sp["council"],status=info.get("status")))
E=pd.DataFrame(rows); OUT["accounting"]=E.groupby(["study","regime","council"]).size().reset_index(name="n").to_dict("records")
OUT["status"]=E.status.value_counts().to_dict()
calls=[]
import glob,os
for f in glob.glob(f"{R}/calls/*.jsonl"):
    if not (os.path.basename(f).split("_")[0] in MAIN or os.path.basename(f) in ("probes_heterogeneous.jsonl","probes_frontier.jsonl")): continue
    for l in open(f,encoding="utf-8"):
        try: j=json.loads(l)
        except: continue
        calls.append(dict(file=os.path.basename(f),role=j.get("role") or j.get("ctx",{}).get("role"),probe=bool((j.get("ctx") or {}).get("probe")),ok=j.get("parse_ok",j.get("ok"))))
CC=pd.DataFrame(calls); OUT["n_calls"]=len(CC); OUT["calls_by_role"]=CC.role.value_counts(dropna=False).to_dict(); OUT["probe_calls"]=int(CC.probe.sum())
OUT["calls_by_layer"]={k:int(v) for k,v in CC.file.str.extract(r"_(A|B)_")[0].value_counts().items()}
# ---------- step data
st=[]
for eid,info in man["episodes"].items():
    sp=info["spec"]; c=sp["cfg"]
    for l in open(f"{R}/episodes/{eid}.jsonl",encoding="utf-8"):
        s=json.loads(l)
        if s["ev"]=="start": shifts=s.get("shift_times"); continue
        st.append(dict(eid=eid,study=sp["study"],layer=sp["layer"],regime=c["regime"],sep=c["separation"],vis=c["visibility"],rat=c["ratifier"],council=sp["council"],t=s["t"],
                       Tb=s["telos_before"],Ta=s["telos_after"],th=s["theta"],y=s["y_norm"],items=s.get("items",[]),prior=s["prior"],shifts=shifts))
S=pd.DataFrame(st)
# shock timing
sh=S.groupby("eid").first()
OUT["shift_times"]={lay:pd.DataFrame([x if isinstance(x,list) else [] for x in sh[sh.layer==lay].shifts]).describe().round(1).to_dict() for lay in ["A","B"]}
# ---------- VIF / correlation of revision regressors
rows=[]
for _,s in S[(S.regime=="sovereign")&(S.study=="mainA")].iterrows():
    for j in range(4):
        rows.append(dict(eid=s.eid,dT=s.Ta[j]-s.Tb[j],gt=s.th[j]-s.Tb[j],gy=s.y[j]-s.Tb[j],gty=s.th[j]-s.y[j],T=s.Tb[j],t=s.t,j=j,lag_gy=None))
D=pd.DataFrame(rows); X=sm.add_constant(D[["gt","gy"]])
OUT["gap_corr"]=float(D["gt"].corr(D["gy"])); OUT["vif"]={c:float(vif(X.values,i)) for i,c in enumerate(X.columns)}
# reparameterized: dT on (theta - y) and (y - T) separately -> avoids common -T; and placebo: future gaps
g=lambda f,d: smf.ols(f,d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d.eid)[0]})
m=g("dT ~ gt + gy - 1",D); OUT["base"]={k:[float(m.params[k]),float(m.bse[k])] for k in m.params.index}
m=g("dT ~ gt + gy + T + C(j) - 1",D); OUT["with_level_FE"]={k:[float(m.params[k]),float(m.bse[k]),float(m.pvalues[k])] for k in ["gt","gy","T"]}
D=D.sort_values(["eid","j","t"]); D["gy_lead"]=D.groupby(["eid","j"]).gy.shift(-2); D["gt_lead"]=D.groupby(["eid","j"]).gt.shift(-2)
D["dT_lead"]=D.groupby(["eid","j"]).dT.shift(-1); D["gy_lag"]=D.groupby(["eid","j"]).gy.shift(1)
d2=D.dropna(subset=["gy_lag"]); m=g("dT ~ gt + gy + gy_lag - 1",d2); OUT["with_lag"]={k:[float(m.params[k]),float(m.bse[k]),float(m.pvalues[k])] for k in m.params.index}
# placebo: regress dT on own-outcome gap observed AFTER revision (t+2) controlling current gaps
d3=D.dropna(subset=["gy_lead"]); m=g("dT ~ gt + gy + gy_lead - 1",d3); OUT["placebo_lead"]={k:[float(m.params[k]),float(m.bse[k]),float(m.pvalues[k])] for k in m.params.index}
# mean-reversion check: change in T should not track noise in y when y is pure noise -> simulate model-based recovery
rng=np.random.default_rng(1); rec=[]
for rep in range(50):
    rr=[]
    for e in range(96):
        T=np.full(4,.25)+rng.normal(0,.03,4); T/=T.sum(); th=np.array([.3,.25,.2,.25])
        for t in range(40):
            if t in (10,22,34): th=rng.dirichlet(np.ones(4)*20)
            y=T+np.array([.05,-.02,-.02,-.01])+rng.normal(0,.08,4)
            gam=0.6; a=0.05; target=(1-gam)*(th+rng.normal(0,.05,4))+gam*y; Tn=(1-a)*T+a*target; Tn=np.clip(Tn,1e-4,1); Tn/=Tn.sum()
            for j in range(4): rr.append((e,Tn[j]-T[j],th[j]-T[j],y[j]-T[j]))
            T=Tn
    dd=pd.DataFrame(rr,columns=["eid","dT","gt","gy"]); mm=smf.ols("dT~gt+gy-1",dd).fit()
    rec.append([mm.params["gt"],mm.params["gy"]])
rec=np.array(rec); OUT["recovery"]=dict(true_bt=0.05*0.4,true_by=0.05*0.6,est_bt=rec[:,0].mean(),est_by=rec[:,1].mean(),sd_bt=rec[:,0].std(),sd_by=rec[:,1].std())
# ---------- beta(tau)
bt={}
for tau in [0,4,8,12,20,30,39]:
    a=S[(S.study=="mainA")&(S.regime=="sovereign")&(S.t==tau)]
    P=np.array(a.prior.tolist()); T=np.array(a.Ta.tolist()); P=P-P.mean(0); T=T-T.mean(0)
    b=float((P*T).sum()/(P*P).sum())
    bs=[]
    for _ in range(500):
        ix=np.random.default_rng(_).integers(0,len(P),len(P)); p=P[ix]-P[ix].mean(0); q=T[ix]-T[ix].mean(0); bs.append((p*q).sum()/(p*p).sum())
    bt[tau]=[b,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
OUT["beta_tau"]=bt
# ---------- ratifier discrimination A and B, with value-based label via proxy distance
it=[]
for _,s in S[(S.rat=="on")&(S.regime=="sovereign")].iterrows():
    n=len(s.Tb)
    dims=["profit","customer","employee","compliance"] if n==4 else ["financial","customer","employee","compliance","innovation"]
    for x in s["items"]:
        if x.get("dimension") not in dims: continue
        j=dims.index(x["dimension"]); b=s.Tb[j]; th=s.th[j]; p=float(x["proposed"])
        it.append(dict(eid=s.eid,study=s.study,council=s.council,t=s.t,war=int(abs(p-th)<abs(b-th)),gain=abs(b-th)-abs(p-th),size=abs(p-b),app=int(bool(x["approved"]))))
I=pd.DataFrame(it); res={}
for (stu,co),gg in I.groupby(["study","council"]):
    r=dict(n=len(gg),app_w=float(gg[gg.war==1].app.mean()),app_u=float(gg[gg.war==0].app.mean()),n_w=int(gg.war.sum()))
    try:
        m=smf.gee("app~war",groups="eid",data=gg,family=sm.families.Binomial()).fit(); lo,hi=m.conf_int().loc["war"]
        r.update(OR=float(np.exp(m.params["war"])),OR_lo=float(np.exp(lo)),OR_hi=float(np.exp(hi)),p=float(m.pvalues["war"]))
        r["AUC"]=float(roc_auc_score(gg.war,gg.app)) if gg.app.nunique()>1 else None
        # risk difference with cluster bootstrap
        eids=gg.eid.unique(); rd=[]
        for k in range(1000):
            ss=pd.concat([gg[gg.eid==e] for e in np.random.default_rng(k).choice(eids,len(eids))]); rd.append(ss[ss.war==1].app.mean()-ss[ss.war==0].app.mean())
        r.update(RD=r["app_w"]-r["app_u"],RD_lo=float(np.nanpercentile(rd,2.5)),RD_hi=float(np.nanpercentile(rd,97.5)))
    except Exception as e: r["err"]=str(e)
    res[f"{stu}|{co}"]=r
OUT["ratifier"]=res
# ---------- numeric covariance floor
items={i["id"]:i for i in json.load(open(str(ROOT/'runner'/'core'/'probes_battery.json')))}
d=pd.DataFrame([json.loads(l) for l in open(f"{R}/probes/probes_heterogeneous.jsonl",encoding="utf-8")])
N=d[d.type=="numeric"].copy(); N["e"]=N.estimate-N.truth
first=N.drop_duplicates(["family","item"],keep="first"); last=N.drop_duplicates(["family","item"],keep="last")
pv=first.pivot(index="item",columns="family",values="e").dropna()
# standardize by item scale (median abs truth) to avoid scale domination: use relative error
scale=first.groupby("item").truth.first().abs().replace(0,1)
pvr=pv.div(scale.loc[pv.index],axis=0)
Sig=np.cov(pvr.values.T); w=np.ones(3)/3
OUT["numeric_cov"]=dict(fams=list(pv.columns),corr=np.corrcoef(pvr.values.T).round(3).tolist(),var_ratio_mean=float(w@Sig@w/np.mean(np.diag(Sig))),
                       var_ratio_best=float(w@Sig@w/np.min(np.diag(Sig))),n=len(pvr))
# binary majority vote of the 3 heterogeneous models
B=d[d.type=="binary"].drop_duplicates(["family","item"],keep="first").copy(); B["ok"]=(B.answer.astype(str).str.strip().str.upper().str[:1]==B.key).astype(int)
pb=B.pivot(index="item",columns="family",values="ok").dropna()
maj=(pb.sum(1)>=2).mean(); OUT["majority"]=dict(acc_individual=pb.mean().round(3).to_dict(),acc_majority=float(maj),acc_if_independent=float(
    sum(np.prod([p if k else 1-p for p,k in zip(pb.mean(),c)]) for c in itertools.product([0,1],repeat=3) if sum(c)>=2)))
json.dump(OUT,open("review_an.json","w"),indent=1,default=float)
for k in ["status","n_calls","calls_by_role","probe_calls","calls_by_layer","gap_corr","vif","base","with_level_FE","with_lag","placebo_lead","recovery","beta_tau","ratifier","numeric_cov","majority"]:
    print(k,json.dumps(OUT[k],default=float)[:900])
