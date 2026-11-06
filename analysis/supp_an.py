import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json,numpy as np,pandas as pd,statsmodels.formula.api as smf,statsmodels.api as sm
from sklearn.metrics import roc_auc_score
R=DATA+"/results"; man=json.load(open(f"{R}/manifest.json")); OUT={}
rows=[];items=[]
for eid,info in man["episodes"].items():
    sp=info["spec"]; c=sp["cfg"]
    for l in open(f"{R}/episodes/{eid}.jsonl",encoding="utf-8"):
        s=json.loads(l)
        if s["ev"]!="step": continue
        n=len(s["telos_before"]); dims=["profit","customer","employee","compliance"] if n==4 else ["financial","customer","employee","compliance","innovation"]
        dl=s.get("delta")
        for j in range(n): rows.append(dict(eid=eid,study=sp["study"],layer=sp["layer"],council=sp["council"],regime=c["regime"],sep=c["separation"],vis=c["visibility"],rat=c["ratifier"],j=j,t=s["t"],
            dT=s["telos_after"][j]-s["telos_before"][j],T=s["telos_before"][j],th=s["theta"][j],y=s["y_norm"][j],T0=s["prior"][j],delta=(dl[dims[j]] if dl else np.nan),yraw=(s["out"]["y"][dims[j]] if s["out"].get("y") and dims[j] in s["out"]["y"] else np.nan)))
        for x in s.get("items") or []:
            if x.get("dimension") in dims:
                j=dims.index(x["dimension"]); b=s["telos_before"][j]; th=s["theta"][j]; p=float(x["proposed"])
                items.append(dict(eid=eid,study=sp["study"],council=sp["council"],rat=c["ratifier"],war=int(abs(p-th)<abs(b-th)),app=int(bool(x["approved"]))))
D=pd.DataFrame(rows); I=pd.DataFrame(items)
def fit(d,form):
    m=smf.ols(form,d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d.eid)[0]})
    return {k:dict(b=float(m.params[k]),se=float(m.bse[k]),p=float(m.pvalues[k])) for k in m.params.index if not k.startswith("C(")}|dict(n_ep=int(d.eid.nunique()))
# S1 perturbation: displayed-profile perturbation delta (dimension-level), controls
Pd=D[(D.study=="supA_perturb")&(D.vis=="perturbed")].copy()
Pd["dc"]=Pd.delta-Pd.groupby(["eid","t"]).delta.transform("mean")   # zero-sum component (relative perturbation)
OUT["S1_perturb_all"]=fit(Pd,"dT~dc+th+y+T+T0+C(j)")
for sep in ["same","distinct"]: OUT[f"S1_perturb_{sep}"]=fit(Pd[Pd.sep==sep],"dT~dc+th+y+T+T0+C(j)")
# displayed profile coefficient directly: y_shown = y + delta
Pd["ys"]=Pd.y+Pd.dc
OUT["S1_yshown"]=fit(Pd,"dT~ys+th+y+T+T0+C(j)")
for sep in ["same","distinct"]:
    OUT[f"S1_ablated_{sep}"]=fit(D[(D.study=="supA_perturb")&(D.vis=="ablated")&(D.sep==sep)],"dT~th+y+T+T0+C(j)")
# revision frequency
chg=D.groupby(["eid","study","sep","vis","rat","council","t"]).dT.apply(lambda x:(x.abs()>1e-9).any()).groupby(["study","sep","vis","rat","council"]).mean()
OUT["change_rate"]={"|".join(map(str,k)):float(v) for k,v in chg.items()}
# ratifier discrimination in supplementary councils
res={}
for (stu,co,rt),g in I[I.rat!="off"].groupby(["study","council","rat"]):
    r=dict(n=len(g),app_w=float(g[g.war==1].app.mean()),app_u=float(g[g.war==0].app.mean()))
    if g.app.nunique()>1 and g.war.nunique()>1:
        try:
            m=smf.gee("app~war",groups="eid",data=g,family=sm.families.Binomial()).fit(); lo,hi=m.conf_int().loc["war"]
            r.update(OR=float(np.exp(m.params["war"])),OR_lo=float(np.exp(lo)),OR_hi=float(np.exp(hi)),AUC=float(roc_auc_score(g.war,g.app)))
        except Exception as e: r["err"]=str(e)
    res[f"{stu}|{co}|{rt}"]=r
OUT["ratifier"]=res
# window outcomes: compare value regret
w=pd.read_csv("windows.csv"); e=pd.read_csv("episodes.csv")
A=w[(w.layer=="A")]
def vr(q): g=A.query(q).groupby("eid").value_regret.mean(); return dict(m=float(g.mean()),se=float(g.std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else None,n=int(len(g)))
OUT["vr"]={
 "het_luna_distinct_raton":vr("study=='mainA' and council=='heterogeneous' and sep=='distinct' and rat=='on' and regime=='sovereign'"),
 "homog_gemini_distinct_raton":vr("study=='mainA' and council=='homogeneous' and sep=='distinct' and rat=='on' and regime=='sovereign'"),
 "homog_gemini_distinct_ratoff":vr("study=='mainA' and council=='homogeneous' and sep=='distinct' and rat=='off' and regime=='sovereign'"),
 "commonrat_gemini_rev_luna_rat":vr("study=='supA_commonrat'"),
 "random25":vr("study=='supA_random' and rat=='random25'"),"random90":vr("study=='supA_random' and rat=='random90'"),
 "perturbed":vr("study=='supA_perturb' and vis=='perturbed'"),"ablated":vr("study=='supA_perturb' and vis=='ablated'"),
 "het_ratoff_hidden":vr("study=='mainA' and council=='heterogeneous' and rat=='off' and vis=='hidden' and regime=='sovereign'"),
 "het_ratoff_shown":vr("study=='mainA' and council=='heterogeneous' and rat=='off' and vis=='shown' and regime=='sovereign'"),
 "claude_exec":vr("study=='supA_claude'"),"gemini_exec_same_het_raton":vr("study=='mainA' and council=='heterogeneous' and sep=='same' and rat=='on' and regime=='sovereign'")}
def rl(q):
    g=e.query(q); return dict(m=float(g.realized.mean()),se=float(g.realized.std(ddof=1)/np.sqrt(len(g))) if len(g)>1 else None,n=int(len(g)),breach=float(g.violations.mean()),chg=float(g.telos_changes.mean()))
OUT["realized"]={
 "A_gemini_sov_all":rl("study=='mainA' and regime=='sovereign'"),"A_gemini_frozen":rl("study=='mainA' and regime=='frozen'"),"A_gemini_periodic":rl("study=='mainA' and regime=='periodic'"),
 "A_gemini_oracle":rl("study=='supA_random' and regime=='oracle' and council=='homogeneous'"),"A_claude_oracle":rl("study=='supA_random' and regime=='oracle' and council=='claude_exec_A'"),
 "A_claude_sov":rl("study=='supA_claude'"),"A_gemini_same_het_raton":rl("study=='mainA' and council=='heterogeneous' and sep=='same' and rat=='on'"),"A_frontier":rl("study=='frontA'"),
 "B_claude_frozen_main":rl("study=='mainB' and regime=='frozen'"),"B_claude_frozen_rep":rl("study=='supB_gemini' and regime=='frozen' and council=='claude_ceo'"),
 "B_claude_oracle":rl("study=='supB_gemini' and regime=='oracle' and council=='claude_ceo'"),"B_claude_periodic":rl("study=='mainB' and regime=='periodic'"),
 "B_claude_exec":rl("study=='mainB' and regime=='sovereign' and sep=='same'"),"B_claude_steward":rl("study=='mainB' and regime=='sovereign' and sep=='distinct'"),
 "B_claude_exec_lunacritic":rl("study=='mainB' and regime=='sovereign' and sep=='same' and council=='claude_ceo'"),"B_claude_steward_lunacritic":rl("study=='mainB' and regime=='sovereign' and sep=='distinct' and council=='claude_ceo'"),
 "B_gemini_frozen":rl("study=='supB_gemini' and regime=='frozen' and council=='gemini_ceo'"),"B_gemini_oracle":rl("study=='supB_gemini' and regime=='oracle' and council=='gemini_ceo'"),
 "B_gemini_exec":rl("study=='supB_gemini' and regime=='sovereign' and sep=='same'"),"B_gemini_steward":rl("study=='supB_gemini' and regime=='sovereign' and sep=='distinct'")}
# executive-model effect in Case A (realized profit), Claude vs Gemini under same council/cell (same, raton, het-like reviser Doubao/Luna)
a=e[(e.study.isin(["supA_claude"]))|((e.study=="mainA")&(e.council=="heterogeneous")&(e.sep=="same")&(e.rat=="on"))].copy(); a["claude"]=(a.study=="supA_claude").astype(int)
m=smf.ols("realized~claude+C(vis)",a).fit(cov_type="HC1"); OUT["A_exec_effect"]=dict(b=float(m.params["claude"]),se=float(m.bse["claude"]),p=float(m.pvalues["claude"]))
# persona probes: within-family cross-role phi
def load(f): 
    d=pd.DataFrame([json.loads(l) for l in open(f"{R}/probes/{f}",encoding="utf-8")]); return d
H=load("probes_heterogeneous.jsonl"); per={}
for fam,f in [("joy_gemini_lite","probes_persona_gemini.jsonl"),("joy_doubao_pro","probes_persona_doubao.jsonl"),("joy_gpt_luna","probes_persona_luna.jsonl")]:
    Q=load(f); a1=H[(H.family==fam)].drop_duplicates("item",keep="first"); a2=Q[Q.family==fam].drop_duplicates("item",keep="first")
    bb=a1[a1.type=="binary"].merge(a2[a2.type=="binary"],on="item")
    ok1=(bb.answer_x.astype(str).str.strip().str.upper().str[:1]==bb.key_x).astype(int); ok2=(bb.answer_y.astype(str).str.strip().str.upper().str[:1]==bb.key_x).astype(int)
    nn=a1[a1.type=="numeric"].merge(a2[a2.type=="numeric"],on="item"); e1=nn.estimate_x-nn.truth_x; e2=nn.estimate_y-nn.truth_x
    per[fam]=dict(n_bin=len(bb),phi=float(np.corrcoef(1-ok1,1-ok2)[0,1]),acc_default=float(ok1.mean()),acc_persona=float(ok2.mean()),agree=float((bb.answer_x.astype(str).str[:1]==bb.answer_y.astype(str).str[:1]).mean()),n_num=len(nn),rho=float(np.corrcoef(e1,e2)[0,1]))
OUT["persona"]=per
json.dump(OUT,open("supp.json","w"),indent=1,default=float)
for k,v in OUT.items(): print(k,json.dumps(v,default=float)[:1500])
