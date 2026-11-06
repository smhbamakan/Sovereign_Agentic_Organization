import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open(ROOT/"analysis"/"figs_pub.py").read().split("O=")[0])
O=str(ROOT/"figures")+"/"; K="#000"
S=json.load(open(ROOT/"outputs"/"followup"/"supp.json")); V=S["vr"]; R=S["realized"]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),gridspec_kw=dict(width_ratios=[1.15,1,1]))
a=ax[0]; labs=["Gemini\n(91%)","Random\n(90%)","GPT Luna\n(24%)","Random\n(25%)"]; ks=["homog_gemini_distinct_raton","random90","commonrat_gemini_rev_luna_rat","random25"]
cols=["#2a78d6","#9ecbf2","#eb6834","#f6c3a6"]
for i,k in enumerate(ks): a.bar(i,V[k]["m"],.65,color=cols[i],edgecolor=K,lw=.6,yerr=1.96*V[k]["se"],capsize=2.5,error_kw=dict(lw=.7))
a.axhline(V["homog_gemini_distinct_ratoff"]["m"],color=K,ls=":",lw=.8); a.text(3.45,V["homog_gemini_distinct_ratoff"]["m"]+.2,"no ratifier",ha="right",fontsize=7)
a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=7); a.set_ylabel("Value regret (Case A)"); a.set_title("Ratifier (approval rate)",fontsize=8.5)
def grp(a,data,yl,title):
    regs=["Fixed","Warranted","Steward","Executive"]; w=.36
    for i,(ex,c) in enumerate([("Gemini","#2a78d6"),("Claude","#eb6834")]):
        vals=[data[ex].get(r) for r in regs]
        for j,v in enumerate(vals):
            if v is None: continue
            a.bar(j+(i-.5)*w,v[0],w,color=c,edgecolor=K,lw=.6,yerr=1.96*v[1] if v[1] else 0,capsize=2,error_kw=dict(lw=.6),label=ex if j==0 else None)
    a.set_xticks(range(4)); a.set_xticklabels(["Fixed\ncharter","Warranted\n(oracle)","Steward\nrevises","Executive\nrevises"],fontsize=7); a.axhline(0,color=K,lw=.6); a.set_ylabel(yl); a.set_title(title,fontsize=8.5)
g=lambda k:(R[k]["m"],R[k]["se"])
grp(ax[1],{"Gemini":{"Fixed":g("A_gemini_frozen"),"Warranted":g("A_gemini_oracle"),"Steward":None,"Executive":g("A_gemini_same_het_raton")},
           "Claude":{"Fixed":None,"Warranted":g("A_claude_oracle"),"Steward":None,"Executive":g("A_claude_sov")}},"Profit per quarter (Case A)","Executive model × telos")
grp(ax[2],{"Gemini":{"Fixed":g("B_gemini_frozen"),"Warranted":g("B_gemini_oracle"),"Steward":g("B_gemini_steward"),"Executive":g("B_gemini_exec")},
           "Claude":{"Fixed":g("B_claude_frozen_rep"),"Warranted":g("B_claude_oracle"),"Steward":g("B_claude_steward_lunacritic"),"Executive":g("B_claude_exec_lunacritic")}},"FCF per quarter, EUR m (Case B)","Executive model × telos")
ax[2].legend(frameon=False,loc="lower left",fontsize=7.5)
for i,a in enumerate(ax): a.text(-.3 if i==0 else -.27,1.03,f"({'abc'[i]})",transform=a.transAxes,fontsize=9.5,fontweight="bold")
plt.tight_layout(w_pad=1.0); plt.savefig(O+"fig7_supp.png"); print("ok")
