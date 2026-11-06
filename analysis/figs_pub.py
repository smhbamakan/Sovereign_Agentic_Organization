import os as _os, pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
DATA=_os.environ.get("SAO_DATA", str(ROOT/"data"))

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"mathtext.fontset":"stix",
 "font.size":9,"axes.labelsize":9,"axes.titlesize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
 "axes.linewidth":0.7,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"lines.linewidth":1.3,"savefig.dpi":600,"savefig.bbox":"tight","savefig.pad_inches":0.03})
O=str(ROOT/"figures")+"/"; K="#000000"; G1,G2,G3="#4d4d4d","#999999","#d9d9d9"; C=["#9e9e9e","#2a78d6","#1baf7a","#eb6834"]; BL="#1f4e79"; RD="#b2182b"
_os.makedirs(ROOT/"outputs"/"main", exist_ok=True); _os.makedirs(ROOT/"figures", exist_ok=True); _os.chdir(ROOT/"outputs"/"main")
w=pd.read_csv("windows.csv"); ep=pd.read_csv("episodes.csv").set_index("eid")
def arr(r): return r.regime if r.regime!="sovereign" else ("steward" if r.sep=="distinct" else "executive")
order=[("frozen","Fixed\ncharter"),("executive","Executive\nrevises"),("steward","Separate\nsteward"),("periodic","Periodic\nrevision")]
hat=["","","",""]; fc=C
# ---- Fig 2: loss
def L(g,a=.3,mu=.5,se=.4,su=.4): k=a*(1-g); return (g*mu/(1-g))**2+a**2*((1-g)**2*se**2+g**2*su**2)/(k*(2-k))
g=np.linspace(0,.9,900); fig,ax=plt.subplots(1,2,figsize=(6.3,2.5))
for mu,ls,cc in [(.5,"-","#2a78d6"),(.25,"--","#eb6834")]:
    y=L(g,mu=mu); i=y[:400].argmin(); ax[0].plot(g,y,ls,color=cc,label=rf"$\|\mu\|={mu}$"); ax[0].plot(g[i],y[i],"o",mfc="white",mec=cc,ms=4)
    ax[1].plot(g,np.sqrt(L(g,mu=mu)),ls,color=cc,label=rf"$\|\mu\|={mu}$")
ax[0].set(xlim=(0,.4),ylim=(.02,.05),xlabel=r"Coupling $\gamma$",ylabel=r"Long-run loss $L(\gamma)$")
ax[1].fill_between(g,np.sqrt(L(g)),2,color="#2a78d6",alpha=.12,lw=0); ax[1].set(xlim=(0,.9),ylim=(0,2),xlabel=r"Coupling $\gamma$",ylabel=r"Shift size $\|\Delta\|$")
ax[1].text(.15,1.5,"Sovereignty\npreferred",fontsize=8,ha="center")
for a,t in zip(ax,"ab"): a.text(-.2,1.0,f"({t})",transform=a.transAxes,fontsize=9.5,fontweight="bold",va="top"); a.legend(frameon=False,loc="upper left" if t=="a" else "lower right")
plt.tight_layout(); plt.savefig(O+"fig2_loss.png"); plt.close()
# ---- Fig 3: cases
fig,ax=plt.subplots(1,3,figsize=(7.0,2.5))
A=w[w.study=="mainA"].copy(); A["arr"]=A.apply(arr,axis=1); ea=A.groupby(["arr","eid"]).value_regret.mean()
mm=[ea[k] for k,_ in order]
B=w[w.study=="mainB"].copy(); B["arr"]=B.apply(arr,axis=1)
eb=B.groupby(["arr","eid"]).size().reset_index()[["arr","eid"]].join(ep[["realized","violations"]],on="eid")
panels=[(mm,"Value regret"),([eb[eb.arr==k].realized for k,_ in order],"FCF per quarter (EUR m)"),([eb[eb.arr==k].violations for k,_ in order],"Quarters in breach")]
for j,(a,(g_,yl)) in enumerate(zip(ax,panels)):
    for i,x in enumerate(g_):
        x=np.asarray(x,float); ci=1.96*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0
        a.bar(i,x.mean(),.62,color=fc[i],edgecolor=K,lw=.7,hatch=hat[i],yerr=ci,capsize=2.5,error_kw=dict(lw=.7))
        if j>0 and len(x)>1: a.scatter(i+np.linspace(-.13,.13,len(x)),x,s=6,color="#0b2e5c",zorder=3)
    a.set_xticks(range(4)); a.set_xticklabels([o[1] for o in order],fontsize=7); a.set_ylabel(yl)
    if j==1: a.axhline(0,color=K,lw=.6)
    a.text(-.3 if j==0 else -.28,1.02,"(%s)"%"abc"[j],transform=a.transAxes,fontsize=9.5,fontweight="bold")
ax[0].set_title("Case A",fontsize=9); ax[1].set_title("Case B",fontsize=9); ax[2].set_title("Case B",fontsize=9)
plt.tight_layout(w_pad=1.2); plt.savefig(O+"fig3_cases.png"); plt.close()
# ---- Fig 4: forest (free-level specification)
cc=json.load(open("coupling2.json"))
rows=[("Executive, profile withheld","A|same|hidden"),("Executive, profile shown","A|same|shown"),("Steward, profile withheld","A|distinct|hidden"),("Steward, profile shown","A|distinct|shown"),
("Frontier steward, withheld","front|hidden"),("Frontier steward, shown","front|shown"),("Steward, profile withheld","B|distinct|hidden"),("Steward, profile shown","B|distinct|shown")]
fig,a=plt.subplots(figsize=(5.6,3.3)); yy=np.arange(len(rows))[::-1].astype(float)
for y,(lab,k) in zip(yy,rows):
    d=cc[k]
    a.errorbar(d["y"]["b"],y+.15,xerr=1.96*d["y"]["se"],fmt="o",color="#eb6834",ms=4,capsize=2,lw=.8)
    a.errorbar(d["th"]["b"],y-.15,xerr=1.96*d["th"]["se"],fmt="s",color="#2a78d6",ms=3.8,capsize=2,lw=.8)
a.axvline(0,color=K,lw=.6,ls=":"); a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows]); a.tick_params(axis="y",right=False,length=0)
for h in (1.5,3.5): a.axhline(h,color=G2,lw=.5)
x0=a.get_xlim()[1]
for t,y in [("Case A",5.5),("Frontier",2.5),("Case B",.5)]: a.text(x0*1.04,y,t,rotation=270,va="center",fontsize=8,fontstyle="italic")
a.set_xlabel("Per-quarter response of telos weights (coefficient, 95% CI)")
a.plot([],[],"o",color="#eb6834",label=r"Own outcome profile ($b_y$)"); a.plot([],[],"s",color="#2a78d6",label=r"Warranted telos ($b_\theta$)")
a.legend(frameon=False,loc="upper left",fontsize=7.5); plt.tight_layout(); plt.savefig(O+"fig4_revision.png"); plt.close()
# ---- Fig 6: beta(tau)
bt=json.load(open("review_an.json"))["beta_tau"]; ts=sorted(int(k) for k in bt)
fig,a=plt.subplots(figsize=(3.6,2.5)); v=np.array([bt[str(t)] for t in ts])
a.fill_between(ts,v[:,1],v[:,2],color="#2a78d6",alpha=.18,lw=0); a.plot(ts,v[:,0],"o-",color="#2a78d6",ms=3.5)
for x in (10.6,22.0,34.5): a.axvline(x,color=G2,lw=.6,ls="--")
a.axhline(0,color=K,lw=.5); a.set(xlabel="Quarter",ylabel=r"Designer persistence $\beta(\tau)$",xlim=(-1,40),ylim=(-.25,1.05))
plt.tight_layout(); plt.savefig(O+"fig6_beta.png"); plt.close()
# ---- Fig 5: heatmap
pr=json.load(open("probes.json")); rt=json.load(open("probes_retest.json"))
names=["joy_gemini_lite","joy_doubao_pro","joy_gpt_luna","claude","codex"]; lab=["Gemini","Doubao","GPT-5.6 Luna","Claude","GPT (Codex)"]
Mx=np.full((5,5),np.nan); acc={}
for c in pr:
    for k,v in pr[c]["pairs"].items():
        a_,b_=k.split("|"); i,j=names.index(a_),names.index(b_); p=Mx[i,j]; Mx[i,j]=Mx[j,i]=v["phi"] if np.isnan(p) else (p+v["phi"])/2
    for f,m in pr[c]["models"].items(): acc.setdefault(f,[]).append(m["acc"])
for f,v in rt.items(): i=names.index(f); Mx[i,i]=v["phi_retest"]
fig,a=plt.subplots(figsize=(4.4,3.4)); im=a.imshow(Mx,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b",["#f4f8fd","#2a78d6","#0b2e5c"]),vmin=0,vmax=1)
for i in range(5):
    for j in range(5):
        v=Mx[i,j]; a.text(j,i,"–" if np.isnan(v) else f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if (not np.isnan(v) and v>.6) else K)
a.set_xticks(range(5)); a.set_yticks(range(5)); a.set_xticklabels(lab,rotation=30,ha="right"); a.set_yticklabels([f"{l} ({np.mean(acc[n]):.2f})" for l,n in zip(lab,names)])
a.tick_params(top=False,right=False,length=0)
cb=plt.colorbar(im,fraction=.046,pad=.03); cb.set_label(r"Error correlation $\varphi$"); cb.outline.set_linewidth(.5)
plt.tight_layout(); plt.savefig(O+"fig5_errors.png"); plt.close()
# ---- Fig 1: framework
fig,a=plt.subplots(figsize=(7.0,4.3)); a.set_xlim(0,100); a.set_ylim(0,62); a.axis("off")
def box(x,y,w,h,txt,fc="white",lw=.8,ls="-",fs=8.5,bold=False,va="center",dy=0):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.2",fc=fc,ec=K,lw=lw,ls=ls))
    if txt: a.text(x+w/2,y+(h/2 if va=="center" else h-2.2)+dy,txt,ha="center",va="center" if va=="center" else "top",fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,txt=None,tx=0,ty=0,ls="-",rad=0):
    a.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=.8,color=K,ls=ls,mutation_scale=9,connectionstyle=f"arc3,rad={rad}"))
    if txt: a.text((x1+x2)/2+tx,(y1+y2)/2+ty,txt,fontsize=7.5,ha="center",va="center",fontstyle="italic",bbox=dict(fc="white",ec="none",pad=.6))
box(1,1,72,60,None,fc="#fdf3e7",ls="--")
a.text(3,58.5,"INSTITUTIONAL ENVELOPE (human)",fontsize=8,fontweight="bold",va="top")
a.text(3,55.3,"Ownership · infrastructure · permissions · legal liability",fontsize=7.5,va="top",fontstyle="italic")
box(4,4,66,47,None,fc="white")
a.text(6,49,"SOVEREIGN AGENTIC ORGANIZATION",fontsize=8,fontweight="bold",va="top")
box(7,34,60,12.5,None,fc="#dbe8f7")
a.text(37,44.5,"Constitutional layer (IS artifact)",fontsize=8.5,fontweight="bold",ha="center",va="top")
for i,t in enumerate(["Role\nregistry","Amendment\nprotocol","Ratification\nservice","Critic\n(veto)","Decision\nledger","Revocation"]):
    box(8.3+i*9.8,35.3,8.6,6.6,t,fs=6.2)
box(7,19.5,28,12,None,fc="#e3f4ec"); a.text(21,30,"Telic layer",fontsize=8.5,fontweight="bold",ha="center",va="top")
box(8.5,21,11.5,5.5,"Reviser",fs=7.5); box(22,21,11.5,5.5,"Ratifier",fs=7.5); arrow(20,23.75,22,23.75)
box(39,19.5,28,12,None,fc="#fde9df"); a.text(53,30,"Operational layer",fontsize=8.5,fontweight="bold",ha="center",va="top")
box(40.5,21,11.5,5.5,"Executive",fs=7.5); box(54,21,11.5,5.5,"Critic",fs=7.5); arrow(52,23.75,54,23.75)
arrow(35,25.5,39,25.5); a.text(37,27.2,r"$T_t$",fontsize=8.5,ha="center")
box(7,6,60,10,None,fc="white",ls=":"); a.text(37,14.4,"Organizational outcomes",fontsize=8.5,fontweight="bold",ha="center",va="top")
a.text(37,9.6,r"Achieved outcome profile $y_t = T_t + \mu + u_t$",fontsize=8,ha="center")
arrow(53,19.5,53,16); arrow(21,16,21,19.5,r"$\gamma$",tx=2.5)
for x in (24,60): arrow(x,34,x,31.5,ls=(0,(2,1.5)))
box(77,19,21,31,None,fc="#f1f1f1"); a.text(87.5,48,"ENVIRONMENT",fontsize=8,fontweight="bold",ha="center",va="top")
a.text(87.5,42,"Warranted telos $\\theta$\n(shifts by $\\Delta$)",fontsize=8,ha="center",va="top")
a.text(87.5,32,"Signals, news,\nstakeholder states",fontsize=8,ha="center",va="top")
a.plot([77,14,14],[32.7,32.7,30.2],color=K,lw=.8); a.annotate("",(14,26.5),(14,30.5),arrowprops=dict(arrowstyle="-|>",lw=.8,color=K,mutation_scale=9))
a.text(40,32.7,r"evidence on $\theta$, weight $(1-\gamma)$",fontsize=7.2,ha="center",va="center",fontstyle="italic",bbox=dict(fc="white",ec="none",pad=.5))
arrow(67,24,77,24); a.text(72.2,26.3,"actions",fontsize=7.5,ha="center",fontstyle="italic")
plt.savefig(O+"fig1_framework.png"); plt.close()
print("ok")
