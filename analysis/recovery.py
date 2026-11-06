import numpy as np,pandas as pd,statsmodels.formula.api as smf,json
mu=np.array([-.09,-.04,-.02,.15]); out={}
def sim(gam,a=.08,revert=0.0,reps=20,seed=0):
    rng=np.random.default_rng(seed); est=[]
    for r in range(reps):
        rr=[]
        for e in range(96):
            T=rng.dirichlet(np.ones(4)*40); th=np.array([.35,.25,.15,.25])
            for t in range(40):
                if t in (10,22,34): th=rng.dirichlet(np.ones(4)*20)
                y=np.clip(T+mu+rng.normal(0,.08,4),0,1)
                tgt=(1-gam)*(th+rng.normal(0,.05,4))+gam*y
                Tn=(1-a)*T+a*tgt - revert*(T-.25); Tn=np.clip(Tn,1e-4,1); Tn/=Tn.sum()
                for j in range(4): rr.append((e,j,Tn[j]-T[j],th[j],y[j],T[j]))
                T=Tn
        d=pd.DataFrame(rr,columns=["eid","j","dT","th","y","T"]); d["gt"]=d.th-d["T"]; d["gy"]=d.y-d["T"]
        m1=smf.ols("dT~gt+gy-1",d).fit(); m2=smf.ols("dT~th+y+T",d).fit(); m3=smf.ols("dT~th+y+T+C(j)",d).fit()
        est.append([m1.params["gt"],m1.params["gy"],m2.params["th"],m2.params["y"],m3.params["th"],m3.params["y"]])
    e=np.array(est); return dict(true_th=a*(1-gam),true_y=a*gam,restricted=e[:,:2].mean(0).round(4).tolist(),free=e[:,2:4].mean(0).round(4).tolist(),freeFE=e[:,4:].mean(0).round(4).tolist())
for name,kw in [("gamma+0.6",dict(gam=.6)),("gamma-0.4",dict(gam=-.4)),("gamma0_revert",dict(gam=0.,a=.02,revert=.06)),("gamma0",dict(gam=0.))]:
    out[name]=sim(**kw); print(name,out[name])
json.dump(out,open("recovery.json","w"))
