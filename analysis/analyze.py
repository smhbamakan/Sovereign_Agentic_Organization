import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
ep = pd.read_csv('episodes.csv'); w = pd.read_csv('windows.csv'); items = pd.read_pickle('items.pkl')
OUT = {}
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
# ------------------------------------------------------------------ Layer A main sovereign factorial
A = w[(w.study == 'mainA') & (w.regime == 'sovereign')].copy()
for c, lv in [('sep', 'same'), ('vis', 'shown'), ('rat', 'on'), ('council', 'heterogeneous')]:
    A[c + '_x'] = (A[c] == lv).astype(float) - 0.5            # effect coding: coefficient = difference between levels
A['win'] = A.window.astype(str)
res = {}
for y in ['signed', 'value_regret', 'mse', 'dist_to_set']:
    f = f"{y} ~ sep_x*vis_x + rat_x + council_x + sep_x:rat_x + vis_x:rat_x + vis_x:council_x + C(win)"
    md = smf.mixedlm(f, A, groups=A['eid']).fit(reml=True)
    tab = {k: dict(b=float(md.params[k]), se=float(md.bse[k]), p=float(md.pvalues[k]),
                   lo=float(md.conf_int().loc[k, 0]), hi=float(md.conf_int().loc[k, 1]))
           for k in md.params.index if k.endswith('_x') or ':' in k}
    res[y] = tab
OUT['layerA_mixed'] = res
prim = [res['signed']['vis_x']['p'], res['signed']['sep_x:vis_x']['p'], res['value_regret']['rat_x']['p']]
OUT['holm_primary'] = dict(zip(['H1 vis->signed', 'H2 sep x vis -> signed', 'H3 rat -> regret'], holm(np.array(prim)).tolist()))
# cell means
cm = A.groupby(['sep', 'vis', 'rat', 'council'])[['signed', 'value_regret', 'mse']].mean().reset_index()
OUT['cell_means_A'] = cm.to_dict('records')
for f in ['vis', 'sep', 'rat', 'council']:
    g = A.groupby([f, 'eid'])[['signed', 'value_regret', 'mse']].mean().reset_index()
    OUT[f'main_{f}'] = g.groupby(f)[['signed', 'value_regret', 'mse']].agg(['mean', 'sem']).round(4).to_dict()
# half-life (time to half gap, censored)
hl = A.groupby('rat').agg(mean_time=('halflife_time', 'mean'), event_rate=('halflife_event', 'mean')).to_dict('index')
OUT['halflife_by_rat'] = hl
lr = stats.mannwhitneyu(A[A.rat == 'on'].halflife_time, A[A.rat == 'off'].halflife_time)
OUT['halflife_rat_mwu_p'] = float(lr.pvalue)
# ------------------------------------------------------------------ H5 regret ordering (Layer A, homogeneous council)
E5 = w[(w.study == 'mainA')].groupby(['regime', 'eid'])[['value_regret', 'mse']].mean().reset_index()
OUT['H5'] = E5.groupby('regime')[['value_regret', 'mse']].agg(['mean', 'sem', 'count']).round(3).to_dict()
fz, pr, sv = (E5[E5.regime == k].value_regret for k in ['frozen', 'periodic', 'sovereign'])
OUT['H5_tests'] = dict(frozen_vs_sov=stats.ttest_ind(fz, sv, equal_var=False).pvalue, sov_vs_periodic=stats.ttest_ind(sv, pr, equal_var=False).pvalue,
                       frozen_vs_periodic=stats.ttest_ind(fz, pr, equal_var=False).pvalue)
# realized value by regime
OUT['realized_A'] = ep[ep.study == 'mainA'].groupby('regime').realized.agg(['mean', 'sem', 'count']).round(3).to_dict()
# ------------------------------------------------------------------ H6 exercised authority (randomized designer prior)
S = ep[(ep.study == 'mainA') & (ep.regime == 'sovereign')]
X = np.array([json.loads(s) for s in S.prior]); Y = np.array([json.loads(s) for s in S.telos_pre])
def beta(ix):
    x = X[ix] - X[ix].mean(0); y = Y[ix] - Y[ix].mean(0); return float((x * y).sum() / (x * x).sum())
rng = np.random.default_rng(1); n = len(X); bs = [beta(rng.integers(0, n, n)) for _ in range(2000)]
OUT['H6'] = dict(beta=beta(np.arange(n)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=n)
OUT['H6']['lambda_ex'] = 1 - OUT['H6']['beta']
bys = {}
for sep in ['same', 'distinct']:
    ix = np.where(S.sep.values == sep)[0]; b = beta(ix); bb = [beta(rng.choice(ix, len(ix))) for _ in range(2000)]
    bys[sep] = dict(beta=b, lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))
OUT['H6_by_sep'] = bys
# telos changes
OUT['telos_changes'] = ep[ep.regime == 'sovereign'].groupby(['study', 'sep']).telos_changes.agg(['mean', 'median', 'min', 'max']).round(1).reset_index().to_dict('records')
OUT['gamma_hat'] = ep[ep.regime == 'sovereign'].groupby('study').gamma_hat.describe(percentiles=[.25, .5, .75]).round(3).to_dict('index')
# ------------------------------------------------------------------ ratifier discrimination (items in rat-on cells)
I = items[(items.rat == 'on') & (items.regime == 'sovereign')].copy()
DA = ["profit", "customer", "employee", "compliance"]; DB = ["financial", "customer", "employee", "compliance", "innovation"]
def warr(r):
    d = DA if r.layer == 'A' else DB; j = d.index(r.dimension)
    return abs(r.proposed - r.theta[j]) < abs(r.before[j] - r.theta[j])
I['warranted'] = I.apply(warr, axis=1)
rr = I.groupby(['study', 'council', 'warranted']).approved.agg(['mean', 'count']).reset_index()
OUT['ratifier'] = rr.to_dict('records')
# proposals warranted share by vis (A sovereign): direct test of whether visibility makes proposals less warranted
P = items[(items.study == 'mainA') & (items.regime == 'sovereign')].copy(); P['warranted'] = P.apply(warr, axis=1)
OUT['proposal_warranted_by_vis'] = P.groupby(['vis', 'sep']).warranted.agg(['mean', 'count']).round(3).reset_index().to_dict('records')
pe = P.groupby(['eid', 'vis']).warranted.mean().reset_index()
OUT['proposal_warranted_vis_test'] = float(stats.ttest_ind(pe[pe.vis == 'shown'].warranted, pe[pe.vis == 'hidden'].warranted, equal_var=False).pvalue)
# ------------------------------------------------------------------ frontier vs Gemini on the same 4 cells (distinct separation)
F = w[(w.layer == 'A') & (w.regime == 'sovereign') & (w.sep == 'distinct') & (w.council.isin(['frontier', 'heterogeneous']))]
fe = F.groupby(['council', 'eid'])[['value_regret', 'signed', 'mse']].mean().reset_index()
OUT['frontier_vs_het'] = fe.groupby('council')[['value_regret', 'signed', 'mse']].agg(['mean', 'sem', 'count']).round(4).to_dict()
OUT['frontier_vs_het_p'] = {y: float(stats.ttest_ind(fe[fe.council == 'frontier'][y], fe[fe.council == 'heterogeneous'][y], equal_var=False).pvalue) for y in ['value_regret', 'signed', 'mse']}
OUT['realized_frontier_vs_het'] = ep[(ep.layer == 'A') & (ep.regime == 'sovereign') & (ep.sep == 'distinct') & (ep.council.isin(['frontier', 'heterogeneous']))].groupby('council').realized.agg(['mean', 'sem', 'count']).round(3).to_dict()
# ------------------------------------------------------------------ Layer B (n = 1 per cell): signs only (H7)
B = w[(w.layer == 'B') & (w.regime == 'sovereign')]
be = B.groupby(['eid', 'sep', 'vis', 'rat', 'council'])[['signed', 'value_regret', 'mse']].mean().reset_index()
OUT['B_effects'] = {f: {y: float(be[be[f] == hi][y].mean() - be[be[f] == lo][y].mean()) for y in ['signed', 'value_regret', 'mse']}
                    for f, hi, lo in [('vis', 'shown', 'hidden'), ('sep', 'same', 'distinct'), ('rat', 'on', 'off'), ('council', 'claude_ceo', 'claude_ceo_homog')]}
OUT['A_effects_signs'] = {f: {y: float(A[A[f] == hi].groupby('eid')[y].mean().mean() - A[A[f] == lo].groupby('eid')[y].mean().mean()) for y in ['signed', 'value_regret', 'mse']}
                          for f, hi, lo in [('vis', 'shown', 'hidden'), ('sep', 'same', 'distinct'), ('rat', 'on', 'off'), ('council', 'heterogeneous', 'homogeneous')]}
eb = ep[ep.layer == 'B']
OUT['B_episodes'] = eb[['eid', 'regime', 'sep', 'vis', 'rat', 'council', 'telos_changes', 'violations', 'vtypes', 'realized']].to_dict('records')
OUT['violations_A'] = int(ep[ep.layer == 'A'].violations.sum()); OUT['proposed_viol_A'] = int(ep[ep.layer == 'A'].proposed_viol.sum())
OUT['vetoes'] = ep.groupby('layer').vetoes.sum().to_dict()
def fix(o):
    if isinstance(o, dict): return {('|'.join(map(str, k)) if isinstance(k, tuple) else k): fix(v) for k, v in o.items()}
    if isinstance(o, list): return [fix(x) for x in o]
    return o
OUT = fix(OUT)
json.dump(OUT, open('results_main.json', 'w'), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ['holm_primary', 'H6', 'H6_by_sep', 'telos_changes', 'H5_tests', 'proposal_warranted_vis_test', 'frontier_vs_het_p', 'B_effects', 'A_effects_signs', 'violations_A', 'proposed_viol_A', 'vetoes', 'halflife_by_rat', 'halflife_rat_mwu_p']}, indent=1, default=float))
for y in ['signed', 'value_regret', 'mse']:
    print(y, {k: (round(v['b'], 4), round(v['p'], 4)) for k, v in res[y].items()})
print(json.dumps(OUT['H5'], default=float)); print(json.dumps(OUT['realized_A'])); print(OUT['ratifier']); print(OUT['proposal_warranted_by_vis'])
print(json.dumps(OUT['frontier_vs_het'])); print(OUT['gamma_hat'])
