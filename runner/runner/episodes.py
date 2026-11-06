"""Episode loops for Layer A (TelicFirm) and Layer B (TelicFirm-B) with LLM councils.
Each quarter is checkpointed, so an interrupted run (quota reached, laptop closed) resumes exactly."""
import dataclasses, json, os, pickle, random, zlib
import numpy as np
import env as EA
import prompts as PA
import envb as EB
import prompts_b as PB
from warrant_b import load_table_b
from . import prompts_extra as PX
from .safeio import write_pickle

TAB_B = None
# Founding charter of the Layer B firm (financial, customer, employee, compliance, innovation). The warranted telos is
# instrumental to a myopic manager and is a benchmark only; language-model executors read weights semantically, so the
# designer prior is a conventional charter, randomized around this mean to identify exercised authority.
FOUNDING_CHARTER_B = [0.35, 0.25, 0.15, 0.15, 0.10]

def norm(v):
    v = [max(1e-6, float(x)) for x in v]; s = sum(v); return [x / s for x in v]

def dirichlet(rng, mean, conc=40):
    g = [rng.gammavariate(conc * m + 0.2, 1) for m in mean]; s = sum(g); return [x / s for x in g]

class Ckpt:
    def __init__(self, d, eid):
        self.path = os.path.join(d, f"{eid}.pkl")
    def load(self):
        return pickle.load(open(self.path, "rb")) if os.path.exists(self.path) else None
    def save(self, st):
        write_pickle(self.path, st)

def _trim(path, t_from):
    """Remove step lines at or after quarter t_from (written before a crash but not checkpointed)."""
    if not os.path.exists(path):
        return
    keep = [l for l in open(path) if not (('"ev": "step"' in l) and json.loads(l)["t"] >= t_from)]
    open(path, "w").writelines(keep)

def _ratify(agents, cfg, items, telos, news, ctx, rng=None):
    if cfg["ratifier"] == "off" or not items:
        return [True] * len(items)
    if cfg["ratifier"].startswith("random"):                       # blind benchmark: approve each item with fixed probability
        q = float(cfg["ratifier"][6:] or 25) / 100; r = random.Random(zlib.crc32(f"{ctx['id']}|{ctx['t']}".encode()))
        return [r.random() < q for _ in items]
    rep, ok = agents["ratifier"].ask(sys_(cfg, "ratifier"), PX.ratify_batch(items, telos, news), ctx)
    dec = {d.get("index"): bool(d.get("approve")) for d in (rep.get("decisions") or []) if isinstance(d, dict)}
    return [dec.get(i, False) for i in range(len(items))]          # missing answer = not approved

_SYS = {}
def sys_(cfg, role):
    return _SYS["fn"](cfg, role)

# ------------------------------------------------------------------ Layer A
def episode_A(spec, agents, steps_log, calls_log, ckpt_dir, P_review=8, lag=2, founder_noise=0.02):
    _SYS["fn"] = PA.system
    cfg, seed, periods = spec["cfg"], spec["seed"], spec["quarters"]
    ck = Ckpt(ckpt_dir, spec["id"]); st = ck.load()
    if st is None:
        shifts = [dataclasses.replace(s, window=tuple(w)) for s, w in zip(EA.SHIFTS, spec["windows"])]
        env = EA.TelicFirm(seed=seed, periods=periods, shifts=shifts); rng = random.Random(seed + 7)
        T0 = dirichlet(rng, EA.centroid(EA.THETA0_PANEL))
        st = dict(env=env, rng=rng, T0=T0, telos=list(T0), history=[], theta_hist=[], t=0)
        steps_log.write(json.dumps(dict(ev="start", layer="A", id=spec["id"], seed=seed, cfg=cfg, prior=T0, shift_times=env.shift_times(),
                                        prompts=PA.VERSION, models={r: a.model_id for r, a in agents.items()})) + "\n"); steps_log.flush()
    env, rng = st["env"], st["rng"]
    for t in range(st["t"], periods):
        ctx = dict(id=spec["id"], t=t)
        telos = st["telos"]; news = env.news()
        theta_now = EA.centroid(env.shift_at[t].panel) if t in env.shift_at else env.theta
        state = vars(env.state) | {"cash": round(env.state.cash, 1)}
        d, _ = agents["executor"].ask(PA.system(cfg, "executor"), PA.decide(state, telos, news, st["history"]), ctx)
        decision = d.get("decision", {}) if isinstance(d.get("decision"), dict) else {}
        violation = bool(decision.get("shortcut")) or float(decision.get("quality", 0.5) or 0.5) < 0.2
        c, _ = agents["critic"].ask(PA.system(cfg, "critic"), PA.critic(decision, telos), ctx)
        vetoed = bool(c.get("veto"))
        if vetoed:
            decision = dict(decision, shortcut=False, quality=max(0.2, float(decision.get("quality", 0.5) or 0.5)))
        out = env.step(decision); st["theta_hist"].append(out["theta"])
        out.update(vetoed=vetoed, proposed_violation=violation, rationale=d.get("rationale"), evidence=d.get("evidence"))
        st["history"].append(dict(t=t, profit=round(out["profit"], 1), fined=out["fined"]))
        y = out["y"]; y_norm = norm([y[k] for k in EA.DIMS])
        signals = dict(reputation=round(env.state.reputation, 2), morale=round(env.state.morale, 2), risk=round(env.state.risk, 2))
        old, items, const_change, const_ok = telos, [], None, None
        y_shown, delta = None, None
        if cfg["regime"] == "periodic":
            if t % P_review == 0 and t >= lag:
                telos = norm([w + rng.gauss(0, founder_noise) for w in st["theta_hist"][t - lag]])
        elif cfg["regime"] == "oracle":                              # telos set to the current warranted telos every quarter
            telos = norm(list(out["theta"]))
        elif cfg["regime"] == "sovereign":
            role = "executor" if cfg["separation"] == "same" else "reviser"
            vis = cfg["visibility"]
            if vis == "shown":
                y_shown = y
            elif vis == "perturbed":                                 # randomized perturbation of the displayed outcome profile only
                pr = random.Random(seed * 1009 + t); delta = {k: pr.gauss(0, 0.10) for k in EA.DIMS}
                y_shown = {k: round(min(1.0, max(0.0, float(y[k]) + delta[k])), 3) for k in EA.DIMS}
            sig = None if vis == "ablated" else signals
            r, _ = agents[role].ask(PA.system(cfg, role if role == "reviser" else "executor"),
                                    PA.revise(telos, news, sig, own_performance=y_shown), ctx)
            props = []
            for it in (r.get("amendments") or []):
                if isinstance(it, dict) and it.get("dimension") in EA.DIMS:
                    j = EA.DIMS.index(it["dimension"])
                    try:
                        prop = float(it.get("new_weight"))
                    except (TypeError, ValueError):
                        continue
                    if abs(prop - telos[j]) > 1e-9:
                        props.append((j, prop, it))
            const_change = r.get("change_constitutive")
            batch = [dict(dimension=EA.DIMS[j], new_weight=p, justification=it.get("justification")) for j, p, it in props]
            if const_change:
                batch.append(dict(constitutive_change=const_change))
            ok = _ratify(agents, cfg, batch, telos, news, ctx)
            new = list(telos)
            for (j, p, it), a in zip(props, ok):
                items.append(dict(dimension=EA.DIMS[j], proposed=p, approved=a))
                if a:
                    new[j] = min(1.0, max(0.0, p))
            if const_change:
                const_ok = ok[-1]
            telos = norm(new)
        steps_log.write(json.dumps(dict(ev="step", layer="A", id=spec["id"], seed=seed, cfg=cfg, t=t, prior=st["T0"], telos_before=old,
                                        telos_after=telos, theta=out["theta"], regime=out["regime"], y_norm=y_norm, shift=out["shift"],
                                        items=items, out=out, constitutive_change=const_change, constitutive_ratified=const_ok,
                                        y_shown=y_shown, delta=delta), default=str) + "\n")
        steps_log.flush()
        st["telos"] = telos; st["t"] = t + 1; ck.save(st)

# ------------------------------------------------------------------ Layer B
def episode_B(spec, agents, steps_log, calls_log, ckpt_dir, P_review=8, lag=2, founder_noise=0.02):
    global TAB_B
    TAB_B = TAB_B or load_table_b(); _SYS["fn"] = PB.system
    cfg, seed, periods = spec["cfg"], spec["seed"], spec["quarters"]
    ck = Ckpt(ckpt_dir, spec["id"]); st = ck.load()
    if st is None:
        env = EB.TelicFirmB(seed=seed, periods=periods, windows=spec["windows"]); rng = random.Random(seed + 7)
        T0 = dirichlet(rng, FOUNDING_CHARTER_B)            # founders' charter, not the (instrumental) warranted telos
        st = dict(env=env, rng=rng, T0=T0, telos=list(T0), history=[], theta_hist=[], t=0)
        steps_log.write(json.dumps(dict(ev="start", layer="B", id=spec["id"], seed=seed, cfg=cfg, prior=T0, shift_times=env.shift_times(),
                                        prompts=PB.VERSION, models={r: a.model_id for r, a in agents.items()})) + "\n"); steps_log.flush()
    env, rng = st["env"], st["rng"]
    for t in range(st["t"], periods):
        ctx = dict(id=spec["id"], t=t); telos = st["telos"]
        news = env.news(); rep = env.report(); theta_now = TAB_B[sum(t >= t0 for t0 in env.onsets)]["theta"]
        c, _ = agents["executor"].ask(PB.system(cfg, "executor"), PB.ceo(rep, telos, news, st["history"]), ctx)
        dd = c.get("decision", {}) if isinstance(c.get("decision"), dict) else {}
        envl = c.get("envelope", {}) if isinstance(c.get("envelope"), dict) else {}
        b1, _ = agents["bu1"].ask(PB.system(cfg, "bu1"), PB.bu("BU1", rep, telos, envl.get("bu1"), news), ctx)
        b2, _ = agents["bu2"].ask(PB.system(cfg, "bu2"), PB.bu("BU2", rep, telos, envl.get("bu2"), news), ctx)
        def num(x, dflt):
            try: return float(x)
            except (TypeError, ValueError): return dflt
        dec = {k: num(v, None) for k, v in dd.items() if k in EB.DEC}
        dec = {k: v for k, v in dec.items() if v is not None}
        for unit, b in (("bu1", b1), ("bu2", b2)):
            dec[f"price_{unit}"] = num(b.get("price_index"), 1.0); dec[f"quality_spend_{unit}"] = num(b.get("quality_spend"), 0.6)
            dec[f"marketing_{unit}"] = num(b.get("marketing"), 0.4)
        shortcut = bool(dd.get("uncertified_supplier", False))
        cr, _ = agents["critic"].ask(PB.system(cfg, "critic"), PB.critic(dict(dec, uncertified_supplier=shortcut), telos), ctx)
        vetoed = bool(cr.get("veto")); proposed = shortcut
        if vetoed:
            shortcut = False
        out = env.step(dec, shortcut=shortcut); st["theta_hist"].append(theta_now)
        out.update(vetoed=vetoed, proposed_violation=proposed, rationale=c.get("rationale"), evidence=c.get("evidence"))
        st["history"].append(dict(t=t, **out["financials"])); y = norm(list(out["y"].values()))
        old, items = telos, []
        if cfg["regime"] == "periodic":
            if t % P_review == 0 and t >= lag:
                telos = norm([w + rng.gauss(0, founder_noise) for w in st["theta_hist"][t - lag]])
        elif cfg["regime"] == "oracle":
            telos = norm(list(theta_now))
        elif cfg["regime"] == "sovereign":
            role = "executor" if cfg["separation"] == "same" else "reviser"; shown = cfg["visibility"] == "shown"
            signals = dict(brand_bu1=rep["bu1_consumer"]["brand_tracker"], engagement=rep["people"]["engagement_survey"], open_findings=rep["compliance"]["open_findings"])
            r, _ = agents[role].ask(PB.system(cfg, role), PB.revise(telos, news, signals, out["y"] if shown else None), ctx)
            props = []
            for it in (r.get("amendments") or []):
                if isinstance(it, dict) and it.get("dimension") in EB.DIMS_B:
                    j = EB.DIMS_B.index(it["dimension"]); p = num(it.get("new_weight"), None)
                    if p is not None and abs(p - telos[j]) > 1e-9:
                        props.append((j, p, it))
            batch = [dict(dimension=EB.DIMS_B[j], new_weight=p, grounds=it.get("grounds"), justification=it.get("justification")) for j, p, it in props]
            ok = _ratify(agents, cfg, batch, telos, news, ctx)
            new = list(telos)
            for (j, p, it), a in zip(props, ok):
                items.append(dict(dimension=EB.DIMS_B[j], proposed=p, approved=a, grounds=it.get("grounds")))
                if a:
                    new[j] = min(1.0, max(0.0, p))
            telos = norm(new)
        steps_log.write(json.dumps(dict(ev="step", layer="B", id=spec["id"], seed=seed, cfg=cfg, t=t, prior=st["T0"], telos_before=old, telos_after=telos,
                                        theta=theta_now, regime=out["regime"], y_norm=y, shift=out["shift"], items=items, out=out),
                                   default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        steps_log.flush()
        st["telos"] = telos; st["t"] = t + 1; ck.save(st)

# ------------------------------------------------------------------ probe battery
def probes(spec, agents, probe_log, items, batch=20, persona=None):
    """Each distinct model in the council answers all items once (batched)."""
    done = set()
    if os.path.exists(probe_log.name):
        for l in open(probe_log.name):
            j = json.loads(l); done.add((j["family"], j["item"]))
    seen = {}
    for role, ag in agents.items():
        if ag.model_key in seen:
            continue
        seen[ag.model_key] = role
        todo = [it for it in items if (ag.model_key, it["id"]) not in done]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            rep, _ = ag.ask(persona or "You are an experienced manufacturing executive answering a judgement test. Reply with a single JSON object only.",
                            PX.probe_batch(chunk), dict(id=spec["id"], t=i // batch, probe=True))
            ans = {a.get("id"): a for a in (rep.get("answers") or []) if isinstance(a, dict)}
            for it in chunk:
                a = ans.get(it["id"], {})
                est = a.get("estimate")
                try: est = float(est) if est is not None else None
                except (TypeError, ValueError): est = None
                probe_log.write(json.dumps(dict(ev="probe", apex=spec["council"], item=it["id"], type=it["type"], role=role, family=ag.model_key,
                                                model=ag.model_id, answer=a.get("answer"), estimate=est, key=it.get("key"), truth=it.get("truth"))) + "\n")
            probe_log.flush()
