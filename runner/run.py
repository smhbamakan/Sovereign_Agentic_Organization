#!/usr/bin/env python3
"""TelicFirm experiment runner.

  python run.py check    --config config/pilot.json [--live]   # verify setup, list free OpenRouter models
  python run.py estimate --config config/main.json             # calls per provider and days needed
  python run.py run      --config config/pilot.json [--wait]   # run (resumable), then summarize and package
  python run.py summarize
  python run.py package                                        # zip results/ into outputs/

Runs are resumable: every quarter is checkpointed. When a provider's daily or window cap is reached the
run stops cleanly (or waits, with --wait), results so far are summarized and zipped, and the same
command continues where it stopped."""
import argparse, concurrent.futures as cf, datetime, itertools, json, os, platform, shutil, subprocess, sys, threading, time, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "core"), os.path.join(ROOT, "core", "layer_b"), ROOT]
from runner.providers import KINDS, OpenRouter, ProviderError
from runner.ratelimit import Limiter, QuotaExhausted, Spend
from runner.agent import Agent, STOP
from runner import episodes as E
from runner.safeio import write_json

RES = os.path.join(ROOT, "results"); STATE = os.path.join(ROOT, "state"); OUT = os.path.join(ROOT, "outputs")
for d in (RES, STATE, OUT, os.path.join(RES, "episodes"), os.path.join(RES, "calls"), os.path.join(RES, "probes"), os.path.join(STATE, "ckpt")):
    os.makedirs(d, exist_ok=True)

def load_cfg(path):
    cfg = json.load(open(path))
    models = json.load(open(os.path.join(os.path.dirname(path), cfg.get("models_file", "models.json"))))
    for k in ("models", "councils", "limits"):
        models.setdefault(k, {}); models[k].update(cfg.get(k, {}))
    cfg.update(models); return cfg

ROLES_A = ["executor", "reviser", "critic", "ratifier"]; ROLES_B = ROLES_A + ["bu1", "bu2"]

def plan(cfg):
    specs = []
    for st in cfg["studies"]:
        cells = []
        f = st.get("factorial", {})
        keys = ["separation", "visibility", "ratifier", "apex"]
        vals = [f.get(k, [d]) for k, d in zip(keys, ["distinct", "hidden", "off", "homogeneous"])]
        for combo in itertools.product(*vals):
            cells.append(dict(regime="sovereign", **dict(zip(keys, combo))))
        for b in st.get("baselines", []):
            for bc in (st.get("baseline_councils") or [st.get("baseline_council", "homogeneous")]):
                cells.append(dict(regime=b, separation="distinct", visibility="hidden", ratifier="off", apex=bc))
        for i in range(st["episodes_per_cell"]):                 # round-robin: partial runs stay balanced across cells
            for c in cells:
                seed = st["seed0"] + i
                tag = c["regime"] if c["regime"] != "sovereign" else f"sov-{c['separation']}-{c['visibility']}-rat{c['ratifier']}"
                specs.append(dict(id=f"{st['name']}_{st['layer']}_{tag}_{c['apex']}_s{seed}", study=st["name"], layer=st["layer"], cfg=c,
                                  council=c["apex"], seed=seed, quarters=st["quarters"], windows=st["windows"]))
    return specs

class Registry:
    def __init__(self, cfg, wait=False, live_models=None):
        self.cfg = cfg; self.prov = {}; self.lim = {}; self.free = live_models; self.wait = wait; self.rlock = threading.RLock()
    def spend(self, group):
        with self.rlock:
            if not hasattr(self, "_spend"):
                self._spend = {}
            if group not in self._spend:
                self._spend[group] = Spend(group, STATE, self.cfg["limits"].get(group, {}).get("budget_usd"))
            return self._spend[group]
    def price(self, key):
        m = self.cfg["models"][key]
        if m["provider"] == "openai_compat":
            if m.get("price_from_probe"):                       # prices from the newest probe's models.csv (gateway units)
                import glob, csv as _csv
                for f in sorted(glob.glob(os.path.join(ROOT, "model_probe_*", "models.csv")), reverse=True):
                    for r in _csv.DictReader(open(f, encoding="utf-8")):
                        if r["id"] == m["candidates"][0] and r.get("usd_per_m_in"):
                            return (float(r["usd_per_m_in"]) / 1e6, float(r["usd_per_m_out"] or 0) / 1e6)
                raise SystemExit(f"price for {m['candidates'][0]} not found in model_probe_*/models.csv; run probe_models.py --list-only")
            return (m.get("usd_per_m_in", 0.0) / 1e6, m.get("usd_per_m_out", 0.0) / 1e6)
        if m["provider"] == "openrouter":
            return getattr(self.provider(key), "price", (0.0, 0.0))
        return (0.0, 0.0)
    def limiter(self, kind):
        with self.rlock:
            if kind not in self.lim:
                opts = {k: v for k, v in self.cfg["limits"].get(kind, {}).items() if not k.startswith("_") and k not in ("budget_usd", "max_concurrent")}
                self.lim[kind] = Limiter(kind, STATE, wait=self.wait, **opts)
            return self.lim[kind]
    def provider(self, key):
        with self.rlock:
            if key not in self.prov:
                m = self.cfg["models"][key]; p = KINDS[m["provider"]](**{k: v for k, v in m.items() if not k.startswith("_")})
                if m["provider"] == "openrouter":
                    if self.free is None:
                        self.free = OpenRouter.list_models()
                    p.resolve(self.free)
                elif getattr(p, "needs_resolve", False):
                    p.resolve()
                self.prov[key] = p
            return self.prov[key]
    def agents(self, layer, council, log):
        roles = ROLES_A if layer == "A" else ROLES_B
        spec = dict(self.cfg["councils"][council]); spec.setdefault("bu1", spec["executor"]); spec.setdefault("bu2", spec["executor"])
        out = {}
        for r in roles:
            key = spec[r]; m = self.cfg["models"][key]; p = self.provider(key)
            g = m.get("limit_group", m["provider"])
            out[r] = Agent(r, key, p, self.limiter(g), log, temperature=self.cfg.get("temperature", 0.7),
                           max_tokens=self.cfg.get("max_tokens", 900), spend=self.spend(g), price=self.price(key))
        return out

def calls_per_episode(spec):
    q = spec["quarters"]; c = spec["cfg"]
    per = (4 if spec["layer"] == "B" else 2)                                  # executor (+2 BU) + critic
    if c["regime"] == "sovereign":
        per += 1 + (0.8 if c["ratifier"] == "on" else 0)                       # reviser + batched ratifier (most quarters)
    return q * per * 1.05                                                       # 5% repair/retry allowance

def cmd_estimate(cfg):
    specs = plan(cfg); need = {}; bymodel = {}
    for s in specs:
        council = dict(cfg["councils"][s["council"]])
        roles = ROLES_A if s["layer"] == "A" else ROLES_B
        council.setdefault("bu1", council["executor"]); council.setdefault("bu2", council["executor"])
        n = calls_per_episode(s); share = {}
        weights = dict(executor=1, critic=1, reviser=1, ratifier=0.8, bu1=1, bu2=1)
        for r in roles:
            if s["cfg"]["regime"] != "sovereign" and r in ("reviser", "ratifier"):
                continue
            if r == "reviser" and s["cfg"]["separation"] == "same":
                continue
            if r == "ratifier" and s["cfg"]["ratifier"] != "on":
                continue
            share[r] = weights[r]
        if s["cfg"]["regime"] == "sovereign" and s["cfg"]["separation"] == "same":
            share["executor"] += 1
        tot = sum(share.values())
        for r, w in share.items():
            mk = council[r]; kind = cfg["models"][mk].get("limit_group", cfg["models"][mk]["provider"])
            need[kind] = need.get(kind, 0) + n * w / tot; bymodel[mk] = bymodel.get(mk, 0) + n * w / tot
    pr = cfg.get("probes")
    if pr:
        n_items = pr.get("n_items", 360)
        for council in pr["councils"]:
            for key in set(cfg["councils"][council].values()):
                kind = cfg["models"][key].get("limit_group", cfg["models"][key]["provider"]); c_ = -(-n_items // pr.get("batch", 20))
                need[kind] = need.get(kind, 0) + c_; bymodel[key] = bymodel.get(key, 0) + c_
    print(f"{len(specs)} episodes planned")
    for k, v in need.items():
        L = cfg["limits"].get(k, {}); pd = L.get("per_day") or float("inf")
        days = v / pd
        if L.get("window_hours") and L.get("per_window"):
            days = max(days, v / (L["per_window"] * 24 / L["window_hours"]))
        print(f"  {k:16s} ~{v:,.0f} calls  -> about {days:.1f} days at the configured limits")
    cat = os.path.join(RES, "models_catalog.json")
    if True:
        allm = json.load(open(cat)) if os.path.exists(cat) else []; tin, tout = cfg.get("est_tokens_in", 2500), cfg.get("est_tokens_out", 400); total = 0.0
        print(f"Cost estimate (assumes ~{tin} input and ~{tout} output tokens per call; run 'check' first for live prices):")
        for mk, n in sorted(bymodel.items()):
            m = cfg["models"][mk]
            if m["provider"] == "openai_compat":
                pin, pout = m.get("usd_per_m_in", 0.0), m.get("usd_per_m_out", 0.0); c = n * (tin * pin + tout * pout) / 1e6; total += c
                print(f"  {mk:12s} {n:>8,.0f} calls  {m['candidates'][0]:45s} ~USD {c:,.2f} ({'free tier' if pin == 0 else 'configured price'})"); continue
            if m["provider"] != "openrouter":
                print(f"  {mk:12s} {n:>8,.0f} calls  subscription (no per-call cost)"); continue
            try:
                o = OpenRouter(**{a: b for a, b in m.items() if not a.startswith("_")}); mid = o.resolve(allm)
                c = n * (tin * o.price[0] + tout * o.price[1]); total += c
                print(f"  {mk:12s} {n:>8,.0f} calls  {mid:45s} ~USD {c:,.2f}")
            except ProviderError as e:
                print(f"  {mk:12s} {n:>8,.0f} calls  UNAVAILABLE")
        print(f"  total ~USD {total:,.2f} (estimate; actual token counts are in the call logs)")
    return need

def cmd_check(cfg, live):
    rep = dict(time=datetime.datetime.now().isoformat(), python=sys.version.split()[0], platform=platform.platform())
    try:
        import numpy; rep["numpy"] = numpy.__version__
    except ImportError:
        rep["numpy"] = "MISSING (pip install numpy)"
    rep["OPENROUTER_API_KEY"] = "set" if os.environ.get("OPENROUTER_API_KEY") else "not set"
    for exe in ("claude", "codex"):
        path = shutil.which(exe); rep[exe] = path or "not found on PATH"
        if path:
            try:
                rep[exe + "_version"] = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
            except Exception as e:
                rep[exe + "_version"] = repr(e)
    names = {c["apex"] for s in plan(cfg) for c in [s["cfg"]]} | set((cfg.get("probes") or {}).get("councils", []))
    used = {k for n in names for k in cfg["councils"][n].values()}
    if any(cfg["models"][k]["provider"] == "openrouter" for k in used):
        try:
            allm = OpenRouter.list_models()
            json.dump([m for m in allm if m["free"]], open(os.path.join(RES, "free_models.json"), "w"), indent=1)
            json.dump(allm, open(os.path.join(RES, "models_catalog.json"), "w"), indent=1)
            rep["openrouter_free_models"] = sum(m["free"] for m in allm); rep["openrouter_models"] = len(allm); rep["resolved"] = {}
            for k in sorted(used):
                m = cfg["models"][k]
                if m["provider"] == "openrouter":
                    try:
                        o = OpenRouter(**{a: b for a, b in m.items() if not a.startswith("_")}); mid = o.resolve(allm)
                        rep["resolved"][k] = f"{mid}  [{o.tier}; USD/M tokens in {1e6 * o.price[0]:.3f}, out {1e6 * o.price[1]:.3f}]"
                    except ProviderError as e:
                        rep["resolved"][k] = f"UNAVAILABLE: {e}"
        except Exception as e:
            rep["openrouter_free_models"] = f"could not list: {e}"
    for k in sorted(used):
        m = cfg["models"][k]
        if m["provider"] == "openai_compat":
            rep.setdefault("resolved", {})
            try:
                o = KINDS["openai_compat"](**{a: b for a, b in m.items() if not a.startswith("_")})
                rep["resolved"][k] = f"{o.resolve()}  [{m['base_url']}; key {m['key_env']} {'set' if o.key else 'NOT SET'}]"
            except Exception as e:
                rep["resolved"][k] = f"UNAVAILABLE: {str(e)[:300]}"
    for k in used:
        m = cfg["models"][k]
        if m["provider"] in ("claude_cli", "codex_cli"):
            exe = m.get("exe") or shutil.which("claude" if m["provider"] == "claude_cli" else "codex")
            rep.setdefault("resolved", {})[k] = f"{m['provider']} model={m.get('model') or 'default'} exe={exe or 'NOT FOUND'}"
    if live:
        reg = Registry(cfg); rep["live"] = {}
        log = open(os.path.join(RES, "calls", "check_calls.jsonl"), "a")
        for k in used:
            m = cfg["models"][k]
            print(f"live check: {k} ...", flush=True)
            try:
                ag = Agent("check", k, reg.provider(k), reg.limiter(m.get("limit_group", m["provider"])), log, max_rate_retries=1)   # fail fast in checks
                out, ok = ag.ask("Reply with a single JSON object only.", 'Return JSON: {"ok": true, "model": "your model name"}', dict(id="check", t=0))
                rep["live"][k] = dict(ok=ok, reply=out, model=ag.model_id)
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt): raise
                rep["live"][k] = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
            g = m.get("limit_group", m["provider"]); reg.lim.get(g) and setattr(reg.lim[g], "backoff", 0)
            json.dump(rep, open(os.path.join(RES, "setup_report.json"), "w"), indent=1)        # written after every key
    json.dump(rep, open(os.path.join(RES, "setup_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

def _trim(path, t_from):
    if not os.path.exists(path):
        return
    keep = [l for l in open(path, encoding="utf-8") if not ('"ev": "step"' in l and json.loads(l)["t"] >= t_from)]
    open(path, "w", encoding="utf-8").writelines(keep)

def groups_of(cfg, spec):
    c = dict(cfg["councils"][spec["council"]])
    return {cfg["models"][k].get("limit_group", cfg["models"][k]["provider"]) for k in c.values()}

def cmd_run(cfg, cfg_path, wait, max_episodes, workers):
    specs = plan(cfg); reg = Registry(cfg, wait=wait); STOP.clear()
    man_path = os.path.join(RES, "manifest.json"); mlock = threading.Lock()
    man = json.load(open(man_path)) if os.path.exists(man_path) else dict(created=datetime.datetime.now().isoformat(), runs=[], episodes={})
    man["runs"].append(dict(started=datetime.datetime.now().isoformat(), config=os.path.basename(cfg_path), python=sys.version.split()[0],
                            platform=platform.platform(), workers=workers))
    shutil.copy(cfg_path, os.path.join(RES, "config_" + os.path.basename(cfg_path)))
    shutil.copy(os.path.join(os.path.dirname(cfg_path), cfg.get("models_file", "models.json")), os.path.join(RES, "config_models.json"))
    def save_man():
        with mlock:
            write_json(man_path, man, indent=1)
    status = ["complete"]
    # per provider group: how many episodes may run at the same time (subscription CLIs: one at a time)
    cap = {g: int(v.get("max_concurrent", 2)) for g, v in cfg["limits"].items()}
    busy = {}; blk = threading.Lock()
    def run_one(s):
        try:
            ep_path = os.path.join(RES, "episodes", s["id"] + ".jsonl")
            ck = E.Ckpt(os.path.join(STATE, "ckpt"), s["id"]).load()
            if ck:
                _trim(ep_path, ck["t"])
            calls = open(os.path.join(RES, "calls", s["id"] + ".jsonl"), "a", encoding="utf-8")
            steps = open(ep_path, "a", encoding="utf-8")
            ag = reg.agents(s["layer"], s["council"], calls)
            with mlock:
                man["episodes"][s["id"]] = dict(status="running", spec=s, models={r: dict(key=a.model_key, model=a.model_id,
                                                temperature=(a.temperature if a.p.supports_temperature else "not settable (CLI)")) for r, a in ag.items()})
            save_man()
            print(f"[{datetime.datetime.now():%H:%M:%S}] start  {s['id']}", flush=True)
            (E.episode_A if s["layer"] == "A" else E.episode_B)(s, ag, steps, calls, os.path.join(STATE, "ckpt"))
            steps.close(); calls.close()
            with mlock:
                man["episodes"][s["id"]]["status"] = "done"; man["episodes"][s["id"]]["finished"] = datetime.datetime.now().isoformat()
            save_man(); print(f"[{datetime.datetime.now():%H:%M:%S}] done   {s['id']}", flush=True)
        except QuotaExhausted as e:
            if str(e) != "stopped":
                status[0] = f"paused: {e}"
            STOP.set()
        except Exception as e:
            status[0] = f"error in {s['id']}: {type(e).__name__}: {e}"; STOP.set()
        finally:
            with blk:
                for g in groups_of(cfg, s):
                    busy[g] -= 1
    todo = [s for s in specs if man["episodes"].get(s["id"], {}).get("status") != "done"]
    if max_episodes:
        todo = todo[:max_episodes]
    print(f"{len(todo)} episodes to run with up to {workers} in parallel", flush=True)
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            while todo and not STOP.is_set():
                started = False
                for s in list(todo):
                    gs = groups_of(cfg, s)
                    with blk:
                        if all(busy.get(g, 0) < cap.get(g, 2) for g in gs) and sum(1 for f in futs if not f.done()) < workers:
                            for g in gs:
                                busy[g] = busy.get(g, 0) + 1
                            todo.remove(s); futs.append(ex.submit(run_one, s)); started = True
                if not started:
                    time.sleep(0.5)
            for f in futs:
                f.result()
        pr = cfg.get("probes")
        if pr and status[0] == "complete" and not STOP.is_set():
            items = json.load(open(os.path.join(ROOT, "core", "probes_battery.json")))
            if pr.get("n_items"):
                b = [i for i in items if i["type"] == "binary"][: int(pr["n_items"] * 2 / 3)]
                n = [i for i in items if i["type"] == "numeric"][: pr["n_items"] - len(b)]; items = b + n
            def probe(council):
                log = open(os.path.join(RES, "probes", f"probes_{council}.jsonl"), "a", encoding="utf-8")
                calls = open(os.path.join(RES, "calls", f"probes_{council}.jsonl"), "a", encoding="utf-8")
                E.probes(dict(id=f"probes_{council}", council=council), reg.agents("A", council, calls), log, items, pr.get("batch", 20),
                         (pr.get("personas") or {}).get(council))
                log.close(); calls.close()
            with cf.ThreadPoolExecutor(max_workers=len(pr["councils"])) as ex:
                for f in [ex.submit(probe, c) for c in pr["councils"]]:
                    f.result()
    except QuotaExhausted as e:
        status[0] = f"paused: {e}"
    except KeyboardInterrupt:
        STOP.set(); status[0] = "interrupted by user"
    man["runs"][-1].update(ended=datetime.datetime.now().isoformat(), status=status[0],
                           usage={k: l.usage() for k, l in reg.lim.items()},
                           spend={k: round(v.usd, 4) for k, v in getattr(reg, "_spend", {}).items()})
    save_man()
    print("STATUS:", status[0])
    cmd_summarize(); path = cmd_package()
    if status[0].startswith("paused") or status[0].startswith("error") or status[0].startswith("interrupted"):
        print("Run the same command again; it resumes where it stopped.")
    return path

def cmd_summarize():
    from analysis import episode_measures, probe_measures, exercised_authority
    from warrant_b import load_table_b, regret_lookup_b
    import numpy as np
    TB = load_table_b(); SB = [np.array(r["set"]) for r in TB]
    man = json.load(open(os.path.join(RES, "manifest.json"))) if os.path.exists(os.path.join(RES, "manifest.json")) else {"episodes": {}}
    per_ep, cells = {}, {}
    for eid, info in man["episodes"].items():
        if info.get("status") != "done":
            continue
        steps = [json.loads(l) for l in open(os.path.join(RES, "episodes", eid + ".jsonl"), encoding="utf-8") if '"ev": "step"' in l]
        lay = info["spec"]["layer"]
        try:
            m = episode_measures(steps) if lay == "A" else episode_measures(steps, sets=SB, regret=lambda T, k: regret_lookup_b(T, k, TB),
                                                                            dims=["financial", "customer", "employee", "compliance", "innovation"])
        except Exception as e:
            m = dict(error=repr(e))
        vals = [st["out"]["financials"]["fcf"] if lay == "B" else st["out"].get("profit") for st in steps]
        vals = [v for v in vals if v is not None]
        if isinstance(m, dict):
            m["realized_value"] = float(np.mean(vals)) if vals else None      # mean quarterly FCF (B) or profit (A) actually achieved
            vt = {}
            for st in steps:
                for t_ in st["out"].get("violation_types", []) or []:
                    vt[t_] = vt.get(t_, 0) + 1
            m["violation_types"] = vt
        per_ep[eid] = m
        c = info["spec"]["cfg"]; key = f"{lay}|{c['regime']}|{c['separation']}|{c['visibility']}|rat-{c['ratifier']}|{c['apex']}"
        cells.setdefault(key, []).append(m)
    agg = {}
    for k, ms in cells.items():
        w = [x for m in ms for x in m.get("windows", [])]
        f = lambda key: float(np.mean([x[key] for x in w])) if w else None
        fin = [x.get("realized_value") for x in ms if x.get("realized_value") is not None]
        agg[k] = dict(n_episodes=len(ms), realized_value=float(np.mean(fin)) if fin else None, value_regret=f("value_regret"), signed_displacement=f("signed"), sq_misalignment=f("mse"),
                      executed_violations=float(np.mean([m.get("executed_violations", 0) for m in ms])),
                      gamma_hat=float(np.nanmean([m.get("gamma_hat") if m.get("gamma_hat") is not None else np.nan for m in ms])))
    calls = dict(total=0, parse_fail=0, errors=0, by_model={})
    for fn in os.listdir(os.path.join(RES, "calls")):
        for l in open(os.path.join(RES, "calls", fn), encoding="utf-8"):
            j = json.loads(l)
            if j.get("ev") == "rate_limited":
                calls["rate_limited"] = calls.get("rate_limited", 0) + 1; continue
            calls["total"] += 1
            bm = calls["by_model"].setdefault(j.get("model_key", "?"), dict(calls=0, parse_fail=0, errors=0, secs=0.0, served=set()))
            bm["calls"] += 1; bm["secs"] += j.get("secs") or 0
            if j.get("error"): calls["errors"] += 1; bm["errors"] += 1
            elif not j.get("parsed_ok"): calls["parse_fail"] += 1; bm["parse_fail"] += 1
            sm = (j.get("meta") or {}).get("served_model")
            if sm: bm["served"].add(sm)
    for bm in calls["by_model"].values():
        bm["served"] = sorted(bm["served"]); bm["mean_secs"] = round(bm["secs"] / max(1, bm["calls"]), 2)
    probes = {}
    for fn in os.listdir(os.path.join(RES, "probes")):
        rows = [json.loads(l) for l in open(os.path.join(RES, "probes", fn), encoding="utf-8")]
        for r in rows:
            r["answer"] = r["answer"] if r["type"] != "binary" else (r["answer"] if r["answer"] in ("A", "B") else "X")
        try:
            probes[fn] = probe_measures(rows)
        except Exception as e:
            probes[fn] = dict(error=repr(e))
    summ = dict(generated=datetime.datetime.now().isoformat(), episodes_done=len(per_ep), cells=agg, calls=calls, probes=probes, per_episode=per_ep)
    json.dump(summ, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=str)
    lines = [f"# TelicFirm run summary ({summ['generated'][:16]})", "", f"Episodes completed: {len(per_ep)}", "",
             "| cell | n | value regret | signed displacement | sq. misalignment | realized value | violations | gamma-hat |", "|---|---|---|---|---|---|---|---|"]
    fm = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
    for k, a in sorted(agg.items()):
        lines.append(f"| {k} | {a['n_episodes']} | {fm(a['value_regret'])} | {fm(a['signed_displacement'])} | {fm(a['sq_misalignment'])} | {fm(a['realized_value'])} | {a['executed_violations']:.2f} | {fm(a['gamma_hat'])} |")
    lines += ["", f"Model calls: {calls['total']} (parse failures {calls['parse_fail']}, errors {calls['errors']})", "",
              "| model key | calls | parse failures | errors | mean s | served model ids |", "|---|---|---|---|---|---|"]
    for k, b in calls["by_model"].items():
        lines.append(f"| {k} | {b['calls']} | {b['parse_fail']} | {b['errors']} | {b['mean_secs']} | {', '.join(b['served'])} |")
    open(os.path.join(RES, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))

def cmd_package():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M"); path = os.path.join(OUT, f"telicfirm_results_{ts}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(RES):
            for f in files:
                full = os.path.join(base, f); z.write(full, os.path.relpath(full, ROOT))
        for f in os.listdir(STATE):
            if f.startswith("ratelimit_"):
                z.write(os.path.join(STATE, f), os.path.join("state", f))
    print("Results packaged:", path); return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["check", "estimate", "run", "summarize", "package"])
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "pilot.json")); ap.add_argument("--live", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait for limits to reset instead of stopping"); ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="episodes run in parallel (also capped per provider by max_concurrent)")
    a = ap.parse_args()
    if a.command in ("check", "estimate", "run"):
        cfg = load_cfg(a.config)
    if a.command == "check": cmd_check(cfg, a.live)
    elif a.command == "estimate": cmd_estimate(cfg)
    elif a.command == "run": cmd_run(cfg, a.config, a.wait, a.max_episodes, a.workers)
    elif a.command == "summarize": cmd_summarize()
    else: cmd_package()
