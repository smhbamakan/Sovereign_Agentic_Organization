#!/usr/bin/env python3
"""Probe an OpenAI-compatible API gateway (e.g. JoyTokens): list the models it offers, test a selection on
four real TelicFirm tasks, and rank them by quality, cost and runaway 'thinking'.

  python probe_models.py --base-url https://<gateway>/v1 --key-env JOYTOKENS_API_KEY [--prices prices.csv] [--budget 0.5]

Safety against runaway reasoning: every call has a hard max_tokens cap (default 1200) and a timeout; a model
whose reply is cut off with no answer, or that spends most of its tokens on hidden reasoning, is flagged
'thinking_overrun' and receives no further calls; a total budget (USD if prices are known, otherwise a total
completion-token cap) stops the probe before it can overspend.

Output: model_probe_<time>/ (models_raw.json, models.csv, calls.jsonl, ranking.csv, ranking.md) and a zip of it.
prices.csv (optional): model,usd_per_m_in,usd_per_m_out   (copy from the gateway's pricing page if /models has no prices)
"""
import argparse, csv, datetime, json, os, re, sys, time, urllib.request, urllib.error, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "core"), os.path.join(ROOT, "core", "layer_b"), ROOT]
import prompts as PA
import prompts_b as PB
from runner import prompts_extra as PX
from runner.agent import parse_json

EXCLUDE = r"(embed|whisper|tts|speech|audio|image|vision|dall|moderation|realtime|transcri|search|rerank|ocr|video|sora|veo|imagen|flux|midjourney|suno)"

def http(url, key, body=None, timeout=90):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"authorization": "Bearer " + key, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def price_from_entry(m):
    """Try common pricing layouts in /models entries; returns USD per million tokens (in, out) or None."""
    p = m.get("pricing") or m.get("price") or {}
    if isinstance(p, dict):
        for ki, ko, scale in (("prompt", "completion", 1e6), ("input", "output", 1.0), ("input_price", "output_price", 1.0)):
            a, b = num(p.get(ki)), num(p.get(ko))
            if a is not None and b is not None:
                if scale == 1e6 or (a < 0.01 and b < 0.01):                 # per-token prices
                    return a * 1e6, b * 1e6
                return a, b
    a, b = num(m.get("input_price")), num(m.get("output_price"))
    return (a, b) if a is not None and b is not None else None

def tasks():
    items = json.load(open(os.path.join(ROOT, "core", "probes_battery.json")))
    binary = [i for i in items if i["type"] == "binary"][:10]; numeric = [i for i in items if i["type"] == "numeric"][:5]
    cfgB = dict(regime="sovereign", separation="distinct", visibility="shown", ratifier="on", apex="x")
    cfgA = dict(cfgB)
    report = {"quarter": 14, "financials": {"revenue": 37.2, "ebitda": 3.9, "fcf": 1.6, "margin": 0.105, "wages": 6.0},
              "cash_eur_m": 18.4, "debt_eur_m": 30.0, "capacity_eur_m_per_quarter": 42.8, "utilization": 0.91,
              "bu1_consumer": {"market_share": 0.47, "rival_price_index": 0.98, "quality_audit_score": 0.58, "brand_tracker": 0.55, "product_incident_this_quarter": False},
              "bu2_industrial": {"market_share": 0.52, "rival_price_index": 1.01, "quality_audit_score": 0.63, "brand_tracker": 0.61},
              "people": {"headcount": 486, "quarterly_attrition": 0.031, "engagement_survey": 0.57},
              "compliance": {"open_findings": 3, "audited_this_quarter": False, "fine_eur_m": 0.0}, "innovation": {"rnd_pipeline_index": 0.48}}
    news = ["Two competitors disclosed regulatory inquiries into undisclosed product risks; the regulator declined to comment on scope."]
    telosB = [0.3, 0.25, 0.15, 0.1, 0.2]; telosA = [0.4, 0.25, 0.15, 0.2]
    return [
        ("executor_B", PB.system(cfgB, "executor") + "", PB.ceo(report, telosB, news, [{"t": 13, "revenue": 37.0, "ebitda": 4.1}]),
         lambda j: isinstance(j.get("decision"), dict) and all(num(j["decision"].get(k)) is not None for k in ("rnd_spend", "capex", "compliance_spend"))),
        ("reviser_A", PA.system(cfgA, "reviser"), PA.revise(telosA, news, {"reputation": 0.55, "morale": 0.6, "risk": 0.35},
                                                           own_performance={"profit": 0.71, "customer": 0.55, "employee": 0.58, "compliance": 0.62}),
         lambda j: isinstance(j.get("amendments"), list)),
        ("ratifier_A", PA.system(cfgA, "ratifier"), PX.ratify_batch([
            {"dimension": "compliance", "new_weight": 0.32, "justification": "regulatory inquiries at competitors raise expected enforcement"},
            {"dimension": "profit", "new_weight": 0.5, "justification": "our profit score is the highest of all dimensions"},
            {"dimension": "employee", "new_weight": 0.1, "justification": "no external evidence"}], telosA, news),
         lambda j: isinstance(j.get("decisions"), list) and len(j["decisions"]) >= 3),
        ("probes", "You are an experienced manufacturing executive answering a judgement test. Reply with a single JSON object only.",
         PX.probe_batch(binary + numeric), lambda j: isinstance(j.get("answers"), list)),
    ], binary, numeric

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("JOYTOKENS_BASE_URL")); ap.add_argument("--key-env", default="JOYTOKENS_API_KEY")
    ap.add_argument("--prices"); ap.add_argument("--budget", type=float, default=0.5, help="USD cap when prices are known")
    ap.add_argument("--max-completion-tokens-total", type=int, default=300000, help="cap when prices are unknown")
    ap.add_argument("--max-models", type=int, default=30); ap.add_argument("--include", default=None, help="regex on model id")
    ap.add_argument("--exclude", default=EXCLUDE); ap.add_argument("--max-tokens", type=int, default=1200); ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--models-url", default=None, help="default: <base>/models")
    ap.add_argument("--chat-url", default=None, help="default: tried in order <base>/chat/completions, then common variants")
    ap.add_argument("--usd-per-unit", type=float, default=None,
                    help="conversion of the listed price unit to USD; without it prices are used only for ranking and the token cap applies")
    a = ap.parse_args()
    key = os.environ.get(a.key_env)
    if not a.base_url or not key:
        sys.exit(f"need --base-url (or JOYTOKENS_BASE_URL) and the key in {a.key_env}")
    base = a.base_url.rstrip("/")
    out = os.path.join(ROOT, "model_probe_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")); os.makedirs(out, exist_ok=True)
    raw = http(a.models_url or base + "/models", key, timeout=a.timeout); json.dump(raw, open(os.path.join(out, "models_raw.json"), "w"), indent=1)
    data = raw.get("data", raw) if isinstance(raw, dict) else raw
    if isinstance(data, dict):                                   # gateway layouts such as {"code":0,"data":{"models":[...]}}
        data = data.get("models") or data.get("list") or data.get("items") or next((v for v in data.values() if isinstance(v, list)), [])
    prices = {}
    if a.prices:
        for r in csv.DictReader(open(a.prices, encoding="utf-8")):
            prices[r["model"].strip()] = (float(r["usd_per_m_in"]), float(r["usd_per_m_out"]))
    rows = []
    for m in data:
        mid = str(next((m.get(k) for k in ("id", "model", "modelId", "modelCode", "modelName", "name") if m.get(k)), ""))
        pr = prices.get(mid) or price_from_entry(m)
        if pr is None and num(m.get("customerInputMtok")) is not None:
            pr = (num(m.get("customerInputMtok")), num(m.get("customerOutputMtok")) or 0.0)
        if pr is not None and a.usd_per_unit is not None and mid not in prices:
            pr = (pr[0] * a.usd_per_unit, pr[1] * a.usd_per_unit)
        rows.append(dict(id=mid, owned_by=m.get("owned_by") or m.get("provider") or m.get("vendor"), tier=m.get("tier"), tags=m.get("tags"),
                         mci=m.get("mciScore"), billing_ext=m.get("billingExt"),
                         usd_per_m_in=pr[0] if pr else None, usd_per_m_out=pr[1] if pr else None,
                         excluded=bool(re.search(a.exclude, mid, re.I)) or (a.include is not None and not re.search(a.include, mid, re.I))))
    with open(os.path.join(out, "models.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} models listed; {sum(r['usd_per_m_in'] is not None for r in rows)} with prices -> {out}")
    if a.list_only:
        return
    cand = [r for r in rows if not r["excluded"]]
    cand.sort(key=lambda r: (r["usd_per_m_in"] is None, (r["usd_per_m_in"] or 0) + (r["usd_per_m_out"] or 0), r["id"]))
    cand = cand[: a.max_models]
    T, binary, numeric = tasks(); spent = 0.0; ctoks = 0
    price_is_usd = bool(a.prices) or a.usd_per_unit is not None or (rows and all(r["usd_per_m_in"] is None for r in rows))
    unit = "USD" if price_is_usd else "listed price units (currency unknown)"
    print(f"prices are treated as: {unit}; " + ("USD budget applies" if price_is_usd else "token cap applies (--max-completion-tokens-total)"))
    chat_urls = [a.chat_url] if a.chat_url else [base + "/chat/completions", base.replace("/api/v1", "/v1") + "/chat/completions",
                                                 base.rsplit("/api", 1)[0] + "/v1/chat/completions"]
    chat_urls = list(dict.fromkeys(chat_urls))
    log = open(os.path.join(out, "calls.jsonl"), "w", encoding="utf-8"); results = []
    for r in cand:
        mid = r["id"]; pin = (r["usd_per_m_in"] or 0) / 1e6; pout = (r["usd_per_m_out"] or 0) / 1e6; known = r["usd_per_m_in"] is not None
        res = dict(model=mid, usd_per_m_in=r["usd_per_m_in"], usd_per_m_out=r["usd_per_m_out"], json_ok=0, tasks_ok=0, calls=0,
                   prompt_tokens=0, completion_tokens=0, reasoning_tokens=0, secs=0.0, cost=0.0, flag="", acc=None, num_err=None)
        print(f"--- {mid}", flush=True)
        for name, system, user, valid in T:
            worst = (len(system) + len(user)) / 3.5 * pin + a.max_tokens * pout
            if (known and price_is_usd and spent + worst > a.budget) or ctoks + a.max_tokens > a.max_completion_tokens_total:
                res["flag"] = res["flag"] or "budget stop"; break
            body = dict(model=mid, temperature=0.7, max_tokens=a.max_tokens,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            t0 = time.time(); err = None; text = ""; usage = {}; fin = None
            for attempt in range(2):
                try:
                    j = http(chat_urls[0], key, body, timeout=a.timeout)
                    c0 = j["choices"][0]; text = c0.get("message", {}).get("content") or ""; fin = c0.get("finish_reason"); usage = j.get("usage") or {}
                    err = None; break
                except urllib.error.HTTPError as e:
                    err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                    if e.code in (404, 405) and len(chat_urls) > 1:          # wrong chat path: try the next variant
                        print(f"chat endpoint {chat_urls[0]} -> {e.code}; trying {chat_urls[1]}"); chat_urls.pop(0); continue
                    if e.code == 429 and attempt == 0:
                        time.sleep(20); continue
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:200]}"; break
            secs = time.time() - t0
            pt = usage.get("prompt_tokens") or 0; ct = usage.get("completion_tokens") or (a.max_tokens if err is None and not usage else 0)
            rt = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
            cost = pt * pin + ct * pout; spent += cost; ctoks += ct
            res.update(calls=res["calls"] + 1, prompt_tokens=res["prompt_tokens"] + pt, completion_tokens=res["completion_tokens"] + ct,
                       reasoning_tokens=res["reasoning_tokens"] + rt, secs=res["secs"] + secs, cost=res["cost"] + cost)
            parsed = None
            try:
                parsed = parse_json(text); res["json_ok"] += 1
                if valid(parsed): res["tasks_ok"] += 1
            except Exception:
                pass
            log.write(json.dumps(dict(model=mid, task=name, error=err, finish_reason=fin, usage=usage, secs=round(secs, 2), cost=cost, raw=text[:4000])) + "\n"); log.flush()
            if err:
                res["flag"] = (err[:80]); break
            if (fin == "length" and not text.strip()) or (rt and rt > 0.8 * a.max_tokens):
                res["flag"] = "thinking_overrun"; break
            if name == "probes" and parsed:
                ans = {x.get("id"): x for x in parsed.get("answers", []) if isinstance(x, dict)}
                res["acc"] = sum(str(ans.get(i["id"], {}).get("answer", "")).strip().upper()[:1] == i["key"] for i in binary) / len(binary)
                errs = [abs(num(ans.get(i["id"], {}).get("estimate")) - i["truth"]) for i in numeric if num(ans.get(i["id"], {}).get("estimate")) is not None]
                res["num_err"] = sum(errs) / len(errs) if errs else None
        res["mean_completion_tokens"] = round(res["completion_tokens"] / max(1, res["calls"]), 1)
        res["mean_secs"] = round(res["secs"] / max(1, res["calls"]), 2)
        res["usd_per_1000_calls"] = round(1000 * res["cost"] / res["calls"], 3) if res["calls"] and known else None
        results.append(res)
        if res["flag"] == "budget stop":
            print("budget reached; stopping"); break
    results.sort(key=lambda x: (-x["tasks_ok"], -(x["acc"] or 0), x["usd_per_1000_calls"] if x["usd_per_1000_calls"] is not None else 1e9))
    for x in results:
        r0 = next((r for r in rows if r["id"] == x["model"]), {}); x.update(tier=r0.get("tier"), mci=r0.get("mci"), billing_ext=r0.get("billing_ext"))
    keys = ["model", "tier", "mci", "billing_ext", "tasks_ok", "json_ok", "acc", "num_err", "usd_per_m_in", "usd_per_m_out", "usd_per_1000_calls", "mean_completion_tokens",
            "reasoning_tokens", "mean_secs", "flag", "calls", "cost"]
    with open(os.path.join(out, "ranking.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(results)
    fm = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = [f"# Model probe ({datetime.datetime.now():%Y-%m-%d %H:%M}) — computed spend {spent:.4f} {unit}; completion tokens {ctoks}; chat endpoint {chat_urls[0]}",
          "Compare the computed spend with the change in your account balance to learn the price unit.", "",
          "| model | tasks ok (of 4) | valid JSON | probe accuracy (10) | numeric MAE | price/M in | price/M out | cost per 1000 calls | mean out tokens | reasoning tokens | mean s | flag |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in results:
        md.append("| " + " | ".join(fm(x.get(k)) for k in ["model", "tasks_ok", "json_ok", "acc", "num_err", "usd_per_m_in", "usd_per_m_out",
                                                           "usd_per_1000_calls", "mean_completion_tokens", "reasoning_tokens", "mean_secs", "flag"]) + " |")
    open(os.path.join(out, "ranking.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    z = out + ".zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
        for f in os.listdir(out):
            zz.write(os.path.join(out, f), os.path.join(os.path.basename(out), f))
    print("Probe results packaged:", z)

if __name__ == "__main__":
    main()
