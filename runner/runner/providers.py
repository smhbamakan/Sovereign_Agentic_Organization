"""Model back-ends.

openrouter  : HTTPS API (OPENROUTER_API_KEY). Free models end in ':free'. The first available model
              from each role's candidate list is resolved at start-up against the live model list,
              and the model that actually served each call is logged.
claude_cli  : Claude Code in print mode ('claude -p'), using the account you are logged into
              (e.g. via VS Code). All tools are disabled, so each call is a plain completion. Temperature cannot be set through the CLI; this is recorded in the manifest.
codex_cli   : OpenAI Codex CLI in non-interactive mode ('codex exec'), read-only sandbox, using your
              logged-in account. Temperature cannot be set either.
mock        : offline stand-in that returns valid random JSON, for testing the pipeline.
"""
import json, os, random, re, shutil, subprocess, tempfile, time, urllib.request, urllib.error

class RateLimited(Exception):
    pass

class ProviderError(Exception):
    pass

def _http(url, headers, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={**headers, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")[:500]
        if e.code == 429:
            raise RateLimited(txt)
        raise ProviderError(f"HTTP {e.code}: {txt}")

class OpenRouter:
    kind = "openrouter"; supports_temperature = True
    def __init__(self, candidates, max_tokens=None, extra_body=None, tier="free", max_usd_per_m_in=0.5, max_usd_per_m_out=1.5, **_):
        self.key = os.environ.get("OPENROUTER_API_KEY"); self.tier = tier; self.price = (0.0, 0.0)
        self.cap = (max_usd_per_m_in / 1e6, max_usd_per_m_out / 1e6)      # price ceiling guards against expensive matches
        self.candidates = candidates; self.model = None; self.json_mode = True
        self.max_tokens_override = max_tokens; self.extra_body = extra_body or {}
    @staticmethod
    def list_models():
        """All text models on OpenRouter with pricing (USD per token) and a free flag."""
        r = _http("https://openrouter.ai/api/v1/models", {})
        out = []
        for m in r.get("data", []):
            pr = m.get("pricing", {}); mod = str(m.get("architecture", {}).get("modality") or "text->text")
            if "->text" not in mod:
                continue
            f = lambda x: float(x) if x not in (None, "") else 0.0
            free = m["id"].endswith(":free") or (f(pr.get("prompt")) == 0 and f(pr.get("completion")) == 0)
            out.append(dict(id=m["id"], name=m.get("name"), context=m.get("context_length"), free=free,
                            price_in=f(pr.get("prompt")), price_out=f(pr.get("completion")),
                            params=m.get("supported_parameters", []), modality=mod))
        return out
    @staticmethod
    def list_free():
        return [m for m in OpenRouter.list_models() if m["free"]]
    _SKIP = ("vl", "vision", "coder", "embed", "guard", "image", "audio", "tts", "ocr")
    def resolve(self, models):
        """Exact ids first, then 'vendor/prefix*' wildcards (alphabetical, skipping vision/coder/guard variants).
        tier 'free' uses only free endpoints; tier 'paid' only paid ones (more reliable, no shared free pool)."""
        allowed = {m["id"]: m for m in models if (m.get("free", m["id"].endswith(":free")) == (self.tier == "free"))
                   and m.get("price_in", 0) <= self.cap[0] and m.get("price_out", 0) <= self.cap[1]}
        expanded = []
        for c in self.candidates:
            if c.endswith("*"):
                expanded += sorted(i for i in allowed if i.startswith(c[:-1]) and not any(s in i.lower() for s in self._SKIP))
            else:
                expanded.append(c)
        for c in expanded:
            if c in allowed:
                self.model = c; info = allowed[c]
                self.json_mode = "response_format" in (info.get("params") or [])
                self.price = (info.get("price_in", 0.0), info.get("price_out", 0.0))
                return c
        raise ProviderError(f"none of the candidate {self.tier} models is currently available within the price cap: {self.candidates}")
    def complete(self, system, user, temperature, max_tokens):
        if not self.key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        body = dict(model=self.model, temperature=temperature, max_tokens=self.max_tokens_override or max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra_body)
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        hdr = {"authorization": "Bearer " + self.key, "HTTP-Referer": "https://github.com/smhbamakan", "X-Title": "TelicFirm research"}
        try:
            r = _http("https://openrouter.ai/api/v1/chat/completions", hdr, body)
        except ProviderError as e:
            if self.json_mode and "response_format" in str(e):
                self.json_mode = False; body.pop("response_format"); r = _http("https://openrouter.ai/api/v1/chat/completions", hdr, body)
            else:
                raise
        if "error" in r:
            msg = json.dumps(r["error"])[:400]
            if "429" in msg or "rate" in msg.lower():
                raise RateLimited(msg)
            raise ProviderError(msg)
        c0 = r["choices"][0]; ch = c0.get("message", {})
        text = ch.get("content") or ""
        meta = dict(usage=r.get("usage", {}), served_model=r.get("model"), provider_route=r.get("provider"),
                    finish_reason=c0.get("finish_reason"), had_reasoning=bool(ch.get("reasoning")))
        if not text.strip() and c0.get("finish_reason") == "length":
            # reasoning models can spend the whole token budget on hidden reasoning: retry once with a larger budget
            body["max_tokens"] = 3 * body["max_tokens"]
            r = _http("https://openrouter.ai/api/v1/chat/completions", hdr, body)
            c0 = r["choices"][0]; ch = c0.get("message", {}); text = ch.get("content") or ""
            meta.update(retried_for_length=True, finish_reason=c0.get("finish_reason"), usage=r.get("usage", {}))
        if not text.strip() and ch.get("reasoning") and "{" in ch["reasoning"]:
            text = ch["reasoning"]; meta["content_from_reasoning"] = True     # some free routes put the answer there
        if not text.strip():
            meta["empty_reply"] = True
        return text, meta

_LIMIT_PAT = re.compile(r"(usage limit|rate limit|limit reached|too many requests|quota)", re.I)

class ClaudeCLI:
    kind = "claude_cli"; supports_temperature = False
    def __init__(self, model="sonnet", timeout=300, exe=None, **_):
        self.model = model; self.timeout = timeout; self.exe = exe or shutil.which("claude") or "claude"
        self.cwd = tempfile.mkdtemp(prefix="telicfirm_claude_")   # empty dir: no project context is loaded
    def resolve(self, _=None):
        return f"claude-code:{self.model}"
    def complete(self, system, user, temperature, max_tokens):
        sysf = os.path.join(self.cwd, f"system_prompt_{time.time_ns()}.txt")   # unique per call (parallel workers)
        open(sysf, "w", encoding="utf-8").write(system)          # file + stdin: no shell quoting issues (Windows-safe)
        cmd = [self.exe, "-p", "--output-format", "json", "--model", self.model, "--system-prompt-file", sysf,
               "--no-session-persistence", "--strict-mcp-config", "--tools", ""]      # --tools is variadic: keep it last
        p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd, encoding="utf-8")
        out = (p.stdout or "").strip()
        try:
            j = json.loads(out)
        except Exception:
            if _LIMIT_PAT.search(out + p.stderr):
                raise RateLimited((out + p.stderr)[:300])
            raise ProviderError(f"claude CLI returned non-JSON (code {p.returncode}): {(out + p.stderr)[:300]}")
        text = j.get("result") or ""
        if j.get("is_error"):
            if _LIMIT_PAT.search(text):
                raise RateLimited(text[:300])
            raise ProviderError(text[:300])
        return text, dict(usage=j.get("usage", {}), cost_usd=j.get("total_cost_usd"), served_model=j.get("model") or self.model)

class CodexCLI:
    kind = "codex_cli"; supports_temperature = False
    def __init__(self, model=None, timeout=300, exe=None, **_):
        self.model = model; self.timeout = timeout; self.exe = exe or shutil.which("codex") or "codex"
        self.cwd = tempfile.mkdtemp(prefix="telicfirm_codex_"); self.stdin_ok = None
    def resolve(self, _=None):
        return f"codex-cli:{self.model or 'default'}"
    def complete(self, system, user, temperature, max_tokens):
        outf = os.path.join(self.cwd, f"last_{time.time_ns()}.txt")
        prompt = f"{system}\n\n{user}\n\nDo not run any commands or edit files. Reply with the JSON object only."
        cmd = [self.exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-o", outf]
        if self.model:
            cmd += ["--model", self.model]
        if self.stdin_ok is not False:                         # prompt via stdin ('-'): avoids shell quoting issues
            p = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd, encoding="utf-8")
            if os.path.exists(outf) or self.stdin_ok:
                self.stdin_ok = True
            else:
                self.stdin_ok = False
        if self.stdin_ok is False:
            p = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd, encoding="utf-8")
        text = open(outf, encoding="utf-8").read() if os.path.exists(outf) else ""
        if not text.strip():
            msg = (p.stdout + p.stderr)[-400:]
            if _LIMIT_PAT.search(msg):
                raise RateLimited(msg)
            raise ProviderError(f"codex exec produced no message (code {p.returncode}): {msg}")
        return text, dict(served_model=self.model or "codex-default")

class OpenAICompat:
    """Any OpenAI-compatible chat endpoint with an API key from an environment variable:
    Google Gemini API (free tier), Mistral La Plateforme (free Experiment plan), Alibaba Model Studio
    (Qwen, international endpoint), and others. The model is resolved against the endpoint's /models list
    when it is available (exact ids first, then 'prefix*' wildcards); otherwise the first candidate is used."""
    kind = "openai_compat"; supports_temperature = True; needs_resolve = True
    def __init__(self, base_url, key_env, candidates, max_tokens=None, extra_body=None, json_mode=True, chat_url=None, **_):
        if base_url.startswith("env:"):                      # e.g. "env:JOYTOKENS_BASE_URL"
            base_url = os.environ.get(base_url[4:], "") or "MISSING_" + base_url[4:]
        self.base = base_url.rstrip("/"); self.key_env = key_env; self.key = os.environ.get(key_env)
        self.candidates = candidates; self.model = None; self.max_tokens_override = max_tokens; self.chat_url = chat_url
        self.extra_body = dict(extra_body or {}); self.json_mode = json_mode
    def _hdr(self):
        if not self.key:
            raise ProviderError(f"{self.key_env} is not set")
        return {"authorization": "Bearer " + self.key}
    def list_models(self):
        r = _http(self.base + "/models", self._hdr())
        return [str(m.get("id", "")).replace("models/", "") for m in r.get("data", [])]
    def resolve(self, _=None):
        try:
            avail = set(self.list_models())
        except Exception:
            avail = None
        if not avail:
            self.model = self.candidates[0]; return self.model
        for c in self.candidates:
            hits = sorted(i for i in avail if i.startswith(c[:-1])) if c.endswith("*") else ([c] if c in avail else [])
            if hits:
                self.model = hits[0]; return self.model
        raise ProviderError(f"none of {self.candidates} is offered at {self.base} (available: {sorted(avail)[:40]})")
    def complete(self, system, user, temperature, max_tokens):
        body = dict(model=self.model, temperature=temperature, max_tokens=self.max_tokens_override or max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra_body)
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        url = self.chat_url or self.base + "/chat/completions"
        try:
            r = _http(url, self._hdr(), body)
        except ProviderError as e:
            msg = str(e)
            if msg.startswith("HTTP 400") and (self.json_mode or self.extra_body):
                # an optional parameter is not supported by this endpoint/model: drop optional parameters once
                self.json_mode = False; self.extra_body = {}
                for k in ("response_format",):
                    body.pop(k, None)
                body = {k: v for k, v in body.items() if k in ("model", "temperature", "max_tokens", "messages")}
                r = _http(url, self._hdr(), body)
            elif "quota" in msg.lower() or "rate" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                raise RateLimited(msg)
            else:
                raise
        c0 = r["choices"][0]; ch = c0.get("message", {}); text = ch.get("content") or ""
        if isinstance(text, list):
            text = "".join(x.get("text", "") for x in text if isinstance(x, dict))
        return text, dict(usage=r.get("usage", {}), served_model=r.get("model") or self.model, finish_reason=c0.get("finish_reason"))

class Mock:
    kind = "mock"; supports_temperature = True
    def __init__(self, seed=0, **_):
        self.rng = random.Random(seed)
    def resolve(self, _=None):
        return "mock"
    def complete(self, system, user, temperature, max_tokens):
        r = self.rng; u = user
        if '"amendments"' in u:
            dims = re.findall(r'"dimension": "([a-z|]+)"', u)[0].split("|")
            k = r.choice(dims); return json.dumps({"amendments": [{"dimension": k, "new_weight": round(r.uniform(.05, .5), 3), "grounds": "external evidence", "justification": "mock"}], "change_constitutive": None, "rationale": "mock", "evidence": ["mock"]}), {}
        if '"decisions"' in u:
            n = u.count('"index"') or 1
            return json.dumps({"decisions": [{"index": i, "approve": r.random() < .6, "reason": "mock"} for i in range(n)]}), {}
        if '"veto"' in u:
            return json.dumps({"veto": r.random() < .05, "reason": "mock"}), {}
        if '"answers"' in u:
            ids = re.findall(r'"id": "([a-z0-9]+)"', u)
            return json.dumps({"answers": [{"id": i, "answer": r.choice("AB"), "estimate": r.uniform(-10, 40)} for i in ids]}), {}
        if '"envelope"' in u:
            return json.dumps({"decision": {"net_hiring_fte": r.randint(-10, 20), "wage_premium": round(r.uniform(-.02, .08), 3), "rnd_spend": round(r.uniform(.5, 2), 2),
                                            "capex": round(r.uniform(.8, 2.5), 2), "compliance_spend": round(r.uniform(.3, 1), 2), "net_borrowing": 0, "uncertified_supplier": False},
                               "envelope": {"bu1": 1.2, "bu2": 1.1}, "rationale": "mock", "evidence": ["mock"]}), {}
        if '"price_index"' in u:
            return json.dumps({"price_index": round(r.uniform(.95, 1.1), 3), "quality_spend": round(r.uniform(.4, 1), 2), "marketing": round(r.uniform(.2, .8), 2), "rationale": "mock"}), {}
        if '"decision"' in u:
            return json.dumps({"decision": {"price": round(r.uniform(.3, .7), 2), "quality": round(r.uniform(.4, .9), 2), "compliance_spend": round(r.uniform(.2, .8), 2),
                                            "wage": round(r.uniform(.3, .7), 2), "shortcut": r.random() < .03}, "rationale": "mock", "evidence": ["mock"]}), {}
        return "{}", {}

KINDS = dict(openrouter=OpenRouter, openai_compat=OpenAICompat, claude_cli=ClaudeCLI, codex_cli=CodexCLI, mock=Mock)
