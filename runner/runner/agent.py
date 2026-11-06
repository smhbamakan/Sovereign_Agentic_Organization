"""Role agent: rate limiting, retries, JSON parsing with one repair attempt, and full call logging."""
import json, re, threading, time

STOP = threading.Event()          # set to stop all workers cleanly at their next call
from .providers import RateLimited, ProviderError
from .ratelimit import QuotaExhausted

def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))

class Agent:
    def __init__(self, role, model_key, provider, limiter, log, temperature=0.7, max_tokens=900, max_rate_retries=4,
                 spend=None, price=(0.0, 0.0)):
        self.role, self.model_key, self.p, self.lim, self.log = role, model_key, provider, limiter, log
        self.spend, self.price = spend, price                         # price: USD per token (in, out)
        self.max_rate_retries = max_rate_retries
        self.temperature, self.max_tokens = temperature, max_tokens
        self.family = model_key
        self.model_id = provider.model if getattr(provider, "model", None) else provider.resolve()

    def _call(self, system, user):
        tries = 0
        while True:
            if STOP.is_set():
                raise QuotaExhausted("stopped")
            self.lim.acquire(); t0 = time.time()
            mt = getattr(self.p, "max_tokens_override", None) or self.max_tokens
            worst = (len(system) + len(user)) / 3.5 * self.price[0] + mt * self.price[1]
            if self.spend:
                self.spend.check(worst)
            try:
                text, meta = self.p.complete(system, user, self.temperature, self.max_tokens)
                self.lim.ok()
                u = (meta or {}).get("usage") or {}
                cost = (u.get("prompt_tokens") or 0) * self.price[0] + (u.get("completion_tokens") or 0) * self.price[1]
                if not u and self.price != (0.0, 0.0):
                    cost = (len(system) + len(user)) / 3.5 * self.price[0] + mt * self.price[1]     # no usage reported: worst case
                if meta is not None:
                    meta["cost_usd"] = round(cost, 6)
                if self.spend:
                    self.spend.add(cost, worst)
                return text, meta, round(time.time() - t0, 2)
            except RateLimited as e:
                if self.spend:
                    self.spend.add(0.0, worst)
                tries += 1
                msg = str(e)[:400]
                self.log.write(json.dumps(dict(ev="rate_limited", role=self.role, model_key=self.model_key, model=self.model_id,
                                               attempt=tries, time=time.strftime("%Y-%m-%d %H:%M:%S"), message=msg)) + "\n"); self.log.flush()
                print(f"[{self.model_key}] rate-limited (attempt {tries}): {msg[:200]}", flush=True)
                if tries > self.max_rate_retries:
                    raise QuotaExhausted(f"{self.model_key} ({self.model_id}) keeps being rate-limited: {msg[:200]}")
                self.lim.rate_limited()
            except (ProviderError, OSError, TimeoutError) as e:
                if self.spend:
                    self.spend.add(0.0, worst)
                tries += 1
                if tries > 3:
                    raise
                time.sleep(10 * tries)

    def ask(self, system, user, ctx):
        """Returns (parsed dict or {}, ok flag). Every attempt is logged."""
        prompt = user
        for attempt in range(2):
            try:
                text, meta, secs = self._call(system, prompt)
            except ProviderError as e:
                self.log.write(json.dumps(dict(ev="call", **ctx, role=self.role, model_key=self.model_key, model=self.model_id,
                                               attempt=attempt, error=str(e)[:500])) + "\n"); self.log.flush()
                return {}, False
            ok, parsed, err = True, None, None
            try:
                parsed = parse_json(text)
            except Exception as e:
                ok, err = False, str(e)
            self.log.write(json.dumps(dict(ev="call", **ctx, role=self.role, model_key=self.model_key, model=self.model_id, attempt=attempt,
                                           system=system if attempt == 0 and ctx.get("t", 0) == 0 else None, prompt=prompt, raw=text,
                                           parsed_ok=ok, parse_error=err, secs=secs, meta=meta), default=str) + "\n"); self.log.flush()
            if ok:
                return parsed, True
            prompt = user + "\n\nYour previous reply was not a valid JSON object. Reply again with the JSON object only, no other text."
        return {}, False
