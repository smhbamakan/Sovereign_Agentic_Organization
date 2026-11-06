"""Persistent, conservative rate limiting per provider.

Each provider has: min_interval_s (spacing between calls), per_minute, per_day (calendar day, UTC),
and an optional rolling window cap (window_hours, per_window) for subscription plans such as
Claude Code or Codex. Call timestamps are persisted in state/ so limits hold across restarts.
When a daily or window cap is reached the limiter either waits until the cap frees up
(wait=True) or raises QuotaExhausted so the run stops cleanly and can be resumed later."""
import json, os, time, datetime, threading
from .safeio import write_json

class QuotaExhausted(Exception):
    pass

class Limiter:
    def __init__(self, name, state_dir, min_interval_s=4.0, per_minute=15, per_day=900,
                 window_hours=None, per_window=None, wait=False, max_wait_s=6 * 3600):
        self.name, self.min_interval, self.per_minute, self.per_day = name, min_interval_s, per_minute, per_day
        self.window_s = window_hours * 3600 if window_hours else None
        self.per_window, self.wait, self.max_wait = per_window, wait, max_wait_s
        self.path = os.path.join(state_dir, f"ratelimit_{name}.json")
        self.calls = json.load(open(self.path)) if os.path.exists(self.path) else []
        self.backoff = 0; self.lock = threading.Lock()

    def _save(self):
        keep = max(86400 * 2, self.window_s or 0)
        now = time.time(); self.calls = [c for c in self.calls if now - c < keep]
        write_json(self.path, self.calls)

    def _today_count(self, now):
        day0 = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        return sum(c >= day0 for c in self.calls), day0 + 86400

    def acquire(self):
        while True:
            with self.lock:
                w = self._try()
            if w == 0:
                return
            time.sleep(w)

    def _try(self):
        """Returns 0 and records the call if allowed now, otherwise the seconds to wait (thread-safe via lock)."""
        if True:
            now = time.time(); waits = []
            if self.calls:
                waits.append(self.calls[-1] + self.min_interval - now)
            last_min = [c for c in self.calls if now - c < 60]
            if self.per_minute and len(last_min) >= self.per_minute:
                waits.append(last_min[0] + 60 - now)
            hard = None
            n_day, reset = self._today_count(now)
            if self.per_day and n_day >= self.per_day:
                hard = ("daily cap", reset - now)
            if self.window_s and self.per_window:
                in_win = [c for c in self.calls if now - c < self.window_s]
                if len(in_win) >= self.per_window:
                    w = in_win[0] + self.window_s - now
                    hard = ("window cap", max(w, hard[1] if hard else 0))
            if hard:
                if not self.wait or hard[1] > self.max_wait:
                    raise QuotaExhausted(f"{self.name}: {hard[0]} reached; frees up in {hard[1] / 3600:.1f} h")
                print(f"[{self.name}] {hard[0]} reached, waiting {hard[1] / 60:.0f} min", flush=True)
                return hard[1] + 5
            w = max(waits + [0])
            if w > 0:
                return w
            self.calls.append(time.time()); self._save(); return 0

    def rate_limited(self):
        """Called after an HTTP 429 or a 'usage limit' message: exponential backoff."""
        self.backoff = min(300, max(30, self.backoff * 2 or 30))
        print(f"[{self.name}] rate-limited by provider; backing off {self.backoff} s", flush=True)
        time.sleep(self.backoff)

    def ok(self):
        self.backoff = 0

    def usage(self):
        now = time.time(); n_day, _ = self._today_count(now)
        out = dict(today=n_day, per_day=self.per_day)
        if self.window_s:
            out.update(in_window=sum(now - c < self.window_s for c in self.calls), per_window=self.per_window)
        return out


class Spend:
    """Persistent USD spend guard per limit group. Cost of a call is computed from the token usage the
    endpoint reports and the configured prices; before each call the worst case (prompt estimate plus the
    full max_tokens budget at the output price) must fit under the budget, otherwise the run pauses."""
    def __init__(self, name, state_dir, budget_usd=None):
        self.name, self.budget = name, budget_usd
        self.path = os.path.join(state_dir, f"spend_{name}.json")
        self.usd = json.load(open(self.path))["usd"] if os.path.exists(self.path) else 0.0
        self.lock = threading.Lock(); self.reserved = 0.0
    def check(self, worst_case):
        """Reserve the worst case before a call (so parallel calls cannot jointly overshoot the budget)."""
        with self.lock:
            if self.budget is not None and self.usd + self.reserved + worst_case > self.budget:
                raise QuotaExhausted(f"{self.name}: spending budget {self.budget} reached (spent {self.usd:.4f})")
            self.reserved += worst_case
    def add(self, usd, worst_case=0.0):
        with self.lock:
            self.usd += usd; self.reserved = max(0.0, self.reserved - worst_case)
            write_json(self.path, dict(usd=self.usd))
