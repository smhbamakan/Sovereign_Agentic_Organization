# TelicFirm experiment runner (v2.5)

## v2.5 — supplementary study (response to review)
`config/supplementary.json`, 56 episodes (~8,000 JoyTokens calls, ~590 Claude calls):
S1 randomized perturbation of the displayed outcome profile (sd 0.10, logged as `delta`) and full signal ablation;
S2 common GPT-5.6 Luna ratifier/critic in a Gemini council and blind random ratifiers (25%, 90%);
S3 Gemini CEO in Case B, oracle (telos = warranted telos each quarter) and fixed-charter baselines for both CEOs;
S4 Claude executor in Case A; probe battery under a ratifier persona.
```
python run.py estimate --config config/supplementary.json
python run.py run --config config/supplementary.json --wait
```

Runs Study 1 of *The Sovereign Agentic Organization* with real language-model councils and packages all
data, logs and model outputs into one zip file.

## راهنمای سریع (فارسی)

۱. پایتون ۳.۱۰ یا بالاتر نصب باشد. سپس در پوشهٔ همین بسته اجرا کنید:
```
pip install -r requirements.txt
```
۲. کلیدهای API را تنظیم کنید (پیش‌فرض: `MISTRAL_API_KEY`، `GEMINI_API_KEY` و `DASHSCOPE_API_KEY`؛ برای OpenRouter: `OPENROUTER_API_KEY`). نمونه برای OpenRouter: در Windows PowerShell: `$env:OPENROUTER_API_KEY="sk-or-..."`، و در macOS/Linux: `export OPENROUTER_API_KEY=sk-or-...`.
   مدل‌های رایگان بدون خرید اعتبار فقط **۵۰ درخواست در روز** دارند. با یک‌بار خرید ۱۰ دلار اعتبار، سقف به **۱۰۰۰ درخواست در روز** می‌رسد و استفاده از مدل‌های رایگان از این اعتبار کم نمی‌کند. اگر اعتبار نخریده‌اید، در `config/models.json` مقدار `per_day` را برای openrouter به 45 تغییر دهید.
۳. Claude Code و Codex باید در ترمینال با همان حسابی کار کنند که در VS Code وارد آن شده‌اید. با `claude --version` و `codex --version` بررسی کنید. اگر متغیر `ANTHROPIC_API_KEY` تنظیم شده باشد، Claude Code از API پولی استفاده می‌کند. برای استفاده از اشتراک، آن را خالی کنید.
۴. محیط را بررسی کنید و با هر مدل یک تماس آزمایشی بگیرید:
```
python run.py check --config config/smoke_live.json --live
```
۵. آزمون کوتاه زنده را اجرا کنید (حدود ۱۵۰ تماس):
```
python run.py run --config config/smoke_live.json
```
فایل zip در پوشهٔ `outputs/` ساخته می‌شود. همان را بفرستید.
۶. بعد از آن به ترتیب پایلوت (`config/pilot.json`)، اجرای اصلی (`config/main.json`) و اجرا با مدل‌های پیشرو (`config/frontier.json`) را اجرا کنید.
   برای اجرای چندروزه از `--wait` استفاده کنید تا اجرا با رسیدن به سقف روزانه منتظر بماند:
```
python run.py run --config config/main.json --wait
```
اگر اجرا قطع شود، مثلاً با بسته شدن لپ‌تاپ یا رسیدن به سقف مصرف، **همان فرمان را دوباره اجرا کنید**. اجرا از همان فصلی که متوقف شده ادامه پیدا می‌کند. هر بار که اجرا متوقف یا تمام شود، خلاصه و فایل zip به‌روز ساخته می‌شوند.

پیش از اجرای اصلی می‌توانید تعداد تماس‌ها و زمان لازم را ببینید:
```
python run.py estimate --config config/main.json
```

## What the configurations do

| config | content | approx. calls | time at default limits |
|---|---|---|---|
| smoke_mock | offline pipeline test, no API | — | seconds |
| smoke_live | every model answers; one short episode per layer; Claude/Codex answer 10 probe items | ~150 | < 1 hour |
| pilot | Layer A: 16 cells + 2 baselines × 1 episode; Layer B: 2 cells; 60 probe items | ~3,000 OpenRouter | a few hours (paid) / ~3 days (free) |
| main | Layer A: 16 cells × 6 + baselines; Layer B: 16 cells × 2 + baselines; 360 probe items | ~23,000 OpenRouter | ~1–2 days (paid) / ~24 days (free) |
| frontier | Layer A key cells with Claude (executor) and Codex/GPT (reviser), 2 episodes each; 120 probe items | ~340 Claude, ~340 Codex, ~480 OpenRouter | ~2 days |

## Councils and roles (config/models.json)

Default councils (configs smoke_live, pilot, main, frontier) need three API keys plus your Claude Code
and Codex logins:

| key (environment variable) | service | cost |
|---|---|---|
| `MISTRAL_API_KEY` | Mistral La Plateforme, free "Experiment" plan (phone verification) | free |
| `GEMINI_API_KEY` | Google Gemini API free tier (Google AI Studio) | free |
| `DASHSCOPE_API_KEY` | Alibaba Cloud Model Studio, international (Singapore) endpoint, Qwen Flash | pay-as-you-go, cents |

| role | homogeneous | heterogeneous | frontier |
|---|---|---|---|
| executor (Layer B: CEO + two BU heads) | Mistral Small | Mistral Small | Claude (Claude Code CLI) |
| telos reviser / steward | Mistral Small | Qwen Flash (Alibaba) | GPT (Codex CLI) |
| critic | Mistral Small | Gemini Flash-Lite (Google) | Gemini Flash-Lite |
| ratifier | Mistral Small | Gemini Flash-Lite | Qwen Flash |

The executor is identical in the homogeneous and heterogeneous councils. Alternatives through OpenRouter:
`*_or.json` (paid tier, needs credits) and `*_free.json` (free endpoints; shared pools, often rate-limited).

## Parallel runs (v1.9)

Episodes run in parallel threads (`--workers`, default 12). Each provider group also has `max_concurrent`
in `config/models.json`: JoyTokens 12, Claude Code 1, Codex 1. `config/all.json` runs the main study and the
frontier replication together, so the subscription-limited Claude/Codex episodes proceed while the JoyTokens
episodes run. Rate limits, the spending cap (which reserves the worst case of every in-flight call) and
checkpoints are thread-safe; an interrupted or paused run resumes with the same command.
If the gateway returns many 429 errors, lower `--workers` (e.g. 6).

## Probing an API gateway (e.g. JoyTokens)

```
python probe_models.py --base-url <gateway base URL ending in /v1> --key-env JOYTOKENS_API_KEY --budget 0.5
```
Lists all models, tests up to 30 text models (cheapest first when prices are known) on four real tasks
(Layer B executor decision, Layer A telos revision, batched ratification, 15 simulator-keyed judgement items),
and writes `model_probe_<time>.zip` with a ranking: task success, valid JSON, judgement accuracy, cost per
1,000 calls, output and reasoning tokens, latency and flags. Each call is capped (max_tokens 1200, 90 s);
a model that runs out of tokens while 'thinking' is flagged `thinking_overrun` and gets no further calls;
the whole probe stops at `--budget` USD (or a total-token cap if prices are unknown). If `/models` has no
prices, copy them into a `prices.csv` (model,usd_per_m_in,usd_per_m_out) and pass `--prices prices.csv`.
During experiment runs, `budget_usd` in `limits` is a hard spending cap per provider group.

## Rate limits

Limits are enforced per provider and persisted in `state/`, so they hold across restarts:
Mistral 40/min; Gemini 8/min and 450/day (free tier, resets at midnight Pacific time); Qwen 60/min; OpenRouter paid 50/min; OpenRouter free 15/min and 950/day (the platform allows 20/min and 1,000/day after
$10 credits); Claude Code and Codex 3/min, at most 40 calls per rolling 5 hours and 300 per day.
Adjust `config/models.json` to your plans. On HTTP 429 or a "usage limit" message the runner backs off
exponentially and, if limits persist, stops cleanly and packages what it has.

## Output (zip)

```
results/manifest.json        configs, resolved model ids, temperature support, status of every episode, usage
results/episodes/*.jsonl     every quarter: telos before/after, warranted telos, outcomes, decisions, amendments
results/calls/*.jsonl        every model call: prompt, raw reply, parse status, latency, usage, served model id
results/probes/*.jsonl       probe answers for error-correlation estimates
results/summary.json, report.md   per-cell measures (value regret, signed displacement, violations, gamma-hat), call statistics
results/setup_report.json, free_models.json
```

Notes: temperature cannot be set for the Claude Code and Codex CLIs (recorded in the manifest). Free
OpenRouter providers may log prompts; all content is synthetic. No personal data is sent.
