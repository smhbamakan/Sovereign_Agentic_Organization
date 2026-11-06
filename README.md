# The Sovereign Agentic Organization

Code, data and analysis for the research article

> **The Sovereign Agentic Organization: Telic Agency, Goal Displacement, and the Internal Constitution of Organizations Governed by AI Agents**
> Seyed Mojtaba Hosseini Bamakan, Department of Business Management, Faculty of Economics and Administration, Masaryk University, Brno.
> Manuscript under review (*Journal of Management Information Systems*, special issue “Agentic Organizations”).

As AI agents move from executing delegated tasks to roles in which they decide what an organization should prioritize, the goal itself becomes an object of delegation. The project studies the limiting case: a **sovereign agentic organization (SAO)**, in which AI agents hold residual control over actions, protocols and priority weights (the *telos*) within a human institutional envelope. A formal model of telic revision is tested in experiments where councils of large language model (LLM) agents govern two simulated firms for ten simulated years.

This repository contains everything needed to rerun the experiments with your own model access and to reproduce every number, table and figure in the manuscript from the archived decision ledger.

---

## Repository layout

```
runner/            Experiment runner: simulated firms, prompts, LLM providers, rate limits, checkpoints
  core/            Case A firm (env.py), warranted-telos computation, probe battery, measures
  core/layer_b/    Case B firm (Halvorsen Industries), its prompts and warranted telos
  runner/          Episode loop, agents, providers (OpenAI-compatible APIs, Claude Code CLI, Codex CLI)
  config/          Study configurations (main.json, frontier.json, supplementary.json, ...) and models.json
analysis/          Analysis scripts (one per output file, see below)
model/             Monte Carlo verification of the formal propositions
data/results.zip   Complete decision ledger of all 190 episodes (every prompt, reply, state and amendment)
outputs/           Derived results (JSON/CSV) produced by run_analysis.sh
figures/           Figures 1-7 of the manuscript (600 dpi PNG)
run_analysis.sh    Reproduces outputs/ and figures/ from data/results.zip
```

## Quick start: reproduce the results

```bash
python -m pip install -r requirements.txt
./run_analysis.sh            # about 2 minutes; unpacks data/results.zip on first run
```

No API access is needed. The script rebuilds every derived file in `outputs/` and every figure in `figures/`; the committed versions are the ones reported in the manuscript.

| Output | Script | Content (manuscript element) |
|---|---|---|
| `outputs/main/windows.csv`, `episodes.csv` | `build_data.py` | Window- and episode-level measures |
| `outputs/main/results_main.json` | `analyze.py` | Mixed models of constitutional factors (Table 8), arrangements (Table 6), H6–H7 |
| `outputs/main/coupling2.json` | `coupling2.py` | Unrestricted revision dynamics (Table 7, Fig. 4) |
| `outputs/main/coupling.json`, `spec_an.json` | `coupling.py`, `spec_an.py` | Planned gap specification and specification comparison |
| `outputs/main/recovery.json` | `recovery.py` | Recovery study for the coupling estimator |
| `outputs/main/probes.json`, `probes_retest.json` | `probes_an.py`, `probes_retest.py` | Error correlations between models (Fig. 5) |
| `outputs/main/review_an.json` | `review_an.py` | Ratifier discrimination (Table 9), β(τ) (Fig. 6), numeric error floor, majority vote, episode accounting |
| `outputs/followup/supp.json`, `perm.json` | `supp_an.py`, `perm.py` | Follow-up experiments S1–S4 and permutation tests (Fig. 7) |
| `outputs/model/verify*_results.json` | `model/verify*.py` | Numerical verification of Propositions 1–5 |
| `figures/fig1–fig7` | `figs_pub.py`, `fig7.py` | All manuscript figures |

## Experimental design

**Two firms.** *Case A* is a parsimonious manufacturer (4 decisions per quarter, 3 state variables, 4 telos dimensions: profit, customer, employee, compliance), built for mechanism identification. *Case B*, Halvorsen Industries, is a fictitious European manufacturer (about 500 employees, EUR 150 million turnover, two business units, 12 decisions in monetary units, 20 state variables, a reacting rival, a 3.5 × EBITDA leverage covenant), calibrated to Eurostat data on EU manufacturing. Each episode runs 40 quarters with three regime shifts (stricter liability enforcement, a safety-driven demand change, a labor shortage).

**Constitutional manipulations** (2 × 2 × 2 × 2): telos revision by the executive or by a separate steward; the achieved outcome profile shown to or withheld from the reviser; amendments ratified by an independent agent or not; councils drawn from one model family or from several developers. A critic agent can veto decisions that breach constitutive commitments. Benchmarks: fixed charter, periodic founder revision, and (follow-ups) an oracle that receives the warranted telos every quarter.

**Warranted telos.** Agents are evaluated against an environment-defined benchmark: the priority weights whose induced reference policy maximizes long-run firm value (`runner/core/warrant.py`, `runner/core/layer_b/warrant_b.py`). Agents never see it.

**Episodes.**

| Study | Config | Episodes |
|---|---|---|
| Case A main study (16 cells × 6, fixed charter, periodic) | `main.json` | 108 |
| Case A frontier replication (Claude executor, GPT/Codex reviser) | `frontier.json` | 8 |
| Case B (16 cells × 1, fixed charter, periodic) | `main.json` | 18 |
| S1 perturbed or ablated performance information | `supplementary.json` | 16 |
| S2 common ratifier, blind random ratifiers, oracle telos | `supplementary.json` | 20 |
| S3 Gemini chief executive in Case B, oracle and fixed charter | `supplementary.json` | 16 |
| S4 Claude executor in Case A | `supplementary.json` | 4 |
| **Total** | | **190** |

Models: Gemini 3.1 Flash-Lite, Doubao Seed 2.0 Pro and GPT-5.6 Luna through an OpenAI-compatible gateway; Claude Sonnet through the Claude Code CLI; GPT through the Codex CLI. The served model identifier of every call is logged in the ledger.

## Main findings

1. **Compensatory coupling.** Revising agents responded to the organization’s own outcomes whenever the profile was visible, raising the priority of dimensions on which achievement lagged. A randomized perturbation of the displayed profile confirms the response causally: raising displayed achievement by 0.10 lowered a weight by 0.74 percentage points (b = −0.074, SE 0.005). Frontier models followed achievement instead; withholding performance information removed the response.
2. **Ratification without discrimination.** Ratifiers approved amendments toward and away from the warranted telos at indistinguishable rates (e.g. odds ratio 1.05 [0.82, 1.35], AUC 0.51), and blind random ratifiers with the same approval rate reproduced their outcomes.
3. **Correlated errors.** Models from different developers shared errors (φ 0.30–0.59; numeric ρ 0.57–0.89); majority vote of three models was less accurate than the best member; role personas did not decorrelate a model (φ 0.67–0.84).
4. **Exercised vs nominal authority.** Designer-set charters shaped agent-governed priorities for several years (β falls from 0.94 to about 0 by quarter 30), and capable executives holding revision rights rarely used them.
5. **Arrangements and capability.** In Case A, periodic human revision < agent sovereignty < fixed charter in value regret (3.1 < 7.6 < 14.7). The capability of the executing agent mattered more for realized performance than any constitutional arrangement, and supplying the warranted telos did not raise realized value for LLM executives.

## Running new experiments

The runner needs Python ≥ 3.10 and access to at least one OpenAI-compatible endpoint; Claude Code and Codex CLIs are optional (used through the user’s subscription).

```bash
cd runner
python -m pip install -r requirements.txt
export JOYTOKENS_API_KEY=...            # or any key referenced by key_env in config/models.json
python run.py check --config config/smoke_mock.json          # offline pipeline test, no API calls
python run.py estimate --config config/supplementary.json    # calls and time per provider
python run.py run --config config/supplementary.json --wait --workers 8
```

- Providers, model identifiers, councils (role → model), per-provider rate limits and hard spending caps are set in `config/models.json`; API keys are read only from environment variables and never written to disk.
- Runs checkpoint every quarter and resume with the same command after an interruption; results are packaged as a zip in `runner/outputs/`.
- `probe_models.py` screens the models of any OpenAI-compatible gateway on the experiment’s four task types and flags models whose hidden reasoning exhausts the token budget.
- See `runner/README.md` for configuration details.

## Data

`data/results.zip` (≈ 10 MB, 77 MB unpacked) contains `results/manifest.json` (configuration and status of every episode), `results/episodes/*.jsonl` (quarter-level telos before and after revision, warranted telos, outcomes, decisions, amendments and approvals), `results/calls/*.jsonl` (every model call: prompt, raw reply, parse status, latency, token usage, served model) and `results/probes/*.jsonl` (probe-battery answers). All content is synthetic; no personal data is included.

## Citation

If you use this code or data, please cite the article (citation details will be added on publication):

```
Hosseini Bamakan, S. M. The Sovereign Agentic Organization: Telic Agency, Goal Displacement,
and the Internal Constitution of Organizations Governed by AI Agents. Manuscript under review, 2026.
```

## Funding

This research was supported by the project “MSCAfellow7_MUNI” (registration number CZ.02.01.01/00/22_010/0008854).

## License

No license has been chosen yet; until one is added, all rights are reserved by the author.
