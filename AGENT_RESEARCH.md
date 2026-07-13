# Agent Research Pipeline

This document describes the multi-agent workflow for running fine-grained image
classification experiments in this repo. It's a reference for how the agents
are organized and how work flows from an idea to a reviewed result — not an
implementation of the agents themselves (see "Open items" below).

## Research directions

The initial set of directions we're pursuing:

1. **Interpretability of fine-grained performance** — which heads/layers carry
   fine-grained discriminative signal, and how this differs across
   architecture paradigms (conv, transformer, hybrid).
2. **Scaling laws for label count** — how performance and training dynamics
   change as the number of classes grows (e.g. toward 30k/40k/50k labels).
3. **Training paradigms for fine-grained recognition** — augmentations,
   adapters, or training modifications that specifically help fine-grained
   (vs. coarse) classification.
4. **Other fine-grained testbeds** — identifying and evaluating additional
   benchmarks beyond the Kaggle herbarium datasets.
5. **Zero-shot recognition with contrastive learning** - can we replicate the success of BioCLIP
in zero-shot classification in the herbarium domain? Investigate different combinations of
text and image encoders to assess their impact. 

New directions can be added following the same structure.

## Philosophy

This is a **collaborative** pipeline, not a fully autonomous one. Agents
propose, implement, and smoketest; the user approves real job launches and
drives result interpretation. Automation is reserved for the tedious,
low-judgment part of the loop — polling SCC job status — not for decisions
about what to run or what results mean.

## Agent roles

| Agent | Scope | Model tier | Responsibilities |
|---|---|---|---|
| Experimentation | Per-direction | Fast/small | Implements code changes, writes and runs smoketests, launches approved SCC jobs, updates the ledger |
| Exploration | Per-direction | Fast/small | Proposes new hypotheses/experiments within its direction, from scratch or user nudging |
| Reflection / Evaluation | Cross-cutting (all directions) | Strongest available | Interprets completed results, gives feedback and suggests next steps, reviewed on user request |
| Documentation | Cross-cutting (all directions) | — | Synthesizes objectives, results, and milestones across all directions into manuscript-ready writeups |

Experimentation and exploration agents are instantiated **per research
direction** so their context stays focused on one thread. Reflection and
documentation are **singletons** that see across all directions, since
cross-direction synthesis (e.g. "does the augmentation finding also explain
the scaling law result?") is their job.

## Workspace layout

Each research direction gets its own top-level folder (not nested under a
shared `research/` directory):

```
<direction>/
  ledger.md      # one row per SCC job: status, smoketest result, checkpoint path
  notes.md       # running log of hypotheses, what was tried, what the exploration agent proposed
```

Example: `scaling-laws/ledger.md`, `interpretability/notes.md`.

Agent configs live in `.claude/agents/`, one file per (direction, role) pair,
e.g. `.claude/agents/interpretability-experimentation.md`.

## Job lifecycle

1. **Implement** — the experimentation agent for a direction writes or edits
   a training/eval script.
2. **Smoketest** — the agent runs a smoketest before any real launch (tiny
   subset, few steps). There's no shared smoketest harness; each
   experimentation agent implements this however fits its script. The agent
   reports pass/fail directly to the user and does not proceed on its own.
2. **Approve** — the user approves the real run based on the smoketest result.
3. **Launch** — the agent submits the SCC job (`qsub`) and adds an entry to
   that direction's `ledger.md`.
4. **Track** — a single cron routine scans **all** directions' ledgers
   (not one cron job per direction), checks `qstat` and `trainer_state.json`
   for jobs marked `running`, and flips their status to `done`/`failed`. It
   only updates status and notifies the user — it does not interpret results.
5. **Review** — when the user is ready, they bring in the reflection agent
   for that direction, pointing it at the ledger entry, checkpoint, and eval
   output. This step is user-triggered, not automatic.

## Ledger schema

Each job entry in a direction's `ledger.md` (or `ledger.json`) tracks:

```
job_id          # SCC job id
script          # path to the training/eval script
args            # key args/config used for the run
smoketest       # pass | fail | pending
status          # queued | running | done | failed
checkpoint_path # output checkpoint location, once available
launched_at     # timestamp
notes           # free-form context
```

## Open items / next steps

Built so far:

- Per-direction top-level folders and `ledger.md` / `notes.md` for
  `interpretability/` and `scaling-laws/`
- Experimentation and exploration agent configs for those two directions
  (`.claude/agents/`)

Still to do:

- The cross-direction cron routine that polls SCC job status
- Reflection and documentation agent configs
- Folders/agents for the remaining directions (training paradigms, other
  testbeds, zero-shot contrastive) once we're ready to scaffold them
