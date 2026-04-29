# Praxis: Project memory for Claude Code

Praxis is a Phase 1 prototype for a Bittensor subnet that distributes mechanically-verifiable RL environments. Bonded creator-miners submit gymnasium envs; validators replay and verify against declared claims. Pure Python, no Bittensor SDK yet.

Current phase: Phase 1 mechanism prototype. NOT a deployable subnet. Validator side only; bonded chain integration is Phase 2.

## How to work in this repo

### Commands

- `uv sync`: install deps
- `uv run pytest -v`: run all tests
- `uv run ruff check .`: lint (must be clean before any commit)
- `uv run mypy src`: type check (strict mode, must be clean)
- Python version: 3.12+ floor (see `.python-version`)
- All commands use `uv`. Never `pip install` directly.

### Verification before every commit

- `git status --porcelain` MUST be empty before commit.
- After commit, `git show --stat HEAD` MUST list every file the task expected.
- All three checks (ruff, mypy, pytest) MUST be green. Stop and report if any are red.

### Commit conventions

- Conventional Commits format: `type(scope): subject`. Types: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`, `ci`.
- Add `!` for breaking changes, with `BREAKING CHANGE:` trailer in the body.
- Body should reference relevant prior commits, decision records, and red-team findings (e.g. DR-001, F-005, RT-001).
- Never amend a pushed commit. Local-only commits may be amended if the original message is misleading.

### Scope rules (sub-agents)

Each agent edits only within its scope. Cross-scope changes split into staged commits.

- `env-architect`: `src/praxis/protocol/`, `src/praxis/envs/`
- `validator-engineer`: `src/praxis/checks/`
- `rl-researcher`: `src/praxis/solver/`, `src/praxis/baselines/`
- `property-tester`: `tests/` (excluding `tests/adversarial/`)
- `mechanism-red-teamer`: `tests/adversarial/`, `docs/red-team/`. Read anywhere, write only here. Never patches validator code; raises findings instead.
- `infra-engineer`: `pyproject.toml`, `.github/`, `.pre-commit-config.yaml`, `justfile`, `.python-version`, `uv.lock`, root-level docs (README, CLAUDE).

If a prompt requires changes outside its primary scope, the agent stops and reports rather than reaching across.

### Style and quality bars

- No em dashes anywhere in code, comments, commit messages, or docs.
- No `# type: ignore` in `src/`. If a generic is needed (e.g. `gym.Env`), parameterise with `Any` (e.g. `gym.Env[Any, Any]`). `Any` at runtime-dynamic boundaries is principled, not a band-aid.
- No band-aids. If a fix papers over a real issue, surface it for discussion instead of committing it.
- No new pip dependencies without explicit approval in the prompt.
- src layout (`src/praxis/...`), not flat. New modules with leading underscore (`_foo.py`) are internal.

### Public-facing content

- Author/brand is `vaNlabs` (lowercase v, capital N, lowercase labs). Never use any other personal name in README, docs, repo prose, or commit messages visible to others.
- GitHub org: `vanlabs-dev`. Repo: `praxis-subnet`.

### Prompt response protocol

- After completing the task in a prompt, stop and report. Do not proceed to the next stage of a multi-stage prompt without an explicit go.
- Report includes: folder tree of changed area, pytest output, ruff output, mypy output, `git show --stat HEAD`.
- If a verification step fails, do not commit. Stop and surface the failure.
- If the task surfaces an unrelated issue, flag it but do not fix it in the same commit.

## Architecture

```
src/praxis/
  protocol/        # EnvManifest, hashing, types -- the bonded contract
  envs/            # reference gymnasium envs (gridworld)
  solver/          # Solver protocol, TabularQLearning, SOLVER_REGISTRY
  checks/          # determinism, reward_bounds, reset_correctness, solver_baseline
                   # plus shared primitives: _rollout, _seeds
docs/red-team/     # adversarial attack catalogs (RT-001 etc.)
scripts/           # operator helpers
tests/             # pytest suites mirroring src layout
```

Key conventions:

- Validator never calls `gym.make()`. Envs load via `entry_point` + `importlib`, wrapped with `TimeLimit` from `manifest.max_episode_steps`. See DR-001 in commit history.
- Sample seeds for sample-based checks derive from env-defining manifest fields plus a per-check salt (see `src/praxis/checks/_seeds.py`). Each check has a unique salt: `b"reward_bounds"`, `b"reset_correctness"`, `b"determinism_self_consistency"`, `b"solver_baseline"`, `b"solver_baseline_eval"`.
- New checks reuse `derive_validator_seeds` and `iter_rollout` from `_seeds.py` and `_rollout.py`. Do not reimplement.

## Phase 1 status

8 of 11 steps complete. 182 tests, CI green on push and PR.

Closed: scaffold; protocol; one reference env (gridworld); four validator checks (determinism with self-consistency, reward bounds, reset correctness, solver baseline); pluggable solver registry; consolidated post-step-8 red-team Stage 1 (RT-001 through RT-005, 38 attacks catalogued, 40 findings, 5 CRITICAL, 3 of 5 RT-001 findings closed in code).

Remaining: validator pipeline orchestration; bonded-submission shim; end-to-end demo. Plus a triage fix-pass on the cheap CRITICAL findings (F-021 band declaration, F-003/F-032 sys.modules guard, F-023 reference_solver) before step 9, and a Stage 2 PoC pass at the end of Phase 1.

## Known limitations carrying forward

- F-002 (canonical action lookup): DEFERRED, HIGH (Phase 2 chain-beacon entropy)
- F-003 (importlib sandbox): DEFERRED, CRITICAL reassessed by RT-005 (Phase 2 process isolation; interim sys.modules guard planned for Phase 1 fix-pass)
- F-031 (cross-cutting composition): DEFERRED, CRITICAL (Phase 2 architectural; mitigated indirectly by closing F-021, F-023, and the F-003/F-032 interim guard)
- Solver baseline is lower-bound only. `trivial_random_warning` flag surfaces likely upper-bound gaps; Phase 2 hardens this.
