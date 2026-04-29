# Praxis

[![CI](https://github.com/vanlabs-dev/praxis-subnet/actions/workflows/ci.yml/badge.svg)](https://github.com/vanlabs-dev/praxis-subnet/actions/workflows/ci.yml)

A Phase 1 prototype validating the mechanism for a Bittensor subnet that distributes mechanically-verifiable reinforcement-learning environments. Bonded creator-miners submit gymnasium-compliant envs with declared properties; validators replay them deterministically and verify against the declared claims.

## Status

Phase 1 (mechanism prototype). 8 of 11 planned steps complete. NOT a deployable subnet. The current scope proves out the validator's verification logic against synthetic envs in pure Python; no Bittensor SDK integration, no on-chain components, no production network.

What works today:

- `EnvManifest` protocol (Pydantic v2, content-addressable env IDs, declared reward bounds, anchor trajectories, kwargs, env_version, reference solver)
- One reference env: parameterised gridworld in three difficulty bands
- Four validator checks composed against any conforming manifest:
  - Determinism (anchor-match + self-consistency at validator-derived seeds)
  - Reward bounds (per-step strict, per-episode strict on naturally-terminated episodes)
  - Reset correctness (seven-category multi-seed adversarial coverage)
  - Solver baseline (normalized return at or above per-band threshold, with random-policy diagnostic)
- Pluggable solver registry. Phase 1 ships tabular Q-learning; Phase 2 will add cleanrl PPO under the same interface.
- 182 tests (pytest), CI on push and PR (Python 3.12, ruff, mypy strict, pytest).

What does not exist yet:

- The remaining three Phase 1 steps: validator pipeline that orchestrates all four checks into a single report, a bonded-submission shim against a local anvil node, an end-to-end demo.
- Anything Bittensor-specific (neurons, axon/dendrite, subtensor calls, on-chain identity, weight-setting). Phase 2 work.
- Sandboxing of creator-supplied env code. Documented as RT-001 finding F-003 (deferred to Phase 2).

## Architecture

````
src/praxis/
  protocol/        EnvManifest, hashing, types: the bonded contract
  envs/            reference gymnasium envs (gridworld)
  solver/          Solver protocol, TabularQLearning, SOLVER_REGISTRY
  checks/          determinism, reward_bounds, reset_correctness, solver_baseline
                   plus shared primitives: _rollout, _seeds
docs/red-team/     adversarial attack catalogs (RT-001)
scripts/           operator helpers (build_gridworld_manifest, etc.)
tests/             pytest suites mirroring the src layout
````

The validator never calls `gym.make()`. Envs load via `entry_point` + `importlib`, wrapped with `TimeLimit` from the manifest's declared `max_episode_steps`. This decouples the on-chain identifier (`env_id` slug) from the runtime resolver and is the pattern that scales unchanged into Phase 2 when bonded creator-miners ship their envs as installable packages. See decision record DR-001 in the commit history.

Validator-derived sample seeds are computed deterministically from the env-defining parts of the manifest (`env_id`, `env_version`, `entry_point`, `kwargs`) plus a per-check salt. Tweaking the declared properties a creator can lie about (reward bounds, anchor trajectories) does not change which seeds the validator samples, closing the obvious cherry-picking attack class. See `src/praxis/checks/_seeds.py`.

## Quickstart

Requires Python 3.12+ and `uv`.

```bash
git clone https://github.com/vanlabs-dev/praxis-subnet.git
cd praxis-subnet
uv sync
uv run pytest -v
```

Optional dev checks:

```bash
uv run ruff check .
uv run mypy src
```

## Running a check against a manifest

```python
from praxis.checks.determinism import check_determinism, check_determinism_self_consistency
from praxis.checks.reward_bounds import check_reward_bounds
from praxis.checks.reset_correctness import check_reset_correctness
from praxis.checks.solver_baseline import check_solver_baseline
from scripts.build_gridworld_manifest import build_easy_manifest

manifest = build_easy_manifest()

print(check_determinism(manifest).passed)
print(check_determinism_self_consistency(manifest).passed)
print(check_reward_bounds(manifest).passed)
print(check_reset_correctness(manifest).passed)
print(check_solver_baseline(manifest).passed)
```

A unified validator pipeline that orchestrates all four checks into a single report is Phase 1 step 9 (not yet implemented).

## Red-team posture

`docs/red-team/RT-001-determinism.md` catalogues attacks against the determinism check, with severity, repro details, and resolution status. Three of the original five HIGH/MEDIUM findings closed in Phase 1 (F-001 anchor cherry-picking, F-004 info side channel, F-005 budget invariant). Two HIGH findings deferred to Phase 2 because they require architectural work beyond Phase 1 scope:

- F-002 canonical action lookup. Mitigation requires chain-beacon entropy added to the seed derivation (validator-derived seeds become unpredictable per-validation-epoch).
- F-003 importlib sandbox. Mitigation requires process isolation or a module allow-list around the entry-point loader.

A consolidated red-team pass on the remaining checks (reward_bounds, reset_correctness, solver_baseline) and cross-cutting concerns is the next item after Phase 1 step 8.

## License

MIT. See [LICENSE](LICENSE).

## Author

[vaNlabs](https://github.com/vanlabs-dev).
