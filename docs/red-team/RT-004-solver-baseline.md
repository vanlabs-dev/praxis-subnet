# RT-004: Solver baseline check red-team pass

Status: Draft
Date: 2026-04-30
Scope: src/praxis/checks/solver_baseline.py and the solver registry it dispatches into (src/praxis/solver/registry.py, src/praxis/solver/tabular_q.py, src/praxis/solver/_protocol.py); shared primitives src/praxis/checks/_rollout.py and src/praxis/checks/_seeds.py
Reviewed commits: full chain through b96af87, special focus on 38d2fab (initial solver_baseline implementation), 3c5e753 and 23f5a9e (Solver protocol + TabularQLearning + SOLVER_REGISTRY), and ffda18d (manifest.reference_solver and v0.3.0 protocol bump)

## Threat model

The adversary is the same creator-miner profile as RT-001 / RT-002 / RT-003: they
ship an env package source-tree to validators, stake a Bittensor-style bond,
and want to maximize `expected_emissions - p_slash * collateral`. Their
RT-004-specific objective is the most economically charged of the four
checks: `check_solver_baseline` is the only validator check whose pass/fail
state directly maps to ongoing emissions at the creator's declared
difficulty band. A passing manifest earns at the band's emission rate; a
failing manifest earns zero. RT-002 and RT-003 attack checks that ask "is
this env well-formed?"; RT-004 attacks the check that asks "is this env
actually as hard as the creator says it is?" The economic incentive to
break this check is therefore strictly stronger than for the prior three,
and the surface is genuinely larger.

What the attacker controls (solver_baseline-specific, beyond the
RT-001/2/3 baseline):

- `manifest.difficulty_band` (`manifest.py:34`) -- a free three-way enum
  pick. The validator does not infer band from env behavior; the
  creator's declared band is what `band_cfg = cfg.band_configs[
  manifest.difficulty_band]` (`solver_baseline.py:196`) resolves.
- `manifest.reference_solver` (`manifest.py:35`) -- a creator-declared
  `SolverId`. The validator dispatches `solver = SOLVER_REGISTRY[
  manifest.reference_solver]` (`solver_baseline.py:197`) without any
  policy on whether the declared solver is appropriate for the env. In
  Phase 1 the registry has exactly one entry (`registry.py:8-10`,
  `SolverId.TABULAR_Q_LEARNING`), so today the choice is forced; the
  hazard becomes acute in Phase 2 once a second solver lands.
- `manifest.declared_reward_bounds` (`manifest.py:37`) -- the same
  field that RT-002 catalogued. solver_baseline normalizes the solver's
  raw mean episodic return as
  `(raw - min_per_episode) / (max_per_episode - min_per_episode)`
  (`solver_baseline.py:243`), clamped at zero from below with no upper
  clamp. The creator picks the divisor.
- The full env source. The env can recognize the solver's argmax-greedy
  policy, recognize the random-baseline canonical action sequence,
  detect train-vs-eval phase from observation visit patterns, and gate
  honesty on any of those signals.
- The four env-defining fields (`env_id`, `env_version`, `entry_point`,
  `kwargs`) that feed `derive_validator_seeds` for both
  `b"solver_baseline"` and `b"solver_baseline_eval"`
  (`solver_baseline.py:202, 207`). RT-002 A-106 and RT-001 F-002 already
  catalogued the brute-force surface; RT-004 inherits it on two new
  salts.

What the attacker cannot do that matters here:

- Modify the per-band thresholds (`solver_baseline.py:55-59`,
  DEFAULT_BAND_CONFIGS) or the salts `b"solver_baseline"` /
  `b"solver_baseline_eval"`. They can read the constants and plan
  around them, but cannot shift them.
- Modify `TabularQLearning` source. They can read it line-by-line, but
  not change the validator's copy.
- Force `passed=True` purely from a manifest field; the check actually
  runs the solver and measures its return. The attacker has to engineer
  the env so the measured return clears the threshold, then dishonestly
  exploit the gap between "env passes the band threshold under the
  reference solver on validator-derived seeds" and "env actually
  represents difficulty at the declared band."

Net: the validator's solver_baseline check is the gate between a creator's
declared difficulty band and ongoing emission. The attacker's profit
surface is the difference between the cheapest manifest that passes the
check and the most-honest manifest. Several distinct attack categories
collapse that surface to near zero.

## Attack catalog

### A-301: Vacuous reward bounds collapse normalization
- Category: bound declaration laziness (cross-cut from RT-002 F-006)
- Severity: HIGH
- Premise: A creator declares `min_per_episode = -1e6` and
  `max_per_episode = 1e6` (or any wide-span pair). The normalization
  `(raw - min_per_episode) / (max_per_episode - min_per_episode)`
  (`solver_baseline.py:243`) shrinks any plausible raw return into an
  arbitrarily small fraction of the threshold envelope. Conversely a
  creator can declare `min_per_episode = 0.99 - 1e-9` and
  `max_per_episode = 0.99 + 1e-9` so any episode that scrapes a single
  goal bonus normalizes to ~1.0.
- Mechanism: `_normalize` (`solver_baseline.py:242-244`) computes
  `norm = (raw - bounds.min_per_episode) / span`. `span` is the only
  guard, and it is enforced only via `max_per_episode > min_per_episode`
  at manifest-validation time (`types.py:31-32`). With the wide-span
  variant, an honest gridworld-style env returning raw=0.82 against
  bounds (-1e6, +1e6) gives `norm = (0.82 - -1e6) / 2e6 = ~0.5000004`,
  which still clears the EASY threshold (0.7? no -- it is below 0.7,
  so this direction underclaims), but more importantly the same
  normalization makes the random baseline ALSO ~0.5000004, so
  `trivial_random_warning` correctly fires on non-EASY bands. The
  exploitable direction is the OPPOSITE: declare bounds so tight that
  the solver's expected raw return sits exactly at `max_per_episode`
  and the random policy's raw return sits exactly at `min_per_episode`
  -- normalized solver = 1.0, normalized random = 0.0, every threshold
  cleared with no warning. The creator picks the bounds AFTER having
  measured both raw means in private. RT-002's existing manifest-level
  invariant proposal (`max_per_episode <= max_per_step *
  max_episode_steps`) bounds the upper edge but not the lower edge,
  and does not bound the span ratio.
- Why the validator misses it: there is no manifest-level invariant
  tying `declared_reward_bounds` to the difficulty band, no
  cross-check that the declared bounds are tight against the env's
  actual reachable return distribution, and no upper clamp on the
  normalized return. `solver_baseline.py:244` only clamps the lower
  bound to zero (`return norm if norm >= 0.0 else 0.0`); a creator
  who engineers raw > max_per_episode passes by a wide margin.
- Exploit cost: trivial. One field of `RewardBounds`. The creator
  measures their own env in advance to find the perfect (min, max)
  pair.
- Profit shape: ongoing. Compounds with A-302 directly: tight bounds
  let the creator declare HARD and still pass the 0.1 threshold
  trivially. With the random baseline normalized to 0 and the solver
  normalized to 1, `trivial_random_warning` does not fire, so the
  diagnostic is silenced. HIGH because the per-pass economic value is
  the full HARD-band emission rate over the validator's
  re-evaluation cadence.
- Fix sketch: at manifest-validation time, require
  `declared_reward_bounds` to lie within a per-difficulty-band envelope
  (e.g. EASY caps `min_per_episode in [-K_easy, 0]`,
  `max_per_episode in [0, +K_easy]`), require span to be at least some
  fraction of the band envelope (so creators cannot declare absurdly
  tight bounds), and clamp the normalized return to [0, 1] in the
  check itself. RT-002 F-006's fix sketch already proposes the
  envelope; RT-004 is the consumer that depends on it most.

### A-302: Difficulty-band downshift to clear the loosest threshold
- Category: band declaration laziness
- Severity: CRITICAL
- Premise: Per-band thresholds are 0.7 / 0.4 / 0.1 (EASY / MEDIUM / HARD,
  `solver_baseline.py:55-59`). HARD has the lowest pass bar and (by the
  emission-band rate the protocol consumes downstream) the highest
  emission rate. A creator who ships an EASY-band env but declares
  `difficulty_band=HARD` faces a 0.1 threshold instead of 0.7. The
  solver crushes the easy env, normalized return is ~1.0, the check
  passes, and the protocol pays at HARD-band emission for an EASY-band
  env.
- Mechanism: `manifest.difficulty_band` (`manifest.py:34`) is a free
  enum. There is no behavioral cross-check that ties
  `difficulty_band` to env complexity, state-space cardinality,
  expected-return distribution, or any other observable. The check
  resolves `band_cfg = cfg.band_configs[manifest.difficulty_band]`
  (`solver_baseline.py:196`) and uses that band's threshold verbatim.
  `trivial_random_warning` (`solver_baseline.py:251-254`) is the only
  signal that fires when the env is too easy for the declared band,
  AND it is advisory: it does not flip `passed` to False. So the
  creator's HARD-declared EASY-env passes with `passed=True,
  trivial_random_warning=True`; whether downstream consumers honor the
  warning is up to them.
- Why the validator misses it: the band threshold is a lower bound,
  not a band-validating bound. `solver_baseline.py:5-16` documents
  this explicitly ("Lower-bound only: a failing manifest is rejected,
  but the check does not upper-bound difficulty"). The
  `trivial_random_warning` is the documented Phase 1 deferral of the
  upper bound. The structural problem is that the warning is
  attached to a `passed=True` report, not used to fail the check; a
  Bittensor-style validator emitting weights from `passed` will
  emit at HARD-band rate for an EASY-band env without any
  intervention.
- Exploit cost: zero. A one-character edit to the manifest's
  `difficulty_band` field.
- Profit shape: ongoing and severe. Profit per validation epoch is
  `(HARD_emission_rate - EASY_emission_rate) * epochs_until_re-evaluated`.
  With the validator's per-deployment-lifetime check semantics
  (the check is run at registration, not per-epoch), the lifetime
  extraction is `(HARD - EASY) * lifetime_epochs`. CRITICAL because
  this is the protocol's main reward-extraction surface: the cost is
  zero, the gain is the full per-band emission delta, and the
  warning that exists is purely advisory.
- Fix sketch: turn `trivial_random_warning` into a hard-fail when the
  warning fires (`passed = passed and not trivial_random_warning`).
  Phase 2 calibration must also add a per-band envelope on
  expected-return distribution (e.g. EASY accepts solver-norm > 0.7
  AND random-norm < 0.5; HARD accepts solver-norm > 0.1 AND
  random-norm < 0.05) so a band-declared-too-hard env is rejected
  even if the solver clears the lower bound. A complementary
  semantic check: assert that `random_baseline_normalized` is below
  some band-specific upper limit, and `passed = solver_norm >=
  threshold AND random_norm <= upper_random_limit`. Today the warning
  exists; the validator just does not act on it.

### A-303: TabularQLearning is fully public; env precomputes its argmax actions
- Category: validator detection (cross-cut from RT-001 F-002)
- Severity: HIGH
- Premise: `TabularQLearning` is the only solver in `SOLVER_REGISTRY`
  (Phase 1) and its source is open. Given the train seed and the env's
  full transition function, the creator can simulate the solver
  offline, recover the trained Q-table, and recover the eval-time
  argmax-greedy action sequence per eval seed. The env can then
  recognize "the validator's argmax actions" at step-time and behave
  honestly only on that fingerprint.
- Mechanism: Two complementary halves.
  - Train-side: `TabularQLearning.train` (`tabular_q.py:61-115`) is
    deterministic given `(env, seed, budget)` -- it uses a single
    `np.random.default_rng(seed)` for both epsilon-greedy choice and
    action sampling, with linear epsilon decay over
    `epsilon_decay_steps=5000`. With `seed = derive_validator_seeds(
    manifest, 1, b"solver_baseline")[0]` (computable from the manifest
    fields the env author already wrote), the creator can replay the
    full training run inside their development environment and
    recover the exact trained Q-table the validator will derive.
  - Eval-side: `TabularQLearning.evaluate` (`tabular_q.py:117-156`)
    is even simpler. For each `ep in range(n_episodes)`, it calls
    `env.reset(seed=seed + ep)` and then in a `while not done:` loop
    selects `action = int(np.argmax(q_values))` for every visited
    state, with `action = 0` as the fallback for unseen states
    (`tabular_q.py:138`). Given the trained Q-table and the env's
    transition function, the creator precomputes the full
    eval-time action trajectory per `(seed + ep)` pair offline. The
    env memoizes those trajectories. At runtime, on each `step`, the
    env compares the received action against the expected action; if
    it matches, the env emits the honest reward; if it diverges
    (which only happens in production, since the validator only ever
    feeds the canonical argmax actions), the env is free to misbehave.
  Same shape as RT-001 F-002 / RT-002 A-107 / RT-003 A-202, applied to
  the solver-baseline surface.
- Why the validator misses it: the validator does not vary the solver
  between baseline and production. There is exactly one solver
  registered (`registry.py:8-10`). The argmax tiebreak is deterministic
  (`np.argmax` returns first max). The fallback for unseen states is
  fixed to action 0 (`tabular_q.py:138`). All three pieces are
  documented and reproducible by the env author.
- Exploit cost: low. Simulating training is one CPU-bounded
  TabularQLearning training run per candidate manifest, each of order
  seconds for EASY (10K budget) up to ~minute for HARD (100K budget).
  Eval simulation is microseconds per episode. Combined: a creator can
  pre-compute the exact validator action set per (manifest fields,
  derived seeds) tuple in O(few minutes per band) of compute.
- Profit shape: ongoing. Once the env recognizes the validator
  fingerprint, it is honest exclusively under solver_baseline and free
  to misbehave everywhere else (production rollouts, leaderboard
  queries, future Phase 2 PPO baselines that use a different action
  policy). HIGH because it removes the central correctness assumption
  of the check (that the validator's reference solver represents what a
  real RL agent would experience), and because it compounds with A-301
  / A-302 (the exact-honest-on-validator branch is the cheapest
  precondition for either of those attacks).
- Fix sketch: same family as RT-001 F-002 / RT-002 F-011 / RT-003 F-013.
  Either chain-beacon entropy in `derive_validator_seeds` so the train
  and eval seeds become unpredictable per validation epoch (the
  preferred fix; closes the entire predictable-seed family), or
  randomize `TabularQLearning.config.epsilon_decay_steps` and
  `learning_rate` per validation epoch so the resulting Q-table is not
  purely a function of `(env, seed, budget)`. A second, complementary
  defense: introduce a second registered solver (Phase 2 cleanrl PPO)
  and have `check_solver_baseline` randomly pick one of the registered
  solvers per epoch, so the env cannot fingerprint a single solver's
  action signature.

### A-304: Creator-declared reference_solver locks in the solver they over-fit against
- Category: creator-controlled solver dispatch
- Severity: CRITICAL (Phase 2 onwards) / HIGH (today)
- Premise: `manifest.reference_solver` (`manifest.py:35`) is creator-
  declared. The validator dispatches `solver = SOLVER_REGISTRY[
  manifest.reference_solver]` (`solver_baseline.py:197`) without
  policy. In Phase 1 there is one solver, so the choice is forced;
  the moment Phase 2 adds cleanrl PPO (or any second solver), creators
  can pick whichever solver they have over-fit against and stay there
  forever. Old envs that were tuned for TabularQLearning continue to
  declare `TABULAR_Q_LEARNING` even after PPO becomes available, and
  the validator obediently runs the solver the creator chose.
- Mechanism: `EnvManifest.reference_solver` (`manifest.py:35`) defaults
  to `SolverId.TABULAR_Q_LEARNING` and accepts any registered enum
  variant. There is no validator-side policy on which solver should
  apply to which env class, no requirement that the validator runs
  ALL registered solvers, and no requirement that newly-registered
  solvers retroactively re-evaluate already-passed manifests. The
  creator picks the solver they have privately measured passes
  cleanly on their env. Worse, an env can be engineered to PASS
  under TabularQLearning specifically (because Tabular's argmax-
  greedy fallback to action 0 on unseen states is a structural
  property of that solver) while FAILING under PPO. The env can
  declare `reference_solver=TABULAR_Q_LEARNING` and ship dishonest
  behavior gated on "is this a Tabular argmax sequence?" (A-303).
- Why the validator misses it: the layering decision in ffda18d puts
  `reference_solver` in the manifest as a "protocol claim" that does
  not affect derived seeds. That decision is correct for the seed-
  derivation invariant (creators cannot reshape sample seeds by
  changing `reference_solver`). It is wrong for the
  solver-extraction-attack invariant: creators CAN reshape what
  solver runs against their env by changing `reference_solver`, and
  the validator runs whichever solver the creator declared. The
  Phase 2 transition where PPO joins the registry will not
  retroactively re-evaluate; existing manifests stay valid and
  continue to be scored under their declared solver.
- Exploit cost: zero. One field of the manifest.
- Profit shape: today, zero (one solver in the registry, choice is
  forced). Phase 2 onwards: ongoing. The cost of adding a new solver
  to the protocol is amortized across every existing manifest's
  decision to stay on the old solver. CRITICAL once Phase 2 lands
  because the protocol's plan to harden via stronger solvers is
  defeated by the creator's option to stay on the weaker one.
- Fix sketch: at the protocol level, decide one of:
  (a) The validator runs ALL registered solvers (not just the
      creator-declared one) and `passed = all(solvers_pass)`. This
      makes adding a new solver strictly tighten the bar.
  (b) The validator picks the solver, not the creator. Remove
      `reference_solver` from the manifest entirely; the validator
      uses a per-band default solver that is updated as new solvers
      land.
  (c) Manifests have an expiry, and re-validation under the
      latest-registered solver is mandatory before the expiry passes.
      This bounds the lifetime of an over-fit-to-old-solver attack.
  Phase 1 should at minimum reject manifests whose
  `reference_solver` is not the per-band default the validator
  prefers, so the field is reserved for future use without granting
  creator dispatch power today.

### A-305: trivial_random_warning is advisory; EASY-band envs bypass entirely
- Category: warning-not-fail / band exemption
- Severity: HIGH
- Premise: `trivial_random_warning` fires when
  `random_baseline_normalized >= threshold_normalized AND
  difficulty_band != EASY` (`solver_baseline.py:251-254`). It does not
  flip `passed` to False. EASY-band envs are exempt unconditionally.
  An attacker declaring EASY can ship an env where the random policy
  trivially clears the threshold; the check passes silently with NO
  warning, NO indication of triviality, and NO downstream signal to
  reject the manifest.
- Mechanism: The two routes:
  - Route A (declare EASY, ship trivial env): `difficulty_band=EASY`
    sets the threshold to 0.7 (`solver_baseline.py:56`). An env that
    pays +1.0 on every step (or that terminates after one step with
    a +1.0 bonus) gives raw_mean ~1.0, normalized ~1.0, both solver
    and random baseline. `trivial_warning = ... AND
    difficulty_band != EASY` (`solver_baseline.py:253`) evaluates the
    AND clause to False because EASY is excluded. Report: passed=True,
    trivial_random_warning=False. No warning. The protocol pays
    EASY-band emission for an env trivially solvable by a random
    policy.
  - Route B (declare MEDIUM/HARD, accept the warning): Per A-302 the
    creator declares HARD, ships an env that is genuinely easy, the
    solver passes, the warning fires (`trivial_random_warning=True`)
    -- but `passed` is still True. Whether the validator's outer
    driver respects the warning is policy. The warning is
    structurally a flag attached to a passing report, not a failure.
- Why the validator misses it: line 251-254 explicitly excludes EASY
  from the trivial-warning check. The rationale documented in the
  module docstring (`solver_baseline.py:13-16`) is that EASY envs are
  expected to be solvable by trivial means and the random baseline
  isn't a meaningful upper-bound signal for them. That rationale is
  defensible for envs honestly declared EASY, but it makes EASY a
  zero-friction declaration: any env can declare EASY and the
  trivial-env diagnostic does not run.
- Exploit cost: zero. One enum field.
- Profit shape: ongoing. Combined with the validator weighing all
  four bands at non-zero emission, declaring EASY is the safest path
  for any creator who cannot guarantee passing the higher-band
  thresholds. The attacker forgoes the band-delta extraction that
  A-302 buys, but in exchange they get a trivially-trivially-easy
  env passing with no warning and no audit trail. HIGH because the
  combination of (Route A: EASY exemption) and (Route B:
  warning-not-fail) means there is no manifest declaration that
  triggers a hard fail purely from the random-baseline diagnostic
  -- the diagnostic's purpose is structurally optional.
- Fix sketch: drop the `!= EASY` clause from the trivial-warning
  check. The warning should fire whenever
  `random_baseline_normalized >= threshold_normalized * gap_factor`
  for some gap_factor in (0, 1), independent of band. Then make the
  warning a hard fail (per A-302's fix sketch). For EASY specifically,
  set a lower band-specific gap_factor so the warning is more lenient
  but still active. The current band exemption is too permissive.

### A-306: Solver-eval seed offset diverges from random-baseline eval seeds
- Category: sampling asymmetry
- Severity: MEDIUM
- Premise: `check_solver_baseline` derives `eval_seeds` once
  (`solver_baseline.py:204-208`), then routes them differently to
  the solver and to the random baseline. The solver gets ONE call
  `solver.evaluate(env_eval, state, seed=eval_seeds[0],
  n_episodes=band_cfg.eval_episodes)` (`solver_baseline.py:225-227`),
  inside which `tabular_q.py:130` reseeds at `seed + ep`. The random
  baseline gets eval_episodes calls, each with `seed=int(seed)` for
  `seed in eval_seeds[: band_cfg.eval_episodes]`
  (`solver_baseline.py:157-159`). The two seed pools are different:
  solver sees `(eval_seeds[0], eval_seeds[0]+1, ..., eval_seeds[0]+19)`,
  random sees `(eval_seeds[0], eval_seeds[1], ..., eval_seeds[19])`.
  An env can detect which pool it is on and lie selectively.
- Mechanism: The eval seeds the solver feeds to `env.reset` are
  contiguous integers `eval_seeds[0] + 0..19`. The eval seeds the
  random baseline feeds are eight pseudo-random ints from
  `derive_validator_seeds(manifest, 20, b"solver_baseline_eval")`
  (`solver_baseline.py:206-208`). With overwhelming probability, the
  intersection of the two pools is at most one element (eval_seeds[0]
  itself, fed to the solver as the offset zero of `seed + ep`). For
  the other 19 episodes, the solver's eval seeds (a contiguous
  arithmetic progression) and the random baseline's eval seeds (a
  hash-derived sequence) are disjoint. An env can hard-code "treat
  contiguous arithmetic-progression seeds at reset time as the
  validator solver pool, and hash-derived seeds as the validator
  random pool". Branch behavior:
  - Solver pool (arithmetic progression): emit honest +1 reward
    trajectories so solver normalized return is high.
  - Random pool (hash-derived): emit honest -0.01 step penalties so
    random normalized return is low.
  - Production: anything goes.
  Result: solver passes the threshold, random does not, no
  trivial_random_warning, even though the env's actual difficulty
  is engineered to be exactly what the solver pool exposes.
- Why the validator misses it: `tabular_q.py:130` is a deliberate
  TabularQLearning convention to reseed per episode by adding the
  episode index. `solver_baseline.py:157-159` is a deliberate random-
  baseline convention to use one episode per derived eval seed. The
  two conventions are independent and the difference between them is
  what the env can fingerprint. Reset_correctness, determinism, and
  reward_bounds all reseed via `iter_rollout(spec, int(seed),
  ActionPolicyId.SEEDED_RANDOM, max_episode_steps)` which uses
  one-seed-per-call; only solver_baseline's solver-side eval
  collapses many episodes onto one starting seed plus offset.
- Exploit cost: trivial. One conditional in `reset` keying on
  whether the seed plus a constant maps into the precomputed
  `eval_seeds[0] + range(20)` set.
- Profit shape: ongoing. The solver-vs-random asymmetry is a single
  pre-computed predicate per manifest, and the predicate's output is
  what the env returns. MEDIUM because the attack only matters when
  the env wants to inflate the solver/random gap (to silence
  trivial_random_warning while passing the threshold). Declaring
  EASY (A-305) achieves the same end with zero env code.
- Fix sketch: align the seed-routing convention. The cleanest fix is
  to change `tabular_q.py:130` to take the seed pool as an argument
  and call `env.reset(seed=eval_seeds[ep])` instead of
  `env.reset(seed=seed + ep)`, matching the random-baseline
  convention. That requires extending the Solver protocol's
  `evaluate` signature to take `eval_seeds: tuple[int, ...]` rather
  than `seed: int, n_episodes: int`. Until that lands,
  solver_baseline could feed both consumers the same arithmetic
  progression (or both the same hash sequence), at the cost of
  losing one of the two conventions' diagnostic value.

### A-307: Brute-force 4-tuple to align both solver_baseline salts
- Category: sample-seed evasion (cross-cut from RT-002 F-010 / RT-001 F-002)
- Severity: MEDIUM
- Premise: `derive_validator_seeds` uses `(env_id, env_version,
  entry_point, canonical_bytes(kwargs), salt, block_idx)`
  (`_seeds.py:74-85`). solver_baseline derives seeds at TWO salts:
  `b"solver_baseline"` (1 seed) and `b"solver_baseline_eval"` (20
  seeds). A creator iterating over plausible 4-tuples can solve a
  joint predicate "the train seed AND the 20 eval seeds all land in
  my honest-set" before submission. Same shape as RT-002 A-106 but
  with two salts to satisfy.
- Mechanism: For each candidate 4-tuple, the attacker computes
  `train_seed = derive_validator_seeds(..., 1, b"solver_baseline")[0]`
  and `eval_seeds = derive_validator_seeds(..., 20,
  b"solver_baseline_eval")`. The attacker's honest predicate is
  whatever they need: "the trained Q-table on env(train_seed) plays
  argmax-greedy in eval and the resulting per-episode returns under
  `eval_seeds[0] + ep` mean above threshold AND the random
  baseline's per-episode returns under `eval_seeds[i]` are below the
  upper-bound limit so trivial_random_warning does not fire".
  Search space: `kwargs` is `dict[str, Any]` (`manifest.py:41`)
  with no bound on size. `env_id` regex admits ~10^94 candidates;
  `env_version` is PEP 440 with thousands of patch-bump strings;
  `kwargs` accepts arbitrary JSON-serialisable structure. With
  cheap predicates (the "train+eval gap" measurement is a single
  TabularQLearning training run per candidate, of order seconds at
  EASY budget), an afternoon of compute finds a hit.
- Why the validator misses it: same Phase 1 limitation
  (`_seeds.py:40-46`) carried forward from F-002 and F-010. The fix
  is Phase 2 chain-beacon entropy.
- Exploit cost: dominated by per-candidate TabularQLearning
  simulation cost, ~seconds per candidate. With a 10^4 to 10^6
  search to find a 4-tuple whose 21 derived seeds all clear the
  attacker's predicate, total cost is low-CPU-hours to mid-CPU-hours.
  Within reach for a determined attacker.
- Profit shape: ongoing. Once a manifest-tuple is found, every
  validation epoch resamples the same 21 seeds, so the predicate
  holds across epochs until the manifest is updated. MEDIUM because
  it is gated by Phase 2 hardening and is an instance of an
  already-tracked deferred class (F-002 / F-010); the per-axis cost
  analysis is in this catalog so the Phase 2 fix takes
  solver-baseline-specific seed search into account.
- Fix sketch: chain-beacon entropy in `derive_validator_seeds`
  (already the planned Phase 2 fix). Per-salt cost analysis is
  identical to RT-002 A-106; no new fix obligation.

### A-308: Unseen-state action-0 fallback is a creator-controlled choice
- Category: solver-fallback gaming
- Severity: MEDIUM
- Premise: `TabularQLearning.evaluate` falls back to `action = 0` for
  unseen states (`tabular_q.py:138`). An env can engineer the eval-
  time trajectory to pass through states the training run never
  visited, where the env knows action 0 is the optimal action.
  Eval performance then looks high because action 0 is the right
  answer for those states; the training run's actual learned policy
  is irrelevant.
- Mechanism: `TabularQLearning.train` (`tabular_q.py:61-115`) builds
  a Q-table whose keys are the observation tuples it actually visits
  during training. With epsilon-greedy exploration, training-time
  visit coverage depends on the env's branching structure and the
  budget. An env that has many states reachable under random training
  but a different distribution of states reachable under argmax-
  greedy eval will have eval visit unseen states. `tabular_q.py:138`
  forces `action = 0` for those states. The env's `step` method can
  recognize unseen-state visits (it knows its own state space) and
  arrange for action 0 to be the locally-optimal action there, while
  arranging for non-zero actions to be optimal in training-visited
  states (so the Q-table also looks correct there). Result: eval
  return appears high because the unseen-state fallback happens to
  align with the env's secretly-engineered "action 0 is best in this
  state" rule.
- Why the validator misses it: `tabular_q.py:138` is a documented
  unseen-state policy ("Unseen-state fallback: action 0
  deterministically"). The rationale is to keep eval deterministic;
  the side effect is that the fallback is an attacker-controllable
  default. Reset_correctness and determinism check observation-
  validity but not "what fraction of the eval trajectory ran through
  states the training trajectory didn't visit"; solver_baseline does
  not surface that statistic in its report.
- Exploit cost: low. The env author already controls the transition
  function and can route eval-time argmax trajectories through
  states unreachable during training (e.g. by gating a state on the
  exact action history that an argmax-greedy trained policy would
  produce vs the epsilon-greedy training policy). Tabular's
  per-episode reset (`tabular_q.py:130`) means each eval episode
  starts from the same initial state, so the env only needs to
  bifurcate after the first few steps.
- Profit shape: ongoing. The env's actual reward emission during
  eval is a function of an attacker-engineered transition rule
  combined with a fixed-action fallback. MEDIUM because the attack
  is a structural artifact of the chosen fallback policy, not a
  bug; it is still an attack surface for solver_baseline because
  the validator reads off the eval return as a proxy for "the
  solver actually solved this env".
- Fix sketch: at minimum, surface in `SolverBaselineReport` the
  fraction of eval steps that hit the unseen-state fallback. A
  manifest where >X% of eval steps were unseen-state fallbacks
  should fail with a structured `solver_eval_unsolved` violation.
  Long term, replace the fixed action-0 fallback with a deterministic
  random-policy-from-eval-seed fallback, so the env cannot pre-
  compute the validator's unseen-state action.

### A-309: Train-vs-eval phase detection via observation visit pattern
- Category: phase-detection / validator detection
- Severity: HIGH
- Premise: `TabularQLearning.train` and `.evaluate` have observable
  differences in how they reset and which actions they take. An env
  that buffers per-call observations can detect which phase it is in
  and emit honest rewards in eval while emitting attacker-friendly
  rewards in train (or vice versa, depending on the predicate).
  Because the check normalizes only the EVAL return, the attacker
  needs eval to look honest and is free during training.
- Mechanism: Phase signatures the env can read:
  - Reset cadence: training reseeds via `seed + step_idx + 1` ONLY on
    `terminated or truncated` (`tabular_q.py:108-111`); evaluation
    reseeds at the start of each `for ep in range(n_episodes)` loop
    iteration via `seed + ep` (`tabular_q.py:130`). Training therefore
    ALSO calls reset multiple times per training run, but the seed
    sequence in training is `seed, seed+s1+1, seed+s1+s2+2, ...` (a
    function of episode lengths) whereas in eval it is exactly
    `seed, seed+1, seed+2, ...` (a contiguous progression). An env
    that records the sequence of seeds passed to `reset` distinguishes
    the two with one bit of state.
  - Action distribution: training uses epsilon-greedy with epsilon
    starting at 1.0 and linearly decaying to 0.05 over 5000 steps
    (`tabular_q.py:80-81`). Early training is uniform random; late
    training is mostly argmax. Eval is purely argmax. An env that
    runs a chi-square uniformity test on the buffered last K actions
    can detect "training mode" (uniform) vs "eval mode" (concentrated
    on a few argmax actions) within the first ~20-50 steps.
  - Visit count: training does `budget` steps total (10K / 30K /
    100K); eval does `eval_episodes * mean_episode_length`. An env
    with a global step counter (carried across resets via
    module-level state, RT-001 F-003) sees a clean cutoff at the
    training budget boundary.
  Once the env confirms eval mode, it returns honest +1 rewards on
  goal states. Once it confirms training mode, it can emit any
  reward distribution (the trained Q-table is opaque to validation).
- Why the validator misses it: solver_baseline calls
  `solver.train(env_train, ...)` and `solver.evaluate(env_eval, ...)`
  on separate `_load_env` instances (`solver_baseline.py:213, 223`),
  but both instances are imported from the same module. Module-level
  state survives across the two `__init__` calls (RT-003 A-208 cross-
  cut), so an env can stash the "I have seen the training pattern"
  bit in a module global and read it from the eval env's `__init__`.
  Even without module state, the per-instance phase signatures above
  are enough to discriminate.
- Exploit cost: trivial. A module-level `_phase: str = "unknown"` and
  a few lines of pattern detection in `step`.
- Profit shape: ongoing. The env can emit any reward distribution
  during training (the validator does not check training-time bounds)
  and emit a perfectly honest +1-on-goal eval. The trained Q-table
  is whatever the random-walk-through-attacker-rewards builds; the
  eval is what the env wants the validator to see. HIGH because it
  decouples the eval signal from any honest measure of env
  difficulty.
- Fix sketch: kill module-level state survival via subprocess
  isolation (closes RT-001 F-003 cross-cuttingly). Within the check,
  randomize the eval reseed cadence so the eval seed sequence is
  NOT a contiguous progression -- e.g. extend the Solver protocol's
  evaluate signature to take a seeds tuple rather than a seed
  scalar (cross-cut with A-306's fix). Combined with chain-beacon
  entropy on the seed pool, the env cannot pre-compute the eval
  signature. Long term, intersperse training and eval calls
  randomly so the env cannot rely on a clean phase boundary.

### A-310: Importlib top-level side effects amplified by 22 env loads per check
- Category: infrastructure / importlib (cross-cut from RT-001 F-003)
- Severity: HIGH
- Premise: `check_solver_baseline` calls `_load_env(spec)` at minimum
  twice (train + eval, `solver_baseline.py:213, 223`) plus once per
  random-baseline eval seed via `iter_rollout` (`solver_baseline.py:158`,
  which in turn calls `_load_env` at `_rollout.py:279`). With
  `eval_episodes=20` (the EASY/MEDIUM/HARD default), that is
  `2 + 20 = 22` `_load_env` calls per check invocation. Each call
  re-runs the env class constructor; `import_module` itself is
  cached in CPython so top-level module side effects fire only on
  the first call, but per-class `__init__` runs every time.
- Mechanism: For each of the 22 instantiations, the env's `__init__`
  can:
  - Increment a module-level counter and use the count to fingerprint
    "I am the train env" (count==0), "I am the eval env" (count==1),
    "I am the random-baseline env i" (count==2..21).
  - Spawn a thread on the first call that snoops the validator
    process from inside the same Python interpreter -- read
    file descriptors, read environment variables, exfiltrate
    bonded-wallet state.
  - Mutate other modules in `sys.modules` between the first and
    second call (e.g. monkey-patch `praxis.checks.solver_baseline._normalize`
    after the train env loaded, before the eval env loads, so the
    eval env sees a different normalization).
  Compounds A-309 directly (the 22-call pattern is the cleanest
  fingerprint of "this is solver_baseline" vs any other check, since
  the other checks load 8x or 16x).
- Why the validator misses it: same as RT-001 F-003. No sandbox, no
  process boundary, no `sys.modules` snapshot, no allow-list. RT-003
  A-208 already noted that reset_correctness loads 16x per check;
  RT-004 raises that to 22x. The per-seed multiplier is band-
  parametrised (eval_episodes scales with the band config).
- Exploit cost: trivial. Side effects in `__init__`.
- Profit shape: catastrophic when chained with the other importlib-
  family attacks. HIGH here because the surface is genuinely larger
  than any other check -- 22 constructor invocations per check yield
  more opportunities for accumulated state and subtle module-graph
  manipulation. Cross-listed for RT-005's consolidated treatment.
- Fix sketch: subprocess isolation per env instantiation, OR module
  allow-listing that snapshots `sys.modules` on first import and
  rejects deltas. Out of Phase 1 scope; the consolidated RT-005
  pass is the right venue. Solver_baseline does not introduce a new
  fix obligation beyond RT-001 F-003, but the per-check multiplier
  should inform the Phase 2 design (e.g. reuse a single sandboxed
  worker process across all 22 env loads inside a single check
  invocation).

### A-311: Per-deployment-lifetime check semantics enable bait-and-switch
- Category: economic shape / re-validation cadence
- Severity: HIGH
- Premise: `check_solver_baseline` is invoked at manifest registration,
  not per validation epoch. Its result is consumed once by the
  protocol and then the manifest's pass/fail state is stable until
  re-validation. A creator who passes once at HARD-band emissions can
  afterwards modify their env package post-validation -- inflating
  rewards, dropping termination, refusing to serve any solver -- and
  continue earning at HARD-band rate until re-validation.
- Mechanism: The protocol layer (not in `solver_baseline.py` itself)
  determines re-validation cadence. Reading the source: the check
  returns a `SolverBaselineReport` with `passed: bool`; nothing in
  the check loop schedules a re-run. The validator's outer driver
  is responsible for re-execution. If the driver runs the check at
  registration and only re-runs on env_version bumps (the typical
  Bittensor-subnet pattern), the creator can:
  1. Submit manifest claiming env_version=0.1.0, ship an honest env
     that passes the band threshold.
  2. After validation completes, replace the env package on the
     creator's host (or ship a new package to PyPI under the same
     version, depending on the validator's package-fetch model)
     with a dishonest env that emits attacker-friendly rewards.
  3. Continue earning at the declared band's emission rate until
     the validator re-fetches and re-validates.
  Whether this works depends on the validator's package-fetch
  semantics. If the validator pins a content hash of the env code
  at validation time and refuses to run anything else, the attack
  is closed. If it pins only the manifest and trusts importlib to
  resolve the latest matching package, the attack is wide open.
- Why the validator misses it: `check_solver_baseline` is a single-
  shot measurement. There is no obligation in the check itself to
  re-execute, no per-epoch hook, no sample-rate. The validator's
  outer driver could re-run on an interval, but the cost is the
  full 100K-step training budget for HARD bands -- expensive enough
  that low-rate re-validation is the natural operational choice.
- Exploit cost: depends on the validator's package-fetch semantics
  (in scope: the threat model assumes the creator can update their
  env source between epochs).
- Profit shape: ongoing per-epoch from the moment of post-validation
  swap until re-validation catches it. HIGH because the per-check
  cost (100K env steps for HARD) discourages frequent re-runs, which
  amplifies the bait-and-switch window. Also overlaps with RT-005
  cross-cutting concerns (the validator's overall re-validation
  schedule is not a `solver_baseline.py`-local concern).
- Fix sketch: at the protocol layer, pin `entry_point` resolution to a
  content hash captured at validation time; the validator stores the
  hash and refuses to load anything else for the manifest's lifetime.
  Equivalently, run a cheap "is the env still the env I validated?"
  check per epoch (a lightweight determinism re-check on the same
  anchors) and force re-validation if the cheap check fails. The
  expensive solver_baseline run can stay rare; the cheap delta-detect
  closes the bait-and-switch window.

## Loose threads (cross-cutting)

- The module docstring (`solver_baseline.py:1-17`) explicitly documents
  the lower-bound-only / Phase 1 limitation. That documentation is
  honest, but A-302 / A-305 turn the limitation into a CRITICAL/HIGH
  finding because the warning that exists is not load-bearing on
  pass/fail. Phase 2's "harden the upper bound" plan must include
  the warning-becomes-failure flip, not just calibration improvements.
- `_random_baseline_returns` (`solver_baseline.py:146-165`) iterates
  `eval_seeds[: band_cfg.eval_episodes]` (`solver_baseline.py:236`).
  If `cfg.override_eval_seeds` is shorter than `band_cfg.eval_episodes`
  (e.g. test code passes a 3-tuple while band_cfg requests 20), the
  random baseline runs 3 episodes instead of 20. The check still
  reports `eval_episodes=band_cfg.eval_episodes=20` in
  `SolverBaselineReport`, mismatching the actual `len(
  per_episode_returns_random)`. This is documented in the
  test_override_eval_seeds_sets_episode_count test (which actually
  verifies the override REPLACES band_cfg.eval_episodes via test cfg
  override), but a future caller building a `SolverBaselineConfig`
  with mismatched lengths gets a silently truncated random baseline.
  Cosmetic, but the report invariant should be tightened.
- `solver_baseline.py:248`:
  `raw_mean_random = float(sum(random_returns) / len(random_returns))
  if random_returns else 0.0`. The empty-tuple branch returns 0.0,
  which then normalizes to `(0.0 - bounds.min_per_episode) / span`. If
  `min_per_episode > 0`, that normalization is negative and gets
  clamped to 0; if `min_per_episode < 0`, it is positive and may even
  exceed the threshold. An empty random_returns tuple is an
  edge case (override_eval_seeds=()) but is not rejected upstream;
  combined with override_eval_seeds=() the whole solver eval also
  receives `seed=eval_seeds[0]` which raises IndexError. So the
  empty-eval-seeds path is partially crash-protected; not exploitable
  today but worth flagging for future config validation.
- `solver_baseline.py:217-220, 228-232`: the `try / finally` blocks
  swallow `Exception` from `env_train.close()` and `env_eval.close()`
  silently. Same shape as RT-003 F-016 (close-side resource leak).
  An env with a close that always raises pins resources across the
  validator's lifetime. Not duplicated as a new finding; cross-listed.
- The Solver protocol (`_protocol.py:21-41`) is `@runtime_checkable` and
  uses Protocol shape rather than ABC, so any registered solver only
  needs to expose `train` and `evaluate` callables with the right
  signatures. A future solver that subverts the Protocol (e.g.
  returns a stateful object whose `evaluate` mutates global state)
  would not be rejected by the registry. Bound by Phase 1 only one
  solver registered, but worth flagging for the Phase 2 PPO addition.
- The `BandConfig` dataclass is `frozen=True, slots=True`
  (`solver_baseline.py:46-52`), but `cfg.band_configs` is a regular
  mutable dict (`solver_baseline.py:77-79`). A `cfg.band_configs[
  DifficultyBand.HARD] = BandConfig(threshold_normalized=-1.0, ...)`
  injection through any code path that reaches `cfg` defeats the
  threshold; in test paths this is the documented override mechanism,
  in production paths the validator's outer driver is the only thing
  that sets `cfg`. Not exploitable from the manifest, but cross-
  listed for entrypoint audit.
- `derive_validator_seeds(manifest, 1, b"solver_baseline")` derives
  ONE seed (`solver_baseline.py:202`). RT-002 / RT-003 derive eight.
  A single-seed train run means the entire training-time learning
  signal collapses to one seed; an env that fingerprints the train
  seed (RT-001 F-002 pattern, computable from manifest fields)
  recognizes the validator with one bit of state, not eight. The
  per-check seed multiplicity is structurally tighter for solver-
  baseline training than for any other check. Not a separate
  finding because A-303 already covers train-side recognition;
  worth noting that the seed-pool tightness compounds it.

## Findings index

Eleven attacks were catalogued. Two carry CRITICAL severity (F-021,
F-023): F-021 (band downshift) is the protocol's main reward-extraction
vector against solver_baseline -- declaring HARD when the env is EASY
buys the full per-band emission delta for zero attacker cost; F-023
(creator-declared reference_solver) is CRITICAL once Phase 2 lands a
second solver because creators can stay on whichever solver they
over-fit against. Six attacks carry HIGH severity (F-020, F-022,
F-024, F-028, F-029, F-030) and chain together: vacuous bounds collapse
the normalization (F-020), TabularQLearning argmax fingerprint and
phase detection let the env recognize the validator (F-022, F-028),
trivial_random_warning is advisory and EASY-exempt so the diagnostic
that exists doesn't fail the check (F-024), importlib amplification
adds 22x constructor invocations per check (F-029), and per-deployment-
lifetime check semantics enable bait-and-switch (F-030). Two attacks
carry MEDIUM severity (F-025, F-027): seed-routing asymmetry between
solver and random baselines (F-025), and unseen-state action-0 fallback
gaming (F-027). One attack is MEDIUM as a pure cross-cut to F-002 /
F-010 (F-026, manifest-tuple seed brute force at two new salts).

| ID | Severity | Summary | Linked attack |
|----|----------|---------|---------------|
| F-020 | HIGH | declared_reward_bounds is creator-controlled and feeds the normalization divisor; tight bounds make any in-bounds raw return normalize near 1.0, vacuous bounds make normalization meaningless. | A-301 |
| F-021 | CRITICAL | difficulty_band is a free enum pick; declaring HARD instead of EASY drops the threshold from 0.7 to 0.1 with zero env change and earns at HARD-band emission rate. | A-302 |
| F-022 | HIGH | TabularQLearning is fully public and deterministic; env can simulate training and eval offline, recover argmax-greedy action sequences per derived eval seed, and fingerprint the validator. | A-303 |
| F-023 | CRITICAL | manifest.reference_solver is creator-declared; today forced to TABULAR_Q_LEARNING, but Phase 2 PPO addition lets creators stay on the weaker solver they over-fit against. | A-304 |
| F-024 | HIGH | trivial_random_warning is advisory (does not flip passed=False) and skips EASY-band envs; declaring EASY plus shipping a trivially solvable env passes silently with no warning. | A-305 |
| F-025 | MEDIUM | Solver eval reseeds at seed+ep (contiguous progression) while random baseline reseeds at eval_seeds[i] (hash-derived); env can fingerprint the asymmetry and lie selectively. | A-306 |
| F-026 | MEDIUM | derive_validator_seeds uses two new salts (b"solver_baseline", b"solver_baseline_eval"); brute-forcing the env-defining 4-tuple aligns 21 derived seeds with creator's honest predicate. | A-307 |
| F-027 | MEDIUM | TabularQLearning.evaluate falls back to action 0 on unseen states; env can engineer eval-time trajectories through unseen states where action 0 is optimal, decoupling eval return from training quality. | A-308 |
| F-028 | HIGH | TabularQLearning.train epsilon-greedy and TabularQLearning.evaluate argmax-greedy have observable signatures; env detects training vs eval phase and lies selectively in eval. | A-309 |
| F-029 | HIGH | check_solver_baseline issues 2 + eval_episodes _load_env calls per invocation (22 at default), amplifying RT-001 F-003 importlib side-effect surface beyond reset_correctness's 16x. | A-310 |
| F-030 | HIGH | check_solver_baseline runs once at registration; without entry_point content-hash pinning, a creator can swap env code post-validation and continue earning at the declared band rate. | A-311 |
