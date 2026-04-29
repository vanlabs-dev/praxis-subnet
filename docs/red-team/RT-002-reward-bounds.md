# RT-002: Reward bounds check red-team pass

Status: Draft
Date: 2026-04-30
Scope: src/praxis/checks/reward_bounds.py and the iter_rollout primitive it consumes
Reviewed commits: full chain through 1c7f541, special focus on 6078765 and 630135e

## Threat model

The adversary is the same creator-miner profile as RT-001: they ship an env
package source-tree to validators, stake a Bittensor-style bond, and want to
maximize `expected_emissions - p_slash * collateral`. Their RT-002-specific
goal is to ship a manifest such that `check_reward_bounds` returns
`passed=True` while the env in production pays out rewards a validator would
have rejected if it had observed them.

What the attacker controls (reward_bounds-specific):

- Every field of `RewardBounds` (`types.py:21-33`). Pydantic enforces only
  `max_per_step > min_per_step` and `max_per_episode > min_per_episode`
  (`types.py:27-32`); both bounds are `float` with no upper magnitude cap, no
  finiteness check, and no relation between per-step and per-episode bounds.
- `manifest.max_episode_steps` (`manifest.py:36`, `gt=0`), which controls the
  TimeLimit truncation budget the rollout primitive will use.
- The full env source. Inside `step` the env can observe everything it has
  ever stepped over (its own internal state and counters), and it can branch
  on whatever it can recover about the validator (action histogram, action
  arrival cadence, info side channels, environment variables, walltime).
- `env_id`, `env_version`, `entry_point`, `kwargs` (`manifest.py:32-41`). These
  are the only manifest fields that feed `derive_validator_seeds` for
  `salt=b"reward_bounds"` (`_seeds.py:74-85`). `kwargs` is a free-form
  JSON-serialisable dict (`manifest.py:53-58`); `env_version` is constrained
  only to PEP 440 (`manifest.py:43-50`); `env_id` is constrained only to a
  64-char regex (`manifest.py:9`). All four can be search-space inputs.

What the attacker cannot do that matters here:

- Modify the salt `b"reward_bounds"` or the `derive_validator_seeds`
  algorithm. After 630135e the seeds depend only on env-defining fields, so
  cherry-picking sample seeds the way A-001 cherry-picked anchor seeds is
  closed. The remaining seed-prediction surface is brute-forcing the
  env-defining 4-tuple (carried forward as a Phase 2 concern; see A-106).
- Influence the iteration order of seeds inside `check_reward_bounds`
  (`reward_bounds.py:250`); seeds are processed in the order
  `derive_validator_seeds` returns them.
- Force `iter_rollout` to skip steps. Each `StepRecord.reward` is forced
  through `float(reward)` (`_rollout.py:297`), so the reward type is always
  Python float by the time `reward_bounds` sees it.

Net: the attacker cannot reshape which seeds are sampled by tweaking declared
bounds (closed by 630135e), but they have wide latitude to declare bounds so
loose that the check is trivially satisfied, to mismatch per-step versus
per-episode bounds, to exploit the truncation-only escape hatch baked into
the per-episode rule, and to introduce float values whose comparison
semantics blur the bounds. The check is strictly behavioral; it has no upper
sanity check on declared values.

## Attack catalog

### A-101: Unbounded declared bounds (vacuous-bounds pass)
- Category: bound declaration laziness
- Severity: HIGH
- Premise: A creator declares `min_per_step = -1e308`, `max_per_step = 1e308`,
  `min_per_episode = -1e308`, `max_per_episode = 1e308`. The check trivially
  passes for any env that emits finite rewards while the manifest tells
  honest downstream code nothing.
- Mechanism: `RewardBounds` (`types.py:21-33`) only enforces `max > min` per
  axis. It does not enforce `min_per_step >= -K` or `max_per_step <= +K` for
  any K, and does not enforce relations such as
  `max_per_step <= max_per_episode`. `check_reward_bounds` compares
  `r < bounds.min_per_step` and `r > bounds.max_per_step`
  (`reward_bounds.py:269`); both are False for any finite `r` and bounds set
  to `(-1e308, 1e308)`. Same for the per-episode arm
  (`reward_bounds.py:286`). The env can therefore emit rewards of any
  magnitude in production and the validator will still report
  `passed=True`. Downstream protocol code that composes on the manifest's
  declared bounds (e.g. solver_baseline normalization, future bond-sizing
  logic, leaderboard scoring) inherits the lie.
- Why the validator misses it: there is no upper sanity check on declared
  bounds. The manifest validator (`manifest.py:21-82`) does not know what
  reward magnitudes are plausible for a given `difficulty_band` or env
  family. `check_reward_bounds` accepts whatever the manifest declares as
  ground truth.
- Exploit cost: zero. A four-line edit to the manifest.
- Profit shape: ongoing. Every check that consumes
  `manifest.declared_reward_bounds` downstream (today: solver_baseline's
  normalization, see Loose threads) becomes degenerate. The attacker can
  simultaneously underclaim difficulty and emit arbitrary reward inflation
  in production rollouts that validators do not re-execute. HIGH because it
  permits ongoing dishonest reward emission while passing the check.
- Fix sketch: at manifest validation time, require declared bounds to lie
  within a per-difficulty-band envelope (e.g. EASY caps `|r_step| <= 10`,
  `|r_episode| <= 100`), and require
  `max_per_episode <= max_per_step * max_episode_steps` and the symmetric
  min relation. Reject manifests outside the envelope. Optionally also
  reject `min_per_step == -inf`, `max_per_step == +inf`, and any NaN.

### A-102: Per-step / per-episode bound desynchronization
- Category: bound declaration laziness
- Severity: HIGH
- Premise: The creator declares per-step bounds tight (e.g.
  `min_per_step=-0.01`, `max_per_step=+0.99`, the honest gridworld values)
  but per-episode bounds wildly loose (`max_per_episode=+1e9`). The
  per-step check pins down what each step can pay; the per-episode check
  is a no-op.
- Mechanism: There is no manifest-level invariant requiring
  `max_per_episode <= max_per_step * max_episode_steps` or
  `min_per_episode >= min_per_step * max_episode_steps`. `RewardBounds`
  only enforces `max > min` per axis (`types.py:27-32`). The two pairs are
  declared independently. An env that, on rare seeds, emits a sequence of
  in-bounds per-step rewards that sum to a per-episode total far outside
  any honest range will be caught only by the per-episode arm. If the
  per-episode arm is `[-1e9, +1e9]`, no terminated episode will violate
  it, and `reward_bounds.py:286` will never fire.
- Why the validator misses it: `reward_bounds.py:269` and
  `reward_bounds.py:286` enforce only the declared bounds verbatim. There
  is no cross-axis sanity invariant. The two arms are checked independently
  with no shared budget logic.
- Exploit cost: trivial. One field of the manifest.
- Profit shape: bounded but real. The attacker cannot inflate any single
  step beyond the tight per-step cap, but they can push cumulative episode
  reward arbitrarily high (up to the loose per-episode cap times whatever
  multiplier the production protocol applies to long episodes). Consumers
  that key on episode totals (leaderboard, solver-baseline normalization)
  see inflated returns. HIGH because it permits sustained reward inflation
  along the per-episode axis while the per-step check looks honest.
- Fix sketch: at manifest validation time, enforce
  `min_per_episode >= min_per_step * max_episode_steps` and
  `max_per_episode <= max_per_step * max_episode_steps`. That is the
  arithmetic upper bound an honest env can ever achieve given the declared
  per-step bounds and the truncation budget. Looser per-episode bounds
  cannot be honest under any per-step bound.

### A-103: Truncation-only escape hatch (per-episode-unverified production env)
- Category: rollout interaction (truncation versus termination)
- Severity: HIGH
- Premise: `check_reward_bounds` only checks per-episode bounds on
  naturally-terminated episodes (`reward_bounds.py:285`). A creator who
  ensures every validator-sampled rollout truncates rather than terminates
  bypasses the per-episode arm entirely while still passing the check
  with `per_episode_unverified=True` and `passed=True`.
- Mechanism: Two complementary routes:
  - Route A (env never terminates on validator inputs): the env's `step`
    is engineered so that `terminated=False` is returned for any state
    reachable from the validator-sampled (seed, action-sequence) pairs
    given `manifest.max_episode_steps`. Honest gridworld already exhibits
    this on the HARD band test fixture (`test_reward_bounds_truncation.py`)
    where a 20x20 grid with `max_episode_steps=10` always truncates. A
    dishonest env can simply hard-code `terminated=False` always, or set
    `terminated` only when the state encodes a marker the validator's
    actions will not generate.
  - Route B (env terminates only on canonical-action prefixes): the env
    examines the action sequence it is being fed. Since
    `SeededRandomPolicy.actions` (`_rollout.py:140-178`) is fully public
    given the seed, the env can recompute the canonical action sequence
    for each derived seed it sees in `reset(seed=s)` and refuse to
    terminate on any deviation. Inside `iter_rollout` the validator only
    ever feeds canonical actions; the env terminates and pays an honest
    total. In production the env sees non-canonical actions and either
    emits unbounded rewards (the per-step bound still applies) or simply
    terminates with attacker-friendly cumulative total (no per-step
    violation if individual steps stay in bounds).
- Why the validator misses it: `reward_bounds.py:285` reads `if terminated:`
  before evaluating the per-episode arm. The check intentionally treats
  truncated episodes as out-of-scope for cumulative-bound enforcement, on
  the rationale that a truncated total is incomplete. But the rule does
  not require ANY terminated episode in the sample; if zero of eight
  terminate, `per_episode_unverified=True` is set and `passed=True` is
  still returned (`reward_bounds.py:311-312`). Downstream callers that
  treat `passed=True` as "the per-episode bound is verified" inherit a
  silent gap.
- Exploit cost: trivial for Route A (a hard-coded `terminated=False` is
  one line). Route B is the canonical-action lookup pattern from RT-001
  F-002, which is already DEFERRED Phase 2; the marginal cost of also
  using it to suppress termination is zero.
- Profit shape: ongoing along the per-episode axis. The attacker can ship
  any per-episode bounds they like and never get them tested. Combined
  with A-101, the per-episode arm is a no-op for any creator willing to
  declare loose per-episode bounds and engineer non-termination. HIGH
  because it removes a major check entirely, with measurable downstream
  effect on solver_baseline and any reward-weighted scoring.
- Fix sketch: distinguish "per-episode arm did not run" from "per-episode
  arm ran and produced no violations". When `terminated_episode_count == 0`,
  treat the check as `passed=False` with a structured failure (or at minimum
  refuse to surface `passed=True` from a validator entrypoint that depends
  on per-episode verification). Alternatively, when no rollout terminates,
  fall back to a semantic upper bound: cumulative reward never exceeds
  `max_per_step * actual_steps` and never falls below
  `min_per_step * actual_steps`; reject the manifest if the per-episode
  bound declared is not derivable from the per-step bound and the actual
  steps observed across the truncated rollouts (closes A-102 and A-103
  jointly).

### A-104: Non-finite reward bypass (NaN, +inf, -inf, negative zero)
- Category: numerical edge cases
- Severity: HIGH
- Premise: `check_reward_bounds` compares each reward against declared
  bounds via `<` and `>`. Python's IEEE-754 comparison semantics treat
  `NaN < x` and `NaN > x` as False for every x. An env that emits
  `float('nan')` as a reward bypasses both arms of the per-step check
  silently.
- Mechanism: At `reward_bounds.py:269`,
  `r < bounds.min_per_step or r > bounds.max_per_step` is evaluated
  literally. With `r = float('nan')`, both comparisons are False, so no
  StepViolation is appended. `total += r` (`reward_bounds.py:267`) makes
  `total = nan`. At `reward_bounds.py:286`,
  `total < bounds.min_per_episode or total > bounds.max_per_episode` is
  also False for NaN. `min_r = math.inf`, `max_r = -math.inf` and the
  `if r < min_r` / `if r > max_r` updates use `<` and `>`, so NaN never
  updates min_r or max_r. The SeedSample records `min_reward_seen=inf`
  and `max_reward_seen=-inf` (because `n > 0` but no comparison ever
  fired) which is internally inconsistent but does not flag the report
  as failed. `report.passed` becomes True. `+inf` and `-inf` are caught
  by the `<` / `>` comparisons against finite declared bounds (so they
  produce StepViolations correctly, assuming declared bounds are
  finite), but if the creator combines A-101 with `+inf` rewards
  (`bounds.max_per_step = +inf`), the comparison `inf > inf` is False
  and the violation is suppressed. Negative zero (`-0.0`) is honestly
  comparison-equal to `0.0` and is not exploitable on its own.
- Why the validator misses it: there is no `math.isnan(r)` or
  `math.isfinite(r)` guard before the bound comparisons. The protocol
  trusts that whatever `float(reward)` (`_rollout.py:297`) returns is a
  well-formed real number. NaN propagates silently through `total +=`
  and contaminates the SeedSample's `episode_total` field.
- Exploit cost: trivial. `return obs, float('nan'), False, False, {}`
  inside the env's `step`.
- Profit shape: ongoing and severe. NaN rewards are entirely
  unconstrained by the check; the env can emit them on production
  inputs while staying silent on validator inputs (route via the
  canonical-action gate from F-002 if needed). Downstream consumers
  (solver_baseline, leaderboard) that average or sum rewards now
  consume NaN, which corrupts their outputs in ways that depend on
  their own handling. HIGH because it is a silent bypass with broad
  downstream contamination, not just a bound miss.
- Fix sketch: at the top of the per-step bound check, reject any reward
  with `math.isnan(r) or math.isinf(r)` as a hard StepViolation
  regardless of declared bounds. (Optionally accept `-inf` / `+inf`
  only when declared bounds are also non-finite, to allow envs with
  unbounded reward semantics; but defaults must reject.) Also reject
  `min_per_step` or `max_per_step` declared as NaN at manifest
  validation time.

### A-105: Float-comparison edges and exact-bound games
- Category: floating-point comparison gaming
- Severity: LOW
- Premise: `check_reward_bounds` uses strict `<` and `>` comparisons
  (`reward_bounds.py:269, 286`). A reward exactly equal to `min_per_step`
  or `max_per_step` is in-bounds. A creator can declare `max_per_step =
  0.99` and an env that pays exactly `0.99 + 1e-17` per step; in float64,
  `0.99 + 1e-17 == 0.99` (the increment is below ULP at that magnitude),
  so the comparison `r > 0.99` is False and no violation fires. This is
  the same property that lets the check be honest for honest envs;
  exploitation requires the creator to also pay non-trivial reward
  excess somewhere else. The attacker can also declare bounds whose
  least-significant bits straddle a specific reward value emitted by
  the env so the comparison goes their way on one validator's hardware
  and not another's, but as RT-001 A-004 documents, modern CPython on
  IEEE-754 platforms is bit-stable.
- Why the validator misses it: there is no fuzzing of the bound by epsilon,
  no slack tolerance, no second comparison at adjacent floats. The
  comparison is a single deterministic IEEE-754 ordered compare. This is
  correct behavior; the alternative (epsilon slack) introduces its own
  attack surface (an env can emit `max_per_step + 0.5*epsilon` and be
  in-bounds despite obviously cheating).
- Exploit cost: trivial.
- Profit shape: bounded by the size of the smallest float increment that
  rounds to the bound, which at `r = 1.0` is 2^-53 = 1.1e-16. A creator
  cannot inflate reward measurably this way; the boundary handling is
  arithmetically correct under IEEE-754. LOW because the practical
  exploit yield is below detection precision; listed for completeness so
  future passes do not waste cycles.
- Fix sketch: not needed. Document explicitly that bounds are inclusive
  endpoints (`r in [min, max]` with `<` / `>` for the violation check)
  and accept the consequence: rewards exactly equal to the declared
  bounds pass.

### A-106: Brute-force search of env-defining fields to align derived seeds
- Category: sample-seed evasion (post-630135e residual surface)
- Severity: MEDIUM
- Premise: After 630135e, the validator's sample seeds are
  `derive_validator_seeds(manifest, 8, salt=b"reward_bounds")` which
  hashes only `env_id`, `env_version`, `entry_point`,
  `canonical_bytes(kwargs)`, and the salt (`_seeds.py:74-85`). A
  determined attacker iterates over plausible variants of these four
  fields until the resulting eight derived seeds, when fed through
  `SeededRandomPolicy.actions`, produce action sequences whose induced
  trajectories all happen to stay inside whatever the env's
  "honest-on-this-seed" predicate is. This is the inverse direction
  of A-103-Route-B: instead of teaching the env to recognize canonical
  actions, the attacker teaches the manifest to land on seeds whose
  canonical actions the env already handles cleanly.
- Mechanism: For a candidate `(env_id, env_version, entry_point, kwargs)`,
  the attacker computes the eight derived seeds; for each seed `s`, the
  canonical action sequence is `np.random.Generator(np.random.PCG64(s))
  .integers(0, n_actions, size=max_episode_steps, dtype=np.int64)`
  (`_rollout.py:177-178`). The env's "honest" branch can be e.g. "the
  total reward over this canonical action sequence falls inside
  declared bounds"; "honest" is fully a function of the seed and the
  action-space size given fixed env code. The search is over four
  attacker-controlled inputs plus a single attacker-controlled
  predicate. With a one-int kwarg (e.g. `grid_size`) and 100
  candidate values, plus a small `env_version` patch-bump space (a
  few thousand PEP 440 strings), and `env_id` allowing 64-char names
  (huge entropy budget), the attacker has at minimum 10^7 candidates
  cheaply.
- Why the validator misses it: `derive_validator_seeds` is documented to
  carry exactly this Phase 1 limitation (`_seeds.py:40-46`). The fix is
  Phase 2 chain-beacon entropy. RT-001 F-002 carries the same shape on
  the determinism side and is DEFERRED.
- Exploit cost: search cost is what the manifest-level entropy permits.
  `env_id` regex is `^[a-z][a-z0-9_-]{2,63}$` (`manifest.py:9`), giving
  ~37^60 ~ 10^94 candidate env_ids; PEP 440 admits trillions of
  versions; `kwargs` accepts arbitrary JSON-serialisable structure.
  The attacker only needs to find one tuple satisfying their predicate,
  not all. With cheap predicates ("is the canonical action sequence on
  the validator's chosen seeds compatible with my honest branch") on
  the order of seconds to minutes per candidate, finding a hit in an
  afternoon of compute is plausible. The lower bound on cost depends
  on how restrictive the predicate is; a creator who only needs the
  eight seeds to agree is probably 10^4 to 10^6 candidates, well
  within reach.
- Profit shape: ongoing. Once a manifest-tuple is found, every
  validation epoch resamples the same eight seeds (deterministic in the
  manifest), so the cheap predicate persists across epochs until the
  manifest is updated. MEDIUM because it is gated by Phase 2 hardening
  and is an instance of an already-tracked deferred class (F-002), but
  the per-axis cost analysis must be in this catalog so the Phase 2
  fix takes reward_bounds-specific seed search into account, not just
  determinism.
- Fix sketch: chain-beacon entropy in `derive_validator_seeds` (already
  the planned Phase 2 fix for F-002); equivalently, derive seeds from
  the validator's per-epoch nonce. Until then, document the
  manifest-tuple search cost explicitly in the docstring (it currently
  says "search cost grows quickly with n" without a quantitative
  estimate).

### A-107: Validator detection via seeded-random action distribution
- Category: validator detection
- Severity: MEDIUM
- Premise: An env can detect that it is running under a validator
  reward_bounds rollout (rather than a real solver) by inspecting the
  statistical signature of the actions it receives. Once detected, the
  env behaves honestly; otherwise it pays attacker-friendly rewards.
- Mechanism: `SeededRandomPolicy.actions` (`_rollout.py:177-178`) draws
  `n_steps` uniform-random actions from a Discrete space using
  `np.random.Generator(np.random.PCG64(seed))`. The env can buffer the
  last K actions it has received and run a chi-square uniformity test,
  or compare the observed action sequence against a precomputed library
  of `SeededRandomPolicy.actions(s, n_steps, action_space)` for every
  seed `s` in the eight `derive_validator_seeds(manifest, 8, b"reward_bounds")`
  outputs (computable inside the env at import time, since both the
  manifest's env-defining fields and the salt are public, and
  `derive_validator_seeds` is in the public package). A direct compare
  is exact rather than statistical: at step 0 the env knows whether it
  is on a validator-canonical action sequence, and `step` can branch
  accordingly with zero false positives.
- Why the validator misses it: the validator does not vary the action
  policy between sampling and production. There is only one policy
  registered (`_rollout.py:181-183`, `POLICY_REGISTRY` has only
  `SEEDED_RANDOM`). The env always knows it is being driven by
  canonical actions because that is the only thing the validator ever
  drives it with. This compounds A-103 (Route B) and A-106 directly.
- Exploit cost: cheap. The action-sequence comparison is `O(K)` per
  step for buffered K, or a single hash-table lookup if the env
  precomputes the validator action library at import. The eight
  validator-derived seeds can be recomputed inside `__init__` from the
  manifest fields the env code already knows (the env author wrote
  the manifest).
- Profit shape: ongoing, multiplicative with any bound-violating
  behavior the env wants to gate on "production-only". MEDIUM because
  it is the same conceptual gap as F-002 from RT-001 (canonical-action
  predictability) but in the other direction: F-002 lets the env detect
  the seed, A-107 lets the env confirm the seed-driven action pattern.
  The attacker needs both for the cleanest exploit; either alone is
  weaker.
- Fix sketch: introduce a second action policy at minimum
  (`SEEDED_RANDOM_OFFSET` or `SEEDED_RANDOM_BIASED`) and have
  `check_reward_bounds` run a fraction of its sample budget under
  the alternate policy. Combined with chain-beacon seeds (A-106 fix),
  the env cannot precompute the validator's exact action sequence.
  Long-term, the validator should occasionally drive the env with a
  trained solver's actions (out-of-distribution from random), making
  validator-versus-production indistinguishability a serious cost
  for the attacker.

### A-108: StepRecord aliasing and rollout-primitive shared state
- Category: validator logic bypass / rollout primitive
- Severity: LOW
- Premise: `iter_rollout` (`_rollout.py:229-307`) is shared by
  `check_determinism`, `check_reward_bounds`, and (per CLAUDE.md) the
  other sample-based checks. A bug in the primitive that conflates
  StepRecord fields between consumers, or that reuses state across
  invocations, would corrupt all consumers simultaneously. RT-002
  examines this for reward_bounds.
- Mechanism: Inspection of `_rollout.py:229-307`. `iter_rollout` is
  called once per seed inside `check_reward_bounds`
  (`reward_bounds.py:251`); each call constructs a fresh env via
  `_load_env(env_spec)` (`_rollout.py:279`) and a fresh generator
  closure (`_rollout.py:289`). `StepRecord` is `frozen=True, slots=True`
  (`_rollout.py:191-221`) so individual records cannot be mutated by
  later consumers. `info` defaults to an empty dict via `field(default_factory=dict)`
  (`_rollout.py:221`), and is set per-step from the env's return
  (`_rollout.py:300`). The generator's `try / finally: env.close()`
  (`_rollout.py:304-305`) guards env release; if the consumer raises
  before exhausting the iterator, env.close() runs only at GC, but
  reward_bounds always exhausts `it` via the `for record in it:` loop
  (`reward_bounds.py:260`), so this is not a reward_bounds-side leak
  on success. On exception inside the loop body, the generator is
  GC'd; that is a determinism-shared concern flagged by RT-001 loose
  threads, not a reward_bounds-specific finding.
- Why the validator catches it: `iter_rollout` does not mutate any
  module-level state. The action sequence (`_rollout.py:283`) is a
  fresh ndarray per call. The env is fresh per call. There is no
  shared cache, no module-global dict, no class-attribute side
  channel.
- Exploit cost: not exploitable in the current code. Listed because
  RT-002 is supposed to consider the primitive surface and the
  threat model attached to it. If a future refactor introduces
  caching of `_load_env` or the action-policy `actions(...)` array,
  this becomes exploitable.
- Profit shape: zero today. Useful as a regression-watch item for
  any future refactor that adds caching or resource pooling to
  iter_rollout.
- Fix sketch: not required today. If caching is introduced, ensure
  the cache key includes the seed AND the salt (or check identity)
  so a determinism rollout cannot pollute a reward_bounds rollout
  state. Document the no-shared-state invariant in the iter_rollout
  docstring.

### A-109: Negative-zero per-episode total under exact symmetry
- Category: numerical edge cases
- Severity: LOW
- Premise: An env that engineers `total = -0.0` exactly (e.g. by emitting
  rewards that sum to negative zero through canceling positive and
  negative contributions) sits at a comparison boundary that may surprise
  readers. `-0.0 < 0.0` is False in IEEE-754; `-0.0 == 0.0` is True. So
  if `min_per_episode = 0.0` exactly, an env emitting `total = -0.0`
  passes the per-episode check; under integer accounting it would have
  failed.
- Mechanism: The per-episode check is `total < bounds.min_per_episode`
  (`reward_bounds.py:286`). With `total = -0.0` and `min_per_episode = 0.0`,
  `-0.0 < 0.0` is False; no EpisodeViolation. This is the IEEE-754
  contract and matches A-105 in shape: bounds are inclusive endpoints.
- Why the validator catches it: this is not a bypass; it is the
  documented inclusive-endpoint behavior. Negative zero compares as
  zero, which is what the bound says. A creator who wanted to emit
  -0.0 to evade a `total < 0` check would be evading a bound the
  honest declaration `min_per_episode = 0.0` already accepts.
- Exploit cost: zero, but profit is zero too.
- Profit shape: none. Listed because it appears in the attack-surface
  enumeration for completeness and because it is the kind of numerical
  edge a future change might inadvertently break.
- Fix sketch: not needed.

## Loose threads (cross-cutting)

- `solver_baseline` consumes `manifest.declared_reward_bounds` for
  reward normalization (cross-checked by reading `checks/solver_baseline.py`
  if present in the tree; flagged here for the next pass to verify). If
  it is true that solver_baseline normalizes by `max_per_episode -
  min_per_episode`, A-101's vacuous-bounds attack collapses normalization
  to roughly zero, which would silently change the solver-baseline
  comparison. This compounds A-101 into a cross-check exploit; it
  belongs in RT-004 (solver_baseline pass) but RT-002 should flag the
  shape so the next pass picks it up.
- `derive_validator_seeds` in `_seeds.py` hashes
  `canonical_bytes(manifest.kwargs)`. If `kwargs` is allowed to contain
  large nested structures (the Pydantic typing is `dict[str, Any]`,
  `manifest.py:41`), the attacker has effectively unbounded entropy in
  one place: an attacker manipulating `kwargs` does not change env
  behavior (because the env probably only reads a small subset of keys)
  but does change the derived seeds. This is the cheapest channel for
  the A-106 brute force. RT-002 raises it; the fix likely lives at the
  manifest layer (e.g. constrain kwargs to the keys the env constructor
  accepts, deny extras).
- `reward_bounds.py:253-254` initializes `min_r = math.inf, max_r =
  -math.inf`. For NaN rewards (A-104), neither updates, so the
  SeedSample reports `min_reward_seen=inf, max_reward_seen=-inf`. That
  is internally inconsistent (min > max) and a future code reviewer
  reading the SeedSample would be confused. Cosmetic, but the asymmetry
  is the visible artifact of the NaN bypass; if NaN handling is added,
  this initialization should pivot to a sentinel that signals "no real
  reward seen".
- `iter_rollout` always calls `env.reset(seed=seed)` (`_rollout.py:285`).
  reward_bounds runs eight rollouts back-to-back inside the same Python
  process; if the env retains module-level state across instantiations
  (e.g. a module-global cache of validator-friendly seeds), reward_bounds
  cannot detect the leak because each rollout is a separate `_load_env`
  call but they share `import_module` returning the same module object
  (`_rollout.py:92`). This is the same import-time-side-effect surface
  as RT-001 F-003 (DEFERRED), in a different lighting.
- The per-step check (`reward_bounds.py:269`) appends a fresh
  StepViolation for every offending step. A pathological env that emits
  one violating reward per step for `max_episode_steps` steps over
  eight seeds produces `8 * max_episode_steps` StepViolation records,
  each with a Pydantic model overhead. With `max_episode_steps=10000`,
  that is 80000 violation records held in memory. Not exploitable for
  reward inflation, but a cheap memory-pressure / log-flooding vector
  worth noting.
- The per-step bound check uses `r < bounds.min_per_step or r >
  bounds.max_per_step` (`reward_bounds.py:269`), but the per-step
  STATISTICS (`min_r`, `max_r`) use the same form
  (`reward_bounds.py:263, 265`). For a NaN reward, the statistics never
  update; for a `+inf` reward, both `r < min_r` (False) and `r > max_r`
  (True if max_r was `-inf`) update max_r to inf. So `max_reward_seen`
  becomes `inf`, which is faithful, while `min_reward_seen` stays
  `+inf` if no other reward was seen. The statistics are honest under
  finite rewards; they degrade silently under non-finite, in lockstep
  with A-104.

## Findings index

Six findings carry HIGH or MEDIUM severity. F-006 (vacuous bounds), F-007
(per-step / per-episode desynchronization), F-008 (truncation-only escape
hatch), and F-009 (NaN reward bypass) are HIGH and chain together: any one
of them lets a creator pass reward_bounds while shipping a bound-violating
env. F-010 (manifest-tuple seed brute force) and F-011 (validator detection
via canonical-action signature) are MEDIUM and overlap with RT-001's F-002
(canonical-action lookup, DEFERRED) on the determinism side; both must be
addressed jointly by the Phase 2 chain-beacon hardening, which should now
cover both `b"determinism_self_consistency"` and `b"reward_bounds"` salts.
The remaining attacks are LOW because they are either correct behavior
under IEEE-754 (A-105, A-109) or not exploitable in the current code
(A-108).

| ID | Severity | Summary | Linked attack |
|----|----------|---------|---------------|
| F-006 | HIGH | RewardBounds accepts arbitrarily wide declared bounds; check trivially passes for any finite-reward env. | A-101 |
| F-007 | HIGH | No invariant tying max_per_episode to max_per_step * max_episode_steps; per-episode bound can be declared loosely while per-step is tight. | A-102 |
| F-008 | HIGH | per_episode_unverified=True still returns passed=True; envs that always truncate bypass the per-episode arm entirely. | A-103 |
| F-009 | HIGH | NaN rewards silently pass both per-step and per-episode comparisons; no isnan/isfinite guard in the check. | A-104 |
| F-010 | MEDIUM | Manifest 4-tuple search lets a creator align derived reward_bounds seeds with their honest predicate; same shape as RT-001 F-002 in a different salt. | A-106 |
| F-011 | MEDIUM | Only one action policy registered; envs can recompute and recognize the canonical action sequence for any derived seed and gate honesty on it. | A-107 |
