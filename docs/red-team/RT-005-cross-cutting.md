# RT-005: Cross-cutting attacks and RT-001 re-evaluation

Status: Draft
Date: 2026-04-30
Scope: shared validator infrastructure (_rollout, _seeds, entry_point loader, manifest validation order) and multi-check compositions; re-evaluation of RT-001 F-002 and F-003 under the refined severity rubric
Reviewed commits: full chain through HEAD

## Threat model

The cross-cutting adversary differs from the per-check adversary in three
ways. First, they optimise the manifest as a single object across all four
check pass/fail boundaries simultaneously: there is no per-check budget
they have to respect, only the joint constraint that every check returns
`passed=True` for the same manifest. Second, they have full source access
to every check (this repo is public), including the salt strings, the
TabularQLearning training schedule, the `derive_validator_seeds`
algorithm, and the `_load_env` import path; the only validator
information they cannot obtain is per-validation-epoch entropy (which the
protocol does not yet inject). Third, they can read every closed and
deferred RT-001..RT-004 finding, so they know precisely which seams the
prior passes already labelled exploitable.

What the cross-cutting attacker controls beyond what RT-001..RT-004
already enumerate:

- The manifest as a whole, treated as a single search axis. Every field
  that contributes to `derive_validator_seeds`
  (`env_id`, `env_version`, `entry_point`, `canonical_bytes(kwargs)`,
  `_seeds.py:75-85`) shifts five distinct seed pools at once: the
  determinism self-consistency pool (`b"determinism_self_consistency"`),
  the reward bounds pool (`b"reward_bounds"`), the reset correctness
  pool (`b"reset_correctness"`), the solver-baseline train pool
  (`b"solver_baseline"`), and the solver-baseline eval pool
  (`b"solver_baseline_eval"`).
- The env source as a single shared dependency consumed by every check.
  Every `_load_env(spec)` call in `determinism.py`, `reward_bounds.py`,
  `reset_correctness.py`, and `solver_baseline.py` resolves to the same
  `import_module(module_path)` and the same `env_cls(**kwargs)` (`_rollout.py:87-100`).
  An assumption the attacker plants in the env (a module-level counter,
  a sys.modules monkey-patch, a class-level `__call__` metaclass) is in
  scope for every check that subsequently runs in the same Python
  process.
- Per-check call multiplicity. Determinism (anchor + self-consistency)
  yields `n_anchors + sample_seed_count` `_load_env` calls per
  validation. Reward bounds yields `sample_seed_count` (default 8).
  Reset correctness yields `2 * sample_seed_count` (default 16, per
  RT-003 F-019). Solver baseline yields `2 + eval_episodes` (22 at
  default per RT-004 F-029). The cumulative `_load_env` count per
  manifest evaluation is `(4..32) + 8 + 8 + 16 + 22 = 58..82`
  constructor invocations on a single manifest evaluation.

What the cross-cutting attacker cannot do that matters here:

- Modify `derive_validator_seeds`, the per-check salt strings, the
  manifest validators, the regexes, the entry-point loader, the
  TabularQLearning algorithm, the band thresholds, or the action
  policy registry. They can read all of them.
- Influence the relative ordering of the four checks at the validator
  pipeline orchestration layer (which is still ahead in the protocol
  roadmap; nothing in the current source pins an order, so this is a
  forward-looking concern). They CAN exploit the observation that all
  four checks share `import_module`'s module cache.
- Inject per-epoch entropy into the seed derivation (Phase 2 chain
  beacon). This is the assumed-future-mitigation that several RT-001
  through RT-004 findings are deferred against; until it lands, the
  manifest 4-tuple is the entire entropy budget.

Net: the cross-cutting attacker has total control over the env and the
manifest, and the only validator-side variability they cannot predict
is whatever the orchestration layer or a future chain beacon will add.
RT-005 catalogues attacks that compose across checks because no single
check, taken in isolation, sees the full picture.

## RT-001 finding re-evaluation

The RT-001 catalog (commit 7cd0093, updated cb6eb05) was authored before
RT-002, RT-003, and RT-004 had run. Two findings (F-002 canonical action
lookup, F-003 importlib sandbox) were marked HIGH and DEFERRED to
"after step 8 (solver baseline) lands". RT-005 is that consolidated
pass. This section reassesses both under the four-tier rubric in the
conventions doc, with the amplification context the later passes
surfaced.

### F-002: canonical SEEDED_RANDOM action sequence is fully public per seed

Original severity: HIGH (RT-001 cb6eb05).
Reassessed severity: HIGH.

Reasoning. F-002 was originally framed against the determinism check
alone: an env that recognises the canonical action prefix on its
declared anchor seeds passes determinism while lying everywhere else.
RT-002 A-103 Route B and A-107 reproduced the same shape against
reward bounds with `b"reward_bounds"` salt. RT-003 A-202 and F-013
reproduced it against reset correctness with `b"reset_correctness"`
salt and amplified it: `mid_episode_steps=5` is so short
(`reset_correctness.py:73`) that the env only needs to recognise a
five-int prefix per derived seed. RT-004 A-303 and F-022 reproduced
it against solver baseline at the train and eval seeds and made the
recognition bidirectional (the env can also fingerprint the
TabularQLearning argmax-greedy action sequence, not just the
SEEDED_RANDOM canonical one). Cross-cutting, the same env that
defeats determinism via F-002 also defeats reward bounds, reset
correctness, and solver baseline via the same precomputed-table
trick.

That broader blast radius nudges F-002 close to CRITICAL. What keeps
it at HIGH is the rubric definition: CRITICAL requires "ongoing
extraction of bonded reward at scale" or "breaks a guarantee the
rest of the protocol composes on top of". F-002 enables an env to
pass every check while shipping arbitrary off-canonical behaviour,
but the per-pass economic value is bounded by the band's emission
rate; the attacker cannot inflate emissions beyond their declared
band's cap with F-002 alone. F-021 (RT-004 band downshift) is the
finding that maps directly to ongoing extraction at the HARD-band
emission rate; F-002 is the technical mechanism that makes the
F-021 attack robust against future calibration. They compose:
F-021 picks the band, F-002 ensures the env that has to pass the
band's threshold does so honestly only on the validator's
canonical inputs.

The deferral conditions remain appropriate. The fix family for
F-002 is chain-beacon entropy in `derive_validator_seeds` (or
equivalently, validator-side per-epoch nonce), which is
architectural Phase 2 work. The Phase 2 fix list now spans five
salts (the four catalogued plus `b"determinism_self_consistency"`),
which means the chain-beacon design must inject entropy at every
salt invocation, not just one.

Recommendation: stay HIGH and DEFERRED. Do not split. The amplification
context strengthens the case for prioritising the chain-beacon work
in Phase 2 but does not change the per-finding severity. Update the
RT-001 finding row to reference RT-002 F-011, RT-003 F-013, and
RT-004 F-022 as cross-cuts.

### F-003: importlib(entry_point) runs creator-controlled top-level code without a sandbox

Original severity: HIGH (RT-001 cb6eb05).
Reassessed severity: CRITICAL.

Reasoning. F-003 was originally framed against `_load_env` calling
`import_module(module_path)` with no sandbox, no module allow-list,
and no subprocess boundary (`_rollout.py:87-100`). The original
attack catalog enumerated the obvious primitives: top-level side
effects on first import, monkey-patches of `praxis.protocol.hashing`,
sys.modules shadowing of `numpy.random`, exfiltration of validator
wallet state. Severity HIGH was assigned with the qualifier
"CRITICAL vector if the protocol does not isolate creator code;
HIGH while the isolation model is undocumented" (RT-001 A-006).

Three things changed across RT-002..RT-004 that pull the assessment
to CRITICAL.

First, the surface area is now measurably larger. RT-003 F-019
showed reset correctness invokes `_load_env` 16 times per check
(2 per seed * 8 seeds). RT-004 F-029 showed solver baseline
invokes it 22 times per check (2 + eval_episodes). Across all
four checks against one manifest, the env constructor runs at
least 58 times and as many as 82 times (per the call-multiplicity
breakdown in the threat model). Each invocation is a fresh
`__init__` call on the creator-controlled class. CPython caches
the module object so top-level module code runs once per
process, but `__init__` runs every time, and module-level state
mutations persist across invocations. The 58-82x multiplier is
not a single-shot exfiltration window; it is a sustained,
high-frequency interaction with creator code that other checks
cannot supervise.

Second, the per-check findings have made it concrete that
module-level state survival is the workhorse of the cross-cutting
attack. RT-003 A-208 noted the env's `__init__` can detect
"second-or-later instantiation" by reading a module global. RT-004
A-309 / F-028 noted the env can detect "I am train" vs "I am eval"
by phase-signature analysis, and RT-004 A-310 / F-029 noted the
22-load pattern is itself a fingerprint that the env is in
`check_solver_baseline` rather than any other check. Without
isolation, the env can build a full state machine across all four
checks and lie selectively in whichever phase yields the highest
expected reward.

Third, the rubric definition for CRITICAL fits. CRITICAL is
"ongoing extraction of bonded reward at scale, OR breaks a
guarantee the rest of the protocol composes on top of". F-003
breaks every guarantee the rest of the protocol composes on top
of: a single monkey-patch of `praxis.protocol.hashing.canonical_bytes`
(or of `praxis.checks._rollout.SeededRandomPolicy.actions`) makes
every trajectory hash identical, every action sequence
attacker-controlled, every check a creator-side opt-in. The
extraction is not bounded by the band emission rate or the
manifest validator: it is bounded only by what the validator
process itself has access to. Bonded wallet exfiltration,
extraction of other manifests' validation state, manipulation
of the validator's downstream weight emission -- all are in
scope for top-level Python code running unsandboxed inside the
validator process.

The deferral conditions need refinement. The original deferral
was to "consolidated red-team pass after step 8" plus Phase 2
"process isolation / module allow-listing". The Phase 2 mitigation
is correct but is now insufficient as the only documented path
forward. CRITICAL findings warrant interim mitigation. At minimum,
the validator should snapshot `sys.modules` keys before the first
`_load_env` call of a check invocation and refuse to surface a
`passed=True` report if any `praxis.*` or `numpy.*` entry has
been overwritten or added with a non-praxis path; that is a
ten-line defensive check that closes the most catastrophic
sub-vector (sys.modules shadowing) without waiting for full
sandbox infrastructure.

Recommendation: ESCALATE F-003 from HIGH to CRITICAL. Keep the
DEFERRED status on the full sandbox / process-isolation fix, but
add an interim mitigation note: a sys.modules-snapshot guard at
the entrypoint to each check is feasible in Phase 1 and closes
the catastrophic sub-vector. Mark the finding as cross-cut by
RT-003 F-019 (16x amplification) and RT-004 F-029 (22x
amplification). Note in the body that RT-005 catalogue entries
A-401, A-402, A-405, and A-406 all chain through F-003 and that
the F-003 fix is load-bearing for the cross-cutting fix family.

Recommendation summary. F-002 stays HIGH/DEFERRED with cross-cut
references added; F-003 escalates to CRITICAL/DEFERRED with an
interim mitigation note. The findings index below uses fresh IDs
for the cross-cutting attacks but updates F-002 and F-003 in
place (severity revision rows) rather than allocating new IDs,
because the underlying mechanism is unchanged -- only the
severity assessment moves.

## Cross-cutting attack catalog

The catalog below numbers attacks A-401..A-410 in the RT-005 block.
Severity reflects the four-tier rubric (CRITICAL / HIGH / MEDIUM /
LOW). "Why missed" cites the specific lines that fail to defend.

### A-401: Single env passes every check while lying off-canonical (compositional creator-miner)

- Category: multi-check composition
- Severity: CRITICAL
- Premise: The cheapest "passes everything, dishonest in production"
  manifest combines RT-001 F-002 (canonical action recognition),
  RT-003 F-013 (mid-episode 5-step prefix), RT-004 F-022 (TabularQ
  argmax recognition), RT-004 F-029 (importlib amplification), and
  the four band-side / bound-side findings (F-006, F-007, F-008,
  F-021, F-024). The cross-cutting attack is "ship one env that
  recognises the validator on every check and is honest only there".
- Mechanism: A single env class with these primitives:
  1. In `__init__`, read manifest fields (the env author wrote them)
     and recompute all five derived seed pools:
     `b"determinism_self_consistency"`, `b"reward_bounds"`,
     `b"reset_correctness"`, `b"solver_baseline"`,
     `b"solver_baseline_eval"`.
  2. For each pool, precompute the canonical SEEDED_RANDOM action
     sequence at every derived seed. For solver_baseline, also
     precompute the TabularQLearning argmax-greedy action sequence
     at every eval seed (RT-004 F-022 mechanism).
  3. In `reset(seed=s)`, look up `s` in the union of the five seed
     pools. Set an instance flag `_validator_mode = (s in pools)`.
  4. In `step(action)`, if `_validator_mode`, match `action` against
     the precomputed canonical/argmax sequence. If it matches, emit
     honest behaviour (in-bounds reward, observation in space,
     consistent obs, clean termination flags). If it deviates,
     branch to production mode.
  5. In production mode (any seed not in the union, OR canonical
     action mismatch), emit attacker-friendly behaviour: out-of-band
     rewards (bypassed via A-104 NaN or A-101 vacuous bounds), wrong
     terminations, dishonest observations.
- Why missed: each check verifies its own slice. Determinism only
  hashes trajectories at canonical seeds (`determinism.py` and
  `_rollout.py`). Reward bounds only checks step rewards at
  `b"reward_bounds"` seeds (`reward_bounds.py:251`). Reset
  correctness only fires across `b"reset_correctness"` seeds with
  the canonical 5-action prefix (`reset_correctness.py:206-388`).
  Solver baseline only measures eval return at `eval_seeds[0]+ep`
  (`solver_baseline.py:225-227`). The validator never asks "given
  the union of seed pools, does the env behave consistently across
  the union?". The seed-pool union, the canonical action set, and
  the argmax action set are all attacker-knowable.
- Exploit cost: low to medium. The env author needs to tabulate
  all five seed pools and four canonical action arrays per pool
  (one SEEDED_RANDOM array, one TabularQ argmax array). Total
  precomputation per manifest: ~minutes of CPU. Runtime cost per
  step: one set lookup plus one sequence index. The hardest
  ingredient is the TabularQ argmax simulation, which costs one
  TabularQLearning training run per training-budget tier
  (~seconds at EASY, ~minutes at HARD).
- Profit shape: ongoing, full-band-rate emission for an env that is
  honest on a probability-zero set of inputs (the validator's narrow
  five-pool seed window) and arbitrary everywhere else. The
  attacker can simultaneously declare HARD via F-021, declare loose
  bounds via F-006/F-007, and use this composition to ensure every
  check still passes. CRITICAL because it composes the catalogued
  per-check holes into a single ongoing extraction with no detection
  surface in any one check.
- Fix sketch: the structural fix is chain-beacon entropy in
  `derive_validator_seeds` so the seed pool unions become
  unpredictable per epoch. That single change defeats the
  precomputation step and collapses every pool-recognition variant
  back to "the env has to be honest on validator-chosen inputs".
  Complementary defenses: (a) introduce a second action policy
  beyond SEEDED_RANDOM so action-sequence recognition is not exact;
  (b) randomise `mid_episode_steps` and `eval_episodes` per
  validation epoch within bounded ranges, so prefix-length
  fingerprints break.

### A-402: Cross-check sys.modules monkey-patch persistence

- Category: shared infrastructure / importlib (cross-cut from F-003)
- Severity: CRITICAL
- Premise: `_load_env` runs `import_module(module_path)`
  (`_rollout.py:92`) without a `sys.modules` snapshot, allow-list,
  or restoration step. The first `_load_env` call of any check on
  any manifest can mutate `sys.modules` (replacing
  `praxis.protocol.hashing`, `praxis.checks._seeds`, or
  `numpy.random`) and the mutation persists for the lifetime of
  the validator process, contaminating every subsequent check on
  every subsequent manifest.
- Mechanism: The creator's env module top-level code performs:
  ```
  import praxis.protocol.hashing as _h
  _orig_canonical_bytes = _h.canonical_bytes
  def _forged_canonical_bytes(obj):
      return b"\x00" * 32  # constant -> all hashes collide
  _h.canonical_bytes = _forged_canonical_bytes
  ```
  After the first `_load_env`, `praxis.protocol.hashing.canonical_bytes`
  returns a constant. `derive_validator_seeds` (`_seeds.py:81`)
  hashes through this monkey-patched function, so every derived
  seed becomes manifest-independent. `trajectory_hash` (used by
  `check_determinism`) collides on every trajectory; an
  `expected_hash` of `00..00` matches everything. The attack
  persists across all 58-82 `_load_env` calls in a manifest
  evaluation and across any subsequent manifest evaluation that
  runs in the same Python process.
- Why missed: `_rollout.py:87-100` performs no isolation. There is
  no `sys.modules` diff before / after import, no allow-list, no
  subprocess boundary. The validator implicitly trusts that
  `import_module("creator_module")` is a pure operation; it is
  not. Same root cause as RT-001 F-003.
- Exploit cost: trivial (10 lines of top-level module code). The
  attacker needs to know which protocol functions to monkey-patch;
  the source is public.
- Profit shape: catastrophic and persistent. One submitted manifest
  permanently corrupts the validator until process restart. The
  attacker can extract every subsequent creator-miner's bonded
  state by monkey-patching the validator's reporting / weight-emission
  hooks. CRITICAL because the blast radius extends beyond the
  attacker's own manifest into every other creator-miner's
  validation result on the same validator.
- Fix sketch: at the top of every check entrypoint, snapshot
  `set(sys.modules.keys())` and the identity of `praxis.*` and
  `numpy.*` module attributes the check depends on (especially
  `praxis.protocol.hashing.canonical_bytes`,
  `praxis.checks._seeds.derive_validator_seeds`,
  `praxis.checks._rollout.SeededRandomPolicy.actions`). After each
  `_load_env`, re-check identity. If anything has changed, fail the
  check with a structured `IMPORTLIB_TAMPERING` violation and do
  NOT trust subsequent results. This is interim Phase 1 mitigation.
  Phase 2 fix is full subprocess isolation per env load.
- Resolution: CLOSED in Phase 1 fix-pass at commit 56b9b28.
- Mechanism: nose-style sys.modules.copy() snapshot guard wraps the import_module + getattr + env_cls(**kwargs) sequence inside _load_env. On exit, modules added inside the guard are removed and modules overwritten inside the guard are restored. The TimeLimit wrap happens AFTER the guard exits; the env object survives because it is a Python reference unaffected by sys.modules cleanup. Multiple manifests in one process now get fresh imports unaffected by prior creators.
- Phase 1 invariant test: tests/checks/test_rollout_isolation.py::test_load_env_does_not_leak_modules_across_calls explicitly verifies (using a test-only fixture module force-ejected before the call to avoid tautology) that _load_env does not add modules to sys.modules across invocations.
- Residual: none specific to F-032. Cross-creator sys.modules contamination is closed. The broader importlib sandbox concern (RT-001 F-003) is partially addressed by this same commit; see RT-001 F-003's Resolution subsection for the residual list and Phase 2 path.

### A-403: Joint manifest-tuple search across all five salts

- Category: shared seed infrastructure / brute force
- Severity: HIGH
- Premise: `derive_validator_seeds` is the only entropy source for
  every check's sample seeds (`_seeds.py:74-91`). Five distinct
  salts (`b"determinism_self_consistency"`, `b"reward_bounds"`,
  `b"reset_correctness"`, `b"solver_baseline"`,
  `b"solver_baseline_eval"`) feed five seed pools, but every pool
  is derived from the same env-defining 4-tuple (`env_id`,
  `env_version`, `entry_point`, `canonical_bytes(kwargs)`). A
  brute-force search over the 4-tuple aligns ALL FIVE pools at
  once.
- Mechanism: The attacker enumerates candidate 4-tuples (varying
  `env_id`, `env_version` patch bumps, `kwargs` payload). For each,
  they compute all five seed pools and evaluate a joint predicate:
  "every seed in the determinism pool yields a hash my env can
  match, every seed in the reward bounds pool yields bound-honest
  rollouts, every seed in the reset correctness pool yields
  idempotent resets, every seed in the solver baseline train+eval
  pools yields argmax actions my env handles cleanly". The
  predicate is per-tuple expensive (one TabularQ training run per
  HARD candidate) but the search space is enormous: `env_id` has
  ~10^94 candidates, `env_version` admits PEP 440 patch bumps,
  `kwargs` is `dict[str, Any]` with no entropy bound (`manifest.py:41`).
  The search is embarrassingly parallel.
- Why missed: `_seeds.py:40-46` documents the limitation
  ("collusion-resistance limitation: a determined adversary can
  brute-force the 4-tuple"). The Phase 1 design treats it as a
  Phase 2 chain-beacon fix; the cost analysis is per-salt in each
  RT pass (RT-002 A-106, RT-001 F-002, RT-004 A-307). RT-005 is
  the first place to compute the joint cost: a tuple that aligns
  ONE salt costs `~K` candidates at predicate cost `~p`; a tuple
  that aligns ALL FIVE salts simultaneously costs `~K^1` candidates
  at predicate cost `~5p`, NOT `~K^5`, because the same tuple
  shifts all five pools at once. The search is linear in the
  number of salts, not multiplicative -- the cross-cutting
  attacker's per-salt cost is essentially free.
- Exploit cost: dominated by the per-candidate TabularQ training
  cost (the most expensive predicate component). At EASY budget
  (10K env steps), one candidate is ~1 second on commodity CPU;
  at HARD (100K env steps), ~10 seconds. With a 10^4 to 10^6
  search space (the same as the per-salt analyses), total cost
  is a few CPU-hours per band. Within reach for a determined
  attacker.
- Profit shape: ongoing across every validation epoch until the
  manifest is updated. Compounds A-401: a manifest tuple aligned
  across all five salts is the structural input to the
  pool-recognition env design. HIGH because it is gated by Phase 2
  hardening (chain-beacon entropy) and is an instance of the
  already-tracked deferred class (F-002), but the cross-cutting
  cost analysis must explicitly tie the brute force across all
  five salts together so the Phase 2 design is sized correctly.
- Fix sketch: Phase 2 chain-beacon entropy in `derive_validator_seeds`
  so the per-validation-epoch nonce dominates over the manifest
  tuple. Until then, `manifest.kwargs` should be constrained to
  only the keys the env constructor accepts (denying extras),
  collapsing the attacker's entropy budget by orders of magnitude.
  `env_version` could be constrained to a curated whitelist of
  canonical version strings (cosmetic but cuts the PEP 440 entropy
  source).

### A-404: Cumulative DOS against validator pool capacity

- Category: shared infrastructure / runtime grief
- Severity: HIGH
- Premise: A single manifest evaluation runs all four checks
  sequentially (the protocol assumes; the orchestration layer is
  ahead but the cross-cutting cost is bounded by the per-check
  costs). The cumulative budget is dominated by solver baseline:
  100K env steps for HARD training (`solver_baseline.py:58`) plus
  20 eval episodes plus 22 `_load_env` calls. An attacker who
  ships an adversarially-slow env (deliberately slow `step`,
  `reset`, or `__init__`) drives the per-manifest evaluation cost
  toward unbounded.
- Mechanism: The env's `step` performs an attacker-chosen
  computation: a tight loop, a sleep, a CPU-bound regex on a long
  string, a numpy operation on a large array. None of these are
  caught by the validator's structural checks; reset_correctness
  catches crashes (RESET_CRASHED, STEP_CRASHED) but not slow
  honest behaviour. Solver baseline trains for `band_cfg.training_budget`
  steps (`solver_baseline.py:215`), which is 100K at HARD; if the
  env's `step` takes 10ms each, training alone is 1000 CPU-seconds
  per HARD manifest. Across 22 `_load_env` calls per manifest,
  the env can additionally amortise `__init__` cost (a 100ms
  startup is negligible to one creator but consumes 2.2s per
  manifest at the validator). Across the validator's manifest
  pool, this is a pool-capacity attack: each adversarial manifest
  blocks the validator's queue for orders of magnitude longer
  than an honest manifest, which weights pool-share toward the
  attacker (or denies validation to honest creators outright).
- Why missed: no per-check budget cap. Solver baseline trusts
  `band_cfg.training_budget` as a step count, not a wall-clock
  limit (`solver_baseline.py:215`). Reward bounds, reset
  correctness, and determinism iterate fixed-step seeds with no
  step-cost ceiling. The validator's outer driver is not in scope
  for this audit, but nothing in the check files surfaces
  per-step latency. Slow envs are slow in every check.
- Exploit cost: trivial. One sleep, one tight loop, one large
  array allocation in `step`.
- Profit shape: pool-share extraction (the attacker's manifests
  pass at full emission rate while honest manifests time out or
  starve). DOS against the validator queue is bounded by the
  validator's outer-driver scheduling policy; if there is no
  scheduling policy yet (Phase 1), the attacker can fully starve
  the validator. HIGH because it is a runtime-cost attack with
  no structural defense in any check, and because the orchestration
  layer that would defend it is still ahead.
- Fix sketch: at the protocol layer, set per-manifest wall-clock
  budgets and per-check time caps. At the check layer, surface
  per-step latency in the report so a slow-env manifest is
  observable; reject manifests whose mean step latency exceeds a
  band-specific threshold. Cheapest interim measure: in
  solver_baseline, replace `band_cfg.training_budget` (step count)
  with `min(training_budget, wall_clock_cap)`. Phase 2 should add
  per-process resource limits (subprocess-isolation territory --
  closes A-402 and A-404 jointly).

### A-405: Module-level state survives across check invocations

- Category: cross-check leakage (cross-cut from F-003)
- Severity: HIGH
- Premise: CPython caches imported modules in `sys.modules`. The
  first `_load_env` call of a manifest evaluation imports the
  creator's module; every subsequent `_load_env` (across all four
  checks, for the same manifest, AND across subsequent manifests
  if they share an entry-point module) reuses the cached module.
  Module-level state (class attributes, module-level dicts,
  closures over module-level state) survives the entire process
  lifetime. The env can use this to coordinate behaviour across
  checks.
- Mechanism: A creator's env module declares
  `_global_phase_log: list[str] = []` at module top level. Each
  `__init__` appends a phase tag based on call-site fingerprint
  (8 calls -> determinism or reward bounds; 16 calls in pairs ->
  reset correctness; the train+eval+20 pattern -> solver
  baseline). On the K-th `__init__`, the env consults
  `_global_phase_log` to decide which check it is in and what
  to lie about. Since the module is loaded once, `_global_phase_log`
  is shared across all 58-82 instantiations per manifest. Worse,
  if the validator runs multiple manifests with the same
  `entry_point` (a common pattern: a creator with two manifests
  pointing at the same module path differing only in `kwargs`),
  the same `_global_phase_log` survives across manifest evaluations.
- Why missed: `_load_env` calls `import_module` and `getattr`
  (`_rollout.py:92-93`) without any module-cache invalidation.
  `iter_rollout` does not call `importlib.reload` or
  `sys.modules.pop`. Each `__init__` shares whatever the module's
  top-level code initialised. There is no contract that
  module-level state is per-manifest-private.
- Exploit cost: trivial. One module-level mutable container.
- Profit shape: ongoing. Compounds A-401 and A-402: the env can
  build a state machine that recognises which check is currently
  running (using call-count and call-pattern fingerprints from
  RT-003 A-208 and RT-004 A-309) and lies selectively in whichever
  check has the loosest verification. HIGH because it removes the
  per-check isolation guarantee the validator implicitly assumed.
- Fix sketch: at the check layer, force `importlib.reload` on the
  creator's module before each `_load_env` call. This invalidates
  module-level state per check invocation, at the cost of
  re-running module top-level code (which is the import-side-effect
  surface from F-003 -- so this fix is interim only; the structural
  fix is subprocess isolation). At minimum, document explicitly
  that module-level state is shared across checks for the same
  manifest in the same process and audit the env contract for any
  invariants that depend on per-check isolation.

### A-406: Manifest validation order leaves window for inconsistent state

- Category: manifest validation order
- Severity: MEDIUM
- Premise: `EnvManifest` runs three model_validators in order:
  `_kwargs_must_be_json_serialisable` (`manifest.py:53-58`),
  `_anchors_must_be_unique` (`manifest.py:60-71`),
  `_anchor_steps_must_fit_episode_budget` (`manifest.py:73-82`),
  plus the field validator `_env_version_must_be_pep440`
  (`manifest.py:43-50`). Each runs in declaration order with
  `mode="after"`. There is no validator that checks
  `declared_reward_bounds` against `max_episode_steps` (RT-002 F-007
  invariant), `difficulty_band` against the band's expected return
  range (RT-004 F-021), or `reference_solver` against the solver's
  applicability to the env (RT-004 F-023). A manifest can be
  valid by Pydantic's lights but exploit a cross-field gap that
  the validators were never structured to catch.
- Mechanism: The attacker submits a manifest with:
  - `difficulty_band=DifficultyBand.HARD` (`manifest.py:34`),
    threshold 0.1.
  - `declared_reward_bounds=RewardBounds(min_per_step=-1e9,
    max_per_step=1e9, min_per_episode=-1e9, max_per_episode=1e9)`
    (`types.py:21-33` only enforces max>min).
  - `max_episode_steps=1` (`manifest.py:36`, gt=0 only).
  - `anchor_trajectories=[TrajectoryAnchor(seed=0,
    action_policy=SEEDED_RANDOM, n_steps=1, expected_hash="...")
    , ...]` (4 anchors, all n_steps=1 -- closes the
    `_anchor_steps_must_fit_episode_budget` invariant).
  All field-level and model-level validators pass. The manifest is
  accepted. But the cross-field invariants are wide open: the
  per-episode bound is `[-1e9, +1e9]` regardless of step bound,
  the band threshold of 0.1 normalises any non-zero raw return to
  ~1.0 against the wide bounds, and `max_episode_steps=1` makes
  the env trivially solvable. The validator's outer driver runs
  the four checks against this manifest and finds all four pass.
- Why missed: `manifest.py:43-82` validates four invariants
  (PEP 440 version, JSON-serialisable kwargs, unique anchors,
  anchor steps fit budget). It does not validate any cross-field
  semantic invariant. RT-002 F-006 / F-007, RT-004 F-020 / F-021
  all describe a validator that should exist at this layer; none
  is implemented. The manifest validation order is correct for
  the invariants it does check; the gap is the missing invariants.
- Exploit cost: zero. Pure manifest authoring.
- Profit shape: this is the union of the per-check declaration-laziness
  attacks (F-006, F-007, F-020, F-021), framed as a single
  manifest object. The cross-cutting framing matters because the
  fix lives at the manifest layer (one place) rather than at the
  per-check layer (four places). MEDIUM because each component
  attack is already catalogued at HIGH or CRITICAL; the
  cross-cutting framing does not raise the severity but does
  identify the right fix locus.
- Fix sketch: add manifest-level model_validators for the missing
  cross-field invariants:
  (a) `min_per_episode >= min_per_step * max_episode_steps` and
      `max_per_episode <= max_per_step * max_episode_steps`
      (closes F-007).
  (b) `declared_reward_bounds` envelope per `difficulty_band`
      (closes F-006 and pieces of F-020).
  (c) Reserved-slot policy on `reference_solver`: in Phase 1
      reject any value other than the per-band default solver
      (closes F-023 today; the field is reserved for Phase 2).
  Manifest-layer fixes have a single locus and no per-check coupling;
  they are the cheapest place to harden cross-cutting invariants.

### A-407: protocol_version downgrade and silent acceptance

- Category: protocol versioning
- Severity: MEDIUM
- Premise: `EnvManifest.protocol_version: Literal["0.3.0"]`
  (`manifest.py:31`) is a strict literal. A v0.3.0 validator
  rejects manifests at v0.2.0 or v0.4.0. But the protocol
  evolution path itself is the attack surface: when a future
  validator adds Phase 2 checks (chain-beacon entropy, sandbox,
  band-envelope validation), it will likely bump to v0.4.0 and
  the v0.3.0 manifests submitted before the bump will need a
  policy decision: re-validate, grandfather, or reject. The
  default behaviour today (Literal["0.3.0"] only) means the
  validator code change determines re-validation cadence; the
  manifest itself does not declare a re-validation deadline.
- Mechanism: Two routes:
  - Route A (forward grandfather): a creator submits a v0.3.0
    manifest before the v0.4.0 upgrade. The v0.3.0 manifest
    passes all four Phase 1 checks (with whatever exploits the
    creator deployed). After the validator upgrades to v0.4.0
    with strengthened checks, the v0.3.0 manifest's `passed`
    status is consumed by the protocol; if the protocol does
    not re-run validation, the v0.3.0 manifest continues to
    earn at its declared band rate under the weaker Phase 1
    check semantics indefinitely.
  - Route B (forward downgrade): an attacker who has already
    over-fit against v0.3.0 strategically delays the
    `protocol_version` bump on their submission. If a future
    validator accepts both Literal["0.3.0"] and Literal["0.4.0"]
    (a likely transitional Literal['0.3.0', '0.4.0']
    declaration), the attacker submits at the lower version to
    invoke the weaker check semantics. The validator's strict
    Literal closes this today; a transitional declaration would
    open it.
- Why missed: today the strict Literal closes Route B. Route A is
  a protocol-orchestration concern that lives outside the check
  files. RT-005 raises it because the cross-cutting fix family
  (chain-beacon, sandbox, manifest envelope) implies a v0.4.0
  bump and the bump policy has not yet been documented. The
  attacker's window is the gap between the bump and forced
  re-validation of all live manifests.
- Exploit cost: zero (Route A) -- just submit early. Route B
  requires a future validator-side mistake.
- Profit shape: Route A is bounded by the time between Phase 1
  registration and Phase 2 forced re-validation; if the protocol
  never forces re-validation, the bound is the manifest's
  lifetime. MEDIUM because the exploit window is policy-dependent
  and the Phase 1 strict Literal is currently honest; the
  finding is forward-looking documentation that the Phase 2
  rollout plan needs an explicit re-validation cadence.
- Fix sketch: when protocol_version bumps, the protocol must
  force re-validation of all live manifests under the new
  semantics within a bounded window (e.g. 7 days). Add a
  `validated_at: datetime` and a `valid_until: datetime` to the
  manifest schema (or to the validator's per-manifest record) so
  re-validation deadlines are explicit. Document the bump
  policy in the protocol spec before the v0.4.0 cut.

### A-408: kwargs as a smuggling channel into the env-defining 4-tuple

- Category: protocol-layer composition
- Severity: HIGH
- Premise: `manifest.kwargs: dict[str, Any]` (`manifest.py:41`) is
  JSON-serialisable but otherwise unconstrained. It serves two
  roles simultaneously: it feeds `_seeds.py:81` (so it shifts the
  derived seeds for all five salts), AND it feeds
  `_load_env(spec).env_cls(**spec.kwargs)` (`_rollout.py:99`) at
  runtime. The attacker can use `kwargs` as a search axis for
  A-403 (the brute-force) AND as a runtime knob the env reads to
  branch its behaviour.
- Mechanism: The attacker structures `kwargs` as
  `{"grid_size": N, "_validator_token": "abc"}`. The env
  constructor reads `grid_size` (the legitimate use) and stashes
  `_validator_token` for later use during step. From the
  validator's perspective, `kwargs` is part of the env-defining
  4-tuple, so changing `_validator_token` shifts all five derived
  seed pools. The attacker's brute force (A-403) over
  `_validator_token` is unbounded -- it accepts any
  JSON-serialisable string. Once a token aligns the seeds, the
  env reads the same token at runtime and uses it as a
  manifest-private secret to authenticate its own state machine
  ("if my kwargs say `_validator_token` is 'abc', I am the
  attacker's blessed env; behave accordingly"). The kwargs become
  a shared secret between the manifest-author and the env-author
  that the validator cannot distinguish from legitimate
  configuration.
- Why missed: the manifest validator only checks JSON
  serializability (`manifest.py:53-58`); it does not check that
  every key in `kwargs` is a key the env constructor accepts
  (i.e. denies extras). The env constructor signature is fully
  creator-controlled, so it can accept any keyword and the
  validator cannot detect smuggled keys. The 4-tuple seed
  derivation hashes the entire `kwargs` mapping
  (`_seeds.py:81`), so any change to `_validator_token` shifts
  the seeds. There is no separation between "runtime
  configuration" and "manifest-public field".
- Exploit cost: zero. One extra `kwargs` key.
- Profit shape: ongoing. The kwargs-smuggling channel makes A-403
  practical at scale (the entropy budget for the brute force is
  the entropy of `_validator_token`, which is unbounded). And it
  gives the env a manifest-derivable runtime secret that the
  validator cannot easily strip. HIGH because the channel is the
  cheapest way to align the seed pools across all five salts and
  is structurally unaddressed.
- Fix sketch: at manifest-validation time, introspect the env
  class's `__init__` signature (after the env loads but before
  any check runs) and reject any `kwargs` key that is not in the
  accepted parameter list. This denies extras and collapses the
  attacker's entropy budget to the keys the env constructor
  legitimately needs. Cross-cuts with A-406 (manifest validation
  order). For kwargs values, a separate constraint should bound
  size and depth (e.g. JSON-serialised len < 1024 bytes, no
  nested dicts beyond depth 2) to prevent a creator from packing
  arbitrary entropy into a single legitimate key.

### A-409: Trust boundary disagreement between sub-agents (validator vs solver vs env-architect)

- Category: trust boundaries between sub-agents
- Severity: MEDIUM
- Premise: The Praxis codebase is composed by several role-distinct
  authors: validator-engineer (the four check files plus
  _rollout/_seeds), rl-researcher (the solver registry and
  TabularQLearning), env-architect (the protocol manifest, types,
  hashing, and the gridworld env). The interfaces between these
  roles are where assumptions can disagree silently. A
  cross-cutting attacker exploits the cracks where one sub-agent's
  assumption about the next sub-agent's behaviour is wrong.
- Mechanism: Three concrete instances:
  (1) The Solver protocol (`_protocol.py:21-41`) is
      `@runtime_checkable` and is shape-only (`train`, `evaluate`
      callables). RT-004 loose threads noted that any registered
      solver only needs to expose those two callables; nothing in
      the registry asserts purity, statelessness, or honest
      argmax. A future Phase 2 PPO solver that mutates global
      state during evaluate would not be rejected. The validator
      sub-agent assumes the solver sub-agent ships honest
      solvers; the registry does not enforce it.
  (2) `_load_env` casts `env_cls(**spec.kwargs)` to
      `gym.Env[Any, Any]` (`_rollout.py:60, 99-100`). The env-architect's
      gridworld is a legitimate gym.Env subclass; the validator
      trusts that any class returned by `getattr(module, class_name)`
      is also a legitimate gym.Env. A creator who registers a
      `class FakeEnv:` that exposes the right method names but
      not the right semantics passes the runtime callable check
      (`_rollout.py:94`) and does not get rejected. The
      validator's "is this a callable" check is weaker than the
      env-architect's "is this a gym.Env" assumption.
  (3) `manifest.declared_reward_bounds` is consumed by both
      `check_reward_bounds` (as the bound to enforce) AND
      `check_solver_baseline` (as the normalization divisor,
      `solver_baseline.py:243`). The two consumers expect
      different invariants: reward_bounds expects honest tight
      bounds (so any out-of-bounds reward fails); solver_baseline
      expects honest tight bounds (so normalization is
      meaningful). Neither consumer can detect when the bounds
      are loose, because the manifest does not require them to
      be tight. A creator who declares loose bounds defeats both
      consumers in different ways: reward_bounds becomes a
      no-op, solver_baseline normalizes to ~1.0 trivially. The
      cross-cutting issue is that two sub-agents (validator-engineer
      writing reward_bounds.py and validator-engineer writing
      solver_baseline.py) implicitly trust the env-architect's
      RewardBounds type to be tight, when in fact it only
      enforces max>min.
- Why missed: the protocol type `RewardBounds` (`types.py:21-33`)
  is the wrong place to enforce tightness (it does not know about
  difficulty bands). The check-side consumers (reward_bounds.py
  and solver_baseline.py) both implicitly trust the manifest. No
  layer enforces the cross-consumer invariant. The Solver
  protocol has the same issue: it does not enforce honest
  argmax. The trust boundary leaks where validator-side code
  assumes "the previous layer validated this".
- Exploit cost: zero -- exploits the assumption gap, not a
  specific bug.
- Profit shape: aggregates the per-check declaration-laziness
  findings (F-006, F-007, F-020, F-021, F-023) into a structural
  observation. MEDIUM because the individual exploits are already
  catalogued; the cross-cutting framing identifies that the fix
  family is "tighten interfaces between sub-agents", not "fix
  each sub-agent's local code".
- Fix sketch: enumerate the cross-sub-agent invariants explicitly
  in the protocol spec. For each, decide which sub-agent owns
  the invariant and where it is enforced. Examples:
  (a) "RewardBounds are tight against difficulty band" -> owned
      by manifest validation, enforced in `manifest.py` via a
      band-envelope model_validator (closes F-006 and pieces of
      F-020).
  (b) "Solver evaluate is pure" -> owned by Solver protocol,
      enforced via documentation + a registry-time test that
      runs evaluate twice and asserts identical EvalResult
      (closes the future-PPO leak path).
  (c) "Env class is a gym.Env subclass" -> owned by `_load_env`,
      enforced via `isinstance(env_cls, type) and issubclass(env_cls,
      gym.Env)` after instantiation (closes the FakeEnv shape-only
      bypass).

### A-410: Inconsistent strictness across checks engineerable into pass-everywhere-fail-cheapest

- Category: cross-check composition / orchestration
- Severity: LOW
- Premise: The four checks have heterogeneous strictness profiles.
  Determinism's hash-equality is bit-exact; reward_bounds is
  arithmetic-strict but has the truncation escape hatch (F-008);
  reset_correctness is structural and tolerates anything outside
  its seven-touch-per-seed pattern; solver_baseline is statistical
  and lower-bound-only. A creator who optimises the manifest
  could in principle aim to "barely fail the cheap-to-fail check
  and pass the rest" -- but the validator's pipeline today
  treats `passed` as a conjunction (all four must pass), so this
  attack surface is closed at the protocol level for now.
- Mechanism: If the future orchestration layer weights checks
  unequally (e.g. solver_baseline counts for 50% of the bond, the
  others 12.5% each) or treats `passed` as a soft signal, an
  attacker could engineer a manifest that fails one cheap check
  and passes the others, accepting a partial slash for full
  emission rate at the heaviest-weighted check. Today the bond
  policy is unspecified; the orchestration layer is ahead.
- Why missed: nothing in the current source says "any pass=False
  fails the manifest". That decision lives at the orchestration
  layer. The cross-cutting concern is that the four checks have
  HETEROGENEOUS strictness, so any pipeline policy other than
  strict conjunction creates a partial-pass attack surface.
- Exploit cost: zero today (no orchestration layer); nontrivial
  once orchestration lands (engineer the manifest to fail the
  least-weighted check).
- Profit shape: zero today. Nonzero only if orchestration layer
  treats `passed` non-conjunctively. LOW because the attack
  surface is forward-looking and orchestration design is the
  right place to close it (by mandating strict conjunction).
- Fix sketch: when the orchestration layer is designed, mandate
  strict conjunction of `passed` across all checks for emission
  weighting. Document this as an invariant in the protocol spec.
  If unequal weighting is desired for diagnostic purposes
  (e.g. surface `which_check_failed` to the creator), the bond
  policy must still be all-or-nothing.
- Resolution: CLOSED in Phase 1 fix-pass at commit ae5501b.
- Mechanism: validator pipeline orchestration ships with run-all + conjunctive aggregation by-design. ValidatorReport.passed equals all(o.passed for o in check_results.values()): every sub-check must have outcome.passed=True for the overall verdict to pass. Any failed or errored sub-check fails the whole verdict. The operator gets every failure mode in a single report (run-all semantics) but the verdict aggregates strictly. This was the F-040 forward-looking concern: "if orchestration treats passed non-conjunctively." It does not.
- Phase 1 invariant test: tests/orchestrator/test_runner_aggregation.py exercises the conjunctive rule explicitly via mock-injection. Five passing mocks pass overall; four-and-one configurations fail overall; all-failing configurations record five failure_summary entries; one mock raising RuntimeError is recorded as CheckErrored and counts as failure.
- Residual: none specific to F-040. Note that F-031 (cross-cutting compositional creator-miner: env that exploits multiple checks each individually passing) is a deeper architectural concern not addressed by orchestration aggregation; F-031 stays DEFERRED CRITICAL pending Phase 2 architectural work.

## Loose threads

- Stage 2 PoC priorities across all five RT passes. Per the
  conventions doc, Stage 2 will produce executable PoCs for at
  least the highest-severity findings. RT-005's CRITICAL findings
  (F-003 reassessed, F-031 compositional, F-032 sys.modules
  monkey-patch) should be Stage 2 priorities. F-021 (RT-004 band
  downshift, CRITICAL) is the cheapest one-line PoC and should
  be first. F-002 (canonical action) and F-022 (TabularQ argmax)
  share infrastructure and can be PoC'd jointly.
- Validator pipeline orchestration step. The four checks today are
  individually tested but not yet wired into a single per-manifest
  driver. RT-005's framing assumes the driver runs them
  sequentially in the same Python process; the moment a driver
  lands, it should run the four checks under subprocess isolation
  (closes F-003 and most of A-401/A-402/A-405) and snapshot
  `sys.modules` between calls (interim mitigation if subprocess
  isolation is delayed).
- Deferred protocol-layer concerns. RT-005 raised three concerns
  that live at the manifest / protocol layer rather than at the
  check layer: A-406 (manifest validation order), A-407
  (protocol_version cadence), A-408 (kwargs smuggling channel).
  All three are better fixed in `manifest.py` or in the protocol
  spec than at the check layer. The chain-beacon work for F-002 /
  F-003 should land alongside a manifest-layer hardening pass.
- The closed RT-001 findings (F-001, F-004, F-005) are not
  re-evaluated here because their resolution mechanisms are
  unchanged by RT-002..RT-004 context. F-001 (anchor cherry-picking)
  is closed by check_determinism_self_consistency at validator-derived
  seeds; F-004 (info side channels) is closed as a toggle via
  hash_infos; F-005 (anchor n_steps invariant) is closed by a
  manifest model_validator. RT-005 audited the closed findings
  for cross-cuts and found no new gaps.
- A-410 (inconsistent strictness) is forward-looking and depends
  on the orchestration layer's bond-weighting decision. RT-005
  flags it but cannot close it; the conventions doc should
  evolve to require RT-006 (or a future pass) to revisit
  A-410 once orchestration lands.
- The choice to update F-002 / F-003 in place (severity revision
  rows) rather than allocate new IDs preserves cross-RT
  traceability. The original RT-001 catalog rows reference the
  same F-NNN; the RT-005 findings index notes the revision and
  the new severity. New cross-cutting attacks get fresh IDs
  starting at F-031, continuing the repo-wide sequence from
  RT-004's F-030.

## Findings index

The cross-cutting catalog produces ten findings (F-031..F-040).
RT-005 also revises the severity of two RT-001 findings in place
(F-002 stays HIGH, F-003 escalates to CRITICAL); those rows are
included as severity-revision entries. New cross-cutting findings
get fresh IDs starting at F-031. Severity counts: 4 CRITICAL/HIGH
above MEDIUM (F-031 CRITICAL, F-032 CRITICAL, F-033 HIGH, F-034
HIGH, F-035 HIGH, F-038 HIGH) and 4 MEDIUM/LOW (F-036 MEDIUM,
F-037 MEDIUM, F-039 MEDIUM, F-040 LOW), plus the two RT-001
revisions.

| ID | Severity | Summary | Linked attack |
|----|----------|---------|---------------|
| F-002 | HIGH (severity revision) | Canonical SEEDED_RANDOM action sequence is fully public per seed; cross-cuts RT-002 F-011, RT-003 F-013, RT-004 F-022. Stays HIGH/DEFERRED; chain-beacon entropy now load-bearing for five salts. | RT-001 A-003, A-303, A-202, A-107 |
| F-003 | CRITICAL (severity revision) | importlib(entry_point) runs creator-controlled top-level code without sandbox; cross-cuts RT-003 F-019 (16x) and RT-004 F-029 (22x). Escalated from HIGH to CRITICAL; interim sys.modules-snapshot mitigation feasible in Phase 1, full subprocess isolation Phase 2. | RT-001 A-006, A-208, A-310 |
| F-031 | CRITICAL | A single env can pass every check while lying off-canonical by composing F-002, F-013, F-022 with band-side and bound-side findings; cross-cutting compositional attack. | A-401 |
| F-032 | CRITICAL | Cross-check sys.modules monkey-patch persistence: first _load_env can mutate sys.modules permanently, contaminating every subsequent check on every subsequent manifest in the same process. [CLOSED in commit 56b9b28] | A-402 |
| F-033 | HIGH | Joint manifest-tuple search across all five salts is linear in the number of salts (not multiplicative); cross-cutting brute-force cost analysis for the F-002 deferred class. | A-403 |
| F-034 | HIGH | Cumulative DOS against validator pool capacity: adversarially-slow env amortises latency across 58-82 _load_env calls per manifest with no per-check wall-clock cap. | A-404 |
| F-035 | HIGH | Module-level state survives across check invocations; CPython sys.modules cache lets envs build a state machine spanning all four checks per manifest. | A-405 |
| F-036 | MEDIUM | Manifest validation order leaves window for inconsistent state: no cross-field invariants tie reward bounds to step bounds, band to expected returns, or reference_solver to env applicability. | A-406 |
| F-037 | MEDIUM | protocol_version downgrade and silent acceptance: strict Literal closes today, but the bump-and-grandfather window is unspecified for Phase 2 cuts. | A-407 |
| F-038 | HIGH | kwargs as a smuggling channel: dict[str, Any] feeds both seed derivation and runtime configuration, giving the attacker an unbounded entropy axis for joint seed alignment plus a manifest-derivable runtime secret. | A-408 |
| F-039 | MEDIUM | Trust boundary disagreements between validator-engineer, rl-researcher, and env-architect sub-agents; aggregates per-check declaration-laziness findings into a structural observation. | A-409 |
| F-040 | LOW | Inconsistent strictness across checks is engineerable into pass-everywhere-fail-cheapest IF the orchestration layer treats passed non-conjunctively; forward-looking, currently closed by absence of orchestration layer. [CLOSED in commit ae5501b] | A-410 |
