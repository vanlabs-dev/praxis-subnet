# RT-001: Determinism check red-team pass

Status: Living document -- 3 of 5 findings closed ; F-003 severity reassessed CRITICAL by RT-005 ; F-003 partially closed by 56b9b28
Date: 2026-04-29 (initial), 2026-04-30 (last updated)
Scope: src/praxis/checks/determinism.py and src/praxis/checks/_rollout.py
Reviewed commits: 489a592..126857f

## Threat model

The adversary is a creator-miner who ships an env package source-tree to validators
and stakes a Bittensor-style bond. Their objective is to maximize
`expected_emissions - p_slash * collateral` by getting `check_determinism` to
return `passed=True` on a manifest that does not actually pin down a deterministic,
honest env.

What the attacker controls:

- The full source of the env module that `_load_env` will `import_module` (line 90 of
  `_rollout.py`) and instantiate.
- Every field of the manifest the validator is given: `env_id`, `env_version`,
  `entry_point`, `kwargs`, `max_episode_steps`, `declared_reward_bounds`, the
  full `anchor_trajectories` list (including `seed`, `n_steps`, `expected_hash`,
  `action_policy`), and `creator_metadata`. The manifest pydantic constraints in
  `manifest.py` (lines 9-11, 18, 23-31) bound shapes but not semantics.
- Compute sufficient to brute-force any input that feeds a 64-byte BLAKE2b digest
  with attacker-chosen labels, including the canonical actions of
  `SeededRandomPolicy` for any anchor seed (it is fully public, `_rollout.py`
  lines 125-163).

What the attacker cannot do:

- Modify validator code, the per-check salt strings (`b"reward_bounds"`,
  `b"reset_correctness"`), the regexes in `manifest.py`, or the
  `derive_validator_seeds` algorithm (`_seeds.py`).
- Run arbitrary code on the validator's host outside of what `import_module` of
  their entry-point module loads. (But that boundary is exactly an interesting
  attack surface; see A-006.)
- Influence anchor seeds chosen later by the validator on a per-validation-epoch
  basis, because the determinism check uses the seeds the creator declared in
  the manifest itself (`determinism.py` line 234, `anchor.seed`).

Net: the attacker has total control over the env package and total control over
which (seed, n_steps, action_policy) tuples the determinism check will exercise.
The check only verifies that the env behaves as the manifest claims on the seeds
the manifest itself nominated.

## Closed findings summary

Three of five findings have been closed. F-005 (no anchor n_steps invariant) is
fully closed by a manifest-time model_validator. F-001 (anchor cherry-picking)
is closed by a new check_determinism_self_consistency that runs at
validator-derived seeds; the previous "creator picks the entire test
distribution" gap is gone. F-004 (info side channel) is closed as a toggle:
DeterminismConfig.hash_infos defaults False (backward compatible) but flips
True for paranoid-mode audits. PoC adversarial envs covering F-001 and F-004
live in tests/checks/_adversarial_envs.py.

Two findings remain. F-002 (canonical SEEDED_RANDOM action sequence is fully
public per seed) and F-003 (importlib runs creator-controlled top-level code
without a sandbox) are both HIGH severity but require Phase 2 architectural
work: chain-beacon entropy for F-002 to make derived seeds unpredictable per
validation epoch, and process isolation / module allow-listing for F-003.
Both are deferred to a consolidated red-team pass after step 8 (solver
baseline) lands; RT-001 will be cross-referenced from that pass.
F-003 severity reassessed CRITICAL by RT-005 (commit 7ca1be1); the deferral remains in place pending Phase 2 subprocess isolation, with an interim sys.modules-snapshot mitigation proposed in RT-005 F-032's fix sketch.
F-003 partially closed by commit 56b9b28 (sys.modules guard); full fix remains DEFERRED to Phase 2 subprocess isolation per the seven residual gaps documented in F-003's Resolution subsection.

## Attack catalog

### A-001: Anchor cherry-picking (lying-on-non-anchors env)
- Category: determinism evasion
- Severity: HIGH
- Premise: The creator wants an env that passes `check_determinism` while behaving
  arbitrarily (non-deterministically, dishonestly, or simply differently) on every
  seed not in the anchor list.
- Mechanism: The env stores a hard-coded set of "blessed" seeds matching exactly
  the four-to-thirty-two anchor seeds the creator committed in
  `manifest.anchor_trajectories`. On `reset(seed=s)` the env checks whether `s`
  is in the blessed set. If yes, it runs deterministic, well-behaved transitions
  yielding the precomputed `expected_hash`. If no, it can do anything: emit
  rewards outside the declared bounds (defers to RT-002), return a fixed obs and
  ignore actions, sample non-deterministically from a clock-seeded RNG, etc.
- Why the validator misses it: `check_determinism` (`determinism.py` lines
  204-260) iterates `manifest.anchor_trajectories` and only ever calls `rollout`
  with `seed=anchor.seed` (line 234). It never tries an off-anchor seed and
  there is no "spot-check on a validator-chosen seed" step. The attacker
  literally chooses the entire test distribution.
- Exploit cost: trivial. A 30-line `if seed in {a, b, c, d}: honest else: lie`
  branch in the env's `reset` and `step` is enough. No brute-forcing needed.
- Profit shape: The env passes determinism. It is then handed to `reward_bounds`
  and `reset_correctness`, which use **derived** seeds (`_seeds.py` line 22),
  so the cherry-picked anchor list does not directly defeat those checks. The
  per-check exploit chain therefore needs A-003 / A-007 to compound this. But
  even standalone, A-001 means determinism asserts nothing about the env's
  behavior at any seed the validator might subsequently want to use for any
  off-line analysis (e.g. spot-checks, forensic replays after a fraud claim).
  It also means a creator who later detects a validator's seed pool can swap
  in a fresh env that lies on every other seed without breaking determinism.
- Fix sketch: include at least one validator-chosen seed in the determinism
  pass. Reuse `derive_validator_seeds(manifest, k, salt=b"determinism_spot")`
  with a small `k` (say 2-4), require the env to produce SOME consistent hash
  when re-run twice on each, and record those hashes in the report. The
  spot-check seeds should NOT be declared in the manifest; the validator
  computes them from the env-defining fields and re-runs the rollout twice,
  comparing computed-hash-A to computed-hash-B. That converts the check from
  "does the env match the creator's claim on creator-picked seeds?" to "is
  the env self-consistent on validator-picked seeds?" -- they are different
  guarantees.
- **Resolution**:
  - Status: CLOSED
  - Resolved by: 126857f (feat(checks): self-consistency determinism + hash_infos config)
  - Mechanism: check_determinism_self_consistency runs the env twice at
    validator-derived seeds (salt=b"determinism_self_consistency") and
    asserts trajectory hash equality. Anchor cherry-picking no longer
    suffices since the validator now exercises seeds the creator did
    not declare.
  - Residual gap: F-002 still applies. Validator seeds are derivable from
    the manifest; a creator can in principle brute-force a manifest whose
    derived seeds plus the public canonical action sequence at each are
    all lying-friendly. Phase 2 chain-beacon fix.

### A-002: Walltime / wallclock side-channel non-determinism
- Category: hidden non-determinism the trajectory hash misses
- Severity: MEDIUM
- Premise: The creator wants an env that emits trajectories which are
  bit-identical on replay (so determinism passes) but whose internal state has
  walltime-dependent components that influence other checks or downstream
  scoring.
- Mechanism: `trajectory_hash` (`hashing.py` lines 27-53) folds only
  `observations`, `actions`, `rewards`, `terminations`, `truncations` and
  optionally `infos`. `infos` are explicitly dropped by default and are dropped
  entirely from the determinism rollout (`determinism.py` line 138 passes
  `infos=[]`, line 139 `include_infos=False`). The env can therefore stash
  walltime, `os.getpid()`, `socket.gethostname()`, `time.monotonic_ns()`, the
  current `tracemalloc` snapshot, or any global-state read into `info` dicts.
  Determinism will not see it. If a future validator extension starts
  consuming `info` (e.g. for fraud audit), those audits become invalid.
  Furthermore, the env can use walltime to *choose* its honest-vs-dishonest
  branch in A-001 (e.g. only lie during certain hours, masking the cherry-pick).
- Why the validator misses it: `determinism.py` line 138, `infos=[]`. The
  trajectory hash structurally cannot detect anything the env writes only to
  `info`. Note that the determinism check IS robust against pure
  `info`-observable non-determinism precisely because of this; the severity
  is in how it interacts with audits and with future protocol versions that
  might re-include `info`.
- Exploit cost: trivial. A `time.time()` call in `info` is two lines.
- Profit shape: standalone, this only weakens forensic capability. Compounded
  with A-001, it lets the creator gate cherry-picking on time of day or on a
  message they read from a side-channel without leaving determinism-detectable
  traces. MEDIUM because it does not by itself extract reward but materially
  weakens the audit guarantee.
- Fix sketch: hash a normalized info dict (with allow-listed deterministic
  keys) when `include_infos=True`, and run determinism with `include_infos=True`
  by default after publishing a list of allowed keys. Alternatively, document
  that info is forensically untrusted and require all reward-bearing claims to
  be observable through obs/reward/term/trunc only.
- **Resolution**:
  - Status: CLOSED (as toggle)
  - Resolved by: 126857f
  - Mechanism: DeterminismConfig grows hash_infos: bool = False. When True,
    info dicts from each step are folded into the trajectory hash via
    protocol.hashing.trajectory_hash include_infos parameter. Both
    check_determinism (anchor) and check_determinism_self_consistency
    respect the flag.
  - Residual gap: paranoid mode is opt-in; default validator runs with
    hash_infos=False. Documented as a design choice. Envs that legitimately
    use info for non-essential channels would fail under hash_infos=True,
    so the protocol cannot make True the default without a separate spec
    requiring deterministic info dicts.

### A-003: Anchor-policy lock-in (env honest only on the canonical action sequence)
- Category: action policy
- Severity: HIGH
- Premise: The creator wants an env whose anchor passes determinism, but which
  cheats during reward-bounds sampling or any future check that uses a
  different action sequence at the same seed.
- Mechanism: `SeededRandomPolicy.actions` (`_rollout.py` lines 135-163) is
  fully deterministic and public:
  `np.random.Generator(np.random.PCG64(seed)).integers(0, n_actions, size=n_steps, dtype=np.int64)`.
  The creator can precompute, for every anchor seed `s` in their manifest,
  the exact action sequence `A(s)` of length `n_steps` that the validator will
  apply during `check_determinism`. The env memoizes `A(s)` for each anchor
  seed and behaves honestly only when the i-th action received during a rollout
  starting from `reset(seed=s)` matches `A(s)[i]`. For any deviation, it
  branches into a lying transition. Because reward_bounds also uses
  `SEEDED_RANDOM` by default (`reward_bounds.py` line 54) but with derived
  seeds (different from anchor seeds, by construction), the env will encounter
  unfamiliar (seed, action) pairs in reward_bounds and is free to misbehave.
- Why the validator misses it: there is no "what if a different policy were
  applied" step. The check fixes a single (seed, action_policy, n_steps) triple
  per anchor and replays exactly that. There is no per-anchor jitter, no
  alternate-policy cross-check, no obs-only sanity test. `rollout`
  (`determinism.py` lines 65-147) takes the policy as a fixed input.
- Exploit cost: precomputing the canonical action sequence for each anchor seed
  is O(n_steps * n_anchors) PCG64 calls -- microseconds. Embedding the table in
  the env is trivial.
- Profit shape: cleanly compounds with A-001. The env passes determinism, and
  on **every** off-anchor seed (including the derived seeds reward_bounds and
  reset_correctness will use), it can lie freely. Reward-bounds violations
  could include emitting +0.99 step rewards when the declared max is +0.99
  but cumulative episode rewards above the declared per-episode max -- which
  is exactly what RT-002 will explore. From determinism's perspective: the
  cheap, working version of A-001 reduces to A-003.
- Fix sketch: in addition to the canonical `SEEDED_RANDOM` policy, run the env
  on each anchor seed under a **second** policy (e.g. `SEEDED_RANDOM_OFFSET`
  using `seed XOR known_constant`) and at minimum verify the trajectory has
  internally consistent shape (obs in space, rewards in declared bounds,
  terminations consistent). Determinism cannot get a hash match here -- the
  trajectory is unknown to the manifest -- but the env must not crash or emit
  out-of-space obs. That alone defeats the cheapest forms of A-003.
- **Resolution**:
  - Status: DEFERRED (Phase 2)
  - Reason: defeating canonical-action lookup requires unpredictable seeds
    per validation epoch (chain-beacon entropy or commit-reveal), which is
    architectural work beyond Phase 1 scope.
  - Re-evaluation: consolidated red-team pass scheduled after step 8
    (solver baseline) lands. RT-001 cross-referenced from that pass.

### A-004: Float repr platform-sensitivity (cross-validator hash divergence)
- Category: hash forgery / float repr canonical_bytes attack
- Severity: LOW
- Premise: An attacker wants two honest validators on different platforms to
  compute different hashes for the same trajectory, OR wants `repr(float)` to
  be ambiguous enough to enable second-preimage collisions.
- Mechanism: `canonical_bytes` (`hashing.py` line 69) calls `repr(obj)` for
  `float`. CPython's `repr(float)` since 3.1 implements David Gay's shortest-
  round-trip algorithm and has been stable across CPython 3.1+ on IEEE-754
  platforms. Special values: `repr(float('nan')) == 'nan'`, `repr(float('inf')) ==
  'inf'`, `repr(-0.0) == '-0.0'` (distinguishable from `'0.0'`), subnormals
  round-trip correctly. So in practice on every modern CPython on every modern
  CPU, two repr's of the same `float` produce the same string.
- Why the validator catches it (or doesn't have to defend): there is no
  practical exploit here on CPython 3.10+ on x86-64/ARM64 with IEEE-754. The
  only escape hatches are non-CPython interpreters (PyPy historically diverged
  on subnormals; not relevant if the protocol mandates CPython) and platforms
  without IEEE-754 (none in the deployment target). The hash function uses
  blake2b-256 (`hashing.py` line 24), no truncation, no weak primitive.
- Exploit cost: requires controlling validator runtime, which the threat model
  forbids.
- Profit shape: none in the current threat model. Listed for completeness so
  future passes do not waste cycles.
- Fix sketch: not needed today. If non-CPython runtimes ever become permitted,
  switch float serialization to a fixed-precision IEEE-754 hex form
  (`float.hex()`) which is bit-exact and platform-independent.

### A-005: Anchor-seed brute force into a friendly action sequence
- Category: seed prediction (anchor-side, NOT derived-seed-side)
- Severity: LOW
- Premise: Distinct from A-003: rather than the env memoizing the canonical
  action sequence, the attacker brute-forces the choice of anchor seed itself
  so that the canonical action sequence happens to be one their env handles
  honestly while the env stays "simple" (no big lookup table).
- Mechanism: For each candidate seed `s` the attacker computes
  `SeededRandomPolicy.actions(s, n_steps, Discrete(n_actions))`. If the
  resulting sequence happens to satisfy some property the env can recognize
  cheaply (e.g. "first action is RIGHT", "no two consecutive UPs",
  "action histogram is uniform within X%"), the env serves an honest path;
  otherwise refuses. Pick four seeds that all satisfy the property; declare
  them as anchors.
- Why the validator misses it: same as A-001 -- the validator does not pick
  seeds. The manifest constraint is `min_length=4, max_length=32` and uniqueness
  on `(seed, action_policy)` (`manifest.py` lines 28, 50-61). No constraint on
  seed range, distribution, or "validator-controlled". The brute-forcer can
  search 2^32 candidate seeds in seconds.
- Exploit cost: cheaper than A-003 (no per-seed lookup table) but strictly
  weaker -- the env still has to pretend honestly on the matching seeds. In
  practice A-003 dominates: brute-forcing seeds buys nothing the canonical-
  action-table approach does not already give for free.
- Profit shape: same as A-003.
- Fix sketch: same as A-001 -- add a validator-chosen spot-check seed.

### A-006: Import-time side effects in the entry-point module
- Category: infrastructure / importlib
- Severity: CRITICAL
- Premise: The validator runs `import_module(module_path)` (`_rollout.py` line
  90) where `module_path` is taken from a creator-controlled manifest. Whatever
  side effects fire during import will execute on the validator's host.
- Mechanism: `manifest.py` line 10 enforces
  `ENTRY_POINT_PATTERN = r"^[\w\.]+:[\w]+$"`. `\w` in Python regex is
  `[A-Za-z0-9_]`. So `module_path` is restricted to dot-separated identifiers,
  and `entry_point` cannot contain shell metacharacters. But `import_module`
  obeys `sys.path`: if the creator's package is on `sys.path` (which it must
  be, otherwise the validator could not load it at all), `import_module("foo")`
  runs `foo/__init__.py` top-level code, plus any sibling import. That code
  can:
  - Call `os.environ.update`, polluting environment variables read by other
    validator subsystems (logging, telemetry, downstream RPC).
  - Inject sys.modules entries that shadow other modules. E.g. `sys.modules
    ["numpy.random"] = forged_module` so subsequent
    `np.random.Generator(np.random.PCG64(seed))` calls (used by
    `SeededRandomPolicy`, `_rollout.py` line 162) return attacker-controlled
    values.
  - Monkey-patch `praxis.protocol.hashing.canonical_bytes` to return a
    constant, making every trajectory hash identical and trivially matching
    any `expected_hash`.
  - Spawn a thread that hooks `gymnasium.wrappers.TimeLimit.step` to lie about
    truncation.
  - Read `~/.bittensor/wallets/*` and exfiltrate to an HTTP endpoint.
- Why the validator misses it: there is no sandbox, no `sys.modules` snapshot,
  no `importlib.util.spec_from_file_location` with isolated namespace, no
  subprocess boundary, no allow-list of permitted module paths. `_load_env`
  validates only the regex shape (line 85-88) and the callability of the final
  attribute (line 92-96).
- Exploit cost: depends on how easy it is to get the creator's package onto
  the validator's `sys.path`. If the protocol assumes the validator pip-
  installs creator-published packages (common pattern), this is trivial: write
  side effects in `__init__.py`. If the validator instead clones a pinned git
  ref into an isolated venv, harder but still in scope (the venv shares
  Python and OS env with the validator process).
- Profit shape: catastrophic. Once you have monkey-patched
  `canonical_bytes` or `np.random.Generator`, every check including
  determinism passes against any expected_hash in the manifest. CRITICAL
  vector if the protocol does not isolate creator code; HIGH while the
  isolation model is undocumented.
- Fix sketch: load creator envs in a subprocess with a clean Python interp,
  pre-loaded `praxis.protocol.*` and `numpy` modules, an allow-list of
  importable modules. Communicate via a structured RPC (e.g. msgpack over
  pipe) so the trajectory data crossing the boundary cannot itself execute
  code. Snapshot `sys.modules` keys before import and assert no praxis or
  numpy entry was overwritten when the env is closed. Until that is in place,
  document explicitly that env code is trusted as much as validator code.
- **Resolution**:
  - Status: DEFERRED (Phase 2)
  - Reason: defeating importlib side effects requires process isolation
    or a module allow-list with a sandbox boundary; substantial
    infrastructure work outside Phase 1's scope.
  - Re-evaluation: consolidated red-team pass scheduled after step 8
    (solver baseline) lands. RT-001 cross-referenced from that pass.

  Severity reassessed CRITICAL on 2026-04-30 per RT-005 cross-cutting analysis. Drivers: 58-82 _load_env calls per manifest evaluation (RT-003 F-019 documents 16x amplification on reset_correctness, RT-004 F-029 documents 22x on solver_baseline); sys.modules contamination persists across creator-miners (RT-005 F-032); a single monkey-patch breaks every guarantee the protocol composes on top of (RT-005 F-003 re-evaluation). Original RT-001 entry preserved as historical record. See docs/red-team/RT-005-cross-cutting.md for the full reassessment, including the proposed interim Phase 1 sys.modules-snapshot mitigation that hedges between now and full subprocess isolation.

  Partial closure landed in commit 56b9b28 (Phase 1 fix-pass). A nose-style sys.modules snapshot guard wraps _load_env's import_module + getattr + env_cls(**kwargs) sequence (see RT-005 F-032 closure for the implementation detail). What this closes: the cross-creator sys.modules contamination vector (F-032 in full). What this does NOT close, severity stays CRITICAL: (1) C-extension state mutations (numpy global state, threading state, GIL-released code); (2) monkey-patches of already-loaded modules (e.g. creator's `import numpy; numpy.array = malicious_func` mutates the existing numpy object; sys.modules still points at the same module); (3) filesystem or network side effects from creator's import-time code; (4) side effects inside the env constructor itself other than sys.modules mutations (e.g. os.environ writes); (5) step/close-time imports (the guard wraps import_module + class lookup + instantiation only; lazy imports inside env.step() or env.close() happen AFTER the guard exits and land permanently in sys.modules); (6) sys.path mutations (not snapshotted); (7) gym env registry mutations (registry is a dict on the gym module, mutated in place; benign for Praxis because DR-001 forbids gym.make(), but unprotected in general). Phase 2 subprocess isolation remains the proper full fix; F-003 stays DEFERRED CRITICAL pending that work.

### A-007: TimeLimit / internal-step-counter disagreement
- Category: validator logic bypass / TimeLimit
- Severity: MEDIUM
- Premise: The env has its own internal step counter (gridworld does --
  `gridworld.py` line 78, `_max_episode_steps`, and line 154 emits
  `truncated = (not terminated) and (self._steps >= self._max_episode_steps)`).
  `_load_env` always wraps with `TimeLimit(env, max_episode_steps=spec.max_episode_steps)`
  (`_rollout.py` line 98), and `EnvSpec.max_episode_steps` (line 55) defaults
  to 0. The attacker exploits the two counters disagreeing.
- Mechanism: Two complementary cases.
  Case A (env truncates first): the creator sets the gridworld constructor
  arg so its internal `_max_episode_steps` is, say, 50, but
  `manifest.max_episode_steps = 10000`. The TimeLimit wrapper allows 10000
  steps, but at step 50 the inner env emits `truncated=True` (line 154) which
  bubbles up through the wrapper. From the validator's perspective the rollout
  ends at step 50 with a truncated episode -- it has no way to distinguish
  "env truncated itself at 50" from "TimeLimit truncated at 50". This is the
  honest gridworld behavior so it is not by itself a bug, but it means the
  env author can choose any internal cap independent of the manifest's claim.
  Per-anchor `n_steps` (`manifest.py` line 17, `gt=0`) is also unbounded above
  except by manifest declaration.
  Case B (manifest declares `max_episode_steps=1`, anchor declares
  `n_steps=200`): `manifest.py` line 26 enforces `gt=0` only. Pydantic does
  NOT enforce that `anchor.n_steps <= manifest.max_episode_steps`. So a
  manifest can declare `max_episode_steps=1` and an anchor with `n_steps=200`.
  TimeLimit will truncate at step 1; the rollout records exactly 1 step plus
  obs0; the trajectory hash is computed over a 1-step trajectory; the
  expected_hash in the manifest matches that 1-step trajectory. The validator
  reports `passed=True` despite the manifest's claim that the env can run
  for 200 steps from this seed. Downstream consumers reading the manifest
  may infer wrong things about the env's capability.
- Why the validator misses it: `manifest.py` validators (lines 33-61) check
  PEP 440 on `env_version`, JSON-serializability on `kwargs`, and uniqueness
  on `(seed, action_policy)` -- but no relation between `anchor.n_steps` and
  `max_episode_steps`. `rollout` (`determinism.py` lines 117-130) iterates
  the step records and silently breaks on `terminated or truncated`; it
  never compares `actual_steps` to the requested `n_steps` and never warns
  when they differ. The `RolloutResult` does carry `terminated_early` and
  `truncated_early` (lines 60-63) but `check_determinism` does not act on
  them, only stores them in `AnchorResult`.
- Exploit cost: trivial. One field of the manifest and one constructor kwarg.
- Profit shape: the attacker can ship a manifest declaring "I run 1000-step
  episodes" while the env actually truncates at 1 step. Determinism passes.
  Reward bounds at 1 step is much easier to satisfy. Reset correctness
  unaffected. So the determinism pass becomes a cover for capability
  misrepresentation.
- Fix sketch: at manifest validation time, enforce
  `all(a.n_steps <= max_episode_steps for a in anchor_trajectories)`. At
  determinism-check time, fail the anchor (or attach a warning to the
  report) if `actual_steps < requested_n_steps and not terminated_early` --
  i.e. if truncation was the only reason the rollout did not reach
  `n_steps`. That is a different check ("n_steps requested == n_steps
  produced unless natural termination") but is cheap and informative.
- **Resolution**:
  - Status: CLOSED
  - Resolved by: 14f9886 (fix(protocol): enforce anchor.n_steps <= max_episode_steps invariant)
  - Mechanism: New _anchor_steps_must_fit_episode_budget model_validator
    on EnvManifest. Manifests where any anchor's n_steps exceeds the
    manifest's max_episode_steps now fail Pydantic validation at parse
    time, before the validator ever loads the env.
  - Residual gap: none. The case-A "env truncates first because its
    internal counter is shorter than the TimeLimit budget" is intentional
    flexibility (the env is allowed to truncate inside the wrapper) and
    is not a forgery vector by itself; it surfaces only as a behavior
    consistency question for downstream checks.

### A-008: Numpy obs-array dtype canonicalization gap
- Category: float repr / canonical_bytes attack
- Severity: LOW
- Premise: An env returns observations whose dtype is changed across two
  honest replays in a way the canonical encoding hashes differently.
- Mechanism: `_normalize` (`hashing.py` line 56-78) handles `np.ndarray` by
  emitting `{"__np__": True, "dtype": str(obj.dtype), "shape": ..., "data_b64":
  base64(obj.tobytes())}`. If the env's observation is sometimes
  `np.array([0,0], dtype=np.int32)` and sometimes `np.array([0,0], dtype=np.int64)`
  (different dtype, same logical value), the dtype string and the byte width
  of `tobytes()` both change, producing different hashes. Gridworld is consistent
  (`gridworld.py` line 160 always returns int32) so this is not exploitable
  against gridworld today. But a custom env that, for example, branches on
  `numpy.show_config()` or on `os.cpu_count()` to choose a dtype could exhibit
  per-host hash drift while looking honest by inspection.
- Why the validator misses it: there is no dtype-canonicalization step. The
  policy says floats use `repr` for precision, but no equivalent normalization
  for int-like ndarrays. The validator trusts the env to produce obs in a
  stable dtype.
- Exploit cost: low if the env author wants it; impossible if the env author
  is honest.
- Profit shape: not directly profitable. It enables hash-divergence between
  validators -- a creator could ship a manifest whose anchor expected_hash
  matches what env produces on x86-64 but not on ARM64, which would split
  consensus among validators on different platforms. Profit only if the
  protocol penalizes hash mismatch from one side and rewards from the other.
- Fix sketch: `_normalize` for ndarray could canonicalize integer dtypes to
  the smallest signed dtype that fits the values (or always int64 for
  compactness), then re-encode. Or the protocol could require obs to declare
  a fixed dtype in the manifest and refuse otherwise. Either is overkill for
  Phase 1 given the LOW severity; document instead.

### A-009: NaN / inf in rewards (silent equality semantics)
- Category: float repr canonical_bytes
- Severity: LOW
- Premise: An env emits a reward of `float('nan')` or `float('inf')` at some
  step. Determinism check needs to be self-consistent, not honest, so two
  rollouts emitting NaN at the same step should produce the same hash.
- Mechanism: `repr(float('nan')) == 'nan'`, `repr(float('inf')) == 'inf'`,
  `repr(float('-inf')) == '-inf'`, all stable. So `canonical_bytes` produces
  a stable encoding and `trajectory_hash` is stable across replays. NaN is
  not specially handled; equality comparison `nan == nan` is False but the
  hash is over byte content, not equality. The check thus accepts NaN
  rewards as a valid deterministic trajectory. Reward-bounds is the
  appropriate place to reject them; determinism does not and arguably should
  not.
- Why the validator catches it (or doesn't have to defend): determinism
  treats NaN like any other float and produces a deterministic hash for
  trajectories containing it. RT-002 is responsible for catching NaN as an
  out-of-bounds reward.
- Exploit cost: trivial.
- Profit shape: zero against determinism. Cross-cutting concern flagged for
  RT-002 instead.
- Fix sketch: not a determinism-layer fix. Reward-bounds should explicitly
  reject `math.isnan(r) or math.isinf(r)` before comparing against
  declared bounds.

### A-010: Anchor-list explosion (32-anchor brute-force budget)
- Category: seed prediction (combined with A-001/A-005)
- Severity: LOW
- Premise: The manifest allows 4 to 32 anchors (`manifest.py` line 28). At 32
  the creator gets 32 chances to pre-compute and publish a "this seed is
  blessed" set. Combined with A-001, that means the lying-on-non-anchors env
  has up to 32 honest seeds to declare.
- Mechanism: max-length is a parameter, not a structural defense. The
  `_anchors_must_be_unique` validator (lines 50-61) deduplicates on
  `(seed, action_policy)` but only `SEEDED_RANDOM` is currently registered
  (`_rollout.py` line 167), so effectively 32 unique seeds.
- Why the validator misses it: no distributional check on the seeds. Adversary
  could declare 32 sequential integers `0..31` and the manifest validates
  fine. There is no "seeds must look uniform in [0, 2^32)" or "seeds must be
  derived from a chain beacon" rule.
- Exploit cost: zero.
- Profit shape: linearly grows with the number of anchors; same shape as A-001.
- Fix sketch: tighten anchor seeds: require them to be a deterministic function
  of the manifest hash (so the creator cannot pick them) OR require some of
  them be validator-derived per A-001. The current 4-32 range exists to give
  the creator hash-budget, not to give the attacker friendly-seed budget.

## Loose threads (cross-cutting)

- `derive_validator_seeds` (`_seeds.py`) hashes only
  `env_id, env_version, entry_point, kwargs, salt, block_idx`. A creator who
  iterates `kwargs` (e.g. a single `{"grid_size": int}` field) has a small
  brute-force surface: one int, ~10^4 plausible values. That is small enough
  for cheap brute force to find a kwargs choice whose derived seeds happen to
  align with the creator's lying set. RT-002 should compute the actual cost
  and severity.
- `reward_bounds` declares `per_episode_unverified=True` (`reward_bounds.py`
  line 314-315) when no episode terminated naturally. For a gridworld
  Easy-band env with random walks, the probability that a 25-step random walk
  on a 5x5 grid reaches (4,4) starting from (0,0) is non-trivial. With 8 seeds
  of `SEEDED_RANDOM` actions and `max_episode_steps=100`, what fraction of
  manifests trip `per_episode_unverified=True`? If most do, the per-episode
  bound is effectively unenforced in production. RT-002 should measure.
- `_obs_equal` in `reset_correctness.py` (lines 127-140) falls back to `==`
  for non-ndarray. If the env returns a Python scalar `0` from one reset and
  a `np.int32(0)` from another, `0 == np.int32(0)` is True (numpy scalars
  compare equal to Python ints). But a `dict` obs with one key as bytes vs
  str would compare unequal silently. RT-003 should investigate dict-obs
  envs.
- `iter_rollout` (`_rollout.py` line 264-281) uses a generator with a
  `try/finally: env.close()`. If the consumer (e.g. `rollout` in
  `determinism.py`) raises before exhausting the iterator, the generator's
  `finally` block does not run until garbage collection. On long-running
  validator processes this is a slow leak rather than a correctness bug,
  but if the env's `close()` releases an external resource (file lock,
  network socket), it could matter. Phase-2 / infra concern.
- `manifest.py` line 31, `kwargs: dict[str, Any]`. The `Any` typing means
  pydantic does not deeply validate kwargs values. JSON-serializable is the
  only constraint (line 42-48). A nested dict with float keys, very large
  ints, or pathological structures could pass. `canonical_bytes` handles all
  these but the env constructor might explode. Reset_correctness would catch
  the explosion as `RESET_CRASHED`, but the validator cycles a lot of
  resources before reaching that verdict. Low severity, infra observation.
- The env `close()` (`gridworld.py` line 166) is a no-op, but for envs that
  hold GPU memory or open files, `_load_env` instantiates the env BEFORE the
  `TimeLimit` wrapper succeeds. If `TimeLimit(env, max_episode_steps=0)` ever
  raised (it does not today, but Gymnasium semantics could change), the env
  would leak. Very low severity but a constructor-vs-wrapper boundary worth
  noting.

## Findings index

Five findings carry HIGH or MEDIUM severity. A-001 (anchor cherry-picking),
A-003 (canonical-action lookup table), and A-006 (importlib side effects)
are HIGH and effectively chain together: A-006 alone trivially defeats every
check, A-001+A-003 define the cheapest route to passing determinism while
shipping a dishonest env. A-002 (info side channels) and A-007 (TimeLimit /
internal counter disagreement) are MEDIUM. The remaining attacks are LOW
because they either require capabilities outside the threat model (A-004) or
collapse onto a HIGH attack (A-005, A-010) or are out-of-scope-for-determinism
concerns (A-008, A-009).

| F-NNN | severity | status | one-line summary | resolving commit |
|---|---|---|---|---|
| F-002 | HIGH | DEFERRED | Canonical SEEDED_RANDOM action sequence fully public per seed; env can lie on every off-canonical (seed, action) pair. | DEFERRED |
| F-003 | CRITICAL | DEFERRED | importlib(entry_point) runs creator-controlled top-level code without a sandbox. [RT-005 reassessment] [partial closure 56b9b28; full fix Phase 2] | DEFERRED |
| F-001 | HIGH | CLOSED | Validator only tested creator-declared seeds; anchor cherry-picking trivially passes determinism. | 126857f |
| F-004 | MEDIUM | CLOSED (toggle) | Infos excluded from trajectory hash by default; walltime/pid leak invisibly through info dicts. | 126857f |
| F-005 | MEDIUM | CLOSED | No anchor n_steps invariant; manifests can declare unfittable anchors. | 14f9886 |
