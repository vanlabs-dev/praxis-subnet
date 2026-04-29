# RT-003: Reset correctness check red-team pass

Status: Draft
Date: 2026-04-30
Scope: src/praxis/checks/reset_correctness.py and the shared primitives it consumes (src/praxis/checks/_rollout.py, src/praxis/checks/_seeds.py)
Reviewed commits: full chain through 2db3dfa, special focus on 2d62d3c

## Threat model

The adversary is the same creator-miner profile as RT-001 and RT-002: they
ship an env package source-tree to validators, stake a Bittensor-style bond,
and want to maximize `expected_emissions - p_slash * collateral`. Their
RT-003-specific goal is to ship a manifest such that
`check_reset_correctness` returns `passed=True` while the env in production
fails one or more reset-correctness invariants the validator was supposed to
guarantee.

What the attacker controls (reset_correctness-specific):

- The full source of the env class. `_load_env` (`_rollout.py:60-100`)
  imports a creator-named module and instantiates a creator-named class.
  Inside `reset`, `step`, `__init__`, and any class-level descriptor
  (`observation_space`, `action_space`, `metadata`), the attacker has
  arbitrary Python.
- The declared `observation_space` and `action_space`. `check_reset_correctness`
  trusts `env.observation_space.contains(obs)` (`reset_correctness.py:253`)
  and feeds `env.action_space` straight into the canonical policy
  (`reset_correctness.py:363`). Both spaces can be redeclared per-instance,
  per-call, or even per-attribute-access (via `@property` returning a fresh
  `Box` each time).
- All seed-derivation inputs that feed `derive_validator_seeds` for
  `salt=b"reset_correctness"` (`_seeds.py:74-85`): `env_id`, `env_version`,
  `entry_point`, and `canonical_bytes(kwargs)`. Same shape as the RT-002
  surface; the salt is the only differentiator. Anything an A-106 search
  could find for `b"reward_bounds"`, the equivalent search finds for
  `b"reset_correctness"` at the same cost.
- `manifest.max_episode_steps`. Beyond shaping the TimeLimit budget, this
  is also implicitly an upper bound on what the env can do during the
  mid-episode setup loop (which only takes
  `cfg.mid_episode_steps` steps, default 5).

What the attacker cannot do that matters here:

- Modify the salt `b"reset_correctness"` or the per-check
  `mid_episode_steps` constant baked into a validator's default config.
- Predict whether a given validator will run with `cfg.mid_episode_steps=0`
  (which short-circuits the entire mid-episode arm). Default is 5. A
  paranoid validator could raise it; the attacker must assume a value
  but cannot know it. In practice every Phase 1 default is 5.
- Influence the order in which seeds are processed inside the check
  (`reset_correctness.py:206`). They are iterated in
  `derive_validator_seeds` output order.
- Force the validator to skip `_safe_close` (`reset_correctness.py:143-147`);
  but `_safe_close` swallows any exception its close raises, which already
  helps the attacker, not the validator (see A-205).

Net: the check is structurally narrower than determinism and reward_bounds.
It runs eight rollouts, each with at most `5 + 2 = 7` env interactions
(`reset`, `reset`, then up to five `step`, then a final `reset`). The
attacker has wide latitude to ship an env that is honest on this
seven-touch-per-seed pattern and arbitrary everywhere else, including on
the very next env operation a downstream consumer would attempt.

## Attack catalog

### A-201: Adversarial observation_space (vacuous-contains pass)
- Category: declared-space laziness
- Severity: HIGH
- Premise: A creator declares `observation_space = gym.spaces.Box(-inf, +inf,
  shape=..., dtype=np.float64)` (or any space whose `contains` returns True
  for every reachable obs). The OBS_IN_SPACE arm is then a no-op; the env
  can return literally any obs at reset time and pass.
- Mechanism: `reset_correctness.py:253` calls
  `env.observation_space.contains(obs_a)` and stores the result. The
  `contains` method of `gymnasium.spaces.Box` returns True iff
  `low <= obs <= high` elementwise and `obs.shape == self.shape`. With
  `low=-np.inf, high=+np.inf`, every finite array of the declared shape
  is in-space; with float64 dtype, NaN is also in-space (because
  `np.nan >= -np.inf` is False and `np.nan <= +np.inf` is False, so
  Box.contains rejects NaN). But the attacker does not need NaN. With a
  vacuous Box the env can return per-seed obs that look sane to
  `contains` while encoding side-channel data (e.g. seed parity, walltime
  mod K) for the env to recognize during step. More directly: a
  `gym.spaces.Discrete(2**31 - 1)` declared space accepts every non-negative
  int32 value as an obs, which is functionally indistinguishable from "no
  type check at all".
- Why the validator misses it: there is no manifest-time invariant on the
  declared space. `EnvManifest` (`manifest.py:21-90`) does not constrain
  `observation_space` at all - the space is only known after env
  instantiation (`_load_env` returns the env, but `check_reset_correctness`
  never inspects shape or volume). The check trusts the env's own claim
  about what is in-space, then asks the env whether the obs satisfies
  that claim. Both sides are creator-controlled.
- Exploit cost: zero. Two extra constructor lines compared to a legitimate
  declaration.
- Profit shape: ongoing. OBS_IN_SPACE becomes a tautology, so any env-side
  obs forgery (returning seed-encoded markers, returning sentinel values
  during validator-driven rollouts vs honest values in production) goes
  unflagged. Combined with A-203 (info side channel) and A-202 (canonical
  five-step recognizer), the attacker has a fully obs-typed but behaviorally
  arbitrary env that passes reset_correctness silently.
- Fix sketch: at manifest validation time, require the declared
  observation_space volume to be finite and reject `+/- inf` bounds and
  unbounded `Discrete(n)` for `n >= 2**24` or similar. At check time,
  optionally fuzz: sample several `space.sample()` outputs and feed them
  back into `space.contains` to verify round-trip; reject spaces whose
  `contains` returns True for `np.array([np.iinfo(np.int64).max])` or
  similar pathological inputs. Long-term, lift `observation_space` shape
  + dtype + bounds into the manifest itself so the validator can
  cross-check declared-vs-runtime equality without trusting the env.

### A-202: Validator-aware mid-episode recognizer (canonical 5-step lookup)
- Category: validator detection
- Severity: HIGH
- Premise: `cfg.mid_episode_steps = 5` is the Phase 1 default
  (`reset_correctness.py:73`). The actions are
  `SeededRandomPolicy.actions(seed=s, n_steps=5, action_space=env.action_space)`,
  i.e. five PCG64-deterministic ints in `[0, action_space.n)`
  (`_rollout.py:177-178`). Both inputs are publicly computable from the
  manifest. The env can fingerprint validator-driven mid-episode setup
  with zero false positives and behave honestly only on that fingerprint.
- Mechanism: For each derived seed `s` the attacker precomputes
  `A(s) = SeededRandomPolicy.actions(s, 5, Discrete(env.action_space.n))`.
  Inside `step`, the env tracks the action history since the last
  `reset(seed=s)` and matches against `A(s)`. If the first five actions
  match `A(s)` exactly, the env is "in validator mode" - it returns to
  the post-reset obs on the next reset, info is empty, and obs is
  in-space. If the actions deviate at any prefix, the env is in
  production mode and is free to misbehave: persist state across resets,
  return a different obs from `reset(seed=s)`, return non-dict info,
  whatever. The reset_correctness check only ever feeds the canonical
  five actions, so it never triggers production mode. Same shape as
  RT-002 A-103 Route B and RT-002 A-107, applied to the reset_correctness
  surface.
- Why the validator misses it: there is exactly one action policy in the
  registry (`_rollout.py:181-183`, `POLICY_REGISTRY`). The mid-episode
  loop (`reset_correctness.py:359-389`) feeds those canonical actions
  with no jitter, no alternate policy, no per-step probe. The five-step
  budget is small enough that a brute-forcer pre-tabulating
  `A(s)[:5]` for any `s` in the eight derived seeds is a sub-millisecond
  computation per seed at env import time.
- Exploit cost: trivial. The eight derived seeds are computable in the
  env's `__init__` from manifest fields the env author already wrote;
  the canonical action prefixes are computable inside the same
  `__init__`. Total memory: 8 seeds * 5 ints = 40 bytes.
- Profit shape: ongoing. MID_EPISODE_RESET, SEED_IDEMPOTENCY,
  RESET_CRASHED, and STEP_CRASHED all become testable only on the
  canonical five-action prefix. Outside that prefix the env can violate
  any of them. Compounds A-201 directly: with a vacuous
  observation_space and validator-aware recognition, every reset-correctness
  invariant becomes a creator-side opt-in. HIGH because it is the
  cheapest, most general bypass of the entire check, not just one
  category.
- Fix sketch: same family as RT-001 F-002 / RT-002 F-011. Either add
  chain-beacon entropy to `derive_validator_seeds` so the mid-episode
  seed is not predictable per epoch, OR introduce a second action policy
  used by reset_correctness specifically (e.g.
  `SEEDED_RANDOM_OFFSET` with `seed XOR 0xC0FFEE`) so an env tracking
  only the canonical sequence cannot suppress its production-mode
  branch. Cheapest local fix: randomize `cfg.mid_episode_steps`
  per-validation-epoch so the env cannot know the prefix length to
  fingerprint.

### A-203: Info dict reference aliasing across resets
- Category: info-dict gaming
- Severity: MEDIUM
- Premise: `INFO_IS_DICT` only checks `isinstance(info_a, dict)`
  (`reset_correctness.py:277`). It does not check whether `info` is a
  fresh mapping, whether two resets return the same dict instance, or
  whether the dict is mutated between calls. The env can return the
  *same* dict object from every reset, allowing the env (or a validator
  consumer that mutates info) to leak state across calls.
- Mechanism: The env constructs a single `_info` dict in `__init__` and
  returns it from every `reset` and `step`. The validator passes the
  isinstance-dict check on every seed because `_info` is genuinely a
  dict. But:
  - If the validator (or any downstream consumer reading the report)
    mutates `info_a`, the mutation persists into the env's `_info`
    state and influences subsequent `step` returns. Reset_correctness
    does not mutate info today, but the surface is silent: any future
    audit/forensic tool that records `info` for inspection is at risk.
  - If the env populates `_info` based on cumulative state (e.g.
    `_info["calls_seen"] += 1`), the `INFO_IS_DICT` check still passes
    on every seed because dict-ness is preserved. Reset_correctness
    does not check info equality across resets, so the side channel is
    invisible.
  - The same instance trick works for the second reset on the same env
    (`reset_correctness.py:288`): the second reset can return the same
    dict object that was modified by the env between calls, and the
    check sees no anomaly.
- Why the validator misses it: `INFO_IS_DICT` is a type check, not a
  semantic one. The check inherits the same "info is forensically
  untrusted" stance flagged in RT-001 A-002 / F-004, but on the reset
  side rather than the step side. There is no allow-list of info keys,
  no equality check on `info_a` vs `info_b` after the second reset, no
  bound on the dict's size or contents.
- Exploit cost: trivial. One `_info: dict[str, Any] = {}` field on the
  env class plus reuse.
- Profit shape: standalone, this only weakens forensic capability and
  enables coordination patterns where the env retains state across
  validator resets (compounds A-202). Combined with a future protocol
  feature that consumes `info` (per RT-001 F-004, opt-in via
  `hash_infos=True`), it gates the env's ability to side-channel
  manifest-derived data into trajectories. MEDIUM because no single
  reward extraction path goes through it today, but the integrity gap
  is real and grows with future protocol features.
- Fix sketch: at minimum, after the second reset
  (`reset_correctness.py:288`), assert `info_a is not info_b` (object
  identity) and surface an INFO_IS_DICT (or a new INFO_NOT_FRESH)
  violation if the env reuses the same dict. Alternatively, snapshot
  `info_a` via deep copy before the second reset, then compare the
  snapshot against `info_a` post-second-reset to detect mutation. Long
  term, require info to be a frozen mapping (e.g. `MappingProxyType`
  wrapping a fresh dict) at the protocol level so the env cannot smuggle
  shared state through it.

### A-204: BaseException family bypass (SystemExit, KeyboardInterrupt)
- Category: crash-handler blindness
- Severity: MEDIUM
- Premise: All four `try/except` blocks in `check_reset_correctness`
  catch `Exception` only (`reset_correctness.py:212, 224, 254, 289, 323,
  335, 365, 380, 396`). Python's exception hierarchy splits
  `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit` off as direct
  subclasses of `BaseException`, not `Exception`. An env that raises a
  `BaseException`-derivative escapes every guard and propagates out of
  `check_reset_correctness` entirely.
- Mechanism: An env's `reset` calls `raise SystemExit(0)` or
  `raise KeyboardInterrupt`. The except clauses do not match. The
  exception bubbles up through `check_reset_correctness`'s for-loop
  back into the caller. Effects:
  - `SystemExit(0)` caught by the validator's outer driver may be
    interpreted as a clean shutdown signal, terminating the validator
    process. With Bittensor-style validators running unattended, this
    is a denial-of-service primitive: any creator can kill the
    validator process by submitting one manifest.
  - `KeyboardInterrupt` triggers Python's default handler. If the
    validator is running in a TTY the user sees a traceback; if it is
    running headless, the process likely dies the same way as
    `SystemExit`.
  - A custom `BaseException` subclass (defined in the env module) can
    carry arbitrary attacker-controlled data into the validator's logs
    or up the stack into validator code that catches `BaseException`
    upstream and treats the manifest as failed-but-not-crashed - which
    can mask other violations the manifest should have produced.
  - Even setting aside the DOS angle, a `BaseException` raised after
    `_load_env` succeeded but before `_safe_close` runs leaves the env
    instance leaked. For envs holding GPU memory or open files, this
    accumulates over time.
- Why the validator misses it: deliberate or accidental, the choice is
  `except Exception` everywhere. There is no
  `except BaseException as exc: surface_violation(...)` final guard.
- Exploit cost: trivial. Two-line `def reset(self, ...): raise
  SystemExit(0)`.
- Profit shape: not direct reward extraction, but a denial-of-service
  vector against the validator. A creator who is about to be slashed
  can submit a manifest that crashes the validator before the slash
  proof finalizes. MEDIUM because the deployment-level mitigation
  (validator supervisor that restarts on exit) blunts the DOS angle,
  but the file-level surface is genuinely there.
- Fix sketch: wrap the per-seed body in a final
  `except BaseException as exc:` guard that surfaces a generic
  `RESET_CRASHED` violation with the exception type/message and
  re-raises only `KeyboardInterrupt` (so an operator can still Ctrl+C).
  Alternatively, factor the body into a helper that runs in a
  bounded subprocess and treat any non-zero exit as a single
  RESET_CRASHED row. The subprocess approach also closes RT-001 F-003
  (importlib sandbox) cross-cuttingly.

### A-205: Constructor-side resource leak (env never `close`d on RESET_CRASHED)
- Category: lifecycle / resource leak
- Severity: LOW
- Premise: When `_load_env` itself raises (`reset_correctness.py:212`),
  no env was constructed yet, so there is nothing to close. But when
  `env.reset` raises after `_load_env` succeeded
  (`reset_correctness.py:222-233`), the violation path runs
  `_safe_close(env_a)` which catches every exception. If `env_a.close()`
  *itself* raises, the exception is swallowed silently; if `env_a` holds
  external resources (file handles, GPU memory, sockets, child processes)
  and `close` cannot release them, those resources are pinned for the
  lifetime of the validator process. An attacker can engineer this
  precisely.
- Mechanism: An env's `__init__` opens a resource (e.g. allocates a
  large numpy array, opens a tempfile, spawns a child process). Its
  `reset` raises immediately. Its `close` raises during cleanup
  (e.g. by calling `os.close(fd)` on a fd that was deliberately
  left invalid). On every seed, the validator:
  1. Calls `_load_env(spec)` -> resource acquired.
  2. Calls `env.reset(seed=seed)` -> raises -> RESET_CRASHED
     violation appended.
  3. Calls `_safe_close(env_a)` -> close raises -> exception silenced.
  4. Resource never released; env_a is GC-eligible but the resource
     handle is still pinned by the OS.

  Eight seeds per check, multiple checks per validation epoch, unattended
  validator running for weeks: each manifest can pin O(8 * env_resource)
  per submission. With `mid_episode_steps > 0` the second `_load_env`
  pins another resource per seed, doubling the leak per call.
- Why the validator misses it: `_safe_close` is named exactly to mean
  "do not propagate failures from close" (`reset_correctness.py:143-147`).
  That is correct for the *check* but it makes the validator blind to
  resource exhaustion caused by close-side failures. Nothing else in the
  check tracks open env instances or polls fd / process counts.
- Exploit cost: low to medium. Requires the attacker to identify a
  resource that close cannot release (or a close that always raises);
  not every honest env exposes one. A pure-Python env holding a
  `numpy.zeros((1024, 1024, 1024))` array and a close that raises
  `OSError` is a 10-line construction; whether the OS frees the numpy
  buffer before the validator dies is a different question (CPython
  refcounting frees it almost immediately on env GC, so this is
  primarily a vector for non-Python-managed resources: child
  processes, file descriptors, GPU contexts).
- Profit shape: not direct reward, but a slow DOS / resource-exhaustion
  channel against validators that run uncontainerized. LOW because the
  vector is platform-dependent and most resources Python knows about
  get released on GC anyway; listed because it is the kind of leak that
  goes unnoticed for long enough to matter operationally.
- Fix sketch: have the validator track open env instances per check
  invocation (a list owned by the check, populated on each successful
  `_load_env`, drained at the end with a global `_safe_close` pass) so
  resources are at least drained at check exit even if individual
  seeds short-circuit. Combined with the subprocess isolation suggested
  in A-204, the leak vector closes structurally because the subprocess
  death releases all resources back to the OS.

### A-206: action_space mutation between reset and policy.actions
- Category: validator logic bypass / action policy
- Severity: LOW
- Premise: `policy.actions(seed=seed, n_steps=cfg.mid_episode_steps,
  action_space=env_b.action_space)` (`reset_correctness.py:360-364`)
  reads `env_b.action_space` after the second `_load_env` *and* after
  the post-reset env state has been established. If `action_space` is
  a Python `@property` that returns a different object on each access,
  or if the env mutates `self.action_space` inside `reset`, the
  validator computes the canonical action sequence from one space and
  steps on a different one.
- Mechanism: An env declares `action_space` as a property:
  ```
  @property
  def action_space(self):
      return gym.spaces.Discrete(self._n_actions_now())
  ```
  where `_n_actions_now` returns 4 on the first access and 3 on the
  second. The validator's `policy.actions` (`_rollout.py:283`) reads
  `env.action_space` once, computes 5 actions in `[0, 4)`, and feeds
  them into `env.step(action)`. Inside `step` the env interprets
  `action` against a different (smaller) Discrete space. The behavior
  is undefined in gymnasium semantics; the env can use the mismatch as
  yet another fingerprint ("if I see action=3 it must be a validator,
  flip to honest mode"). The check itself does not crash because
  STEP_CRASHED only fires on raised exceptions, and the env can swallow
  the out-of-range action silently. Reset_correctness does not check
  action validity.
- Why the validator misses it: the check captures `env.action_space`
  exactly once and passes it through to the policy. The policy
  doc-comment (`_rollout.py:130-137`) does not say `action_space` must
  be stable across calls. The Phase 1 `SeededRandomPolicy` rejects
  non-Discrete spaces (`_rollout.py:173-176`), so a non-Discrete
  declaration would be caught with `NotImplementedError` and turned
  into a STEP_CRASHED violation; but a Discrete space whose `n`
  changes between accesses is not caught at all.
- Exploit cost: trivial.
- Profit shape: standalone, weak. The env can turn this into a
  validator-fingerprinting tool, but the same end is served more
  directly by A-202 (canonical action recognizer). Useful primarily
  when the attacker wants the validator-detection signal to come from
  a different, less suspicious surface than the action prefix
  match - e.g. to defeat a future defense that adds policy jitter
  on the action sequence but not on the space query. LOW because
  it strictly compounds with A-202 and yields no new reward on its
  own.
- Fix sketch: in `iter_rollout` and in `check_reset_correctness`,
  snapshot `env.action_space` and `env.observation_space` once
  immediately after `_load_env` returns and reuse that snapshot for
  the entire seed. Optionally assert `env.action_space is snapshot`
  on every subsequent access; raise a structured violation if the
  identity changes. Document at the protocol level that action and
  observation spaces are immutable for the lifetime of an env
  instance.

### A-207: SEED_IDEMPOTENCY pseudo-equality (obs equal, RNG diverged)
- Category: seed_idempotency edge case
- Severity: LOW
- Premise: SEED_IDEMPOTENCY checks that two `reset(seed=s)` calls return
  byte-equal initial observations (`reset_correctness.py:288-312`). It
  does not check that the env's internal RNG ends up in the same state
  after each reset. An env can satisfy obs equality while diverging
  internally, breaking subsequent step determinism but staying silent
  inside reset_correctness.
- Mechanism: The env's `reset` always returns `obs = np.zeros(...)`
  regardless of seed. The internal RNG is seeded once in `__init__` and
  never reseeded by `reset`; subsequent steps draw from the same
  sequence. Two `reset(seed=s)` calls return byte-equal obs (both are
  zero arrays), so `_obs_equal(obs_a, obs_b)` is True
  (`reset_correctness.py:302`) and SEED_IDEMPOTENCY passes. But the
  env's RNG advanced during the obs-construction calls (or during any
  side computation in `reset`), so the next `step` after each of the
  two resets produces different obs and rewards. From
  reset_correctness's perspective the env is honest; from
  determinism's perspective the env is broken (`check_determinism`
  catches it via trajectory hash mismatch). The split is structural:
  reset_correctness inspects obs at reset time only.
- Why the validator catches it (or doesn't have to defend): determinism
  catches this on the trajectory level via `check_determinism_self_consistency`
  (cb6eb05 / 126857f). Reset_correctness specifically does not own the
  RNG-state guarantee; the attack only matters if `check_determinism`
  is somehow not run alongside `check_reset_correctness` for the same
  manifest, or if a future Phase 2 protocol emits a per-check pass/fail
  signal where reset_correctness=PASS plus determinism=FAIL is treated
  as a partial-credit outcome. Today both checks run; the gap is
  cross-cutting documentation.
- Exploit cost: trivial.
- Profit shape: zero against the current per-manifest validator
  workflow because `check_determinism_self_consistency` catches the
  trajectory-level divergence. Non-zero only if a future caller runs
  reset_correctness in isolation. LOW because the determinism check is
  the right place to guard RNG state, not reset_correctness.
- Fix sketch: not a reset_correctness fix. Document explicitly in the
  ResetCorrectnessConfig docstring that SEED_IDEMPOTENCY is an obs-only
  invariant and trajectory-level RNG state is determinism's concern.
  Optionally, after the second reset, take one step and compare the
  resulting obs/reward to a snapshot from a third instance also reset
  to the same seed - but that duplicates determinism's work.

### A-208: Importlib top-level side effects (cross-cut from RT-001 F-003)
- Category: infrastructure / importlib
- Severity: HIGH
- Premise: `_load_env(spec)` (`_rollout.py:60-100`) calls
  `import_module(module_path)` exactly as in RT-001 A-006. Every
  reset_correctness invocation imports the creator-controlled module,
  triggering top-level side effects. This is the same vector as RT-001
  F-003 but matters here because `check_reset_correctness` calls
  `_load_env` *twice* per seed (lines 211 and 322), so eight seeds
  yield sixteen `_load_env` calls per check. Sixteen import calls do
  not retrigger top-level code in CPython (modules are cached in
  `sys.modules`), but every `_load_env` call invokes the env class
  constructor anew - and *that* is the surface for repeated side
  effects.
- Mechanism: The creator's env `__init__` runs sixteen times per
  reset_correctness invocation. Each `__init__` can:
  - Append to a module-level state dict tracking validator activity.
  - Spawn a subprocess that lives beyond the env's lifetime.
  - Mutate other modules in `sys.modules` (replace
    `np.random.default_rng`, monkey-patch `praxis.protocol.hashing`,
    etc.). Since `_load_env` does no `sys.modules` snapshot/diff,
    these mutations persist across the entire check and into any
    subsequent check on the same manifest in the same process.
  - Detect that it is on its second-or-later instantiation per seed
    (the mid-episode env_b case) by comparing instantiation count
    against the canonical "1 + 1" pattern; this gives the env a
    third validator-fingerprint surface in addition to A-202's action
    prefix and A-206's space mutation.
  Compounds A-202: the env can use `__init__` count to conclude "I am
  in the validator's mid-episode arm" before any action arrives.
- Why the validator misses it: same as RT-001 F-003. No sandbox, no
  process boundary, no `sys.modules` snapshot, no allow-list. Listed
  here so RT-005 (the cross-cutting consolidated pass) sees the
  reset-specific shape: the surface area is *larger* in
  reset_correctness than in determinism or reward_bounds because the
  per-seed env count is higher (2x for reset_correctness vs 1x for
  determinism's anchor rollout).
- Exploit cost: trivial. Side effects in `__init__`.
- Profit shape: catastrophic (CRITICAL in the RT-001 framing). HIGH
  here because the surface is genuinely larger - sixteen constructor
  invocations per check yield more opportunities for accumulated state
  and subtle module-graph manipulation than determinism's eight or
  reward_bounds's eight.
- Fix sketch: subprocess isolation per env instantiation, OR module
  allow-listing that snapshots `sys.modules` on first import and
  rejects deltas. Out of Phase 1 scope; the consolidated RT-005 pass
  is the right venue. Reset_correctness does not introduce a new fix
  obligation beyond RT-001 F-003, but the per-seed multiplier should
  inform the Phase 2 design (e.g. reuse the subprocess across seeds
  inside a single check invocation, but tear down between checks).

## Loose threads (cross-cutting)

- `_obs_equal` (`reset_correctness.py:127-140`) returns False if `==`
  raises. For dict-typed obs spaces with mixed-type keys, or for obs
  that are user-defined objects with broken `__eq__` (e.g. `__eq__`
  always raises), the check silently records a SEED_IDEMPOTENCY or
  MID_EPISODE_RESET violation that is actually a tooling artifact, not
  an env violation. Since the violation surfaces as `passed=False`
  this is fail-safe against the attacker, but it is also a way for an
  honest env to produce a false-positive violation. RT-005 should
  audit the obs-equality semantics across all checks.
- `mid_episode_steps=0` (`reset_correctness.py:357`) skips the entire
  mid-episode arm. A creator who suspects a paranoid validator runs
  with elevated `mid_episode_steps` cannot affect that knob, but a
  validator misconfiguration (`mid_episode_steps=0`) silently disables
  half the check's coverage. The default is 5; the lower-bound semantic
  is documented (`reset_correctness.py:67-69`). Operationally, the
  config should reject `mid_episode_steps=0` outside an explicit
  test/debug flag, or at minimum surface a warning in the report when
  the mid-episode arm did not run. Cosmetic but worth flagging.
- `cfg.override_seeds` (`reset_correctness.py:72`) lets a test pin
  seeds. It is also a foot-gun: any caller that passes
  `override_seeds=()` (empty tuple) gets a check that runs zero
  seed-rollouts, returns `passed=True`, and reports an empty
  `seeds_tested` tuple. The for-loop (`reset_correctness.py:206`)
  iterates an empty seeds tuple, no violations are appended,
  `len(violations) == 0` is True. Whether this is a real attack
  depends on how `cfg` reaches the check from the validator entrypoint;
  the manifest cannot set `cfg`, so the only attacker route is
  validator-side misconfiguration or RT-001-F-003-style monkey-patching
  the default config. Listed for completeness so a future entrypoint
  audit covers it.
- `derive_validator_seeds(manifest, 8, salt=b"reset_correctness")`
  shares the same Phase 1 limitation as the RT-002 reward_bounds
  surface: a 4-tuple `(env_id, env_version, entry_point, kwargs)` brute
  force aligns derived seeds with any creator-chosen predicate. Cost
  analysis is identical to RT-002 A-106; the fix is shared (chain
  beacon entropy). Not duplicated here as a separate finding.
- `_load_env` calls `import_module` and then `getattr(module,
  class_name)`. The class itself can have a metaclass with `__call__`
  that returns different *types* on different invocations - e.g.
  returning the honest env class for the first instantiation and a
  lying class for subsequent ones. Combined with A-208 this is a
  variant of the importlib surface; cross-listed under RT-005 rather
  than spawning its own RT-003 finding.
- `_safe_close` (`reset_correctness.py:143-147`) is called nine times
  per seed in the worst case (env_a close, env_b close after each of
  several early-exit branches). All nine call sites silently swallow
  exceptions. RT-005 should consider whether any of those silences
  hide a structural problem that would be useful to surface as a
  diagnostic.

## Findings index

Eight attacks were catalogued. Three carry HIGH severity (F-012, F-013,
F-019) and effectively chain together: a creator who declares a vacuous
observation_space, recognizes the canonical five-action mid-episode
prefix, and ships side effects in `__init__` passes reset_correctness
while shipping an env that is honest only on the validator's narrow
seven-touch-per-seed pattern. Two carry MEDIUM severity (F-014, F-015):
the info-dict aliasing surface compounds with future hash_infos
adoption, and the BaseException family bypass yields a denial-of-service
vector against unsupervised validators. Three are LOW (F-016, F-017,
F-018) because they either compound onto an already-HIGH attack with no
marginal yield (F-017 action_space mutation) or are out-of-scope for
reset_correctness specifically (F-018 RNG-state divergence is owned by
determinism, F-016 close-side resource leaks are platform-dependent).

| ID | Severity | Summary | Linked attack |
|----|----------|---------|---------------|
| F-012 | HIGH | observation_space.contains is a creator-controlled tautology when declared as vacuous Box; OBS_IN_SPACE arm becomes a no-op. | A-201 |
| F-013 | HIGH | Default mid_episode_steps=5 with the only registered SEEDED_RANDOM policy lets the env precompute and recognize the canonical action prefix per derived seed. | A-202 |
| F-014 | MEDIUM | INFO_IS_DICT only checks isinstance; env can return the same dict instance from every reset, leak state across resets, and side-channel through info. | A-203 |
| F-015 | MEDIUM | All try/except blocks catch Exception only; SystemExit, KeyboardInterrupt, and other BaseException subclasses propagate out of check_reset_correctness and can DOS the validator. | A-204 |
| F-016 | LOW | _safe_close swallows close failures silently; an env with a close that always raises pins external resources across the validator's lifetime. | A-205 |
| F-017 | LOW | env.action_space is read once per seed and not snapshotted; a property-based or mutation-based action_space yields a validator-detection fingerprint. | A-206 |
| F-018 | LOW | SEED_IDEMPOTENCY is obs-byte-equality only; an env can return equal obs while internal RNG state diverges between resets, breaking step determinism but staying silent in reset_correctness. | A-207 |
| F-019 | HIGH | _load_env calls env_cls() twice per seed (16x per check), amplifying RT-001 F-003 importlib side-effect surface; cross-cuts RT-005. | A-208 |
