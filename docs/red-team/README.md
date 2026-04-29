# Praxis red-team docs

This directory holds adversarial attack catalogs against the Praxis validator. Each RT pass targets one component (or the cross-cutting surface) and is independently authored, reviewed, and closed.

## Passes

| ID     | Scope                                                     | Status                              |
|--------|-----------------------------------------------------------|-------------------------------------|
| RT-001 | `src/praxis/checks/determinism.py`                        | Living: 3 of 5 findings closed      |
| RT-002 | `src/praxis/checks/reward_bounds.py`                      | Draft                               |
| RT-003 | `src/praxis/checks/reset_correctness.py`                  | Not started                         |
| RT-004 | `src/praxis/checks/solver_baseline.py`                    | Not started                         |
| RT-005 | Cross-cutting + RT-001 F-002/F-003 re-evaluation           | Not started                         |

Update this table whenever an RT pass changes status.

## Conventions

### Attack ID numbering

Each RT pass owns a 100-block of attack IDs:

- RT-001: A-001..A-099
- RT-002: A-101..A-199
- RT-003: A-201..A-299
- RT-004: A-301..A-399
- RT-005: A-401..A-499

Future passes continue the pattern. Attacks within a pass are numbered sequentially from the block's first ID.

### Finding IDs

Findings are repo-wide and sequential: F-001, F-002, F-003... Each pass continues from where the previous pass left off. The Findings index table at the bottom of each RT doc lists only that pass's findings; the global numbering ties them together.

### Severity rubric

Use this exact rubric in every RT pass:

- CRITICAL: validator passes a manifest that allows ongoing extraction of bonded reward at scale, OR breaks a guarantee the rest of the protocol composes on top of.
- HIGH: validator passes a manifest that an attacker can use to extract bonded reward while shipping a broken or dishonest env. Profitable but bounded.
- MEDIUM: validator passes a manifest that materially weakens a guarantee but is hard to exploit profitably without further chaining.
- LOW: theoretical edge case; requires capabilities a real adversary doesn't have, OR already documented as Phase 2 work.

If unsure between two levels, pick the higher one.

### Doc skeleton

Every RT doc follows this structure:

1. Header: ID, status, date, scope, reviewed commits.
2. Threat model: who, what, can, can't.
3. Attack catalog: per-attack format (category, severity, premise, mechanism, why missed/caught, exploit cost, profit shape, fix sketch).
4. Loose threads: cross-cutting observations the pass deferred to another doc.
5. Findings index: F-NNN | severity | status | one-line summary | resolving commit (or DEFERRED).
6. (Living docs only) Closed findings summary: short synopsis at the top once findings are closed.

### Quota

Soft target of 4+ attacks per pass, no upper bound. If the surface is genuinely thin, ship 3 and say so explicitly. Do not stretch to hit numbers; inflated LOW findings dilute signal. RT-005 (cross-cutting) has a higher floor of 6.
