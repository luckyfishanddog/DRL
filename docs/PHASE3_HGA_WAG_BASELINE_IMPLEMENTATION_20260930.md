# Phase 3-1 HGA/WAG Baseline Implementation Handoff — 2026-09-30

## Status and boundary

`PHASE3_BASELINE_IMPLEMENTATION_STATUS = PASS`.

This handoff covers only workbook-role freeze, paper-aligned HGA and WAG+VNS
reconstruction, common-model adaptation, and the first small PPO development
smoke.  It does not freeze a publication comparison protocol and does not use
ID_TEST.  `FORMAL_SCOPE_V1_1`, B32, the SA temperature, Kdp/Kref, the active
ALNS operators, and the formal split domain were not changed.

The implementation is deliberately separated:

- proposed method: `src/mrta_search/`;
- comparison methods: `src/mrta_baselines/`;
- dataset governance: `src/mrta_data/phase3_split.py` and
  `data/manifests/PPO_PHASE3_DATA_SPLIT_V1.json`;
- development smoke only: `scripts/run_phase3_baseline_smoke.py` and
  `data/development/phase3_baseline_smoke_v1.json`.

The only proposed-method source edit in this phase is the requested docstring
correction: rail-serial is a deterministic *feasibility-oriented* fallback,
not an interference-safe construction.

## A. Source and tests

The supplied folder is nested under an unrelated outer Git checkout, whose
HEAD/remote do not identify `luckyfishanddog/DRL`.  Results therefore use the
explicit label `LOCAL_PHASE3_20260930_UNVERIFIED`, record the deterministic
DRL source-tree hash, and set `development_only=true` and
`commit_verified=false`.  They are not represented as results from GitHub
main `d05016436f08e0db49ac6410916166e19dcd2cf4`.

The final pre-handoff full regression command is:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

The final replay after the generated smoke refresh passed `209 tests in
31.45s`.  No pre-existing test was removed or weakened.

## B. Reference PDFs read directly

Both local PDFs were opened, text-extracted, and their algorithm pages were
rendered for visual verification:

1. Wenbo Liu et al., *An effective hybrid genetic algorithm for the
   multi-robot task allocation problem with limited span*, Expert Systems
   With Applications 280 (2025) 127299,
   DOI `10.1016/j.eswa.2025.127299`, local file
   `references/1-s2.0-S0957417425009212-main.pdf` (17 pages).
2. Jongsung Lee, Byung-In Kim, and Mihee Nam, *Novel method for welding
   gantry robot scheduling at shipyards*, International Journal of Production
   Research 61:17 (2023), 5842–5859,
   DOI `10.1080/00207543.2022.2117869`, local file
   `references/Novel method for welding gantry robot scheduling at
   shipyards.pdf` (19 pages).

The detailed PAPER_EXACT/PAPER_UNDERSPECIFIED/COMMON_MODEL_ADAPTATION ledger is
`docs/PHASE3_BASELINE_SOURCE_MAPPING_20260930.md`.

## C–E. HGA/WAG source mapping and adaptation boundary

### HGA

PAPER_EXACT components retained are population initialization, binary
tournament, route-based crossover, M1–M6 VND, alpha-nearness, population
growth to `mu+lambda`, route-edge diversity, and the paper defaults
`mu=20`, `lambda=10`, `alpha=20`, `max_iterations=200000`.

The paper does not provide an executable biased-fitness formula.  The fixed
adaptation `ADAPTED_HGA_BIASED_FITNESS_V1` uses
`(objective_rank + diversity_rank, objective_rank, diversity_rank,
canonical_hash)`.  The common-model bounded VND cap is 256 cheap candidates
per call; it was declared before the 30-second smoke.  All initialization
time is inside the wall clock.

The paper's movable-gantry/two-manipulator region physics and start/return
cycle model are not scientific fitness here.  One deterministic HGA
population seed adapts spatial region decomposition to formal rail eligibility
and a contiguous process-load-balanced boundary.  It is only an initializer,
does not use `x_up/x_low`, and does not replace continuous interference,
same-rail, WAIT, task horizon, B32, or certification.  The rest of the
population uses random-unassigned/shortest-route/least-insertion construction.

### WAG+VNS

PAPER_EXACT concepts retained are multiple balanced WAGs, nearest addition,
farthest insertion, two-direction route cost, 2-opt, factorial edge
combination, improved routes `i_r1/i_r2/i_r3`, route-type combinations, and
MOVE/SWAP/LNS.  Defaults remain `n=10`, `p=0.50`, `Imax=500`, and factorial
window 5.

`PAPER_EXACT` factorial mode exhausts the small fixture neighborhood.
`COMMON_MODEL_BOUNDED` keeps the same operator but fixes `max_windows=8` and
`max_factorial_calls=30720`, with deadline checks and charged runtime.

The paper's three-robot G1/G2/G3 model is not mechanically relabelled as four
robots.  `ADAPTED_WAG_ASSIGNMENT_V1` first respects upper/lower formal
eligibility and then forms deterministic x-ordered, process-balanced variants
inside R0↔R1 and R2↔R3.  Adapted Stage 3 sends bounded route combinations to
the common formal scheduler.  The paper-specific robot-order scheduler is not
used as an easier alternative evaluator.

## F–H. Workbook-level dataset freeze

The split was generated before any PPO HGA/WAG performance run, using only
workbook hashes/seeds and aggregate solver-independent geometry descriptors.
The unit is an entire workbook/generation-seed family; every sheet inherits
that role.  Historical folder names are metadata only.

| Role | Workbooks |
|---|---:|
| DEVELOPMENT_CONSUMED | 23 |
| TRAIN_POOL | 43 |
| VALIDATION | 15 |
| ID_TEST | 15 |
| Total | 96 |

There are 73 previously untouched workbooks.  N>90 instances remain
LARGE_SCALE_STRESS and do not participate in main-range balancing.  Split
hash:

```text
5b5e3a6d43b943c5bad03a9ecbadaefa74c6158b33036da6e0fc236e03af1c5a
```

The fail-closed access helper rejects ID_TEST before workbook loading.  No
ID_TEST instance was passed to HGA, WAG, ALNS, direction DP, reference
evaluator, or certifier in this phase.

## I–J. Implemented comparison methods

`ADAPTED_HGA_V1` includes:

- legal WHOLE/Y pattern chromosome and four canonical routes;
- deterministic region seed plus paper-style randomized population members;
- route crossover with route replacement, duplicate removal, shuffled missing
  block reinsertion, eligibility repair, and split-sibling repair;
- actual M1 relocate, M2 single swap, M3 consecutive-pair swap, M4 intra-route
  2-opt, M5 2-opt*, and M6 alternate 2-opt*;
- endpoint alpha-nearness, cheap route/load VND, optional-Y mutation;
- normalized route-edge distance and deterministic objective/diversity
  survival at `mu+lambda`.

`ADAPTED_WAG_VNS_V1` includes:

- multiple rail-eligible, spatially ordered, balanced WAG variants;
- nearest-addition and farthest-insertion routes;
- two-direction dynamic route cost, 2-opt, exact/bounded factorial combination,
  `i_r1/i_r2/i_r3`, and four-robot route combinations;
- same-rail MOVE/SWAP, WAG-style LNS repair, and optional-Y toggle;
- formal heavy-robot selection and the paper `n/p/Imax` limits.

Neither comparison method calls ALNS destroy/repair or hides formal evaluator
calls in its cheap local neighborhood.

## K–M. Formal domain, direction, and evaluator contract

Both baselines use the same pattern domain as the proposed method: WHOLE,
optional legal Y_SPLIT, mandatory Y_SPLIT, and no X_SPLIT.  Pattern changes use
`generate_y_split_patterns`, enforce at most one pattern per parent, exactly
once block coverage, canonical routes, and robot eligibility.

Every complete baseline candidate follows:

```text
CanonicalSolution
  -> optimize_directions_with_initial_feasibility
  -> FORMAL_SCOPE_V1_1 FormalReferenceEvaluator
  -> independent certify_schedule
  -> official_metrics
```

Only certified FEASIBLE candidates are ranked.  CONSTRUCTION_REJECTED,
DIRECTION_INFEASIBLE, DEADLOCK, INFEASIBLE, and NUMERIC_FAILURE remain distinct
and receive no penalty Cmax.  Reference/certifier calls and all major timing
categories are charged.

SA-OI-ALNS retains its bounded schedule-aware direction refinement as a
proposed-method component.  HGA/WAG receive the same initial-feasibility
direction DP but no hidden schedule-aware refinement.  Phase 3-2 should include
the planned ALNS-without-refinement ablation; this phase does not change ALNS.

## N. Timing and checkpoint contract

The wall clock starts before native initialization.  Records include
time-to-first-certified, Cmax@5, Cmax@30, Cmax@60, runtime, and overshoot.
Checkpoints later than the requested budget are null; failed candidates never
backfill a checkpoint.  Reference evaluations already in progress are allowed
to finish, so overshoot is measured rather than hidden.

The 30-second smoke produced a certified solution before 5 seconds for every
method/instance/seed.  Cmax@5 and Cmax@30 are therefore useful for the next
development comparison.  No 60-second run was performed, so Cmax@60 remains
null and its discriminatory value is unresolved.  A 5/30/60 protocol may be
tested on VALIDATION next, but must be frozen before formal comparison without
choosing checkpoints based on which method wins.

## O. Unit and operator coverage

New tests cover:

- deterministic eligible HGA initialization and region seed;
- route crossover exactly-once coverage and deterministic repair;
- all six HGA neighborhoods, alpha cap, route-edge diversity, population
  survival, optional Y, and fixed defaults/bounds;
- deterministic multiple WAGs, balance/eligibility, nearest/farthest routes,
  2-opt, exact factorial window, MOVE/SWAP/LNS, same-rail topology, optional Y,
  and bounded defaults;
- shared scope/evaluator/certifier path, no-penalty failure classification,
  and checkpoint semantics;
- workbook isolation, consumed-vs-ID_TEST exclusion, historical-folder
  non-authority, exact role counts, stable split validation, and fail-closed
  solver access.

## P–R. Three-method PPO development smoke

Selection was solver-independent: distinct DEVELOPMENT_CONSUMED workbooks,
closest N to 25/55/85 within N20–30/N50–60/N80–90, then geometry hash.

| Tier | N | PPO-relative workbook | Sheet |
|---|---:|---|---|
| small | 25 | `data/PPO_TRAIN/seed_0760656368.xlsx` | `g16_w025` |
| medium | 55 | `data/VALIDATION/seed_0967455456.xlsx` | `g27_w055` |
| large | 85 | `data/PPO_TRAIN/seed_0211781140.xlsx` | `g32_w085` |

Historical directory names above do not define Phase 3 roles; all three are
frozen DEVELOPMENT_CONSUMED.  Seeds are 20260928, 20260929, and 20260930.
Each method received 30 seconds per instance/seed.  All 27 runs completed with
certified final schedules; initialization was 9/9 successful for each method,
with no NUMERIC_FAILURE or certifier mismatch.

Aggregate profile from the completed smoke:

| Method | Certified | Median first certified (s) | Median runtime (s) | Max overshoot (s) | Total reference calls | Total certifier calls | DEADLOCK / recovered / remaining |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SA_OI_ALNS_INIT_POLICY_V2` | 9/9 | 0.879 | 30.332 | 4.351 | 1044 | 583 | 696 / 232 / 464 |
| `ADAPTED_HGA_V1` | 9/9 | 0.456 | 30.046 | 1.607 | 778 | 444 | 513 / 179 / 334 |
| `ADAPTED_WAG_VNS_V1` | 9/9 | 0.447 | 30.185 | 0.806 | 1455 | 1139 | 584 / 268 / 316 |

These figures verify integration and accounting; they are not a statement that
one method is superior.  The largest ALNS overshoot comes from a non-preempted
formal evaluation and remains visible in the record.

The separate `COMMON_CERTIFIED_SEED` diagnostic directly constructs the
existing RAIL_SERIAL_BOOTSTRAP once per selected instance.  N=25/55/85 all
certified with Cmax 4007.837328, 5303.928358, and 10557.156474 respectively,
using one reference and one certifier call each.  It is not injected into the
native comparison because HGA/WAG representations already have valid native
initializers; this avoids replacing initialization quality with a shared seed.

## S. Remaining paper ambiguities

- HGA's exact biased-fitness weights, restart policy, equal-cost tie rules,
  alpha endpoint definition, and one printed comparison direction are not fully
  specified.  All choices are fixed and labelled adaptations.
- WAG's random tie policy and exact repeated local-search stopping order are
  underspecified.  The common-model factorial caps are explicit.
- Paper region/home/return physics and the WAG three-order conflict scheduler
  describe different physical models.  They are not used as common-model
  scientific fitness.
- Flexible-rail WAG alternative breadth and the four-robot HGA spatial seed are
  common-model adaptations, not claims of exact paper reproduction.

## T. Readiness and next phase

The code is ready for **Phase 3-2 — Development/VALIDATION Common-Model
Comparison**.  It is not yet a frozen formal baseline benchmark: 60-second
timing, checkpoint policy, any VALIDATION-only parameter decision, the
ALNS-direction-refinement ablation, and standalone verified DRL Git provenance
remain to be completed before one-time ID_TEST use.

ID_TEST must remain untouched until that protocol is frozen.

## Final verification

- two PDFs read directly: PASS;
- source mapping: PASS;
- split frozen before PPO baseline performance runs: PASS;
- 23 consumed workbooks excluded from ID_TEST: PASS;
- HGA and WAG paper cores implemented: PASS;
- shared formal evaluator/certifier and optional-Y domain: PASS;
- 27/27 PPO development smoke runs certified: PASS;
- no numeric failure/certifier mismatch: PASS;
- full regression: PASS.

`NEXT_PHASE = Phase 3-2 — Development/VALIDATION Common-Model Comparison`.
