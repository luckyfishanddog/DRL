# Phase 3-X — Finite Optional X_SPLIT Mechanism Gate

Date: 2026-10-05  
Evidence role: DEVELOPMENT mechanism evidence only  
Active formal scope: `FORMAL_SCOPE_V1_1` (unchanged)

## Final decision

```text
PHASE3X_EXECUTION_STATUS = PASS
X_SPLIT_MECHANISM_STATUS = NOT_SUPPORTED
X_SPLIT_SEARCH_ACCESS = WEAK
FORMAL_SCOPE_V2_AUTHORIZED = NO
ID_TEST_STATUS = SEALED
NEXT_PHASE = Phase 4-0 — Candidate-Pool Oracle Recall Audit under FORMAL_SCOPE_V1_1
```

The finite optional X_SPLIT mechanism was implemented and exercised successfully,
but it did not satisfy the preregistered support rule. The negative result is
retained without changing thresholds, candidate points, operators, run time, seeds,
or selected instances.

## Frozen evidence

- Gate set: `PPO_X_SPLIT_GATE_SET_V1`
- Gate-set hash: `4bfe998a9caef567a66db24a77e9a598b24bda4be7967e51d0484c1548558c93`
- Protocol: `PHASE3X_XSPLIT_PROTOCOL_V1`
- Protocol hash: `73fb5ff0c17d53439eacd097b79a19a0a98e6ebad762bd7ba21d9d7cc66b7c18`
- Experimental scope: `EXPERIMENTAL_X_SPLIT_SCOPE_V1`
- Experimental scope hash: `28236ee1caa75ce669af2a62ddbd28e2c557ed4b00eb5d4a1d8a29335aa209c8`
- Historical V1.1 scope hash: `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`
- Runs: 12 instances × 3 seeds × 2 arms = 72
- Seeds: `20261004`, `20261005`, `20261006`
- Search budget: 60 seconds per arm, excluding common-seed construction
- Arms: `NO_X_CONTROL` and `FINITE_X_DOMAIN`
- TWO_OPT_STAR: OFF

The selected workbook paths retain some historical directory names such as
`PPO_TRAIN`, `VALIDATION`, or `ID_TEST`, but the frozen Phase 3 split assigns every
selected workbook the role `DEVELOPMENT_CONSUMED`. The runner verifies this role and
exact provenance before loading a workbook. No active `TRAIN_POOL`, `VALIDATION`, or
`ID_TEST` evidence was accessed, and the formal ID test remains sealed.

## Implementation closure

The experimental scope inherits V1.1 semantics and adds only finite optional
rail-specific X_SPLIT. Candidate legality is determined before search from geometry,
frozen rail boundary, `delta_x=0.20 m`, `Lmin=0.20 m`, and rail eligibility. It does
not depend on WAIT, DEADLOCK, incumbent quality, an algorithm result, or a 10 m
threshold.

Each parent still has exactly one final pattern and at most two final welding blocks.
Combined XY splitting, recursive splitting, and three/four-block patterns are
excluded. Mandatory Y_SPLIT cannot be replaced by X_SPLIT. Spatial left/right child
identity controls the fixed assignments `R0/R1` on the upper rail and `R2/R3` on the
lower rail. Direction remains a search decision. The shared split point receives no
collision exception.

Each accepted X split has zero split/cut time but two independent SETUP/WELD/POST
sequences, adding exactly `tpre+tpost=50 s` of base processing. Same-robot adjacent
children canonicalize back to WHOLE. Coverage, length conservation, workspace,
assignment, scheduler, and certifier checks remain active. V1.1 fails closed on any
X_SPLIT input.

Search access uses only the existing split activation/deactivation/point-switch
family. It supports WHOLE→X_SPLIT, X_SPLIT→WHOLE, and X_SPLIT(t1)→X_SPLIT(t2), with
the normal canonicalization, cheap screen, direction DP, reference evaluator, and
certifier. No other X-specific neighborhood or new algorithm was added.

## Execution and certification audit

- 72/72 records completed with `TIME_LIMIT`.
- Runtime range: 60.013–69.660 seconds, including bounded in-flight overshoot.
- 72/72 final solutions certified.
- NUMERIC_FAILURE count: 0.
- All 36 paired arms used the same common certified seed hash.
- All 72 run-key hashes are unique and provenance-exact.
- `Cmax@5`, `Cmax@30`, `Cmax@60`, direction calls, DEADLOCK/recovery counts,
  per-robot process/empty/WAIT values, makespan-robot WAIT, final X patterns, and
  length coverage were recorded.
- Maximum observed split length-conservation residual was
  `1.4210854715202004e-14 m`, within scientific numeric tolerance.
- Full regression: 237 passed.
- `ACTIVE_FORMAL_SCOPE` remains `FORMAL_SCOPE_V1_1`.

## Preregistered support-rule result

Passed:

- all semantic tests;
- 72/72 runs without numeric failure;
- all reported final solutions certified;
- at least one certified X candidate produced a global-best update.

Failed:

- two different instances with three-seed median improvement ≥1%;
- at least one such supporting instance with `N >= 50`;
- a ≥1% supporting final best that actually contains X_SPLIT;
- two distinct X-rich workbooks with X beating NO-X in at least 2/3 seeds.

There were 2 X-arm wins, 10 losses, and 24 ties among the 36 paired runs. Only one
win ended with X_SPLIT in the final solution, and its improvement was 0.8038%, below
the 1% threshold. The other win ended without X_SPLIT and therefore cannot establish
mechanism support.

Certified X candidates were evaluated and sometimes accepted, and certified X
candidates produced global-best updates, but stable final retention and replicated
benefit were weak. Therefore `X_SPLIT_SEARCH_ACCESS = WEAK`; this does not authorize
adding more hand-designed X neighborhoods.

## Required handoff answers

### 1. How many parents had legal X candidates?

Across the 12 frozen gate instances, 429 parents had at least one legal X_SPLIT
candidate. They produced 539 rail-specific finite patterns.

### 2. Did candidates mainly come from Bx intersections or midpoint?

They mainly came from midpoint:

- MIDPOINT: 424/539 (78.7%)
- Bx intersections combined: 115/539 (21.3%)
  - BX_LOWER: 37
  - BX_CENTER: 42
  - BX_UPPER: 36

The selected instances contained 38 parents crossing `x_up` and 44 crossing
`x_low`; Bx points were useful but were not the dominant source.

### 3. Did X-rich and low-opportunity controls behave differently?

Both groups had zero median paired improvement. In the 18 X-rich pairs there were
1 win, 3 losses, and 14 ties. In the 18 low-opportunity-control pairs there were
1 win, 7 losses, and 10 ties. Final X_SPLIT appeared in 3 X-rich runs and 2 control
runs. X-rich instances were somewhat less harmful, but they did not show replicated
positive evidence and did not meet the high-opportunity rule.

### 4. How many instances actually selected X_SPLIT in a final solution?

Three of 12 instances selected X_SPLIT in at least one seed. This occurred in 5 of
36 FINITE_X_DOMAIN final solutions. Every such solution selected exactly one X parent.

### 5. Was Cmax reduction enough to cover the extra pre/post work?

Not reliably. Each retained split added 50 seconds. Of the five final-X runs, four
were worse than their paired NO-X controls. One run improved Cmax by 12.162 seconds
(0.8038%) despite the 50-second overhead, so a local net benefit exists, but it was
below the preregistered 1% threshold and did not replicate across seeds.

### 6. Were improvements associated with load bottleneck or WAIT reduction?

No stable signature was found. The only final-X win had no WAIT reduction and its
process-spread metric worsened by 16.322 seconds; its empty-travel metric improved by
33.779 seconds. The two other X-arm wins did not retain X_SPLIT and cannot be used as
X mechanism evidence. The gate therefore does not support a general process-load or
WAIT-reduction explanation.

### 7. Did X_SPLIT increase WAIT but still improve Cmax?

No. The retained-X runs that added WAIT (+20.003 seconds and +38.434 seconds) both
worsened Cmax.

### 8. Did X_SPLIT reduce WAIT but still lose because of added pre/post work?

Yes. One X-rich retained-X run reduced total and makespan-robot WAIT by 0.495 seconds
and reduced empty travel by 18.111 seconds, yet Cmax worsened by 13.242 seconds. The
50-second extra setup/post burden, together with worse process spread, outweighed
those small savings.

### 9. Did the shared split point create new coordination WAIT?

Yes in the controlled mechanism fixture: the shared point is checked normally and
forces WAIT for the inward-direction case. In real gate results, two retained-X runs
also added 20.003 and 38.434 seconds of total WAIT relative to control. The aggregate
run telemetry cannot attribute every real WAIT second solely to the shared point, but
it confirms that X_SPLIT does not bypass coordination cost.

### 10. Was there a beneficial case shorter than 10 m?

Yes. The controlled 8 m fixture benefits from finite X load sharing. In real gate
evidence, the only retained-X win split a 0.794 m parent and improved Cmax by 0.8038%.
This directly confirms that 10 m has no scientific role in candidate legality or
potential benefit.

### 11. Was there a long weld that was not split?

Yes. Four gate instances had a maximum eligible X span of 12.5 m, and none selected
X_SPLIT in any of their three final finite-domain solutions. The controlled long-span
fixture also preserves WHOLE as a legal, optional choice. Length alone does not decide
whether splitting is useful.

### 12. Should X_SPLIT enter formal V2?

No. The mechanism is semantically valid and occasionally reachable, but the frozen
DEVELOPMENT evidence does not demonstrate the required magnitude, replication, or
stable final retention. `FORMAL_SCOPE_V2_AUTHORIZED = NO`, and no V2 proposal is
created.

## Data and future-work consequences

The historical WHOLE+Y results remain valid only for their original V1/V1.1 domain.
No historical validation artifact or scope hash was changed. Because V2 is not
authorized, no validation repartition is performed, and no HGA/WAG common-X-domain
adaptation is started.

The next phase returns to `Phase 4-0 — Candidate-Pool Oracle Recall Audit under
FORMAL_SCOPE_V1_1`. The observed weak X access may remain useful as a later candidate
ranking research question, but this gate does not authorize further hand-built
X_SPLIT neighborhoods or indefinite mechanism tuning.

