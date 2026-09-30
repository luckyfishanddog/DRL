# Phase 3 Baseline Source Mapping - 2026-09-30

This mapping was prepared from the two local PDFs in `references/`.  It is the
implementation boundary between paper-defined behavior and the four-robot
common-model adaptations.  `PAPER_EXACT` means that the cited paper states the
behavior directly; it does not mean that the complete paper problem is the
same as `FORMAL_SCOPE_V1_1`.

## HGA source

- Paper: Wenbo Liu, Zhian Kuang, Yongcong Zhang, Bo Zhou, Pengfei He, and
  Shihua Li, *An effective hybrid genetic algorithm for the multi-robot task
  allocation problem with limited span*, Expert Systems With Applications 280
  (2025) 127299, DOI `10.1016/j.eswa.2025.127299`.
- Local source: `references/1-s2.0-S0957417425009212-main.pdf`, 17 pages.
- Problem: two manipulators mounted on a movable gantry; limited working span;
  a workpiece is divided into regions and each region is optimized separately.
- Task representation: undirected weld lines represented as required arcs with
  two possible traversal directions.
- Paper objective: minimum cycle time including welding, torch travel, gantry
  movement, and waiting between regions.  Each paper route starts at and
  returns to the designated start point.

### PAPER_EXACT HGA components

1. A population-based hybrid genetic algorithm is run independently inside
   each paper region.
2. Population initialization creates one route per manipulator, repeatedly
   chooses a random unassigned weld line, inserts it into the currently
   shortest route at the least route-length increase, tests both directions of
   an undirected line, and applies local search before population insertion.
3. Binary tournament parent selection is based on objective value.
4. Route-based crossover copies Parent A, replaces one randomly selected route
   with one randomly selected Parent B route, removes duplicates, shuffles
   missing lines, and reinserts each at its least-cost position.
5. Variable-neighborhood descent uses all six neighborhoods:
   M1 relocate a line from the longest route after a line in another route;
   M2 inter-route single-line swap involving the longest route; M3 inter-route
   two-consecutive-line swap involving the longest route; M4 intra-route
   2-opt; M5 inter-route 2-opt* joining `(u,v)` and `(x,y)`; M6 the alternate
   inter-route 2-opt* joining `(u,y)` and `(v,x)`.
6. Local search considers only the alpha nearest weld lines of a selected weld
   line.
7. Population growth continues to `mu + lambda`; survival removes `lambda`
   solutions using objective quality and diversity.  Diversity is normalized
   Hamming distance based on non-common route edges.
8. Paper defaults are `mu=20`, `lambda=10`, `alpha=20`.  The paper stopping
   condition is 200000 iterations, where an iteration improves one solution by
   local search.

### PAPER_UNDERSPECIFIED HGA components

- The paper does not give an executable formula or weight for combining
  objective rank and diversity rank into biased fitness.
- Tie-breaking, restart details, exact alpha-nearness endpoint formula, VND
  acceptance order, and every least-cost tie are not fully specified.
- Algorithm 1 line 10 is printed with a comparison direction inconsistent with
  minimization; the surrounding text and all experiments establish that lower
  cycle time is better.

`ADAPTED_HGA_BIASED_FITNESS_V1` therefore uses the fixed tuple
`(objective_rank + diversity_rank, objective_rank, diversity_rank,
canonical_hash)`.  Objective rank uses certified common-model metrics when
available and the deterministic route/load surrogate otherwise.  Diversity
rank is descending mean normalized route-edge distance.  This is an explicit
adaptation, not a claim about the missing paper formula.

### COMMON_MODEL_ADAPTATION HGA components

- Paper: two manipulators optimize within paper-specific regions.  Ours: four
  robots on two rails; one deterministic population seed adapts region
  division to upper/lower rail by eligibility and a contiguous left/right
  boundary minimizing the larger regional processing load.  This seed is only
  an initializer: it does not replace formal eligibility, interference, WAIT,
  or task-horizon semantics, and it is unrelated to frozen `x_up`/`x_low`.
  The remaining population uses the paper random-unassigned, shortest-route,
  least-insertion construction.
- Paper: movable-gantry regions and paper start/return semantics.  Ours:
  `FORMAL_SCOPE_V1_1` rail eligibility, open routes, task-horizon release,
  continuous interference, same-rail order, WAIT, and B32 recovery.
- Genotype is the common `CanonicalSolution`: one legal WHOLE or Y_SPLIT
  pattern per parent, exactly-once block assignment, and four canonical routes.
- Route crossover remains route-based.  Pattern/block mismatches, ineligible
  inherited blocks, duplicates, missing blocks, and split-sibling changes are
  repaired deterministically with least-cost eligible insertion.
- Optional legal Y_SPLIT states are reached through a pattern mutation layer;
  mandatory Y_SPLIT is always retained.  X_SPLIT is never generated.
- Alpha-nearness uses minimum endpoint-to-endpoint Euclidean distance and is
  capped naturally at the available task count.
- Cheap route/load costs rank local candidates.  Every complete candidate sent
  to the scientific evaluator is counted and passes through the shared initial
  feasibility direction DP, formal reference evaluator, and independent
  certifier.  Paper region physics never becomes common-model fitness.
- Common comparison stops at the shared wall-clock boundary; initialization is
  inside the clock.  A paper-core iteration cap remains available only for
  small operator sanity.
- To prevent one common-model VND call from consuming the whole comparison
  budget, `COMMON_MODEL_BOUNDED` fixes a 256-candidate cap per VND call while
  retaining `alpha=20` and the M1--M6 order.  Up to all `mu=20` native initial
  individuals receive real common-model evaluation before survival selection.

## WAG+VNS source

- Paper: Jongsung Lee, Byung-In Kim, and Mihee Nam, *Novel method for welding
  gantry robot scheduling at shipyards*, International Journal of Production
  Research 61:17 (2023) 5842-5859, DOI
  `10.1080/00207543.2022.2117869`.
- Local source: `references/Novel method for welding gantry robot scheduling at
  shipyards.pdf`, 19 pages.
- Problem: three one-dimensional, non-crossing gantry robots; G1/G3 have
  exclusive end regions; neighboring robots maintain separation; long welds
  may be split.
- Task representation: bidirectional welding edges with setup, welding,
  post-processing, empty travel, and home positions.
- Objective: minimum makespan including collision-avoidance waiting.

### PAPER_EXACT WAG components

1. Stage 1 splits paper-long edges, sorts edges by leftmost x-coordinate,
   respects G1/G3 exclusive regions, and generates multiple balanced work
   assignment groups (WAGs) by moving boundary edges out of G2.
2. Split pieces assigned to the same robot are recombined before routing to
   avoid repeated setup/post costs.
3. Stage 2 constructs routes by both nearest addition and farthest insertion.
   Farthest insertion starts with the edge having the largest x-coordinate and
   then inserts the edge whose best insertion has the largest incremental
   distance; both edge directions are tested.
4. Routes are iteratively improved by 2-opt and factorial edge combination.
   The experiment uses a factorial window of five, hence `5! * 2^5 = 3840`
   directed sequences per full window.
5. Three improved route types are generated: `i_r1` from nearest addition with
   final-edge-changing improvements; `i_r2` from farthest insertion without
   changing the final edge; `i_r3` by further improving `i_r2` while allowing
   the final edge to change.
6. Stage 3 tries every `3*3*3` route-type combination per WAG.  The paper
   scheduler first uses order G1->G2->G3 and inserts WAIT when needed; if any
   waiting occurs it also tries G3->G2->G1 and G2->G1->G3.
7. VNS randomly chooses MOVE, SWAP, or LNS.  MOVE transfers a random edge from
   the longest-completion robot to neighbor robots.  SWAP exchanges a random
   edge with a neighbor.  LNS deletes every G2 edge and `p` percent of G1/G3,
   then repairs through the WAG generation rule.
8. Paper defaults are `n=10`, `p=0.50`, `Imax=500`; factorial window size is 5.

### PAPER_UNDERSPECIFIED WAG components

- Random sampling tie-breaks and seeds are not specified.
- The exact stopping test/order for repeated 2-opt and factorial passes, plus
  several equal-cost insertion ties, are not specified.
- The paper gives experimental `lmax=5000 mm` and `lmin=1000 mm`, but these are
  paper model parameters rather than a transferable split rule.

### COMMON_MODEL_ADAPTATION WAG components

- Paper: three robots in one non-crossing line.  Ours: four robots in rail
  pairs `R0<->R1` and `R2<->R3`; flexible blocks may consider every formally
  eligible robot during repair.
- Paper G1/G2/G3 exclusive-region rules are not mapped mechanically to
  R0/R1/R2.  `ADAPTED_WAG_ASSIGNMENT_V1` first obeys formal upper/lower rail
  eligibility and then creates x-ordered, process-time-balanced variants inside
  each rail pair.  Flexible blocks receive a deterministic load-balanced
  initial rail; bounded MOVE/LNS repair may then consider every formally
  eligible robot, including the other rail.
- Paper long-edge splitting is replaced by the already-frozen WHOLE/Y_SPLIT
  domain.  Mandatory Y_SPLIT is preserved and optional legal Y_SPLIT is
  reachable through a VNS split-toggle candidate.  X_SPLIT is excluded.
- The paper's three fixed robot-order conflict scheduler is source-mapped but
  deliberately not used as a second scientific scheduler.  Adapted Stage 3 is
  real: it orders all `3^4` route-type combinations by the paper-like cheap
  route objective, then sends the bounded leading combinations through the
  shared direction DP, `FORMAL_SCOPE_V1_1` evaluator, and independent
  certifier to generate a conflict-free schedule or an explicit failure.
- `PAPER_EXACT` factorial mode exhausts every sequence/direction for small
  fixtures.  `COMMON_MODEL_BOUNDED` retains the same window operator but uses
  fixed `max_windows=8` and `max_factorial_calls=30720`, with deadline checks;
  all its time is charged.
- Common comparison stops by wall clock, with paper `Imax=500` retained as an
  internal maximum.  Failed candidates have explicit status and no penalty
  Cmax.

## Shared comparison boundary

HGA, WAG+VNS, and SA-OI-ALNS use the same `ScientificConfig`, active scope hash
`5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`,
initial-feasibility direction DP, formal evaluator, certifier, and official
lexicographic metrics.  SA-OI-ALNS retains its schedule-aware direction
refinement as a proposed-method component; neither baseline receives a hidden
equivalent or free evaluator call.  A later ablation must measure that component.
