# Phase 3-0 — PPO Frozen Instance Intake Handoff

## A. Baseline

- Declared GitHub main baseline supplied by the owner: `e4de209d872d46687af4974d030b32904192b906`.
- Git operations were excluded from this local intake after owner clarification.
- Runtime: `D:\pybullet_test\.venv\Scripts\python.exe`, executed from `D:\pybullet_test\MRTA_GA\DRL`.
- `FORMAL_SCOPE_V1_1` remained unchanged, scope hash `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`.
- F4 remained `FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1`, complete-rollout budget B32.

## B. Discovered PPO root

The local data root is `D:\pybullet_test\MRTA_GA\ppo`, a sibling of `DRL`. Production code and manifests do not embed this absolute path. Runtime resolution uses `--ppo-root` or `MRTA_PPO_ROOT`.

The initial intake copied no Excel file into DRL. The frozen PPO workbooks remain in the local PPO directory and are addressed by PPO-root-relative path.

Owner follow-up subsequently authorized a minimal local migration for testing. Exact byte copies of the three smoke workbooks are now stored under the git-ignored `DRL/data/local/ppo_smokeset/` tree, preserving their original `data/...` relative paths. No other PPO workbook was copied.

## C. Excel inventory

- Excel workbooks: 96 `.xlsx`, 0 `.xls`.
- Total workbook bytes: 16,098,278; min/median/max file size: 13,079 / 169,685 / 216,780 bytes.
- Frozen platform instance sheets: 2,942; min/median/max per workbook: 2 / 31 / 31.
- Frozen platform family workbooks: 96.
- Raw source assembly workbooks inside the PPO root: 0.
- Unrecognized workbooks: 0.

The complete per-file inventory, including relative path, filename, file size, SHA-256, sheet names, row counts, and columns, is stored in `data/manifests/PPO_DATASET_MANIFEST_V1.json` under `workbook_inventory`.

## D. Observed schemas

Two actual schemas were observed and implemented:

1. `PPO_FROZEN_FAMILY_SCHEMA_V1`: 95 workbooks. Uppercase `META` uses the 31-column V9.1 family/sheet record format. Each non-META sheet has `weld_id, instance_index, group_id, orig_index, x1, y1, z1, x2, y2, z2, length_m`.
2. `PPO_LEGACY_FROZEN_FAMILY_SCHEMA_V1`: 1 workbook. Uppercase `META` uses the 13-column legacy V9.1 family/sheet record format. Its `N30` and `N40` sheets use the same weld columns.

No `placed_welds` plus key/value `meta`, one-row `meta`, or unrelated single-sheet platform schema was present in the discovered PPO root, so no speculative parser for those formats was added.

## E. Writer and source-code evidence

The family writer/loader is `..\ppo\src\laces_weld_slim\dataset.py`. It defines the observed `META_COLUMNS`, `LEGACY_META_COLUMNS`, and `RAW_WELD_COLUMNS`, writes one frozen sheet per platform instance, and calls historical generation only when building the old dataset. The new DRL adapter never imports or calls that generation path.

The historical platform writer is `..\src\generate_welds.py`, SHA-256 `d781e20970dc8e95e9ae0a86ee535afd4d2efe25a73e17c6fe103ff96475dcd7`. It reads the source workbook in millimetres, converts it to metres, performs historical filtering/packing, and calls `generate_placed_welds(..., output_units="m")` before the family writer serializes `x1..z2` and `length_m`. Its metadata records platform 20 m × 12 m and weld Z = 0.1 m.

`..\src\generate_instance_candidates.py`, SHA-256 `c26d9a4fa1713cc38dd04cb21da42de400f631067fb0b4284f1d8ef444da2006`, was read only for geometry metric definitions. Its random candidate-generation main flow was not executed.

The source assembly path recorded by the historical workbooks is not portable and the source workbook is not inside PPO. Its source SHA is consistently recorded as `f99cdc25ae9dd7b0958cef996921130ce5da26d785f128cb2bf278d927e18670`. The frozen platform workbooks themselves are sufficient for intake and are distinguished from that raw source by the V9.1 family META and writer contract.

## F. Unit rules

- Frozen platform sheet coordinate units: metres.
- `coordinate_scale_to_m`: 1.0.
- Historical source workbook units recorded by all 96 family META blocks: millimetres.
- The manifest therefore stores `input_units="m"` for the coordinates actually loaded and `source_input_units="mm"` for provenance.

This rule comes from writer code and `length_m`/platform metadata, not magnitude inference.

## G. Valid, invalid, and unsupported counts

- Valid platform instances: 2,942.
- Invalid platform instances: 0.
- Unsupported vertical-only weld instances: 0.
- Unsupported non-planar instances: 0.
- Valid unique platform instances: 2,942.
- All Z values are 0.1 m; no weld row was removed, clipped, moved, or regenerated.

## H. Invalid reason distribution

Empty. Validation nevertheless fails closed for non-numeric/NaN/Inf coordinates, duplicate explicit weld IDs, out-of-bounds endpoints, zero XY length, vertical-only welds, non-planar welds, multiple Z planes, formulas in frozen data, and metadata row-count mismatch.

## I. Duplicate analysis

- Identical raw file SHA-256 groups: 0.
- Identical normalized geometry hash groups: 0.
- Inventory entries are retained even when future duplicates are found; non-canonical copies receive `duplicate_of` and reason metadata.

## J. Weld count distribution

Across valid unique instances: min 10, median 55, max 102 welds.

## K. Geometry metric summary

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| bbox_area_ratio | 0.433798 | 0.912471 | 0.985515 |
| x_coverage_ratio | 0.684338 | 0.956536 | 0.998174 |
| y_coverage_ratio | 0.587795 | 0.965070 | 0.995892 |
| midpoint_dispersion | 0.186071 | 0.262347 | 0.319182 |
| grid_occupancy_ratio | 0.375000 | 0.937500 | 1.000000 |
| quadrant_length_cv | 0.026035 | 0.326079 | 1.030214 |
| upper_lower_length_imbalance | 0.000199 | 0.147850 | 0.750692 |
| left_right_length_imbalance | 0.000153 | 0.154365 | 0.862572 |
| weld_length_mean_m | 0.942109 | 1.540352 | 3.326614 |
| weld_length_std_m | 0.441666 | 1.190377 | 3.732542 |
| weld_length_min_m | 0.503873 | 0.503873 | 0.740000 |
| weld_length_max_m | 2.591000 | 5.375030 | 12.500000 |

## L. Formal compatibility summary

Using the unchanged default `ScientificConfig`:

- `n_whole_upper_only`: 77,056.
- `n_whole_lower_only`: 73,998.
- `n_whole_both_rails`: 2,238.
- `n_mandatory_y_split`: 9,928.
- `n_no_legal_pattern`: 0.
- `n_cross_By` (strict endpoint crossing of y=6 m): 13,563.

All 2,942 instances pass the static formal pattern compatibility check.

## M. Dataset manifest identity

- ID: `PPO_DATASET_MANIFEST_V1`.
- Hash: `3a12fc59708fcd6adc20bb19ba44ba8ece585f8e0a81a93304694960f18f2a31`.
- Hash policy: SHA-256 over canonical JSON of valid unique entries, sorted by `instance_id`.
- Dataset identity remains separate from the formal scope hash.

## N. Phase 3 smoke set

`PPO_PHASE3_SMOKESET_V1` was selected before solver execution solely by weld count, with geometry-hash tie-breaking:

| Tier | Instance | N | Total length (m) | Geometry hash |
|---|---|---:|---:|---|
| small | `data/DEV_ONLY/seed_000202::g06_w010` | 10 | 16.499350 | `412c2a5a8a23df1fd6e8ea72417c49d589333185c792b9a675f89437a51f15be` |
| medium | `data/VALIDATION/seed_0967455456::g27_w055` | 55 | 71.894199 | `00ec9cb45f83d42e3c60a600fd01837d1197f463c59eac08acd365a00dcab9a6` |
| large | `data/ID_TEST/seed_1411387107::g45_w102` | 102 | 166.008895 | `7295d29ee4cfc7963c0bf402dd447ac7fe93b1a7176f9651ba7fcc2ef5c24dad` |

Historical directory labels are not adopted as a new TRAIN/VALIDATION/TEST split.

## O. SA-OI-ALNS PPO smoke results

Solver seed was 20260929. M/Kdp/Kref, repair budget, ScientificConfig, F4, and B32 were unchanged.

| Tier | Limit | Initialization | Initial Cmax | Best Cmax | Cmax@1 | Cmax@5 | Cmax@30 | Iterations | Nref | Baseline DEADLOCK | Recovered | Remaining DEADLOCK | Runtime (s) | Overshoot (s) | Scheduler (s) | Repair (s) | Certifier (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 5 | SUCCESS | 1283.634899 | 643.344113 | 643.344113 | 643.344113 | n/a | 27 | 104 | 96 | 93 | 3 | 5.043597 | 0.043597 | 2.448767 | 1.588353 | 0.137190 |
| medium | 5 | INITIALIZATION_FAILED | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 2 | 0 | 2 | 0.835893 | 0 | 0.719329 | 0 | 0 |
| large | 5 | INITIALIZATION_FAILED | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 1 | 0 | 1 | 1.871771 | 0 | 1.708413 | 0 | 0 |
| medium | 30 | INITIALIZATION_FAILED | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 2 | 0 | 2 | 0.891449 | 0 | 0.760727 | 0 | 0 |
| large | 30 | INITIALIZATION_FAILED | n/a | n/a | n/a | n/a | n/a | 0 | 0 | 1 | 0 | 1 | 1.874951 | 0 | 1.713271 | 0 | 0 |

The 30-second setting cannot extend a run that fails before search initialization. These failures are reported without changing the smoke set or evaluator.

## P. Certification

- Small 5-second final schedule: FEASIBLE and independently certified; certification errors empty.
- Medium and large: no FEASIBLE initialization and therefore no final schedule to certify at either requested limit.
- No scheduler-FEASIBLE/certifier-fail mismatch occurred.
- The PASS condition requiring every smoke final schedule to be certified is not satisfied.

## Q. Tests

- Pre-change baseline: 180 passed in 28.49 s.
- PPO adapter tests: 7 passed. They cover both observed META schemas, frozen-sheet metre policy/source-unit provenance, stable weld IDs and geometry identity, absolute-root-invariant data identity, out-of-bounds rejection, vertical-weld rejection, NaN rejection, no row filtering, deterministic manifest order/hash, and duplicate geometry detection.
- Final DRL regression: 187 passed in 29.01 s with pytest cache disabled.
- Tests use temporary workbooks and do not depend on the local PPO directory.

## R. Known limitations

- The historical raw source assembly workbook is not present under the PPO root. Its identity and source-unit policy are preserved in workbook META and corroborated by the historical writer.
- The source layouts are real-geometry-derived frozen platform instances, not proven shop-floor in-place measurements.
- Static pattern compatibility does not guarantee that the bounded reference dispatch policy will initialize every large instance.
- The DRL directory has no standalone Git metadata in this workspace, so smoke provenance is explicitly development-only and uses the owner-supplied commit label. This does not affect dataset identity.
- No TRAIN/VALIDATION/TEST/OOD split is frozen in this round.

Local smoke-cache verification after migration: all three copied raw file SHA-256 values match the manifest; the selected sheets load as VALID with N=10/55/102 and all three normalized geometry hashes match. The cache can be used with `--ppo-root data/local/ppo_smokeset` and is excluded by `.gitignore`.

## S. Next-phase authorization

`PPO_DATA_INTAKE_STATUS = FAIL`

Unique blocking condition: the deterministic medium and large `PPO_PHASE3_SMOKESET_V1` instances do not produce a FEASIBLE initialization under the unchanged `FORMAL_SCOPE_V1_1` B32 evaluator, so final certified schedules do not exist for all smoke instances.

`PHASE3_DATA_SOURCE = PPO_FROZEN_PLATFORM_INSTANCES`

`NEXT_PHASE = BLOCKED; Phase 3-1 Adapted HGA / Adapted WAG is not authorized by this intake round.`
