# FORMAL_SCOPE_V1_1 F4 Closure — 2026-09-29

`FORMAL_SCOPE_V1_1_F4_STATUS = PASS`

`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1`

`NEXT_PHASE = Phase 3 — Adapted HGA / Adapted WAG common-model`

本报告只闭合 F4 的 prefix-depth coupling。没有实现 HGA/WAG、扩展 ALNS operator、修改 exact、重新引入 X_SPLIT、改变 parking/occupancy、生成 ranker data，或使用未来 VALIDATION/TEST/OOD。全部 calibration 与 smoke 为 development-only evidence。

## A. Baseline、Git 与 source provenance

实际执行目录为 `D:\pybullet_test\MRTA_GA\DRL`，解释器为实际存在的 `D:\pybullet_test\.venv\Scripts\python.exe`；用户给出的 `D:\pybullet_test.venv\Scripts\python.exe` 不存在。初始 DRL suite：`163 passed in 29.00s`。

当前本地目录仍嵌套于外层仓库：

- `git rev-parse --show-toplevel`：`D:/pybullet_test/MRTA_GA`
- outer HEAD：`0935ad8a723057084be190e31c6be4733ce3cb62`
- outer remote：`https://github.com/luckyfishanddog/MRTA.git`
- 目标 DRL repository：`https://github.com/luckyfishanddog/DRL`
- provenance：`development_only=true`、`source_commit=UNVERIFIED_LOCAL_TREE`、`commit_verified=false`

因此本轮本地 calibration 不是 publishable `FORMAL_RESULT`。`SOURCE_PROVENANCE_POLICY_V1` 没有放宽；未来正式 benchmark 必须在 standalone clean DRL checkout 自动验证 repository/root/HEAD/tree。

## B. V1 的结构性问题

历史 `FORMAL_SCOPE_V1` 的 recovery budget 是 16 个 popped prefix states。每次 expansion 最多提交一个 template operation，所以 K-template complete leaf 至少需要 K+1 states。现有 corpus 的 N=50/N=100 候选约有 196/396 templates，V1 的 16、32、64，乃至 N=100 的 128/256 都可能在到达任何 complete leaf 前耗尽。

旧报告的 16/32/64 local plateau 数据本身有效，但科学解释已修正为 **depth-censored plateau**。V1 保留原定义、实现与 hash，不再作为 active scope。

## C. 新 complete-rollout policy

`FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1` 保留原 optimized deterministic list scheduler 和 priority `(ESS, -remaining_processing, -completion, robot_id)`。Baseline FEASIBLE 立即返回，operations、WAIT、directions、completion、Cmax 与 canonical schedule JSON 不变。只有 baseline DEADLOCK 才运行 recovery。

一个 recovery budget unit 是一个 **complete alternative continuation**：从保存的 branch snapshot 选择确定性的非 baseline choice，然后用同一 dispatch kernel 一直运行到 FEASIBLE、DEADLOCK 或独立 NUMERIC_FAILURE。Template 数量不限制 continuation 深度。Baseline 本身不计 rollout budget。

每个完成的 FEASIBLE candidate 都保留到预算结束；代表结果按 `(Cmax, canonical schedule JSON)` 唯一选择。预算耗尽未找到结果仍为 DEADLOCK，不写 INFEASIBLE、不产生 penalty Cmax。

## D. Dispatch trace 与 branch state

`DispatchState` 是 immutable snapshot，含 next indices、robot completion、current points、WAIT counters 和 fixed operations。稳定 identity 使用 canonical JSON（sort keys、compact separators、allow_nan=false）与 SHA-256，不使用 object id、set iteration order 或 rounded float text。

当一步存在至少两个 finite choices 时，trace 记录 depth、ordinal、snapshot、原 priority ordered choices 与 chosen rank。Trace 只供 recovery 使用，不进入 scientific schedule timeline。Snapshot continuation 与从 root 强制 replay 的 status/schedule 由 differential test 验证一致；共享 kernel 继续调用既有 template validation、remaining-processing suffix、`earliest_safe_start_optimized`、WAIT construction 与 relevant-fixed index，没有复制 geometry/ESS/interference 科学规则。

## E. Deterministic discrepancy order

Baseline DEADLOCK 的 terminal blocker operation IDs 从既有 wait-for/ESS evidence 提取。与 blocker 对应的 branch points 作为 causal finite frontier；若没有 causal match，则使用完整 branch trace。顺序为 recent depth first、alternative rank ascending、stable plan/state identity。所有 single alternatives 先进入 frontier。

Single continuation 仍 DEADLOCK 时，可加入一个通用 causal continuation：只在当前 baseline choice 的 operation ID 属于该 rollout 的 terminal blocker 集合时选择 rank 1，否则恢复 rank 0。该规则只读取 dispatch/wait-for structure，不读取 family、N、solution hash、algorithm name、incumbent、history 或 RNG。`max_discrepancies_used` 记录 continuation 内实际非零 rank 数，避免把 causal chain 伪报成 single discrepancy。

代码与测试搜索确认没有 `handover_heavy`、`N==100` 或 candidate hash 的 dispatch 特判。

## F. Budget unit 与冻结值

- Unit：`COMPLETE_ALTERNATIVE_ROLLOUTS`
- Calibration：B=1/2/4/8/16/32
- Frozen budget：`deadlock_rollout_budget=32`
- Baseline run：不计 budget
- Active recovery order：`LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1`

16→32 的 recovery count 从 1 增至 9，仍明显变化，所以没有冻结 16。32 是本轮已授权 calibration range 的最小末端选择；没有继续无界扩大，也没有按 ALNS Cmax 选择 evaluator。B=32 的高开销被保留为明确 limitation。

## G. E1–E4

| Case | Baseline | V1.1 | source | Cmax | rollouts | certified |
|---|---|---|---|---:|---:|---|
| E1 | FEASIBLE | FEASIBLE | BASELINE | 2.000000 | 0 | yes |
| E2 | FEASIBLE | FEASIBLE | BASELINE | 14.16227766056838 | 0 | yes |
| E3 | DEADLOCK | FEASIBLE | LIMITED_DISCREPANCY_RECOVERY | 16.63619980225593 | 1 | yes |
| E4 | DEADLOCK | DEADLOCK | BASELINE | null | 2 | n/a |

E1/E2 的 V1、V1.1 与 baseline canonical JSON 完全相同。E4 frontier exhaustion 不被误写为 INFEASIBLE。

## H. Known old-2048 recoverable case

Corpus identity `328926b7612633797dbe0ae62b62964bbf0a6765b1ef0d6147305b07a0ec6511` 是 V1 prefix DFS 首次在 budget 2048 才恢复的 N=100 case。V1.1 在 B=16 时用 14 complete rollouts 恢复：Cmax=`2298.92129651784`、branch points considered=3、max discrepancies used=5、certified=yes。B=32 仍得到同一 Cmax，实际 frontier 在 14 rollouts 后耗尽。

独立 adversarial test 还让共享 complete engine 走完 404 templates，证明 rollout 不受 `B < K+1` 的深度截断。

## I/J. 31-state corpus calibration 与 runtime

复用 `data/development/f4_deadlock_stress_corpus.json` 的 31 个固定 baseline-DEADLOCK states；没有创建第二份 corpus。每项固定 solution、patterns、routes、directions 与 ScientificConfig，所有 FEASIBLE recovery 均由 `certify_schedule(..., scope=FORMAL_SCOPE_V1_1)` 通过。

| B | FEASIBLE / 31 | DEADLOCK | recovery p50 / p95 (s) | overall p50 / p95 (s) | rollout mean / median | max discrepancies |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 31 | 0.368 / 1.304 | 0.912 / 2.621 | 1.000 / 1 | 1 |
| 2 | 0 | 31 | 0.731 / 2.637 | 1.325 / 3.987 | 2.000 / 2 | 1 |
| 4 | 0 | 31 | 1.433 / 5.258 | 1.903 / 6.608 | 4.000 / 4 | 1 |
| 8 | 0 | 31 | 3.224 / 10.604 | 3.742 / 11.953 | 8.000 / 8 | 9 |
| 16 | 1 | 30 | 6.789 / 25.060 | 7.238 / 26.323 | 15.742 / 16 | 186 |
| 32 | 9 | 22 | 13.581 / 37.369 | 14.076 / 38.686 | 24.710 / 22 | 271 |

Baseline scheduler p50/p95 为 0.495/1.334 s。自动检查确认：较小 B 已 FEASIBLE 时较大 B 不回退 DEADLOCK；共同 FEASIBLE 的 Cmax 不恶化。9 个 recovered schedules 全部 certified。其余 22 项明确保持 DEADLOCK。

## K. Budget selection

B=16 满足 known-case 硬验收，却只恢复 1/31；B=32 恢复 9/31，说明 16 不是 complete-rollout plateau。选择 32 的代价是 DEADLOCK-heavy evaluator 的 p50/p95 额外成本增至 13.581/37.369 s。该成本会由 Phase 3 所有方法共同承担；它不是为 SA-OI-ALNS 调整的优势参数。未来改变 budget 必须创建新 scope version，不能按 method/instance 动态修改。

## L/M. V1 与 V1.1 identity

F1/F2/F3 的 pattern domain、optional-X policy、Y rule、split limit、terminal、empty-route、initial deployment、open route、objective、certifier 与 interference fields 逐字段相同。仅 scope id 与 F4 policy/budget/order/count fields 改变。

- V1 historical hash：`8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`
- V1.1 active hash：`5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc`

V1.1 canonical definition：

```json
{"certifier_policy_id":"INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1","deadlock_budget_unit":"COMPLETE_ALTERNATIVE_ROLLOUTS","deadlock_policy_id":"FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1","deadlock_rollout_budget":32,"dispatch_order_id":"DFS_ESS_REMAINING_PROCESS_COMPLETION_ROBOT_V1","dispatch_recovery_order_id":"LIMITED_DISCREPANCY_RECENT_BRANCH_FIRST_V1","empty_route_policy":"UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1","initial_deployment_policy":"FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1","interference_policy_id":"CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1","max_split_per_parent":1,"objective_policy_id":"CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1","open_route_policy_id":"NO_HOME_FIRST_NO_RETURN_HOME_V1","optional_x_split_policy":"EXCLUDED","pattern_domain":["WHOLE","Y_SPLIT"],"recovery_selection_id":"CMAX_THEN_CANONICAL_SCHEDULE_JSON_V1","reference_scheduler_policy_id":"FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1","scope_id":"FORMAL_SCOPE_V1_1","state_count_policy_id":"COMPLETE_ALTERNATIVE_ROLLOUTS_V1","terminal_policy":"TASK_HORIZON_RELEASE_V1","y_split_rule_id":"BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1"}
```

`RunScientificIdentity` 继续承载 scope_id、scope_hash、scientific_config_hash、reference_policy_id、repository_id、source_commit、source_tree_hash、dirty/verified state。最终本地 scientific source tree hash（`pyproject.toml` + `src/**/*.py`）：`d876e9c3276abdd4f84b02cdee56c79d1f5f1f0bb93038e704b270eb3fe338d0`。

## N. Q1–Q6 V1.1 regression

| Case | initial Cref | final Cref | best source | recovery calls | status / certified |
|---|---:|---:|---|---:|---|
| Q1 assignment | 12.84 | 7.15 | ATOMIC | 0 | COMPLETED / yes |
| Q2 route | 11.545657757864154 | 11.468135899374975 | ATOMIC | 2 | COMPLETED / yes |
| Q3 direction | 7.105576065114842 | 6.271416626834708 | DIRECTION_REFINEMENT | 0 | COMPLETED / yes |
| Q4 optional Y | 6.0 | 4.0 | ATOMIC | 10 | COMPLETED / yes |
| Q5 interference/WAIT | 7.105576065114842 | 6.271416626834708 | ATOMIC | 12 | COMPLETED / yes |
| Q6 LNS basin | 19.35791211412712 | 3.557181856472053 | LNS_REPAIRED | 0 | COMPLETED / yes |

每项同 seed 重放的 canonical solution、directions 与 schedule 相同。Development exact C* 只作开发证据；没有跨 scope 声称 formal exact gap。

## O. Development-family smoke

运行 4 families × N=20/50/100 × seeds 20260928/20260929/20260930，固定 2 iterations，共 36 runs。全部 initialization success、status=COMPLETED、final certified、无 NUMERIC_FAILURE。Load-skew、spatial-cluster 与 interference-stress 的 27 runs 全部 C4 FEASIBLE；handover-heavy 保留真实 recovery/DEADLOCK variation。

Handover-heavy N=100 三个 seed 均完成 2 iterations并认证，runtime 分别约 184.15、90.94、209.14 s；baseline deadlocks 分别 5、6、5，recoveries 0、4、0，remaining deadlocks 5、2、5。失败是在完整 continuation 到 DEADLOCK 后保留，不是 prefix depth 小于 template count。

## P. N100 / 5s gate

Search defaults 未降低，`time_limit=5.0`。Runner 保证 successful initialization 后至少完成一个真实 iteration；总 runtime 和 overshoot 如实记录。Anytime events仍以从 run start 到 certified completion 的 elapsed time记账，截止线后完成的 improvement 不回填 5s checkpoint。

| seed | actual / overshoot (s) | iterations / Nref | initial / best Cmax | baseline / recovered / remaining | C4 feasible | scheduler p50 / p95 (s) | repair / certifier (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260928 | 54.499 / 49.499 | 1 / 2 | 2210.926 / 2210.926 | 3 / 0 / 3 | 0.00 | 6.171 / 36.090 | 1.065 / 0.118 |
| 20260929 | 27.564 / 22.564 | 1 / 4 | 2210.926 / 2210.926 | 5 / 4 / 1 | 1.00 | 0.676 / 11.725 | 1.887 / 0.387 |
| 20260930 | 78.078 / 73.078 | 1 / 2 | 2210.926 / 2210.926 | 3 / 0 / 3 | 0.00 | 17.536 / 37.370 | 2.306 / 0.151 |

三项均 initialization success、COMPLETED、final certified。高 overshoot 是当前 non-preemptive candidate evaluation 与 B=32 formal evaluator 的明确成本；5s checkpoint 不宣称得到截止线后的改善。

## Q. Tests 与命令

关键命令：

```powershell
D:\pybullet_test\.venv\Scripts\python.exe scripts/profile_scheduler.py --formal-scope-v1-1 replay
D:\pybullet_test\.venv\Scripts\python.exe scripts/profile_scheduler.py --formal-scope-v1-1 calibration
D:\pybullet_test\.venv\Scripts\python.exe scripts/profile_scheduler.py --formal-scope-v1-1 quality
D:\pybullet_test\.venv\Scripts\python.exe scripts/profile_scheduler.py --formal-scope-v1-1 smoke
D:\pybullet_test\.venv\Scripts\python.exe scripts/profile_scheduler.py --formal-scope-v1-1 performance --seed <seed>
D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests DRL/tests
```

最终 DRL suite：`169 passed in 27.14s`。联合 repository + DRL suite：`541 passed, 1 skipped, 3 subtests passed in 63.89s`。覆盖 baseline bypass、V1/V1.1 equivalence、>400 templates、deterministic trace、snapshot/root differential、recent/rank order、dedup、exact budget cap、status/Cmax monotonicity、certification、known case、E1–E4、V1 replay/hash、F1–F3 equality、fixed-seed ALNS reproducibility、time-limited real iteration 与 anytime no-future-backfill。

## R. Remaining limitations

1. Recovery 是 bounded deterministic heuristic，不是 mathematical feasibility oracle；remaining DEADLOCK 不等于 infeasible。
2. B=32 在 DEADLOCK-heavy N=100 上成本高，non-preemptive evaluation 导致 5s runs 明显 overshoot。Phase 3 比较必须让所有方法使用同一 evaluator 和报告同一 timing boundary。
3. TASK_HORIZON_RELEASE_V1 仍是 task-layer abstraction，不覆盖 physical deployment、retract、parking、IK、joint trajectories 或 3D collision-free execution。
4. 当前 exact 仍是 development Y-scope dispatch+ESS micro backbone；没有 `Cmax_OPT` 正式声明。
5. 本地 nested workspace provenance 未验证；本报告是 development release evidence，不是正式 benchmark result。

## S. Phase 3 authorization

Baseline FEASIBLE 未改变；complete-rollout budget 消除了 template-depth structural censoring；known large case 在 16 rollouts 内恢复；全部 recovered results certified；budget status/Cmax 单调；E1–E4、Q1–Q6、36-run smoke、N100/5s 与 full regression 通过；V1 历史可 replay；F1/F2/F3 未改变。

`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1`

`NEXT_PHASE = Phase 3 — Adapted HGA / Adapted WAG common-model`
