# Phase 2B-2 — Minimal Real ALNS Closure 交接报告

日期：2026-09-28  
范围：`EXACT_Y_SCOPE_CURRENT_SEMANTICS`  
结论：Phase 2B-2 PASS；Phase 2B 可结束，下一阶段为 `FORMAL_SCOPE_GATE`。

本报告记录 development 实现与实测，不是 formal experiment，也没有冻结参数。实现没有启用 X_SPLIT、terminal occupancy、empty-route parking、bounded deadlock repair 或 `TWO_OPT_STAR`。

## A. Baseline

- 开始 SHA：`0935ad8a723057084be190e31c6be4733ce3cb62`
- 开始时 `DRL`：`128 passed in 13.79s`
- 开始时项目联合：`500 passed, 1 skipped, 3 subtests passed in 47.84s`
- 工作树开始时已有大量用户改动；本轮只修改 `DRL/`，未运行 reset/checkout/clean。

## B. CompleteSearchCandidate 架构

新增统一完整候选层：

- `source_kind = ATOMIC | LNS_REPAIRED`
- 完整 `canonical_solution`
- 稳定 `CompleteCandidateIdentity`
- makespan-first cheap features
- provenance
- atomic 候选保留原 `CandidateMove`
- LNS 候选记录 destroy、repair、removed parent IDs、q 和完整 repair trace

LNS identity 只使用 revision、字符串 enum、排序后的 parent IDs 和确定性 repair decisions；不依赖 object id、set/dict 遍历顺序。Partial state 带 source revision，旧 revision replay 会明确失败。

Atomic 与 LNS repaired candidates 共用唯一流程：

`complete pool → cheap score → Kdp → constrained direction DP → exact-empty rerank → Kref → reference → official metrics → SA`

没有给 LNS 建第二条 evaluator。

## C. Bounded Initial Portfolio

实现四种 deterministic construction：

1. `LOAD_FIRST`
2. `RAIL_BALANCED`
3. `X_ORDER_AWARE`
4. `SPATIAL_SPREAD`

Development defaults：`B_init_pool=4`、`Kinit_ref=2`。四个构造先 canonical dedup、constrained direction DP 和 cheap rank，最多两个才进入 reference。NUMERIC_FAILURE 立即隔离；其他失败不触发无界 retry。

已知 `handover_heavy/N=50` 闭合结果：三个固定 seed 均由 `SPATIAL_SPREAD` 得到 certified FEASIBLE initial；每次 4 个 construction、最多 2 个 reference call。没有启用 deadlock repair。

## D. Destroy operators

- `RANDOM_REMOVAL`：seeded、无放回、parent-level removal。
- `CRITICAL_LOAD_REMOVAL`：使用已有 FEASIBLE schedule 的 max-completion robot、completion、process load 和 WAIT 信息作确定性 lexicographic score；destroy 内无新 scheduler call。

Destroy 单位始终是 parent。Y_SPLIT parent 的两个 child 一起移除，partial state 从不伪装成 CanonicalSolution。

`q = clamp(round(rho*N_parent), q_min, q_max)`，development defaults 为 `rho=0.15, q_min=2, q_max=8`；N>1 不删除全部 parent。

## E. Repair operators

- `GREEDY_REPAIR`：descending parent process work，再按 parent ID；选择最优完整 insertion alternative。
- `REGRET_2_REPAIR`：比较每个 parent 至少两个完整 alternative，先最大化 projected-Pmax regret，再比较 travel regret 和稳定 ID。

WHOLE alternative 是单 block placement。Y_SPLIT alternative 是两个 children 的完整联合 placement；mandatory Y 不允许通过 same-robot consecutive placement collapse 为非法 WHOLE。最终 canonicalize 仍是权威。

Repair score 为 lexicographic：projected process makespan、local directed travel proxy、robot/position/direction hint。没有 weighted objective。

## F. Repair complexity 与硬预算

- `I_repair=8`
- `repair_evaluation_cap=4096` / repaired candidate
- regret-2 按剩余 future calls 分配硬预算，避免首个 parent 消耗全部预算
- `repair_insertion_evaluations` 结构化记录
- `repair_reference_calls == 0`
- `repair_full_direction_dp_calls == 0`

运行时 invariant 与 tests 同时验证 raw/repair candidate 在 C3 前没有 reference/full direction DP。

## G. Adaptive operator selection

四个 pair：D1R1、D1R2、D2R1、D2R2。初始 weight 均为 1.0，使用 algorithm RNG 的 deterministic roulette。Development defaults：reaction `0.2`，segment length `20`。

Reward 仅来自真实 C4/SA outcome：GLOBAL_BEST、ACCEPTED_IMPROVEMENT、ACCEPTED_NON_IMPROVEMENT、REJECTED、EVALUATION_FAILED。NUMERIC_FAILURE 不更新 weight；DEADLOCK 保持独立 status，不转换为 infeasible/penalty Cmax。

N=50/30s 中观察到一次完整 segment update，最终 weights：

| pair | weight |
|---|---:|
| RANDOM+GREEDY | 0.7200 |
| RANDOM+REGRET2 | 0.8178 |
| CRITICAL+GREEDY | 0.7067 |
| CRITICAL+REGRET2 | 2.0800 |

N=100/30s 只有 2 个 LNS candidates 真正进入 C4，未达到 segment 20，因此 weights 保持 1.0；这是候选 shortlist 竞争结果，不是隐藏补样。

## H. Atomic + LNS unified pool

Development defaults：`M=64`，其中 `M_atomic=48`、`M_lns=16`；总 attempt 是 hard cap。Invalid、identity、duplicate 和 repair failure 都消耗 attempt，不补到固定合法候选数。

统一 cheap ordering：

`(projected_process_makespan, local_directed_travel_delta, split_count_delta, complete_identity)`

统一 C3 rerank：

`(projected_process_makespan, exact_directed_total_empty, split_count_delta, original_cheap_rank, complete_identity)`

## I. Bounded schedule-aware direction refinement

`refine_directions_bounded` 只生成 deterministic single-flip variants，优先 critical-completion robot、WAIT block、first block、last block；development `Bdir<=4`。

每次 flip 都是显式方向的真实 reference call，并计入：

- `Nref`
- wall-clock
- search scheduler timings
- per-iteration `Kref_total`

默认 `Kref=2`、`Kref_total=4`。只保留 FEASIBLE、certified、official-metric improved 的 direction；assignment/route/pattern 不变。

Q3：base Ccoord `7.1055760681`，refined Ccoord `6.2714166268`，达到 current-scope `C*`。Direction search gap 从 `0.8341594412` 降为 `0`。

## J. Q1–Q6 micro ground truth

FAST_CONFIG，仅作 current-scope development validation：

| case | C* | initial Ccoord | initial Cref | final Ccoord | final Cref | final search gap | final scheduler gap | best source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q1 assignment | 7.150000 | 12.840000 | 12.840000 | 7.150000 | 7.150000 | ~0 | 0 | ATOMIC |
| Q2 route | 11.468136 | 11.545658 | 11.545658 | 11.468136 | 11.468136 | 0 | 0 | ATOMIC |
| Q3 direction | 6.271417 | 7.105576 | 7.105576 | 6.271417 | 6.271417 | 0 | 0 | DIRECTION_REFINEMENT |
| Q4 optional Y | 4.000000 | 6.000000 | 6.000000 | 4.000000 | 4.000000 | 0 | 0 | ATOMIC |
| Q5 interference/WAIT | 6.271417 | 7.105576 | 7.105576 | 6.271417 | 6.271417 | 0 | 0 | ATOMIC |
| Q6 LNS basin | 3.557182 | 19.357912 | 19.357912 | 3.557182 | 3.557182 | 0 | 0 | LNS_REPAIRED |

Q3 initial `Cref-Ccoord=-2.97e-9` 是当前 tolerance 内的两套连续计算浮点差；没有截断或用于最终 gap。所有 final results certified，最终 decomposition 满足 `total ≈ search_gap + scheduler_gap`。

Q6 additional control：相同 seed 的 atomic-only、1 iteration 仍为 `19.357912`；LNS-only、4 iterations 到达 `3.557182=C*`。因此 destroy→repair 有实际 basin-crossing 价值。

## K. Initialization robustness matrix

四类 × N=20/50/100 × 5 个固定 development seeds：60/60 SUCCESS。Initializer 本身 deterministic，因此 seed repetitions 是 replay/robustness diagnostic，不是统计推断。

| family | N20 winner | N50 winner | N100 winner | success |
|---|---|---|---|---:|
| load_skew | RAIL_BALANCED | LOAD_FIRST | LOAD_FIRST | 15/15 |
| spatial_cluster | LOAD_FIRST | LOAD_FIRST | LOAD_FIRST | 15/15 |
| handover_heavy | X_ORDER_AWARE | SPATIAL_SPREAD | X_ORDER_AWARE | 15/15 |
| interference_stress | LOAD_FIRST | LOAD_FIRST | LOAD_FIRST | 15/15 |

Portfolio 内部仍真实观测到 DEADLOCK 和 NO_LEGAL_FIRST_ORIENTATION；它们没有被改写为 INFEASIBLE。NUMERIC_FAILURE 为 0。

## L. 1/5/30 s development runs

Nominal `ScientificConfig`，synthetic development family，seed 20260928：

| N | requested | actual | overshoot | iterations | Nref | Nref/s | atomic accepted | LNS accepted | dir calls | initial/best Cmax |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 1.049 | 0.049 | 5 | 20 | 19.06 | 5 | 0 | 10 | 755.463 / 755.463 |
| 20 | 5 | 5.157 | 0.157 | 26 | 104 | 20.17 | 20 | 2 | 52 | 755.463 / 755.463 |
| 20 | 30 | 30.031 | 0.031 | 150 | 600 | 19.98 | 111 | 5 | 300 | 755.463 / 755.463 |
| 50 | 1 | 1.633 | 0.633 | 2 | 8 | 4.90 | 1 | 1 | 4 | 1896.204 / 1896.204 |
| 50 | 5 | 5.574 | 0.574 | 7 | 28 | 5.02 | 2 | 5 | 14 | 1896.204 / 1896.204 |
| 50 | 30 | 30.629 | 0.629 | 36 | 144 | 4.70 | 13 | 21 | 72 | 1896.204 / 1896.204 |
| 100 | 1 | 1.892 | 0.892 | 1 | 4 | 2.11 | 1 | 0 | 2 | 3607.315 / 3607.315 |
| 100 | 5 | 5.564 | 0.564 | 3 | 12 | 2.16 | 2 | 1 | 6 | 3607.315 / 3607.315 |
| 100 | 30 | 31.363 | 1.363 | 18 | 72 | 2.30 | 16 | 2 | 36 | 3607.315 / 3607.315 |

这些 synthetic instances 的 initial 已经很强，未产生 Cmax improvement；Q1–Q6 才是本阶段 search-quality 判据。所有 9 个 runs 均进入真实 ALNS search 并 certified。

## M. Runtime share

| case | candidate gen | repair | cheap | direction DP | reference | certifier | scheduler share |
|---|---:|---:|---:|---:|---:|---:|---:|
| N100/5s | 0.021s | 2.348s | 0.256s | 0.030s | 1.861s | 0.845s | 33.44% |
| N100/30s | 0.124s | 15.161s | 1.501s | 0.132s | 9.038s | 4.464s | 28.82% |

当前 N100 主瓶颈已转为 bounded repair alternative enumeration；reference scheduler 仍重要，但不是唯一绝对瓶颈。Certifier 也占可见份额。

当前 scheduler sanity：N100 slow `6.430s`，optimized `0.183s`，约 `35.17x`，scientific output equivalent。

## N. Operator contribution

- Q6 global best：`RANDOM_REMOVAL + GREEDY_REPAIR`
- N50/30s：21 次 LNS accepted；adaptive segment 已更新 weights
- N100/30s：2 次 LNS accepted，但没有 global Cmax improvement；LNS 进入 C4 的频率低，weights 未满 segment
- Direction refinement 在 Q3 达到 exact current-scope optimum；synthetic 1/5/30s 中无 direction improvement

## O. Tests

Final commands and actual results：

```text
D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
138 passed in 16.02s

D:\pybullet_test\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests DRL/tests
510 passed, 1 skipped, 3 subtests passed in 59.22s
```

New coverage includes complete identity/replay, parent destroy, WHOLE/Y repair, both repair operators, hard insertion budget, adaptive replay/update, unified pool, Q3 direction improvement, Q6 LNS basin, total reference cap, fixed-budget deterministic trajectory, and handover/N50 portfolio closure.

## P. Current bottleneck

1. N50/N100 repair consumes more time than reference scheduling; bounded Y pair alternatives and repeated regret recomputation are the next performance target.
2. On large synthetic states, cheap ranking rarely promotes LNS candidates into C4, so adaptive weights receive sparse evidence.
3. Wall-clock checks remain at safe call boundaries; N100/30s overshoot 1.36s is dominated by a complete iteration, not cancellation races.
4. Synthetic initial solutions show no Cmax improvement; this is reported directly, not hidden by alternate metrics.

## Q. Phase boundary

Phase 2B requirements are met: real destroy→repair exists, two destroy and two repair operators are active, adaptive selection updates reproducibly, atomic/LNS share one evaluator, Q3/Q6 establish direction/LNS value, initialization robustness is closed, fixed-budget replay passes, and all final best schedules are certified.

The next stage is `FORMAL_SCOPE_GATE`, which must decide F1 optional X_SPLIT, F2 terminal occupancy, F3 empty-route parking, and F4 bounded deadlock repair, forming `FORMAL_SCOPE_V1`. This report does not authorize Phase 3/GAT/MLP/PPO or formal experiments.

`PHASE_2B2_STATUS = PASS`
