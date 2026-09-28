# Phase 2B-1.5 Reference Scheduler Performance Closure 交接报告

> 日期：2026-09-28  
> Scope：`EXACT_Y_SCOPE_CURRENT_SEMANTICS`  
> 阶段名称：Phase 2B-1.5 — Reference Scheduler Performance Closure  
> 科学边界：仅做 semantics-preserving 性能闭合；没有实现 destroy→repair、adaptive destroy/repair、X_SPLIT、deadlock repair 或 F1–F4。

## A. Baseline SHA 与真实 baseline

外层仓库 SHA：

```text
0935ad8a723057084be190e31c6be4733ce3cb62
```

本轮开始前工作树已有大量 `DRL/` 之外的修改、删除和未跟踪内容；本轮未清理、回滚或覆盖这些内容。开始前真实测试：

```text
DRL: 109 passed in 4.17s
项目联合 tests DRL/tests: 481 passed, 1 skipped, 3 subtests passed in 38.16s
```

## B. Profiler 定位到的真实热点

新增 development-only、stdout JSON profiler `scripts/profile_scheduler.py`。它分别记录 canonical、block-map/route build、operation-template build、scheduler main loop、ESS、forbidden interval、interval sweep、conflict validation 和 remaining-processing priority，并记录 operation/ESS/fixed-scan/interval/conflict/WAIT 次数。

Nominal N=100 legacy 结果中：

```text
total                         6.483290 s
ESS                           6.440496 s
forbidden_start_intervals     6.100506 s
remaining-processing sum      0.031906 s
ESS calls                     1578
historical fixed scans        310474
forbidden interval calls      232858
operations_conflict calls     232858
```

因此主热点不是 canonicalization、template build 或 certifier，而是每次 ESS 对全部历史 fixed operations 重复生成 forbidden intervals，再做完整 validation；remaining-processing 的重复 suffix sum 是次要但确定的可消除成本。

## C. Semantics-preserving 优化

`mrta_reference.scheduler` 现在显式保留：

- `reference_schedule_slow()` / `reference_schedule_from_templates_slow()`：原 Phase 1.1 list scheduler，作为 equivalence oracle；
- `reference_schedule_optimized()` / `reference_schedule_from_templates_optimized()`：production optimized path；
- 公共 `reference_schedule()` 保持原 API，差分通过后指向 optimized path。

实施的三项简单优化：

1. **Expired fixed filtering**：仅删除 `fixed.end_time < ready - scaled_numeric_tolerance` 的跨机器人历史 operation；`end==ready`、`ready±epsilon` 和零时长边界保留。按 robot 保存 end-time 单调序列，用 `bisect` 取得 relevant suffix。
2. **Remaining-processing suffix**：每 robot 预计算 suffix；只累加 SETUP/WELD/POST，MOVE/WAIT 不进入 priority，保持原 dispatch tie-break。
3. **ESS deterministic interval sweep**：仍调用相同 `forbidden_start_intervals()`，按 `(lo, hi, blocker robot, blocker operation ID)` 排序，一次向前扫过 nested/overlap/touching union；推进仍使用原 `nextafter/epsilon-after-hi`。最终 candidate conflict validation 与 WAIT conflict validation均保留，只缩小到 relevant fixed subset。

没有更改 ScientificConfig、timeline、连续干涉、同轨不越位、initial feasibility、ESS 数学、dispatch priority、tie、WAIT、四状态、open-route、terminal/empty semantics 或 direction handling。

## D. 没有实施的优化

没有实现 incremental ESS。第一批简单优化已经使 N=100 单次 reference 达到约 0.19 s，并使 5 s run 完成 10 次真实 iteration，继续维护 per-next-operation incremental interval state 会增加等价性证明面而不再是当前 gate 所需。

没有删除 ESS 后置 conflict validation，没有近邻裁剪、时间网格、粗采样、collision approximation、operation 跳过、priority 改写、第三方索引或线程 cancellation。

## E. Slow/optimized equivalence 覆盖

新增 `tests/test_scheduler_equivalence.py`，覆盖：

- expired boundary：`end==ready`、`ready±epsilon`、明确过期、零时长；
- 500 个 seeded ESS differential cases；
- 显式 nested、touching、identical、infinite interval sweep；
- Phase 2A E1–E4 与 M1–M7；
- 300 个 seeded solution-level random cases，含不同 route/assignment/direction/empty route；
- WHOLE、optional/mandatory Y_SPLIT、WAIT、same-rail boundary；
- DEADLOCK status、完整 wait-for graph、cycle/blocker diagnostics；
- INFEASIBLE 和强制 NUMERIC_FAILURE；
- operation-by-operation robot/kind/geometry/time/sequence/block identity；
- 固定 `max_iterations`、无 wall-clock stop 的 slow/optimized search trajectory。

固定预算搜索比较了 final canonical solution、directions、Cmax、reference status sequence、accepted/rejected proposal trajectory 与 best-improvement sequence，全部一致。

## F. Scheduler runtime 与 speedup

同一机器、nominal `ScientificConfig`、seed `20260928`、同一 deterministic synthetic solutions：

| case | operations | ESS calls | slow | optimized | speedup | slow→optimized fixed/conflict scans |
|---|---:|---:|---:|---:|---:|---:|
| N=20 | 76 | 298 | 0.232412 s | 0.034473 s | 6.74× | 10,954/8,218 → 874/874 |
| N=50 | 196 | 766 | 1.647713 s | 0.064030 s | 25.73× | 72,994/54,734 → 1,566/1,566 |
| N=100 | 396 | 1,578 | 6.483290 s | 0.193379 s | 33.53× | 310,474/232,858 → 4,714/4,714 |
| interference stress E2 | 6 | 11 | 0.000793 s | 0.000427 s | 1.86× | 32/16 → 5/5 |

四例 slow/optimized scientific output 均完全等价。E2 实际生成 1 个 forbidden interval，并执行 1 次 WAIT conflict validation，不是无干涉 happy path。N=20/N=50 没有退化。

## G. Search timing 与 N=100、5 s gate

`SearchStats` 分开保存：

```text
init_scheduler_durations
search_scheduler_durations
all_scheduler_durations
```

三组均提供 mean/p50/p95。`Nref` 只计 search-stage actual calls；`init_reference_calls` 单独统计。另记录 requested budget、actual runtime、overshoot、last reference start/end。

N=100、nominal config、`M=64/Kdp=8/Kref=2` 未降低、seed `20260928`：

```text
requested budget             5.000000 s
actual runtime               5.313904 s
overshoot                    0.313904 s
iterations                   10
search-stage Nref            20
initial / best Cmax          3607.314815 / 3607.314815
certified                    true
init scheduler p50/p95       0.180534 / 0.180534 s
search scheduler p50/p95     0.123031 / 0.154050 s
all scheduler p50/p95        0.123793 / 0.180534 s
last reference start/end     5.068406 / 5.189253 s
```

Scheduler 调用仍不可中断；wall clock 只在安全边界检查。Anytime 仍只使用 deadline 前已完成且已成为 best 的 FEASIBLE result，不把 deadline 后完成结果回填早期 checkpoint。

## H. State-aware applicable move mask

`applicable_move_mask(state)` 只让当前至少存在潜在合法 proposal 的 atomic move 参与 quota；applicable set 内仍按 seeded rotation 均衡分配。`M` 仍是 raw generation attempts hard cap，invalid/duplicate/cheap-rejected 仍消耗 M，不补齐合法 candidates。

结构化统计新增 `applicable_by_move`，并保留 attempted/constructed/valid/C4/accepted/improvement counters。测试证明 inactive split moves 得到零 attempts，而 applicable moves 的 quota 差不超过 1，总 attempts 精确等于 M。

## I. Q1–Q5 非平凡 micro quality

配置为 FAST scientific config；每例的 `C*` 均由 Phase 2A `solve_exact_micro()` 在 `EXACT_Y_SCOPE_CURRENT_SEMANTICS` 下实际求得。`Ccoord` 固定 search 的 pattern/assignment/route/directions，仅重新运行 exact coordination。

| case | C* | initial Ccoord/Cref | final Ccoord/Cref | initial→final search gap | scheduler gap | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Q1 assignment trap | 7.150000 | 12.840000 / 12.840000 | 7.150000 / 7.150000 | 5.690000 → 0 | 0 | INTER_RELOCATE 改善 |
| Q2 route-order trap | 11.468136 | 11.545658 / 11.545658 | 11.468136 / 11.468136 | 0.077522 → 0 | 0 | 同 assignment，TWO_OPT 改善 |
| Q3 direction trap | 6.271417 | 7.105576 / 7.105576 | 7.105576 / 7.105576 | 0.834159 → 0.834159 | 约 -3e-9 数值噪声 | 未改善；无纯方向 move 的真实边界 |
| Q4 optional-Y trap | 4.000000 | 6.000000 / 6.000000 | 4.000000 / 4.000000 | 2.000000 → 0 | 0 | SPLIT_ACTIVATE 改善 |
| Q5 interference/WAIT trap | 6.271417 | 7.105576 / 7.105576 | 6.271417 / 6.271417 | 0.834159 → 0 | final 0 | 初始有显式 WAIT；INTER_RELOCATE/SWAP 改善 |

五例初始均严格非 optimal；Q1/Q2/Q4/Q5 改善到 C*，Q3 如实未改善。约 `3e-9` 的 Q3 initial `Cref-Ccoord` 在既有 scale-aware tolerance 内，未截断原始输出。

## J. Development families

Profiler 内置 development-only deterministic families：`load_skew`、`spatial_cluster`、`handover_heavy`、`interference_stress`，各有 N=20/50/100；它们不是 TRAIN/TEST 或论文 dataset。每项执行一轮真实 atomic search (`Kref=1`)：

| family | N20 | N50 | N100 |
|---|---|---|---|
| load_skew | COMPLETED, Cmax 1180.741 | COMPLETED, 2749.259 | COMPLETED, 5458.519 |
| spatial_cluster | COMPLETED, 375.741 | COMPLETED, 960.926 | COMPLETED, 1838.704 |
| handover_heavy | COMPLETED, 470.185 | INITIALIZATION_FAILED | COMPLETED, 2210.926 |
| interference_stress | COMPLETED, 614.074 | COMPLETED, 1532.593 | COMPLETED, 2910.370 |

`handover_heavy/N=50` 的两次 bounded construction 分别得到 reference DEADLOCK 和 `NO_LEGAL_FIRST_ORIENTATION`，因此按既有语义返回 `INITIALIZATION_FAILED`；没有扩大 B_init、启用 repair 或把它伪装成 INFEASIBLE。该结果是后续 Phase 2B-2 需要观察的初始化脆弱性，不是 slow/optimized equivalence failure。

## K. 测试与最终 regression

使用 `D:\pybullet_test\.venv\Scripts\python.exe`：

```text
python -m pytest -q -p no:cacheprovider DRL/tests
128 passed in 14.32s

python -m pytest -q -p no:cacheprovider tests DRL/tests
500 passed, 1 skipped, 3 subtests passed in 48.95s
```

没有删除困难测试。Raw candidate full-DP/reference invariants、每 iteration Kdp/Kref caps、四状态计数、final certification 和旧 Phase 1.1/2A/2B-1 tests 均无 regression。

## L. 修改与新增文件

新增且必要：

```text
scripts/profile_scheduler.py
tests/test_scheduler_equivalence.py
docs/Phase2B1_5_Reference_Scheduler_Performance_Closure_20260928.md
```

修改：

```text
README.md
docs/多机器人焊缝分配与排序实验方案.md
src/mrta_reference/__init__.py
src/mrta_reference/scheduler.py
src/mrta_search/__init__.py
src/mrta_search/initialization.py
src/mrta_search/neighborhood.py
src/mrta_search/pipeline.py
src/mrta_search/stats.py
tests/test_search_neighborhood.py
tests/test_search_sa.py
tests/test_search_stats.py
```

没有创建 profiler 数据副本、实验方案副本、fixture 数据文件、额外 hash、冻结 contract 或发布 gate。

## M. 当前瓶颈与 Phase 2B-2 边界

1. N=100 reference 已不再主导 5 s run，但复杂干涉 family 的单次 scheduler latency 仍明显高于无干涉 case。
2. Q3 证明只按 empty-travel 选方向的 atomic backbone 不能关闭纯 direction search gap；schedule-aware direction refinement 仍未启用。
3. `handover_heavy/N=50` 暴露 bounded initializer 在特定几何上的 DEADLOCK/首方向失败；本轮没有用隐藏 repair 绕过。
4. Synthetic N=100 的 10 iterations 未改善 Cmax，说明性能空间已经打开，但 atomic move 的质量能力仍有限。

允许进入的下一阶段仅为 **Phase 2B-2 — Minimal Real ALNS**：random removal、critical/load removal、greedy repair、regret-2 repair、deterministic reproducible adaptive weights。Repair 内部不得调用 reference scheduler；完整 repaired solution 才进入既有 C0–C4 pipeline。

仍留给 Formal Scope Gate：F1 optional X_SPLIT、F2 terminal occupancy、F3 empty-route parking、F4 bounded deadlock repair。没有实现 TWO_OPT_STAR、HGA/WAG、MLP/GAT/PPO、ranker dataset、LB_LP 或 formal experiments。

```text
PHASE_2B1_5_STATUS = PASS
```
