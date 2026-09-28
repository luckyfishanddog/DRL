# 第三阶段 Phase 2B-1 Bounded SA-OI Backbone 交接报告

> 日期：2026-09-28  
> Scope：`EXACT_Y_SCOPE_CURRENT_SEMANTICS`  
> 阶段结论：Phase 2B-1 bounded SA-OI neighborhood-search backbone  
> 重要边界：本阶段不是完整 SA-OI-ALNS；没有 destroy→repair，也没有 adaptive destroy/repair operator selection。

## A. Baseline SHA 与测试

外层仓库 SHA：

```text
0935ad8a723057084be190e31c6be4733ce3cb62
```

实现前真实 baseline：

```text
项目根目录 pytest -q -p no:cacheprovider:
372 passed, 1 skipped, 3 subtests passed in 40.68s

项目根目录 pytest -q -p no:cacheprovider tests DRL/tests:
455 passed, 1 skipped, 3 subtests passed in 39.63s
```

外层工作树在本轮前已有大量修改、删除和未跟踪内容；本轮只写 `DRL/`，未清理、回滚或覆盖外部工作。

## B. 新增与修改文件

新增实现：

```text
src/mrta_search/__init__.py
src/mrta_search/direction.py
src/mrta_search/initialization.py
src/mrta_search/neighborhood.py
src/mrta_search/pipeline.py
src/mrta_search/stats.py
scripts/profile_phase2b1.py
```

新增测试：

```text
tests/test_search_direction.py
tests/test_search_initialization.py
tests/test_search_neighborhood.py
tests/test_search_sa.py
tests/test_search_stats.py
```

修改 `README.md`，新增本 handoff。没有新增实验方案副本、fixture 数据文件、hash、baseline、冻结 contract 或 gate。

## C. Initial-orientation-constrained direction DP

`optimize_directions_with_initial_feasibility()`：

- empty route 返回 `()`；
- 对最多四条 active routes 枚举最多 16 个 first-orientation combinations；
- 同时检查所有 active robots 的初始理论干涉与 R0/R1、R2/R3 初始顺序；
- 对每条 route 和固定首方向执行 exact empty-travel DP；
- route tie 为 `(empty travel, complete vector)`；全局 tie 为 `(total empty travel, R0..R3 concatenated vectors)`；
- 不调用 reference scheduler，不优化 WAIT，不声称 Cmax 最优；
- 无合法组合返回 `NO_LEGAL_FIRST_ORIENTATION` 及诊断零方向，供获得 C4 名额时显式 reference 确认。

反例中自由 DP 因 tie 得到 `((0,), (0,), (), ())`，违反同轨首序；受约束 DP 找到另一合法组合。测试还对 route 长度 1–6 穷举全部方向，与受约束 DP 的 exact empty-travel optimum 一致，并覆盖 empty route、四 active robots、deterministic tie。Nominal `vw=0.0108, vm=0.20, tpre=20, tpost=30` 下方向、reference 和 N=1 micro exact smoke 均为有限 FEASIBLE/OPTIMAL，无 epsilon/ESS 异常。

## D. Deterministic initial solution

实现严格使用当前 Y-only development scope：

- whole eligible 默认 WHOLE；optional Y 初始关闭；
- mandatory Y 过滤 child 无 eligible robot 的 pattern 后按 `(t, point_id)` 取首个；
- X_SPLIT 不生成；
- blocks 按 `(-process_time, parent_id, block_id)`；
- `I_init=8`，短 route 全位置，长 route 用方案定义的 floor bounded positions；
- 首次 cheap score 为 `(projected_process_makespan, local_empty_delta, robot_id, position, hint)`；
- 第二 construction 为 `(projected_process_makespan, robot_id, local_empty_delta, position, hint)`；
- raw insertion 不调用 full route DP/reference；全部插入后才 canonicalize→constrained DP→explicit-direction reference→certifier；
- `B_init=2`，DEADLOCK/无合法首方向允许一次重构；canonical duplicate 不重评；NUMERIC_FAILURE 立即隔离；两次失败返回 run outcome `INITIALIZATION_FAILED`。

## E. Seven-move bounded generator 与 quota

Active moves：

```text
INTRA_RELOCATE
INTER_RELOCATE
SWAP
TWO_OPT
Y_SPLIT_ACTIVATE
Y_SPLIT_DEACTIVATE
Y_SPLIT_POINT_SWITCH
```

`TWO_OPT_STAR` inactive。Raw proposals 按 seed/revision/move/ordinal 直接有界构造，不先展开 O(N²) 完整邻域。`M` 是 raw attempts hard cap；invalid、duplicate、cheap-rejected 均消耗预算。七类先取 `floor(M/7)`，余数按 seed 确定的 rotation 分配；同 seed/state 的 attempt order 和 candidate replay 一致。

Move tests 覆盖七类的 valid、invalid/boundary、deterministic replay；relocate/swap/2-opt 保持 block 与 parent coverage；mandatory Y 不能 deactivate；point switch 只接受 `generate_y_split_patterns()` 的既有合法 point。

## F. C0-C4 与 cheap score

C0/C1 只执行 route index、coverage/one-split/Lmin、eligibility、mandatory Y、Y point、identity 与 canonical duplicate 检查。Cheap reject 只写 rejection reason，不伪造 scheduler status。

Phase 2B-1 cheap ordering：

```text
(
  projected_process_makespan,
  local_directed_travel_delta,
  split_count_delta,
  CandidateKey deterministic order
)
```

没有 weighted objective。C2 取 top Kdp；C3 对每个执行 constrained direction DP；然后按：

```text
(
  projected_process_makespan,
  exact_directed_total_empty,
  split_count_delta,
  original_cheap_rank,
  CandidateKey
)
```

二次排序并取 top Kref。固定测试证明原 cheap rank 第 3 的 candidate 经 exact direction 后进入 Kref=2 shortlist，因此 Kdp>Kref 的 DP 有实际作用。

每个 C4 显式传完整方向。Iteration-local memo identity 同时包含 scope、`ScientificConfig` identity、canonical solution、explicit directions 和 reference policy；cache hit 不冒充实际 reference call。

## G. SA engine 与四状态

Engine 独立保存 current/best solution、directions、schedule、metrics。多个 FEASIBLE C4 先按 official metrics 与 CandidateKey 选唯一 proposal，再做一次 SA：

- Cmax improve：必接收；
- numeric-equal：接收 current move；best 仍只按完整 official metrics 更新；
- worse：`exp(-(Cnew-Ccurrent)/(T*Cscale))`；
- `T=0`：只接收 non-worse；
- RNG 仅使用 algorithm-seeded `random.Random`；
- 温度按 search-stage actual Nref 衰减，初始化调用分列；
- DEADLOCK/INFEASIBLE/NUMERIC_FAILURE 均不进入 proposal/SA。

Reference status 保持 `FEASIBLE / DEADLOCK / INFEASIBLE / NUMERIC_FAILURE` 四分；DEADLOCK 不等于 mathematical infeasible。Accepted move 后 solution revision 由现有 canonical candidate application 推进；旧 revision candidate 在新 current state replay 会失败。

## H. Instrumentation 与硬 invariant

`SearchStats` 记录 scope/seed/iterations、初始化 construction/calls/status/time、raw/constructed/cheap/duplicate/Kdp、Nref 四状态、候选/筛选/DP/scheduler/certifier timings、scheduler mean/p50/p95、各 move attempted/constructed/valid/duplicate/C3/C4/accepted/best-improvement，以及 completed-best events。

运行时硬检查：

```text
N_feasible + N_deadlock + N_infeasible + N_numeric_failure == Nref
raw candidate full DP calls == 0
raw candidate reference calls == 0
Kdp_count_iteration <= Kdp
Nref_iteration <= Kref
```

Anytime 只从 `(reference completion/成为 best 的 elapsed, Cmax)` 事件生成；deadline 前无 FEASIBLE 时返回 `null + NO_FEASIBLE_BEFORE_DEADLINE`，不会用 deadline 后结果回填。

## I. Optional Y fixture

FAST_CONFIG fixture：单个 `(0,6)→(4,6)` parent。

```text
WHOLE initial Cmax = 6.0
Y_SPLIT_ACTIVATE best Cmax = 4.0
final pattern = Y_SPLIT
final certified = true
```

因此 optional Y_SPLIT 实际进入 neighborhood 并改变搜索结果，不只是数据结构占位。

## J. Micro exact/reference decomposition

配置：`weld_speed=empty_speed=1, tpre=tpost=1`；scope 为 `EXACT_Y_SCOPE_CURRENT_SEMANTICS`。Ccoord 固定 search 输出的 pattern/assignment/route/directions，只重跑 exact coordination。

| N | C* | Ccoord | Cref | search_gap | scheduler_gap | total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.0 | 3.0 | 3.0 | 0.0 | 0.0 | 0.0 |
| 2 | 3.0 | 3.0 | 3.0 | 0.0 | 0.0 | 0.0 |
| 3 | 3.0 | 3.0 | 3.0 | 0.0 | 0.0 | 0.0 |
| 4 | 3.0 | 3.0 | 3.0 | 0.0 | 0.0 | 0.0 |

四例均 exact `OPTIMAL`、search `COMPLETED`、final certified，并验证 `total ≈ search_gap + scheduler_gap`。实现对 `search_gap<0` 或 `Ccoord>Cref+tolerance` 直接报错，不截断。

## K. N=20/50/100 development smoke

脚本 `scripts/profile_phase2b1.py` 使用 nominal `ScientificConfig`、seed `20260928`、deterministic development-only synthetic geometry；不是 formal benchmark。下表保留 0.2/1/5 秒真实 wall-clock 行为（iteration 在 reference 完成后才计入，因此可能越过 deadline）：

| N | budget | runtime | initial/best Cmax | improvement | iterations | cand/s | Nref (Nref/s) | scheduler mean/p95 | deadlock | valid/accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.2 | 0.236 | 755.463 / 755.463 | 0 | 0 | 0 | 0 / 0 | 0.229 / 0.229 | n/a | n/a |
| 20 | 1 | 1.209 | 755.463 / 755.463 | 0 | 2 | 105.91 | 4 / 3.31 | 0.231 / 0.234 | 0% | 32.03% / 4.88% |
| 20 | 5 | 5.044 | 755.463 / 755.463 | 0 | 10 | 126.89 | 20 / 3.97 | 0.229 / 0.238 | 0% | 32.81% / 4.29% |
| 50 | 0.2 | 1.541 | 1896.204 / 1896.204 | 0 | 0 | 0 | 0 / 0 | 1.504 / 1.504 | n/a | n/a |
| 50 | 1 | 1.543 | 1896.204 / 1896.204 | 0 | 0 | 0 | 0 / 0 | 1.507 / 1.507 | n/a | n/a |
| 50 | 5 | 7.910 | 1896.204 / 1896.204 | 0 | 2 | 16.18 | 4 / 0.506 | 1.546 / 1.591 | 0% | 37.50% / 4.17% |
| 100 | 0.2 | 6.472 | 3607.315 / 3607.315 | 0 | 0 | 0 | 0 / 0 | 6.342 / 6.342 | n/a | n/a |
| 100 | 1 | 6.482 | 3607.315 / 3607.315 | 0 | 0 | 0 | 0 / 0 | 6.350 / 6.350 | n/a | n/a |
| 100 | 5 | 6.553 | 3607.315 / 3607.315 | 0 | 0 | 0 | 0 / 0 | 6.424 / 6.424 | n/a | n/a |

全部 initialization 成功、final certified。真实结论不是“搜索已加速”：N=100 单次初始化 reference 已超过 5 秒，因此该预算没有 search-stage Nref。N=50 的 5 秒预算也因完整 reference call 跨界而实耗 7.91 秒。这是下一阶段前必须面对的性能事实。

## L. 测试结果

新增 Phase 2B-1 tests：

```text
26 passed in 0.64s
```

DRL Phase 1.1 + 2A + 2B-1：

```text
109 passed in 4.26s
```

项目根目录联合 regression：

```text
481 passed, 1 skipped, 3 subtests passed in 37.75s
```

困难测试未删除。

## M. 当前主要瓶颈

1. Reference list scheduler 随 operation 数增长明显；nominal N=100 初始化一次约 6.4 秒，是当前首要瓶颈。
2. 每 candidate 的 constrained DP 与 Python canonical/block reconstruction 仍有重复工作，但量级小于 scheduler。
3. 纯原子 moves 在已平衡 synthetic instances 上未改善初始 Cmax；这是 Phase 2B-1 能力边界，不应包装为完整 ALNS 性能。
4. Wall-clock hard stop 只能在不可中断的 reference call 边界检查；长调用可越过请求预算，但 anytime 不会回填早期 checkpoint。

## N. 留给 Phase 2B-2 / Formal Scope Gate

Phase 2B-2：真正的 destroy→repair、多个 destroy/repair operators、adaptive selection；不得把当前 atomic neighborhood 冒充这些层。`TWO_OPT_STAR`、regret repair、critical removal、schedule-aware direction refinement 均仍 inactive。

Formal Scope Gate：F1 optional X_SPLIT、F2 terminal occupancy、F3 empty-route parking、F4 bounded deadlock repair。当前实现没有改变 `FORMAL_SCOPE`，也没有新增这些规则。LB_LP、HGA/WAG、MLP/GAT/PPO、ranker dataset 与 formal experiments 未实现。

```text
PHASE_2B1_STATUS = PASS
```
