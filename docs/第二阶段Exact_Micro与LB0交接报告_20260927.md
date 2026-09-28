# 第二阶段 Exact Micro 与 LB0 交接报告

> 日期：2026-09-27  
> 阶段：Phase 2A — Exact Micro Validation Backbone + Analytical LB0  
> 状态：PASS  
> 当前 exact scope：`EXACT_Y_SCOPE_CURRENT_SEMANTICS`

> Phase 2B0 核对（2026-09-28）：本报告保留 Phase 2A 实测记录；科学声明以主实验方案为准。文中 exact/OPTIMAL 是当前离散空间与 dispatch+ESS generator 的枚举结论，连续时间全局性仍须通过主方案的 EXACT_SCHEDULER_VALIDITY_GATE，不能仅由现有 tests 或 certifier 推出。

## A. Baseline commit 与测试

外层仓库 baseline commit：

```text
0935ad8a723057084be190e31c6be4733ce3cb62
```

实现前实测：

```text
DRL Phase 1.1: 61 passed in 1.72s
联合仓库:       433 passed, 1 skipped, 3 subtests passed in 49.02s
```

外层工作树在接手前已有大量修改/删除/未跟踪内容，且 `DRL/` 整体在外层 Git 中为未跟踪目录。本轮没有清理、重置、覆盖或修改任何 `DRL/` 之外的用户工作。

## B. 新增文件

本轮只新增六个已确认必要的文件：

```text
DRL/src/mrta_exact/__init__.py
DRL/src/mrta_exact/lower_bounds.py
DRL/src/mrta_exact/scheduler.py
DRL/src/mrta_exact/solver.py
DRL/tests/test_exact.py
DRL/docs/第二阶段Exact_Micro与LB0交接报告_20260927.md
```

冻结的 `DRL/src/mrta_reference/` 未修改。没有新增 CLI、配置、fixture 数据文件、hash/baseline/gate 或缓存文件。

## C. Exact scope 精确定义

所有结果均属于：

```text
EXACT_Y_SCOPE_CURRENT_SEMANTICS
```

该 scope 的 pattern domain：

- whole eligible parent：`WHOLE` 加全部现有科学模型定义的合法 `Y_SPLIT`；
- whole 不 eligible parent：仅合法 mandatory `Y_SPLIT`；
- `X_SPLIT` 明确拒绝。

占位与调度语义严格保持 Phase 1.1 当前行为：

- final `POST` 结束后机器人不再参与 interference；
- empty route 不生成 operation、不参与 interference、completion 为 0；
- 不添加 terminal WAIT；
- 不发明 parking point；
- 不执行 deadlock repair。

因此本文的 `Cmax_exact` 只能解释为 `Cmax_OPT_Y_CURRENT`，不是论文最终完整问题的 `Cmax_OPT`。

## D. Analytical LB0

`AnalyticalLowerBound` 返回：

```text
total_base_processing_work
lb_work
max_parent_parallel_lb
lb0
```

实现公式：

```text
W0 = sum_i(t_pre + L_i / weld_speed + t_post)
LB_work = W0 / 4
LB_parent_i = t_pre + t_post + L_i / (2 * weld_speed)
LB0 = max(LB_work, max_i LB_parent_i)
```

空 parent 集合显式返回四个 `0.0`，不会产生 NaN。测试覆盖单/多 parent 手算、空实例和参数修改。

## E. Independent exact coordination scheduler

`exact_schedule_from_templates()` 支持 0–4 个 active robots：

1. 校验 robot identity、operation duration/geometry 和 route 空间连续性；
2. 独立校验 first-task proximity 与 same-rail initial order；
3. 递归枚举所有保持 robot-local precedence 的跨机器人 dispatch interleaving；
4. 对每个 next operation 独立使用连续时间线性约束与 Fourier–Motzkin 投影求 earliest safe start；
5. 延迟以显式 `WAIT` 表示，并单独验证 WAIT 占位；
6. 无可行 next operation 的分支终止；
7. 对完整分支按 `Cmax` 和确定性 schedule JSON 选代表解。

该搜索核心没有调用：

```text
reference_schedule()
reference_schedule_from_templates()
mrta_reference.scheduler.earliest_safe_start()
```

它只复用公共 dataclass/config；连续冲突投影与 ESS 在 `mrta_exact.scheduler` 内独立实现，不使用 coarse time sampling。

## F. Current Y-scope exact micro solver 枚举层次

`solve_exact_micro()` 完整枚举：

```text
parent pattern
-> welding-block robot assignment
-> per-robot route permutation
-> canonicalize + canonical_hash deduplication
-> every route block's forward/reverse direction
-> exact coordination schedule
-> independent certify_schedule
```

Assignment 只使用 `robot_is_eligible()` 允许的 rail robots。方向枚举为 canonical routes 上完整的 `2^m`，没有把 direction DP 的 top-1 当作 exact proof。

最终 `ExactResult` 保存 solution、directions/operations、official metrics、LB0、reference comparison、枚举计数、dispatch states 和 wall-clock 统计。

默认 `max_parents=4`；更大实例必须显式 `allow_larger=True`。支持 `max_states`、`max_schedule_evaluations`、`time_limit`，任一 limit 命中均返回 `LIMIT_REACHED` 且 `optimal=False`。

## G. 安全 pruning

无剪枝 `exhaustive_mode=True` 除 exact canonical duplicate suppression 外不删除分支，且 exact scheduler 也关闭 incumbent pruning。

B&B 模式仅使用：

- canonical hash 完全重复解去重；
- canonical route 已确定的单机器人 processing load 仅在严格高于 incumbent Cmax 容差带时剪枝；
- exact scheduler 内 `completion + remaining local operation duration` 下界与当前 coordination incumbent 比较；Cmax 可能打平时，只有累计 WAIT 已不可能改善 incumbent 才剪枝。

没有 route heuristic、conflict proxy、direction proxy、learned bound 或 LP bound。LB0 本轮用于解释和校验，没有用于搜索排序、ranker 或参数选择，也没有用它提前终止从而影响等值代表解。

## H. Brute-force 与 B&B 对照

开发配置：`weld_speed=empty_speed=1`，`t_pre=t_post=1`，其余参数保持 `ScientificConfig` 默认。

| N | 模式 | 状态 | Cmax | schedule evaluations | dispatch states | pruned | runtime (s) |
|---:|---|---|---:|---:|---:|---:|---:|
| 2 | brute | OPTIMAL | 3.0 | 24 | 408 | 0 | 0.0213 |
| 2 | B&B | OPTIMAL | 3.0 | 16 | 108 | 14 | 0.0055 |
| 3 | brute | OPTIMAL | 8.0 | 192 | 24,912 | 0 | 1.8758 |
| 3 | B&B | OPTIMAL | 8.0 | 144 | 1,392 | 246 | 0.0646 |

两种模式 status 与最优 Cmax 完全一致。

## I. Tiny oracle 与 exact scheduler

| Case | exact status | exact Cmax | tiny oracle Cmax | reference status/Cmax | scheduler gap |
|---|---|---:|---:|---|---:|
| E1 | FEASIBLE | 2.0 | 2.0 | FEASIBLE / 2.0 | 0.0 |
| E2 | FEASIBLE | 13.6982470367 | 13.6982470367 | FEASIBLE / 14.1622776606 | 0.4640306238 |
| E3 | FEASIBLE | 15.5287634621 | 15.5287634621 | DEADLOCK / — | — |
| E4 | INFEASIBLE | — | — | DEADLOCK / — | — |

E1–E4 均与 independent tiny oracle 一致。E3 证明 reference DEADLOCK 不能当成 mathematical infeasibility；E4 中两种实现都未生成可行 schedule，其 INFEASIBLE 仅覆盖各自 dispatch+ESS 枚举域，不是一般连续时间不可行性证明。

## J. M1–M7 与受控 N=4 真实结果

同样使用上述快速开发配置。

| Fixture | 关键决策/验证 | Cmax exact | LB0 | gap to LB0 | reference |
|---|---|---:|---:|---:|---|
| M1 | 单 parent WHOLE；手算 `1+2+1` | 4.0 | 3.0 | 33.33% | FEASIBLE / 4.0 |
| M2 | 两 parent 分给 R2/R3；同机器人对照为 13.0 | 3.0 | 2.5 | 20.00% | FEASIBLE / 3.0 |
| M3 | 固定 assignment 下短路线 15.0，差路线 22.0 | 15.0/22.0 对照 | — | — | 两者均 certified |
| M4 | 固定 route 全 forward 为 9.0，第二 block reverse 为 8.0 | 8.0 | — | — | certified |
| M5 | optional `Y_SPLIT:MIDPOINT`；WHOLE-only 为 6.0 | 4.0 | 4.0 | 0.00% | FEASIBLE / 4.0 |
| M6 | 固定 templates 的 exact coordination 优于 reference | 13.6982470367 | — | — | FEASIBLE / 14.1622776606 |
| M7 | mandatory `Y_SPLIT:BY_CENTER`，children 分属上下轨 | 3.0 | 3.0 | 0.00% | FEASIBLE / 3.0 |

M2 的完整 solver 最优 routes 为 R2=`left`、R3=`right`。M3/M4 的三 parent 完整 solver 最优为 R2=`a,b`、R3=`c`，R2 directions=`0,1`，`Cmax=8.0`；上表的 15/22 和 8/9 是为分别隔离 route 与 direction 因果而运行的固定 assignment 对照。

M5 统计：2 个 pattern combinations、20 个 assignment states、24 个 raw route states、8 个 canonical duplicates；B&B 实际执行 56 次 schedule evaluation。它证明 optional Y pattern enumeration 确实改变最优值。

受控 N=4：

```text
status                 = OPTIMAL
Cmax exact             = 3.0
LB0                    = 3.0
reference Cmax         = 3.0
pattern combinations   = 1
assignment states      = 16
route states           = 36
direction states       = 288
schedule evaluations   = 288
dispatch states        = 5,120
pruned states          = 2,034
runtime total          = 0.2906 s
runtime scheduler      = 0.2067 s
certified              = true
```

2026-09-28 以 `tests/test_exact.py` 的 `_parallel_solution(4).parents`、`FAST_CONFIG` 和 `solve_exact_micro(..., exhaustive_mode=False)` 重新核对，全部离散计数与上述结果一致：dispatch states=5120、pruned states=2034；本次 runtime total=0.3393 s、scheduler=0.2381 s。上表时间仍保留原阶段测量值。

## K. LB0 检查

所有具有 exact optimum 的 parent fixtures 均满足：

```text
LB0 <= Cmax_exact + numeric tolerance
```

观测范围为 M1 的 `3.0 <= 4.0` 到 M5/M7/N4 的等号成立。没有发现 LB0 高于已知 optimum 的反例。

## L. Reference scheduler-only gap

完整 solver 的上述最优 micro fixtures 中 reference 与 exact coordination gap 均为 0。独立 coordination fixture M6/E2 提供严格正 gap：

```text
reference Cmax         = 14.16227766056838
exact Cmax             = 13.698247036726606
scheduler-only gap     = 0.46403062384177396
```

E3 另提供：

```text
reference status = DEADLOCK
exact status     = FEASIBLE
```

没有为 DEADLOCK 情况填写伪 gap。

## M. 测试命令与最终结果

新增 exact 测试：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests\test_exact.py
```

```text
22 passed in 2.78s
```

DRL 全量：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

```text
83 passed in 5.10s
```

联合仓库：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
```

```text
455 passed, 1 skipped, 3 subtests passed in 43.34s
```

联合回归曾有一次旧 `test_cplr_relocate_backend_equivalence` 的 neighbor-stream hash 瞬时不一致；该失败发生在外层旧 CPLR 路径，隔离复跑通过，随后完整联合复跑也全部通过。`mrta_exact` 不被该测试导入，本轮未修改外层 CPLR 代码或测试。

## N. 当前性能瓶颈

主瓶颈是方向/route 组合数乘以 coordination interleaving 数。N=3 brute 已产生 24,912 个 dispatch states，而 B&B 为 1,392；N=4 受控 fixture 的 B&B 为 5,120。当前实现的目标是 correctness backbone，不适用于直接求 N=6 常规实例。

下一步若优化，只能加入有证明的 dominance/lower bound 或 memoization；不得以 heuristic direction、reference priority 或 coarse time grid 删除 exact 分支。

## O. 仍未解决的 scientific ambiguities

本轮没有自行决定：

1. optional X_SPLIT candidate enumeration；
2. bounded deadlock repair；
3. terminal occupancy after final POST；
4. empty-route robot parking/occupancy。

这些问题在正式数据冻结、正式算法比较和论文最终 `Cmax_OPT` 声明前仍必须冻结。当前 terminal/empty-route 行为只是 scope identity 的一部分，不是最终科学结论。

## P. 是否可进入 minimal SA-OI-ALNS

可以进入仅使用同一 `EXACT_Y_SCOPE_CURRENT_SEMANTICS` 的 minimal SA-OI-ALNS development 阶段。exact micro solver 可用于小实例回归、方向/route/operator 正确性检查和 current-scope optimality gap。

在四项歧义冻结前，不得：

- 把 current-scope exact optimum 称为最终完整问题 global optimum；
- 用 current-scope gap 支撑正式论文结论；
- 将 X_SPLIT、terminal/parking 或 repair 的缺省行为静默带入正式数据集。

```text
PHASE_2A_STATUS = PASS
```
