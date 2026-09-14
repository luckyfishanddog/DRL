# 多机器人焊缝分配与排序：第一阶段公共科学模型 AI 交接报告

> 更新日期：2026-09-15  
> 阶段：Phase 1.1 — Reference Evaluator Closure  
> 项目目录：`D:\pybullet_test\MRTA_GA`  
> Python：`D:\pybullet_test\.venv\Scripts\python.exe`（3.11.9）  
> 当前结论：`PHASE_1_1_STATUS = PASS`

## 1. 给接手 AI 的执行摘要

第一阶段公共科学模型和 reference evaluator 已完成收尾，可以作为后续 exact micro solver 与 deterministic backbone 的公共评价基础。

当前实现包括：

- 统一科学配置及配置哈希；
- ParentWeld/WeldingBlock、WHOLE/X_SPLIT/Y_SPLIT；
- Y handover 与 frozen `x_up/x_low`；
- canonicalization 与确定性 solution identity；
- 开放路线方向 DP；
- operation timeline；
- 连续时间解析干涉与同轨不越位；
- deterministic reference list scheduler；
- independent certifier；
- official metrics；
- CandidateMove/CandidateKey；
- 独立 tiny scheduler oracle；
- adversarial、boundary 与 regression tests。

Phase 1.1 已关闭三个关键工程漏洞：

1. certifier 只检查时间、不检查相邻 operation 空间连续性；
2. caller-supplied arbitrary X_SPLIT 可以绕过正式验证；
3. tiny oracle 与 reference scheduler 共用 earliest-safe-start，无法独立发现 reference ESS/dispatch 问题。

不得把旧 V9/V10/PPO 实现当成新科学定义。后续 AI 也不得在没有研究者决定时自行冻结 optional X_SPLIT、deadlock repair、terminal occupancy 或 idle robot parking。

## 2. 仓库状态与保护要求

`DRL/` 当前在 Git 中整体显示为未跟踪目录：

```text
?? DRL/
```

仓库其他位置存在用户既有未提交工作。后续操作必须：

- 继续将本研究的新文件归类在 `DRL/`；
- 不覆盖、删除或回滚用户已有修改；
- 不使用 `git reset --hard`、`git checkout --` 或批量清理；
- 新增文件前先向用户说明必要性并取得同意。

科学方案副本：`DRL/docs/多机器人焊缝分配与排序实验方案.md`。

## 3. 科学边界

### 3.1 研究空间与机器人

```text
workspace = [0,20] × [0,12] m
R0 = 左上
R1 = 右上
R2 = 左下
R3 = 右下
```

本模型是任务层二维模型，不包含 IK、关节空间轨迹、三维本体碰撞、clearance plane、加速度或梯形速度曲线。

### 3.2 统一科学配置

`ScientificConfig` 默认值：

```text
weld_speed       = 0.0108 m/s
empty_speed      = 0.20 m/s
t_pre            = 20 s
t_post           = 30 s
min_child_length = 0.20 m
delta_x          = 0.20 m
delta_y          = 0.20 m
interference_dx  = 0.50 m
interference_dy  = 0.50 m
numeric_epsilon  = 1e-10
```

加工与移动时间：

```text
T_proc(q) = t_pre + L_q / weld_speed + t_post
T_move(i,j) = Euclidean(end_i, start_j) / empty_speed
```

开放路线语义：无 home-to-first、无 return-home；计算相邻任务 empty travel 和 SETUP/WELD/POST/WAIT。免费初始部署仍必须满足同轨初始合法顺序与理论干涉约束。

## 4. 核心数据模型

实现位置：`DRL/src/mrta_reference/model.py`

```text
ScientificConfig
ParentWeld
WeldingBlock
SplitPattern
Route
RobotRoute
Operation
ScheduleResult
ScheduleStatus
CanonicalSolution
CandidateMove
CandidateKey
OfficialMetrics
```

主要 Enum：

```text
SplitKind: WHOLE, X_SPLIT, Y_SPLIT
OperationKind: MOVE, SETUP, WELD, POST, WAIT
ScheduleStatus: FEASIBLE, DEADLOCK, INFEASIBLE, NUMERIC_FAILURE
```

默认科学配置的 canonical SHA-256：

```text
791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e
```

`ScientificConfig.scientific_hash` 只依赖科学参数值，不包含 git commit、机器名、时间戳或测试数量。

## 5. Geometry、handover 与 split

实现位置：`DRL/src/mrta_reference/geometry.py`

### 5.1 Y handover

```text
By = [6-delta_y, 6+delta_y] = [5.8, 6.2]
```

Whole eligibility：整条 parent 满足 `y >= 5.8` 时 upper rail eligible；满足 `y <= 6.2` 时 lower rail eligible；两者可同时成立。若两者都不成立，必须使用合法 Y_SPLIT。

Y_SPLIT 候选只来自 By lower boundary、`y=6` center、By upper boundary 和 legal midpoint。候选按 `t` 确定性去重，每个 child 必须满足 `length >= Lmin`；没有网格枚举。

### 5.2 Frozen x_up/x_low

每个 instance 对每条轨道按 parent midpoint-x 和加工时间权重计算最左 weighted median，再裁剪至 `[delta_x, 20-delta_x]`。某轨无 whole-eligible parent 时使用 `10 m`。同时对上下轨 eligible 的 parent 会进入两个集合。

`x_up/x_low` 是 frozen handover center，不是 hard ownership boundary。

### 5.3 Optional X_SPLIT：fail-closed

已实现给定 split point 后的 length/Lmin validation、单次 split、两 child geometry、length conservation、canonicalization 和 Bx relation。没有实现默认候选枚举规则。

正式保留 X_SPLIT 时必须显式提供 `XSplitValidator`：

- 无 validator：抛出 `ScientificAmbiguityError`；
- validator 拒绝：正式 evaluation 拒绝；
- validator 批准：才可继续 canonicalize/schedule/certify；
- 若同机器人连续 children 已先 canonicalize 回 WHOLE，则已消失的 split 不要求 validator。

生产代码没有 Bx lower/center/upper、毫米网格、旧版本规则、比例阈值或 heuristic split point。

## 6. Canonicalization

实现位置：`DRL/src/mrta_reference/solution.py`

正式评价前会重新 canonicalize，并检查：

- parent ID 唯一且覆盖完整；
- 每个 parent 恰好一个互斥 pattern；
- parent 最多 split 一次、最多两个 blocks；
- child 长度与总长度守恒；
- blocks 不重、不漏、不重复 assignment；
- workspace 边界；
- mandatory Y_SPLIT 由 geometry 重新推导；
- 同机器人相邻 children 按 `0,1` 或 `1,0` 执行均 merge 回 WHOLE；
- children 被其他任务分隔或分配给不同机器人时不 merge；
- merge 在 operation 生成前完成，不会重复收取 SETUP/POST。

Canonical JSON/hash 不包含 solution revision。不同 move 路径得到的相同逻辑解具有相同 canonical hash。

## 7. Direction DP

实现位置：`DRL/src/mrta_reference/geometry.py`

固定 route 中每个 welding block 有 forward/reverse 两个方向。开放路线初始化为 `DP1(0)=DP1(1)=0`。DP 只优化相邻 blocks 的 empty travel，不包含 home、return-home 或 scheduler waiting；tie 使用完整 orientation vector 的确定性字典序。

测试对随机 `m=1..8` route 枚举全部 `2^m` 方向，并覆盖 exact tie、reversed geometry、zero transition 和单任务 open-route。

## 8. Operation timeline 与连续时间干涉

每个非空 route 转为：

```text
SETUP -> WELD -> POST -> MOVE -> SETUP -> WELD -> POST -> ...
```

第一项任务前没有 home MOVE。协调延迟必须表示为显式 WAIT。

- MOVE：端点间二维直线匀速；
- SETUP：停在 weld start；
- WELD：沿正式方向从 weld start 到 weld end；
- POST：停在 weld end；
- WAIT：停在上一 operation 的终点；
- MOVE/WELD 均为 non-preemptive。

对两个 operation 的闭合时间重叠区间，解析求是否存在：

```text
|delta_x(t)| <= Dx + eps_num
|delta_y(t)| <= Dy + eps_num
```

没有 `dt` sampling。覆盖 moving/moving、moving/stationary、stationary/stationary、零速度、endpoint touching、parallel 与 near-parallel。

同轨额外要求：

```text
x_R0(t) + Dx <= x_R1(t)
x_R2(t) + Dx <= x_R3(t)
```

## 9. Reference scheduler

实现位置：`DRL/src/mrta_reference/scheduler.py`

确定性 list scheduler：

1. 读取每个 robot 的 next ready operation；
2. 用解析 forbidden-start intervals 求 earliest safe start；
3. 选择 ESS 最小者；
4. tie 依次使用 remaining processing load 大、current completion 大、robot ID 小；
5. 固定完整 operation；
6. 重新计算其他 ready operations；
7. 延迟写入显式 WAIT；
8. 没有有限选择时返回 DEADLOCK 和 blocker diagnostics/wait-for graph。

`reference_schedule_from_templates()` 用于固定 operation routes 的直接调度和 tiny oracle comparison，并校验 template 的运动时间、静止语义、robot identity 和空间连续性。

### Deadlock repair hook

默认没有 hidden repair。caller-supplied callback 会收到 canonical solution、ScientificConfig、原始 DEADLOCK ScheduleResult、wait-for graph 和 diagnostics。

callback 返回 `None` 时保留原 DEADLOCK；返回显式结果时由 caller 对该 policy 负责。测试中的 FEASIBLE repaired result 会再次经过 independent certifier。这只是工程接口，不是正式 bounded repair 科学策略。

## 10. Independent certifier

实现位置：`DRL/src/mrta_reference/certifier.py`

certifier 不调用 scheduler 的 conflict、ESS 或 schedule 内部检查函数，而是独立重建并验证 parent/pattern/block coverage、one split、length conservation、Lmin、pre/post count、eligibility、precedence、non-preemptive、durations、WAIT、initial order、same-rail non-passing、continuous interference、completion 和 Cmax。

Phase 1.1 新增强空间接口验证：

```text
相邻 op_k.end == op_(k+1).start
SETUP.position == weld_start(q,direction)
WELD.start/end == declared weld_start/weld_end
POST.position == weld_end(q,direction)
MOVE.start == previous operation end
MOVE.end == next SETUP position
WAIT.position == previous operation end
```

所有比较使用 numeric tolerance。篡改 WELD、POST、MOVE 或 WAIT 空间位置都会被拒绝。

## 11. Official metrics

实现位置：`DRL/src/mrta_reference/solution.py`

```text
P_r = sum(t_pre + L_q/vw + t_post)
B_proc = max(P_r) - min(P_r)
T_empty = 全部相邻 directed welding blocks 的 empty travel 总和
W_total = 全部显式 WAIT duration 总和
N_split_opt = 最终非 mandatory split parent 数量
```

正式比较：

```text
(Cmax, N_split_opt, B_proc, T_empty, W_total, deterministic ID order)
```

只有 Cmax 使用 `eps_C = 1e-9 * max(1, abs(Cmax))`；该 epsilon 只判断数值等价，不允许接受更差 Cmax。

## 12. Candidate identity

实现位置：`DRL/src/mrta_reference/candidate.py`

已支持：

```text
INTRA_RELOCATE
INTER_RELOCATE
SWAP
TWO_OPT
TWO_OPT_STAR
SPLIT_ACTIVATE
SPLIT_DEACTIVATE
SPLIT_POINT_SWITCH
```

`CandidateKey` 包含 revision、move type、排序后的 affected parent IDs、source/destination robots、route positions、split pattern ID 与 split point ID。

相同 state/key 会确定性 replay。同 iteration 先按 key 去重，再按 canonical solution hash 去重。本阶段没有 candidate ranker、bounded pool 或 neural scorer。

## 13. Tiny scheduler oracle

实现位置：`DRL/src/mrta_reference/oracle.py`

- `tiny_scheduler_oracle_from_templates()`：恰好两个 active robots，每机器人 2–4 个人工 operations；
- `tiny_scheduler_oracle()`：solution wrapper，每个 active robot 最多两条 welding blocks，约 7 operations。

oracle 完整枚举保持 robot-local precedence 的 dispatch/interleaving，并使用自己实现的连续冲突投影与 ESS。搜索核心不调用 `scheduler.earliest_safe_start()` 或 `scheduler.reference_schedule()`；reference 只在独立搜索完成后用于 comparison。

固定 fixtures：

- Case A：多 operation，reference 与 oracle 可行且 gap=0；
- Case B：二者可行，但 reference Cmax 大于 oracle best Cmax；
- Case C：reference DEADLOCK，oracle 找到可行 dispatch；
- Case D：所有 dispatch alternatives 均不可行，双方都不伪造 FEASIBLE。

输出包括 oracle/reference status、best/reference Cmax、feasibility agreement、scheduler-only gap 和 explored dispatch count。

## 14. Adversarial test coverage

当前测试覆盖：

- zero/near-zero geometry、exact/just-below Lmin；
- Y duplicate `t`、point ID mismatch、workspace boundary；
- duplicate parent、missing/double-assigned block；
- child 正反顺序 merge、分隔/跨机器人不 merge；
- equivalent solution canonical hash；
- direction DP `m=1..8` exhaustive comparison；
- analytic moving/moving、moving/stationary、stationary/stationary；
- threshold、endpoint、parallel、near-parallel；
- same-rail violation without ordinary 2D conflict；
- finite ESS、explicit WAIT、deterministic replay；
- initial illegal order、DEADLOCK diagnostics、wait-for graph、no default repair；
- wrong Cmax/direction/durations；
- missing SETUP/POST、hidden idle gap、same-robot overlap；
- cross-robot interference、same-rail passing；
- SETUP/WELD/POST/MOVE/WAIT 空间瞬移攻击；
- X_SPLIT validator approve/reject/missing；
- candidate replay 与 canonical duplicate detection；
- scientific config hash stability；
- tiny oracle A–D cases。

## 15. Scientific ambiguities

以下问题仍未冻结，必须与普通 coding bug 分开处理。

### 15.1 Optional X_SPLIT candidate enumeration

尚无唯一正式规则。不得自行使用 Bx lower/center/upper、毫米网格、旧 V9/V10 规则、ratio threshold 或 heuristic point。当前通过 `XSplitValidator` fail-closed。

### 15.2 Bounded deadlock repair

尚未冻结 repair trigger、priority alternatives、search order、tie breaking、per-trigger/per-instance budget 和 repaired result acceptance rule。这些会改变正式 Cmax，因此当前默认不 repair。

### 15.3 Terminal occupancy

尚未决定最终 POST 完成后机器人是否持续在终点占位至 global Cmax。

当前实现语义：机器人在最终显式 POST 的 `end_time` 后停止参与 interference，不添加 terminal WAIT。该行为是现状，不代表科学规则已冻结。

### 15.4 Empty-route robot parking/occupancy

尚未定义空 route 机器人的 parking point 与占位时间。

当前实现语义：empty-route robot 不生成 operation、不分配任意 parking position、不参与 interference，completion 为 `0.0`。没有偷偷禁止 empty route。

## 16. 当前未包含的内容

```text
initial solution heuristic
bounded candidate pool
SA-OI-ALNS
simulated annealing acceptance
GNN/GAT/MLP/ranker
PPO
HGA
WAG+VNS
exact MRTA solver
lower bound
TRAIN/VALIDATION/TEST datasets
formal experiments
```

## 17. 文件结构

```text
DRL/
  README.md
  pyproject.toml
  docs/
    多机器人焊缝分配与排序实验方案.md
    第一阶段公共科学模型与Reference_Evaluator_AI交接报告_20260914.md
  src/mrta_reference/
    __init__.py
    model.py
    geometry.py
    solution.py
    candidate.py
    scheduler.py
    certifier.py
    oracle.py
  tests/
    test_model_geometry.py
    test_canonical_candidate.py
    test_interference_scheduler.py
    test_certifier_oracle.py
```

## 18. 安装与测试

Editable install：

```powershell
Set-Location 'D:\pybullet_test\MRTA_GA\DRL'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pip install --no-build-isolation --no-deps -e .
```

Phase 1.1 suite：

```powershell
Set-Location 'D:\pybullet_test\MRTA_GA\DRL'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
```

2026-09-15 最新实测：

```text
61 passed in 1.72s
```

全仓联合命令：

```powershell
Set-Location 'D:\pybullet_test\MRTA_GA'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
```

Phase 1.1 完成时实测：

```text
433 passed, 1 skipped, 3 subtests passed in 38.55s
```

## 19. 后续 AI 接手规则

1. 先运行第 18 节测试，确认当前基线。
2. 以 `DRL/src/mrta_reference` 为公共 evaluator，不把旧 PPO/V9/V10 逻辑混入。
3. 若接入旧数据或算法，使用显式 adapter，不污染科学核心。
4. 不为 optional X_SPLIT 或 bounded deadlock repair 发明规则。
5. 不添加 hidden terminal WAIT 或任意 idle parking。
6. 不使用 coarse time sampling。
7. 不把 DEADLOCK 当成 mathematical INFEASIBLE。
8. 后续 exact micro solver 应独立于 reference list scheduler，并报告 scheduler-only gap。
9. 新增文件前先确认必要性，且新增文件必须归类在 `DRL/`。

## 20. 冻结判定

Phase 1.1 验收项均已满足：空间连续性、X_SPLIT fail-closed、独立 oracle、多 operation cases、dispatch-loss fixture、无 hidden repair、未擅自处理 terminal/idle、最小复现配置、全量测试和无回退均已有实现与证据。

仍存在的四项 scientific ambiguities 已通过接口或明确现状隔离，不阻止进入下一阶段，但下一阶段不得假装它们已经冻结。

```text
PHASE_1_1_STATUS = PASS
```
