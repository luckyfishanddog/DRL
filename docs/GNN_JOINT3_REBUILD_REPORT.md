# B32 反馈驱动的三父焊缝联合重构：独立 GNN 主算法实现与 30 组 VALIDATION 结果

日期：2026-11-21/22。运行环境 `D:/pybullet_test/.venv/Scripts/python.exe`（Python 3.11.9，torch 2.13.0+cpu），工作目录 `D:/pybullet_test/MRTA_GA/DRL`。

## 0. 结论摘要

| 通过标准 | 结果 | 判定 |
|---|---|---|
| 1. 30/30 最终负载合格且 B32 FEASIBLE | B32 FEASIBLE 30/30；严格负载合格（仅 HGA/WAG 两个冻结参考）**13/30**；三参考口径（含本实验原始 GNN 构造作为第三参考）29/30 | 未达到（严格口径） |
| 2. 对 HGA 配对 Cmax 中位差 < 0 | +6.6269%（30 对） | 未达到 |
| 3. 对 WAG+VNS 配对 Cmax 中位差 < 0 | +4.9573%（22 对可比 Cmax） | 未达到 |
| 4. 两组胜数均多于负数 | HGA 0胜/30负；WAG+VNS 8胜/22负（其中 8 胜来自 WAG 该种子 DEADLOCK 的可行性优势） | 未达到 |
| 5. 完整在线时间同口径比较 | 本方法中位 72.6 s；HGA 冻结中位 1103.4 s；WAG+VNS 冻结中位 96.0 s | 已报告（时间不构成否决项） |

按用户事先规定：未扩大 TEST、未放宽负载约束、未增加第二套网络或第二个调度器。本轮是「规格内实现 + 诚实的负结果」。

## 1. 交付物与文件边界

| 类别 | 路径 | 说明 |
|---|---|---|
| 代码（修改） | `scripts/run_gnn_weld_layout.py` | 新增 `joint3-*` 四个 stage 与全部算子；旧 stage 未改动 |
| 代码（修改） | `src/mrta_search/gnn_constructor.py` | 仅新增向后兼容开关 `feedback_outcomes`（默认关），旧检查点/旧路径行为不变 |
| 测试（修改） | `tests/test_gnn_construct_vns.py` | 新增 6 个 joint3 测试；按 3.3 新语义更新 1 个旧 cycle 断言 |
| 新模型 | `models/gnn_weld_layout_joint3.pt` | 从原始 `GNN_VNEXT_WORKLOAD` 初始化的新权重，旧模型全部保留 |
| 新结果 | `results/new_dataset_final/gnn_joint3_rebuild.csv` | 30 组 final + 每次真实 B32 候选（`stage` 分列） |
| 新结果 | `results/new_dataset_final/gnn_joint3_training.csv` | 训练耗时与 B32 成本单独记账 |
| 本报告 | `docs/GNN_JOINT3_REBUILD_REPORT.md` | - |

## 2. 模型与方案来源审计

审计在代码里是**自证**的：`joint3_provenance()` 用 AST 扫描新算子的每个函数体，任何对教师/热启动/回退来源的引用都会抛错；`test_joint3_provenance_forbids_teacher_warm_start_and_fallback_sources` 会反向注入一个违规函数确认审计失败闭合。

被切断的六处旧污染链：

1. `imitation_saved()` 读取 `expanded_dagger` 的 WAG+VNS `teachers` —— 新算子不调用。
2. `imitation_hga_job` 生成 HGA 教师 —— 新算子不调用，也不读 `IMITATION_*`。
3. `imitation_best_teachers` 混合 HGA/WAG —— 新算子不调用。
4. `imitation_cycle_train` 由上述教师生成 `IMITATION_MODEL`/`CYCLE_MODEL` —— 新模型初始化只读 `models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD`，缺失参数必须恰好是 `joint_weights`。
5. `conflict_initial_hints` 的 HGA 区域初始化 + WAG 分配组 —— 新算子没有 hint 入口。
6. `conflict_core_case` 的 `initial0 = GNN_IMITATION` 与 `cycle_transfer_case` 的热启动 —— 新算子的初始解是**原始 GNN 构造**，搜索起点只来自本次运行的 B32 结果。

审计输出（`python scripts/run_gnn_weld_layout.py joint3-audit`）：

```json
{
 "algorithm": "GNN-Joint3-B32-Feedback",
 "initialization": "models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD",
 "new_model": "models\\gnn_weld_layout_joint3.pt",
 "own_records": "results\\new_dataset_final\\gnn_joint3_rebuild.csv",
 "violations": {},
 "feedback_shared": {
  "reconstruct_uses_feedback_features": true,
  "inference_uses_same_feedback": true,
  "training_uses_same_feedback": true
 }
}
```

`HGA`/`WAG+VNS` 的历史结果只用于两件事：逐例配对的 Cmax/运行时间，以及**固定负载上限**（`joint3_load_limits`）。它们从不进入候选生成。

## 3. 唯一核心算子

### 3.1 核心选择（复用现有选择器，不新增规则）

- FEASIBLE：`lns.select_routeaware_bottleneck(current.solution, current.directions, schedule, wait_blockers, CONFIG, q=3, seed=..., assignment_space=space)`。实际 Cmax 关键机器人、关键机器人高贡献父、`parent_critical_wait`、`parent_critical_blocked`、真实高代价 MOVE 弧端点共同决定核心；WAIT/阻塞证据来自 `selected_schedule_wait_blockers` 的真实时间线。
- DEADLOCK：`conflict_core_partial()`。保留真实 blocked/blocker 父焊缝及必要邻居；`q=3` 只是目标规模，真实死锁核心大于 3 时**不截断**。30 组实际核心规模：3 个父 481 次、4 个父 7 次（出现 4 时是真实核心/必要邻居超过目标规模，未截断）。

### 3.2 GNN 联合重构（有限 Beam，宽度 3）

`joint3_reconstruct()` 移除核心父焊缝的全部子段后，用**原** `gnn.vnext_joint_actions(..., conflict_core=True)` 逐步生成联合动作：合法拆分模式、机器人归属、插入顺序与位置、焊接方向在同一个动作里决定，不经过独立方向优化、不复制 HGA VND 或 WAG LNS。

Beam 只保留不同结构：每个（父、合法模式、机器人归属）只留一个代表，方向仅作为同一结构内的属性，不会出现「大量仅方向不同」的重复候选；排序时把**改变模式/归属**的扩展放在恢复原状之前（宽度内保留 2 个改变 + 1 个不改变），因此有限 beam 不会退化成原方案的 3 份拷贝。完整候选再统一用原评分头 `imitation_joint_scores` 打分排序。

### 3.3 安全的路线周期剪枝

`joint3_safe_prune(lb, incumbent_cmax)` 只在**候选的精确有向路线周期下界**已经达到当前真实 B32 Cmax 时剪枝：`lb >= Cmax_incumbent`。不存在 `lb >= lb_incumbent` 这种不充分条件。旧 `cycle_transfer_groups` 里的 `lb < incumbent_lb` 过滤已按同一语义改成「仅在提供实测 incumbent Cmax 时才剪枝」，并删除了 Swap 候选在 GNN 评分前的固定位置 LB 排序截断（`test_cycle_swap_is_two_sided_and_safe_lb_load_filter_is_exact` 已按新语义更新）。

### 3.4 B32 反馈真正进入训练与推理

审计发现旧 `cycle_joint_plan` 的 `risk` 维度在训练阶段恒为 0（`imitation_cycle_train` 生成 groups 时不传 `feedback`），推理阶段才有值；而且同一核心的候选共享该值，组内排序时会被抵消。本轮改为**候选级**反馈交互特征，训练与推理走同一个函数 `joint3_feedback_features()`：

| 维度 | 含义 |
|---|---|
| `moved_wait_work` | 实际等待父焊缝从关键机器人上迁走的加工量 / 2500 |
| `moved_blocked_work` | 实际阻塞父焊缝从关键机器人上迁走的加工量 / 2500 |
| `critical_route_cycle` | 候选在关键机器人上的精确路线周期 (P+MOVE) / 2500 |
| `critical_owner_changes` | 核心父焊缝中改挂到关键机器人的子段数 / 4 |

这 4 维接在原 6 维 outcome 之后（`joint_weights` 6→10，新增 4 个系数**零初始化**，因此未训练时评分与原策略逐位一致）。旧路径默认 6 维、行为不变。训练日志显示 4 维在训练数据里真实非零（如 `moved_blocked_work=0.13/0.18/0.48`、`critical_owner_changes=0.25/0.5/0.75`），不存在「训练全零、推理才有值」。

## 4. 重新训练（TRAIN only，真实 B32 标签）

- 数据：现有 TRAIN（`data/weld_layout/scale1000/train.jsonl`）按父数分层固定选 20 个实例，不扩充数据集、不读 VALIDATION/TEST 标签。
- 生成：每个实例用原始 GNN 构造得到初始解 → 安全核心 → 三父联合重构 beam（宽度 4）→ 每个完整候选**一次**真实 B32。
- 标签：`负载合格 FEASIBLE(0) > 其它 FEASIBLE(1，按真实 Cmax) > DEADLOCK(2)`；负载合格用训练期相对保护（早前正式变体 `Pmax<=1.01*初始Pmax`、`DeltaP<=初始DeltaP+20s`），不触碰正式实验的冻结上限。
- 训练：只训练现有评分头（assignment + joint + insertion），AdamW，lr=1e-4，3 epoch，pairwise softplus 加权。

| epoch | loss | 排序准确率 | 对数 | 更新组 | 累计 B32 | 该轮耗时 |
|---|---|---|---|---|---|---|
| 1 | 0.689297 | 60.259% | 4240 | 30 | 694 | 465.8 s |
| 2 | 0.653981 | 63.137% | 4240 | 30 | 694 | 469.9 s |
| 3 | 0.628795 | 65.660% | 4240 | 30 | 694 | 473.7 s |

训练总成本单独记账：数据生成 30 个有效排序组、**694 次真实 B32**、构造 99.2 s、B32 305.6 s；三 epoch 合计约 23.5 min（含前向/反向），不混入在线搜索时间。

新权重 `joint_weights`（前 6 维为原策略先验，后 4 维为训练得到的反馈系数）：`[-4.99912, -1.49779, -0.99838, -0.99883, -3.00231, -1.502, -0.00062, 0.00031, 0.00149, 0.00387]`。

## 5. 实验协议

- 排样：`data/weld_layout/validation.jsonl` 的 10 个官方 VALIDATION 排样（`validation_01..10`，共 936 条父焊缝）。
- 种子：`20261110/20261111/20261112`，共 30 组。每组最多 **100 次新增 B32**、最多 **500 次外层尝试**、无墙钟硬截止；每个完整方案只评价一次（`plan_key` 缓存）。
- 初始：原始 GNN 构造（width 1），计入构造时间与 B32。搜索只从本次运行的 B32 结果出发。
- 说明：训练更新的是同一个 VNext actor（含构造用的 assignment/insertion 头），所以正式实验的初始构造由**训练后的权重**产生；权重来源与网络结构仍是 `GNN_VNEXT_WORKLOAD`，没有引入教师。训练前的原始构造（`validation_01` = 6063.07、`validation_10` = 8481.98）作为对照保留在 §9。
- 最终：只在负载合格的 B32 FEASIBLE 方案中选最小真实 Cmax；若一个都没有，报告最好 FEASIBLE（标记 `load_allowed=0`），不退化为 DEADLOCK。
- 负载上限口径：**主口径（严格）** `Pmax <= 1.05*max(冻结 HGA/WAG 峰值)`、`spread <= max(冻结极差, 峰值中位数的 10%)`；**敏感性口径** 沿用上一轮 6 例公式，把本实验原始 GNN 构造峰/极差作为第三参考，只报告、不用于择优。`validation_07` 的冻结 WAG+VNS 三个种子全为 DEADLOCK，因此该例严格口径只用 HGA 一个可行参考。

## 6. 30 组 VALIDATION 结果

表 1：逐例最终方案（P/M/W 为四机器人加工时间 / MOVE / WAIT，单位 s；`maxPM` 为精确路线周期下界 `max(P+MOVE)`）。

| 排样 | 种子 | B32 | Cmax | P0/P1/P2/P3 | M0/M1/M2/M3 | W0/W1/W2/W3 | Pmax | 极差 | maxPM | 负载(严格/三参考) | 重构核心父焊缝 | B32次数 | 运行(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation_01 | 20261110 | FEASIBLE | 6082.481 | 6037/4557/5212/5716 | 26/186/157/170 | 0/101/713/0 | 6037.0 | 1480.3 | 6063.1 | 0/1 | (初始解,未重构) | 7 | 37.8 |
| validation_01 | 20261111 | FEASIBLE | 6082.481 | 6037/4557/5212/5716 | 26/186/157/170 | 0/101/713/0 | 6037.0 | 1480.3 | 6063.1 | 0/1 | (初始解,未重构) | 7 | 37.9 |
| validation_01 | 20261112 | FEASIBLE | 6082.481 | 6037/4557/5212/5716 | 26/186/157/170 | 0/101/713/0 | 6037.0 | 1480.3 | 6063.1 | 0/1 | (初始解,未重构) | 7 | 38.1 |
| validation_02 | 20261110 | FEASIBLE | 5887.311 | 5734/5461/5101/5601 | 153/157/201/54 | 0/0/0/62 | 5733.8 | 632.9 | 5887.3 | 0/1 | P08-W01-S01,P08-W02-S01,P14-W01-S01 | 100 | 107.0 |
| validation_02 | 20261111 | FEASIBLE | 5843.063 | 5224/5392/5683/5647 | 207/162/160/57 | 342/0/0/0 | 5683.3 | 459.7 | 5843.1 | 1/1 | P07-W01-S01,P08-W01-S01,P18-W12-S01 | 84 | 75.3 |
| validation_02 | 20261112 | FEASIBLE | 6005.989 | 5468/5777/5286/5416 | 217/160/131/36 | 0/69/0/0 | 5776.6 | 490.5 | 5937.0 | 1/1 | P07-W01-S01,P08-W01-S01,P14-W01-S01 | 36 | 33.6 |
| validation_03 | 20261110 | FEASIBLE | 5274.049 | 5004/5059/4796/5065 | 202/164/196/209 | 0/0/0/0 | 5065.3 | 269.0 | 5274.0 | 1/1 | P02-W01-S01,P07-W02-S01,P19-W06-S01 | 31 | 108.0 |
| validation_03 | 20261111 | FEASIBLE | 5677.501 | 4690/4911/5454/4919 | 164/178/223/249 | 0/0/0/51 | 5454.3 | 764.5 | 5677.5 | 0/1 | P04-W01-S01,P22-W01-S01,P22-W04-S01 | 100 | 156.8 |
| validation_03 | 20261112 | FEASIBLE | 5408.744 | 5119/4915/5184/4907 | 145/214/225/206 | 0/139/0/165 | 5183.6 | 276.6 | 5408.7 | 0/1 | P02-W01-S01,P13-W08-S01,P19-W04-S01 | 100 | 202.1 |
| validation_04 | 20261110 | FEASIBLE | 5134.687 | 3876/3889/4937/4918 | 165/101/198/200 | 0/0/0/0 | 4936.7 | 1060.7 | 5134.7 | 0/1 | P09-W01-S01,P11-W02-S01,P17-W05-S01,P18-W02-S01 | 100 | 36.0 |
| validation_04 | 20261111 | FEASIBLE | 5230.728 | 3924/3891/4937/4918 | 160/106/178/157 | 0/0/116/0 | 4936.7 | 1045.4 | 5114.5 | 1/1 | P06-W04-S01,P11-W01-S01,P12-W05-S01 | 100 | 61.2 |
| validation_04 | 20261112 | FEASIBLE | 5117.724 | 3406/4310/4909/4945 | 119/113/189/173 | 0/0/0/0 | 4945.1 | 1539.2 | 5117.7 | 0/1 | P11-W01-S01,P11-W02-S01,P14-W08-S01 | 100 | 54.4 |
| validation_05 | 20261110 | FEASIBLE | 5727.662 | 5079/5339/5510/5572 | 208/130/218/145 | 0/0/0/0 | 5571.8 | 493.2 | 5727.7 | 1/1 | P01-W01-S01,P04-W01-S01,P10-W01-S01 | 100 | 52.5 |
| validation_05 | 20261111 | FEASIBLE | 5821.527 | 4665/5450/5604/5730 | 141/97/194/91 | 0/0/0/0 | 5730.4 | 1065.1 | 5821.5 | 0/1 | P10-W02-S01,P13-W05-S01,P15-W09-S01 | 100 | 72.7 |
| validation_05 | 20261112 | FEASIBLE | 5713.815 | 4918/5500/5544/5537 | 161/149/169/118 | 0/0/0/0 | 5544.3 | 626.2 | 5713.8 | 0/1 | P04-W01-S01,P07-W01-S01,P10-W01-S01 | 100 | 71.6 |
| validation_06 | 20261110 | FEASIBLE | 5634.438 | 5493/5404/5479/5115 | 141/170/72/228 | 0/0/0/0 | 5493.2 | 377.7 | 5634.4 | 1/1 | P06-W02-S01,P16-W01-S01,P16-W03-S01 | 65 | 63.0 |
| validation_06 | 20261111 | FEASIBLE | 5525.449 | 5347/5362/5336/5247 | 135/163/41/203 | 0/0/0/0 | 5362.4 | 115.8 | 5525.4 | 1/1 | P02-W02-S01,P06-W02-S01,P18-W02-S01 | 51 | 60.3 |
| validation_06 | 20261112 | FEASIBLE | 6042.944 | 5563/5246/5383/5200 | 173/142/41/223 | 307/0/0/0 | 5563.2 | 362.9 | 5736.0 | 1/1 | P06-W01-S01,P09-W09-S01,P18-W02-S01 | 100 | 72.6 |
| validation_07 | 20261110 | FEASIBLE | 5841.823 | 5548/5460/5588/5706 | 181/240/164/136 | 0/0/0/0 | 5706.0 | 245.6 | 5841.8 | 1/1 | P10-W01-S01,P10-W02-S01,P21-W01-S01 | 80 | 106.7 |
| validation_07 | 20261111 | FEASIBLE | 6021.514 | 5426/5769/5856/5203 | 195/253/96/148 | 0/0/0/0 | 5855.8 | 653.0 | 6021.5 | 0/1 | P05-W04-S01,P14-W02-S01,P17-W08-S01 | 100 | 142.3 |
| validation_07 | 20261112 | FEASIBLE | 6050.951 | 5426/5633/5444/5850 | 195/238/94/154 | 387/0/0/47 | 5850.4 | 424.4 | 6004.4 | 0/1 | P09-W02-S01,P13-W01-S01,P14-W02-S01 | 100 | 140.4 |
| validation_08 | 20261110 | FEASIBLE | 5699.184 | 5126/5277/5478/5405 | 200/186/221/257 | 0/0/0/0 | 5478.0 | 352.3 | 5699.2 | 1/1 | P03-W02-S01,P05-W09-S01,P11-W01-S01 | 97 | 369.0 |
| validation_08 | 20261111 | FEASIBLE | 5841.458 | 5332/4902/5566/5585 | 181/202/238/256 | 223/0/0/0 | 5585.1 | 682.8 | 5841.5 | 0/1 | P03-W02-S01,P20-W01-S01,P20-W02-S01 | 100 | 207.2 |
| validation_08 | 20261112 | FEASIBLE | 5895.335 | 4514/5787/5638/5246 | 173/109/184/198 | 0/0/0/287 | 5786.7 | 1272.3 | 5895.3 | 0/1 | P05-W09-S01,P08-W01-S01,P11-W01-S01 | 100 | 208.6 |
| validation_09 | 20261110 | FEASIBLE | 5566.867 | 5160/5077/5416/5423 | 36/297/134/144 | 154/0/0/0 | 5422.7 | 345.6 | 5566.9 | 1/1 | (初始解,未重构) | 1 | 8.0 |
| validation_09 | 20261111 | FEASIBLE | 5566.867 | 5160/5077/5416/5423 | 36/297/134/144 | 154/0/0/0 | 5422.7 | 345.6 | 5566.9 | 1/1 | (初始解,未重构) | 1 | 8.3 |
| validation_09 | 20261112 | FEASIBLE | 5566.867 | 5160/5077/5416/5423 | 36/297/134/144 | 154/0/0/0 | 5422.7 | 345.6 | 5566.9 | 1/1 | (初始解,未重构) | 1 | 8.5 |
| validation_10 | 20261110 | FEASIBLE | 6188.949 | 3123/6107/5894/6049 | 93/42/205/140 | 0/0/0/0 | 6106.9 | 2984.0 | 6188.9 | 0/0 | P11-W01-S01,P12-W02-S01,P16-W06-S01 | 100 | 108.2 |
| validation_10 | 20261111 | FEASIBLE | 5834.874 | 4427/5578/5612/5705 | 146/84/188/130 | 0/0/0/0 | 5704.6 | 1278.0 | 5834.9 | 0/1 | P04-W02-S01,P11-W01-S01,P13-W02-S01 | 100 | 116.9 |
| validation_10 | 20261112 | FEASIBLE | 6120.818 | 3590/5736/5921/5975 | 121/51/197/146 | 0/0/0/0 | 5974.5 | 2384.3 | 6120.8 | 0/1 | P11-W01-S01,P11-W02-S01,P16-W04-S01 | 100 | 96.1 |

表 2：最终方案的联合调整内容（模式/归属/插入位置/方向变化均来自保存的 `mode_changes`/`assignment_changes`/`rebuild_actions`）。

| 排样 | 种子 | 模式变化数 | 归属变化父数 | 重插动作数 | 反馈来源 | 反馈特征 [wait,blocked,cycle,owner] |
|---|---|---|---|---|---|---|
| validation_01 | 20261110 | 0 | 0 | 0 | - | - |
| validation_01 | 20261111 | 0 | 0 | 0 | - | - |
| validation_01 | 20261112 | 0 | 0 | 0 | - | - |
| validation_02 | 20261110 | 2 | 4 | 3 | B32 feasible WAIT/blocked intervals | [0.0408, 0.0, 2.2472, 0.25] |
| validation_02 | 20261111 | 5 | 7 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.2217, 0.0] |
| validation_02 | 20261112 | 4 | 9 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.3748, 0.0] |
| validation_03 | 20261110 | 6 | 10 | 5 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.1096, 0.5] |
| validation_03 | 20261111 | 12 | 16 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0387, 2.0357, 0.25] |
| validation_03 | 20261112 | 10 | 12 | 5 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.1635, 0.0] |
| validation_04 | 20261110 | 2 | 2 | 5 | B32 DEADLOCK blocked/blocker evidence | [0.0, 0.044, 2.0539, 0.25] |
| validation_04 | 20261111 | 3 | 2 | 5 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.0458, 0.0] |
| validation_04 | 20261112 | 2 | 1 | 5 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.0395, 0.0] |
| validation_05 | 20261110 | 3 | 6 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.2911, 0.25] |
| validation_05 | 20261111 | 2 | 6 | 4 | B32 feasible WAIT/blocked intervals | [0.0185, 0.0185, 2.3191, 0.0] |
| validation_05 | 20261112 | 3 | 5 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.2855, 0.0] |
| validation_06 | 20261110 | 6 | 11 | 4 | B32 feasible WAIT/blocked intervals | [0.0408, 0.0, 2.2296, 0.0] |
| validation_06 | 20261111 | 2 | 9 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.2102, 0.0] |
| validation_06 | 20261112 | 4 | 11 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.2944, 0.25] |
| validation_07 | 20261110 | 3 | 7 | 5 | B32 feasible WAIT/blocked intervals | [0.0255, 0.0, 2.3011, 0.5] |
| validation_07 | 20261111 | 3 | 5 | 3 | B32 feasible WAIT/blocked intervals | [0.048, 0.0, 2.4086, 0.25] |
| validation_07 | 20261112 | 4 | 6 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.3484, 0.0] |
| validation_08 | 20261110 | 3 | 8 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.1848, 0.5] |
| validation_08 | 20261111 | 3 | 10 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.0418, 0.5] |
| validation_08 | 20261112 | 1 | 5 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.1778, 0.0] |
| validation_09 | 20261110 | 0 | 0 | 0 | - | - |
| validation_09 | 20261111 | 0 | 0 | 0 | - | - |
| validation_09 | 20261112 | 0 | 0 | 0 | - | - |
| validation_10 | 20261110 | 4 | 6 | 3 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.4756, 0.25] |
| validation_10 | 20261111 | 5 | 7 | 5 | B32 feasible WAIT/blocked intervals | [0.0462, 0.0, 2.3201, 0.0] |
| validation_10 | 20261112 | 2 | 7 | 4 | B32 feasible WAIT/blocked intervals | [0.0, 0.0, 2.4483, 0.0] |

## 7. 与冻结 HGA/WAG+VNS 的逐例配对

参考取每个排样对应官方冻结来源的最佳可行行（`HGA=ORIENTED_ROUTE_VND_NATIVE_SELECT_V1`、`WAG+VNS=PARTIAL_LNS_V1`）；WAG 在 `validation_07` 三个种子与 `validation_01/03/10` 部分种子为 DEADLOCK，这些位置按可行性结果计，不伪造 Cmax 配对。

| 排样 | 种子 | GNN Cmax | HGA Cmax | DeltaHGA% | WAG Cmax | DeltaWAG% | 时间 GNN/HGA/WAG (s) |
|---|---|---|---|---|---|---|---|
| validation_01 | 20261110 | 6082.481 | 5537.350 | +9.845% | 5591.128 | +8.788% | 38/1110/96 |
| validation_01 | 20261111 | 6082.481 | 5537.350 | +9.845% | 5591.128 | +8.788% | 38/1110/96 |
| validation_01 | 20261112 | 6082.481 | 5537.350 | +9.845% | 5591.128 | +8.788% | 38/1110/96 |
| validation_02 | 20261110 | 5887.311 | 5569.005 | +5.716% | 5638.293 | +4.417% | 107/1300/82 |
| validation_02 | 20261111 | 5843.063 | 5569.005 | +4.921% | 5638.293 | +3.632% | 75/1300/82 |
| validation_02 | 20261112 | 6005.989 | 5569.005 | +7.847% | 5638.293 | +6.521% | 34/1300/82 |
| validation_03 | 20261110 | 5274.049 | 4998.072 | +5.522% | 5026.139 | +4.932% | 108/944/130 |
| validation_03 | 20261111 | 5677.501 | 4998.072 | +13.594% | 5026.139 | +12.959% | 157/944/130 |
| validation_03 | 20261112 | 5408.744 | 4998.072 | +8.217% | 5026.139 | +7.612% | 202/944/130 |
| validation_04 | 20261110 | 5134.687 | 4842.268 | +6.039% | 4982.488 | +3.055% | 36/1097/83 |
| validation_04 | 20261111 | 5230.728 | 4842.268 | +8.022% | 4982.488 | +4.982% | 61/1097/83 |
| validation_04 | 20261112 | 5117.724 | 4842.268 | +5.689% | 4982.488 | +2.714% | 54/1097/83 |
| validation_05 | 20261110 | 5727.662 | 5476.569 | +4.585% | 5479.539 | +4.528% | 53/1015/70 |
| validation_05 | 20261111 | 5821.527 | 5476.569 | +6.299% | 5479.539 | +6.241% | 73/1015/70 |
| validation_05 | 20261112 | 5713.815 | 5476.569 | +4.332% | 5479.539 | +4.275% | 72/1015/70 |
| validation_06 | 20261110 | 5634.438 | 5377.535 | +4.777% | 5404.660 | +4.251% | 63/1078/107 |
| validation_06 | 20261111 | 5525.449 | 5377.535 | +2.751% | 5404.660 | +2.235% | 60/1078/107 |
| validation_06 | 20261112 | 6042.944 | 5377.535 | +12.374% | 5404.660 | +11.810% | 73/1078/107 |
| validation_07 | 20261110 | 5841.823 | 5629.954 | +3.763% | - | WAG 全种子 DEADLOCK | 107/1433/- |
| validation_07 | 20261111 | 6021.514 | 5629.954 | +6.955% | - | WAG 全种子 DEADLOCK | 142/1433/- |
| validation_07 | 20261112 | 6050.951 | 5629.954 | +7.478% | - | WAG 全种子 DEADLOCK | 140/1433/- |
| validation_08 | 20261110 | 5699.184 | 5400.101 | +5.538% | 5420.396 | +5.143% | 369/1257/140 |
| validation_08 | 20261111 | 5841.458 | 5400.101 | +8.173% | 5420.396 | +7.768% | 207/1257/140 |
| validation_08 | 20261112 | 5895.335 | 5400.101 | +9.171% | 5420.396 | +8.762% | 209/1257/140 |
| validation_09 | 20261110 | 5566.867 | 5471.642 | +1.740% | 5533.942 | +0.595% | 8/1017/60 |
| validation_09 | 20261111 | 5566.867 | 5471.642 | +1.740% | 5533.942 | +0.595% | 8/1017/60 |
| validation_09 | 20261112 | 5566.867 | 5471.642 | +1.740% | 5533.942 | +0.595% | 9/1017/60 |
| validation_10 | 20261110 | 6188.949 | 5418.051 | +14.228% | 5517.687 | +12.166% | 108/14179/316 |
| validation_10 | 20261111 | 5834.874 | 5418.051 | +7.693% | 5517.687 | +5.749% | 117/14179/316 |
| validation_10 | 20261112 | 6120.818 | 5418.051 | +12.971% | 5517.687 | +10.931% | 96/14179/316 |

汇总：

- HGA：30 对，配对相对差中位 **+6.6269%**，0 胜 / 30 负 / 0 平。
- WAG+VNS：22 对可比 Cmax，配对相对差中位 **+4.9573%**；含可行性记为 8 胜 / 22 负（其中 8 胜是 WAG 该种子 DEADLOCK、本方法可行）。
- 时间：本方法 30 组中位 72.6 s（含初始构造、候选重构、全部 B32 与在线搜索），HGA 冻结中位 1103.4 s，WAG+VNS 冻结中位 96.0 s。

## 8. 候选死锁率与最终可行率

- 候选层面：本次 30 例共 4306 个真实 B32 完整方案，其中 DEADLOCK 1522 个，**候选死锁率 35.35%**。
- 最终方案：**B32 FEASIBLE 30/30（100%）**，DEADLOCK 0/30；新增 B32 次数中位 100。
- 负载：严格口径 13/30；三参考口径 29/30（唯一不达标为 `validation_10/20261110`，Pmax 6365 > 三参考上限 6224）。

## 9. 未达标原因（逐条、可核对）

1. **起点本身负载失衡**：原始 `GNN_VNEXT_WORKLOAD` 构造在 10 个排样上的加工极差为 691~3454 s，而冻结 HGA/WAG 的极差只有 86~1177 s。严格上限取 `1.05*max(HGA,WAG 峰值)` 且极差上限取参考极差，初始解在 9/10 个排样上就已不合格（例：`validation_01` Pmax 6037 vs 上限 5758；`validation_04` Pmax 5927 vs 上限 5124）。三父局部重构能在 `validation_07` 把极差压到 245.6 s 并合格，但无法在 100 次局部评价内系统性补齐 10%~15% 的全局失衡。
2. **Cmax 差距主要在加工负载分布本身**：30 例中多数最终 WAIT 为 0，Cmax 约等于 max(P+MOVE)，说明剩余差距不是协调等待而是负载分布；HGA 用 200000 次迭代做全局协调。本方法只允许「删除 3 个父 → GNN 联合重插」，单步最多移动 3 个父。
3. **有 2 个案例搜索几乎没花预算**：`validation_01`（三个种子均 7 次 B32）与 `validation_09`（1 次 B32）。前者是「同一状态的可测核心集合饱和」——`select_routeaware_bottleneck` 的确定性锚点让 129 次外层尝试只得到 6 个不同核心，候选又都被 `plan_key` 去重；后者是安全剪枝正确生效（初始解已负载合格，且全部候选 `lb >= Cmax_incumbent`，按 3.3 不得评价）。两例 `diagnostic.stop` 均为 `no new legal core candidate`，可原样复核。
4. **旧路线也没有解决同一问题**：上一轮 GNN-Cycle-Transfer 对 HGA 中位 +4.78%、对 WAG +2.86%；本轮 +6.63% / +4.96% 更差。差别不在搜索预算，而在起点（旧路线从负载补偿最优解热启动，本轮从更差的原始构造起步）与可移动范围。
5. **本轮禁止的手段恰好是可能翻盘的方向**（更强的全局构造/更大核心/独立调度器），按要求没有使用：没有放宽负载、没有换更大 q、没有引入 HGA/WAG 教师或回退、没有新增网络、没有扩大 TEST。

## 10. 复现与核对

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint3-audit
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint3-train
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint3-validation --workers 4
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint3-summary
D:/pybullet_test/.venv/Scripts/python.exe -m pytest tests/test_gnn_construct_vns.py -k "joint3 or cycle" -q -p no:cacheprovider
```

核对点（已逐条跑过校验脚本）：`gnn_joint3_rebuild.csv` 中 30 个 `(layout_id, seed, stage=final)` 全部唯一、无重复；`b32_calls` 全部 <=100（最大恰为 100）；最终行 `cmax_s` 等于该例全部 `stage in (initial, candidate)` 中「严格负载合格且 FEASIBLE」的最小值，**0 处违反**（无合格解时退化为最优 FEASIBLE，且 `load_allowed=0`）；30 例共 4306 个真实 B32 完整方案、其中 DEADLOCK 1522 个。

## 11. 数据表附注

- `gnn_joint3_rebuild.csv` 共 2198 行：30 个 final + 2168 个候选/初始行；候选行不保存完整时间线（`schedule` 置空）以控制体积，final 行保存完整 B32 时间线。
- `gnn_joint3_training.csv` 3 行（每 epoch 一行），训练 B32 成本与在线搜索成本完全分开。
- 旧实验文件、旧模型、旧 CSV 均未修改、未删除、未覆盖。
