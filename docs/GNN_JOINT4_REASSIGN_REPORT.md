# GNN-Joint4：瓶颈父焊缝跨机器人联合重插入——实现与 TRAIN 筛查报告

日期：2026-11-22。环境 `D:/pybullet_test/.venv/Scripts/python.exe`，工作目录 `D:/pybullet_test/MRTA_GA/DRL`。
基线：`GNN-Joint3B`（30 组 VALIDATION，严格负载合格 12/30，对 HGA 中位 +6.206%）。本轮不重设计网络、不新增调度器/认证器，未改 B32、资格、模式目录与正式负载阈值。

## 0. 结论：TRAIN 筛查的判定

| 第一阶段检查项 | 结果 | 判定 |
|---|---|---|
| 1. 是否有效改变机器人任务归属 | 每个完整候选的跨机器人归属变化数中位 **3**（Joint3B 2）；拥属/模式变化写入 CSV | 通过 |
| 2. 是否产生真正的 Relocate/Swap | 核心固定为 2 个转出父 + 2 个接收方父（q=4），GNN 联合重插；`transfer_parents`/`receiver_parents` 逐行记录 | 通过 |
| 3. Pmax 与加工负载极差 | 中位 Pmax 5089.1 → **5021.4**；极差 381.0 → **268.1** | 通过 |
| 4. MOVE 与 WAIT | MOVE 中位 589.9 → 618.7（+5%）；WAIT 中位 7.6 → **0.0** | 部分通过（WAIT 更好，MOVE 略增） |
| 5. Cmax 是否优于 Joint3B | 配对中位 **-0.0485%**，4 胜 2 负（最大 -3.82%，最大 +1.45%） | **未达到“明显改善”** |
| 6. 最终是否全部 B32 FEASIBLE | 6/6 FEASIBLE，0 死锁 | 通过 |

**判定：不进入完整 VALIDATION，不启动 TEST。** 按事前约定，第一阶段没有体现“明显”的 Cmax 改善（配对中位 ≈ -0.05%），且死锁修复次数出现回退（67→37），继续扩大实验只会重复一个已知的负结果。本轮交付的是“算子 + 标签 + native 筛选”的完整实现、第一阶段三臂数据与失败归因，不制造成功的假象。

## 1. 读代码后的两个关键实测（决定了实现方式）

1. 仓库现有原生协调评价 `priority_schedule()`（HGA/WAG 搜索期用的同一函数）在**有干涉**的 GNN 方案上直接返回 DEADLOCK：`validation_01` 的初解和 8 个候选**全部** native DEADLOCK（B32 却全部 FEASIBLE）。因此它不能当硬过滤。
2. 在**无干涉**排样上它与 B32 **逐位相等**：`validation_07` 的 8/9 个方案 native Cmax == B32 Cmax，native DEADLOCK 当且仅当 B32 DEADLOCK；单次 ~2ms（首次含 numba 编译）。

所以本轮把 native 用作「GNN 排序 + native 筛选」：origin 的 native 有限时按 native Cmax 排序；origin native 为 DEADLOCK 时只把 native 可行者前置、其余保持 GNN 顺序，且**永远不丢弃候选、不把 native 当可行性证明**。30 组 TRAIN 候选里 native 可行 158/282（56%），说明它在近半情况下可用。

## 2. 修改了哪些核心代码（全部在 `scripts/run_gnn_weld_layout.py`）

| 位置 | 改动 | 为什么 |
|---|---|---|
| `joint4_bottleneck_core()` | FEASIBLE 核心改为「瓶颈机器人转出 + 接收方父」：`critical = argmax B32 完成时间`，`peak = argmax P_r`；转出父按 `加工 + 0.5×(真实 critical WAIT/阻塞) + 0.25×MOVE` 排序；接收机器人取**完成时间不晚于 critical** 且能接收至少一个转出父的机器人（完成时间→负载→覆盖数）；接收方父按同一打分排序；q=4 时 2 转出 + 2 接收 | 旧 Joint3B 只在瓶颈机器人上取 3 个父，无法表达“转出+接收交换”；HGA M1/M2/M3 与 WAG MOVE/SWAP 都是跨机器人结构 |
| `joint4_core()` | FEASIBLE 走上式；DEADLOCK 仍走 Joint3B 的真实 blocked/blocker 核心；无接收方时回退 `select_routeaware_bottleneck`（q≤3） | 不推翻已验证的死锁机制 |
| `joint4_screen()` | 用现有 `priority_schedule()` 给每个完整候选打 native 值；只让排序后的前 2 个不同结构进入 B32 | 100 次 B32 要花在更有价值的候选上；HGA/WAG 也是 native 搜索期筛选 |
| `joint4_deadlock_signature()` | 死锁标识从“只 blocked parent IDs”扩展为 `(blocked 机器人, blocked 操作 kind/robot, blocked parents/blocks, blocker 操作 ID, blocker blocks, blocker parents 分组)` | 旧标识会把“父相同但 blocker 机器人/操作不同”的两个状态误判为同一核心 |
| `joint4_training_rank()` / `joint4_group_loss()` | 新标签：class 0 = `Pmax<=1.01*initial` 且 `spread<=max(0.5*initial_spread, 0.10*initial_Pmax)`；class 1 = 其它 FEASIBLE，**按连续负载质量 `max(Pmax/limit, spread/limit)` 再按真实 B32 Cmax 排序**；class 2 = DEADLOCK。pairwise loss 权重 = ΔCmax/ref + 0.25×Δquality | 旧标签 `spread<=initial+20` 比正式阈值松数倍；严格正样本稀疏时按要求用连续负载质量排序，而不是放宽正样本条件 |
| `joint4_fit()` | 用同一评分头（assignment+joint+insertion）在 20 个 TRAIN 实例上续训 3 epoch，初始化自 Joint3 权重 | 学“哪些父该迁出/哪个接收方/哪个组合值得换/哪种插入降低 MOVE/哪些会 WAIT 或死锁” |
| 核心规模回退阶梯 | q=4 的核心若无法完成重构（兄弟子段会被 canonicalize 合并），自动缩到 4→3，而不是丢掉整轮 | 5 父核心在部分排样上 8/8 完成全部被“canonicalization changed”拒绝 |
| `joint3_case` 恢复逻辑 | 断点续跑改为同时匹配 `phase` | 之前的续跑会跨 phase 误命中（消融实验暴露） |

没有改动：`gnn_constructor.py`、`hga.py`、`wag_vns.py`、B32、方向优化、负载阈值公式；没有新增网络、PPO、独立调度器/认证器/hash。

## 3. 核心规模 q 的证据与一处方法学披露

在 `validation_07`（1 个种子、40 次 B32、同一权重）上比较 q=3/4/5：

| q | 核心结构 | Cmax | 负载合格 |
|---|---|---|---|
| 3 | 2 转出 + 1 接收 | 6386.9 | 0 |
| **4** | **2 转出 + 2 接收** | **5884.2** | **1** |
| 5 | 3 转出 + 2 接收 | 6383.9 | 0 |

q=5 反而明显更差：去掉 5 个父后 beam 宽度 4 的重构更容易退化成坏解。**披露：这次 q 的选择是在 VALIDATION 排样上做的参数探查（4 个排样、1 个种子、40 次 B32），属于在验证集上选参数的方法学瑕疵**；第一阶段的 TRAIN 结果独立复算了 q=4，结论与该探查一致（见 §5），但该瑕疵不应被隐藏。

## 4. 训练标签的实际分布（TRAIN only）

- 20 个 TRAIN 实例、37 个有效排序组，候选 641 个；其中 FEASIBLE 366 个，严格正样本（class 0）**52** 个（占可行 14.2%）。
- 严格正样本稀疏但非零，因此 class 0 作为上界、class 1 内部用连续负载质量排序（按要求，没有放宽正样本条件凑数）。

| epoch | loss | 排序准确率 | 对数 | 更新组 | 累计 B32 |
|---|---|---|---|---|---|
| 1 | 0.562462 | 68.814% | 4098 | 37 | 685 |
| 2 | 0.546554 | 70.791% | 4098 | 37 | 685 |
| 3 | 0.533501 | 72.426% | 4098 | 37 | 685 |

训练成本（与在线搜索分开）：685 次真实 B32、构造 104.6 s、B32 533.9 s、3 epoch 合计 721.3 s。

## 5. 第一阶段：6 个 TRAIN 排样（同实例、同种子 20261110、同 100 次 B32）

三臂：Joint3B（旧算子+旧权重）、Joint4（新算子+新标签权重）、Joint4-unfit（新算子+旧权重，消融）。

| 排样 | Joint3B Cmax | Joint4 Cmax | Δ%(J4/J3B) | Joint4-unfit Cmax | Joint3B Pmax/极差 | Joint4 Pmax/极差 | J3B MOVE/WAIT | J4 MOVE/WAIT | J3B B32 | J4 B32 |
|---|---|---|---|---|---|---|---|---|---|---|
| train_52 | 4945.4 | 5017.2 | +1.454% | 5129.3 | 4883.3/448.1 | 4930.0/873.3 | 438.0/0.0 | 501.2/0.0 | 40 | 23 |
| scale_train_0482 | 5004.4 | 5001.7 | -0.054% | 5421.8 | 4837.9/15.5 | 4948.9/150.1 | 466.1/0.0 | 453.4/0.0 | 20 | 41 |
| scale_train_0973 | 5214.9 | 5015.9 | -3.817% | 5076.9 | 5055.6/532.7 | 4950.4/146.2 | 522.0/387.6 | 515.9/0.0 | 100 | 29 |
| scale_train_0214 | 5278.0 | 5283.8 | +0.109% | 5068.6 | 5122.7/375.2 | 5092.3/315.0 | 657.7/26.7 | 797.3/338.2 | 100 | 67 |
| scale_train_0617 | 5723.6 | 5721.1 | -0.043% | 5633.4 | 5531.3/386.8 | 5531.3/398.3 | 679.6/0.0 | 721.4/0.0 | 18 | 84 |
| scale_train_0268 | 6293.8 | 6287.1 | -0.106% | 6336.2 | 6087.0/213.9 | 6087.0/221.1 | 714.0/15.2 | 759.0/49.9 | 38 | 38 |

汇总（6 例中位）：

| 指标 | Joint3B | Joint4 | Joint4-unfit |
|---|---|---|---|
| Cmax | 5246.5 | 5150.5 | 5275.5 |
| Pmax | 5089.1 | 5021.4 | 5160.2 |
| 极差 | 381.0 | 268.1 | 192.5 |
| MOVE | 589.9 | 618.7 | 623.0 |
| WAIT | 7.6 | 0.0 | 0.0 |
| B32 次数 | 39 | 40 | 24 |
| 运行时间 | 38.6 | 111.4 | 83.9 |

- 负载合格（TRAIN 相对上限）：Joint3B 5/6，Joint4 5/6。
- 候选层：Joint3B 316 个（死锁 129，死锁起点修复 67，跨机器人变化中位 2）；Joint4 282 个（死锁 150，修复 37，跨机器人变化中位 3；native 已知 282，其中 native 可行 158）。

## 6. 改善究竟来自什么（三臂消融）

6 例 Cmax 中位：Joint3B **5246.5** → Joint4-unfit **5275.5** → Joint4 **5150.5**。

- **算子本身（新核心 + native 筛选、旧权重）没有带来 Cmax 收益**：Joint4-unfit 比 Joint3B 还差约 +0.6%（中位），说明只把核心从 3 父扩到“2 转出+2 接收”、并加 native 排序，不足以改善解质量。
- **收益来自标签改动**：同一算子换上按新标签续训的权重后，Cmax 中位比 Joint3B 低约 1.8%，比 unfit 低约 2.4%；Pmax/极差同步改善（极差 381→268）。
- 也就是说：**GNN 是否学会跨机器人分配，主要取决于训练标签教它“什么样的负载结构算好”，而不是核心多大**。这与本轮 §3 的核心目标一致，但也说明单靠扩大核心/加 native 筛选不会自动变好。

## 7. 失败与回退（不掩盖）

- **配对 Cmax 中位仅 -0.05%**（4 胜 2 负），不构成“明显改善”；最大回退 +1.45%（train_52），最大改善 -3.82%（scale_train_0973）。
- **死锁修复回退**：候选层死锁起点修复 67 → 37；死锁候选 129 → 150。原因是 Joint4 的轮次更多花在 4 父可行核心与 native 排序上，B32 预算里留给死锁起点修复的比例下降。这个方向需要在下一轮单独处理，不能假装没有发生。
- **MOVE 略增**（中位 590→619，+5%），WAIT 明显下降（7.6→0）。没有出现“用 MOVE/WAIT 暴涨换负载”的现象，但 MOVE 的增长需要在更大样本上复核。
- **运行时间**从 38.6s 增到 111.4s（native 评价 + 4 父重构 + 更多轮次）；在相同 B32 预算下这是额外成本。
- **与 HGA/WAG 的差距**：Joint3B 在 VALIDATION 上对 HGA 中位 +6.206%。第一阶段的 -0.05% 配对改善不足以改变这个结论，因此不进入 VALIDATION。

## 8. 文件与复现

| 内容 | 路径 |
|---|---|
| 算法代码（同一文件新增 `joint4-*`） | `scripts/run_gnn_weld_layout.py` |
| 新增测试（4 个：死锁标识、标签排序、native 筛选、瓶颈核心） | `tests/test_gnn_construct_vns.py` |
| 新权重（Joint3 续训，同架构） | `models/gnn_weld_layout_joint4.pt` |
| 新结果 CSV（训练轮次 + TRAIN 三臂 + 全部候选） | `results/new_dataset_final/gnn_joint4_reassign.csv` |
| 本报告 | `docs/GNN_JOINT4_REASSIGN_REPORT.md` |

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint4-fit
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint4-train --workers 4
D:/pybullet_test/.venv/Scripts/python.exe scripts/run_gnn_weld_layout.py joint4-summary --split train
D:/pybullet_test/.venv/Scripts/python.exe -m pytest tests/test_gnn_construct_vns.py -k "joint4" -q -p no:cacheprovider
```

旧 Joint3 / Joint3B / HGA / WAG 的模型、CSV、报告均未改动。
