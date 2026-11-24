# WAG+VNS 与 HGA 对照算法实现、性能与实验报告

创建：2026-11-11；更新：2026-11-12（Asia/Shanghai）  
项目：`D:\pybullet_test\MRTA_GA\DRL`  
运行环境：`D:\pybullet_test\.venv\Scripts\python.exe`  
用途：提供给 GPT 或研究人员，分析算法适配的忠实性、计算瓶颈、实验有效性与下一步优化方向。

**版本说明：第 1–12 节记录基准 `86e32c7` 的上一轮结果；本次审查与修正见第 13 节。旧结果不能作为修正后算法的实测结果。**

**最新状态（2026-11-12，Asia/Shanghai）：新版 WAG 已完成90次正式运行（VALIDATION可行22/30、TEST可行60/60）。用户确认的显式方向HGA已实现，最新63项相关测试通过，来源为 `ORIENTED_COORDINATED_VND_V1`。三个既有代表排样各1000子代的最终诊断为47.47/68.91/121.94s，相比同一显式定义初版下降约29%–38%，仍未达到本轮完整预算效率目标。本版本的三个200000有效子代验证及后续正式实验尚未完成，目前没有后台完整预算长跑。不能把历史版本完整结果或本版本短测当作本轮验证通过；最新修改、计时、撤回原型与可比性说明见13.9末尾。**

**第13.7节的54项测试及吞吐数字属于此前全路线方向DP版本；显式编码改变方向搜索定义，旧实验不可混入新来源。第13.9节按阶段列出62项及最终63项测试与新计时。固定预算仍为mu20/lambda10/alpha20/200000有效子代和完整六邻域，搜索内不调用B32或GNN；本阶段未新增文件。**

实验记录环境为 Python 3.11.9、NumPy 2.4.6、Numba 0.68.1。数值内核使用 float64，未启用 fastmath；精确性测试按浮点容差比较，不声称逐位相同。

## 1. 基准86e32c7的历史结论与完成范围

本轮已经实际修改代码并执行实验，结果如下：

- **WAG+VNS**：完成全部 VALIDATION 的 30 次运行和 TEST 的 60 次运行，全部达到 500 轮 VNS。VALIDATION 的 B32 可行率为 19/30，TEST 为 60/60。
- **HGA**：完成小、中、大三个 VALIDATION 代表排样各一次运行，均达到 200,000 次有效子代，最终 B32 均可行。但单例耗时约 27–38 分钟，效率仍未达到本轮目标。
- **未完成部分**：HGA 剩余 27 次 VALIDATION 和全部 60 次 TEST 没有执行。不能把三个代表样本写成完整 VALIDATION，也没有 HGA 的正式 TEST 配对结论。
- 本对照组修复轮次没有修改数据、焊缝几何、机器人资格、拆分合法性、B32 或 GNN 主算法。
- 已有实验记录为 37 项 WAG/HGA/共同评价测试、194 项 GNN 相关测试通过。这是该修复轮次的验证记录，不表示之后其他开发修改已经重新通过同一组检查。

需要区分两种判断：WAG 已有完整预算和完整 TEST 的可报告结果；是否足以称为原论文的严格复现，还需要审查第 10 节列出的项目适配细节。HGA 已证明能够完成规定预算并生成 B32 可行方案，但尚不具备本轮要求的完整实验覆盖和运行效率。

## 2. 研究对象与共同实验条件

研究对象为 20 m × 12 m 平台上的双滑轨四悬臂机器人焊缝任务分配与排序，机器人为 R0、R1、R2、R3。目标是最小化最大完工时间：

\[
\min C_{\max}=\min\max(C_0,C_1,C_2,C_3).
\]

| 条件 | 本轮口径 |
|---|---|
| 数据 | 当前固定排样，不重新生成焊缝 |
| 科学配置 | `ScientificConfig()` |
| 正式范围 | `FORMAL_SCOPE_V2` |
| 合法任务模式 | 现有 WHOLE、Y_SPLIT、X_SPLIT |
| 最短合法子焊缝 | 0.50 m；此限制针对拆分子段，不代表删除较短合法 WHOLE 父任务 |
| 焊接速度 | 0.0108 m/s |
| 空走速度 | 0.20 m/s |
| 准备／后处理 | 每个 canonical 焊接块 20 s／30 s |
| 最终调度评价 | 原有 B32 |
| 独立认证 | `certify=False`，不重复认证 |
| 搜索时间停止条件 | 无人工墙钟截止；正式入口使用 `math.inf` |

| 数据集 | 排样数 | 父焊缝数 |
|---|---:|---:|
| TRAIN | 40 | 3,749 |
| VALIDATION | 10 | 936 |
| TEST | 20 | 2,060 |

WAG 完整 VALIDATION/TEST 的随机种子为 `20261110、20261111、20261112`。HGA 三个代表排样均使用 `20261110`，尚未完成另外两个种子。

不同算法可以选择不同合法拆分，因此焊接总长度固定，但 canonical 块数量及准备、后处理总时间可能不同。不能无条件假设所有方法的四机加工时间之和完全相等。

## 3. 必须区分的三个目标

| 层次 | 定义与用途 | 是否包含冲突等待 |
|---|---|---|
| 路线周期代理目标 | 四条独立开放路线的加工与最优方向空走时间最大值；用于路线改进及 HGA VND | 不包含 WAIT |
| 快速原生调度目标 | 共享四机器人优先级调度器输出的 Cmax；WAG 用于搜索比较，HGA 用于完整个体评分、选择及最优解保留 | 包含其调度规则产生的 MOVE/WAIT 和干涉检查 |
| 最终 B32 目标 | 搜索结束后，对最终完整方案进行一次原 B32 调度 | 正式系统评价 |

HGA 当前 VND 并不是直接以 B32 Cmax 为邻域目标，也不是以快速原生 Cmax 搜索到局部最优。其完整 VND 指在原路线周期目标、α 近邻、机器人资格、拆分一致性和首任务合法条件下，六个邻域均无严格改进后终止。

每次完整预算运行结束后只对最终完整方案调用一次 B32。搜索过程中不调用 B32。CSV 的 `native_cmax_s` 与 `cmax_s` 独立记录，不能用前者替代后者。

原生调度按每条轨道的两台机器人分别安排优先级，必要时尝试反向优先级；随后检查跨轨干涉。当前跨轨检查发现冲突后拒绝该原生候选，并不继续搜索跨轨等待调整方案。因此它不是 B32，也不是完整四机可行性判定器。

## 4. WAG+VNS 实现与修复

参考文献：`references/Novel method for welding gantry robot scheduling at shipyards.pdf`，主要为第 4、5 节。当前方法是四机器人共同模型适配版，应与原文三机器人场景区分。

| 参数 | 正式设置 |
|---|---:|
| Move / Swap 参数 n | 10 |
| LNS 破坏比例 p | 0.50 |
| VNS 迭代数 | 500 |
| Factorial 窗口 | 5 |
| Factorial 模式 | `PAPER_EXACT` |
| 主初始化 WAG 数量上限 | `None` |
| 路线组合 top-k 截断 | 不使用 |

保留的核心过程：

1. 根据共同模型的可达资格和双滑轨结构生成 WAG。
2. 每台机器人使用 Nearest Addition、Farthest Insertion，并结合 2-opt、Factorial Edge Combination 得到三条候选路线。
3. 按论文顺序构造三条改进路线：nearest 路线允许改变末焊缝；farthest 路线先保持末焊缝；第三条从该改进 farthest 路线继续允许改变末焊缝。两种改进算法交替执行至均不再改进。
4. 每个 WAG 处理 `3^4=81` 个逻辑路线组合。重复路线复用相同原生评分；其余组合按安全下界排序和剪枝。
5. 比较全部实际生成的初始 WAG，取消首个可行初解后的 `break`。
6. 每轮随机选择 Move、Swap 或 LNS，以更新后的当前最优 WAG 继续搜索，实际执行 500 轮。

路线组合下界为：

\[
LB=\max_r\left(P_r+\min_{\text{方向}}MOVE_r\right).
\]

它忽略非负等待，并放松首任务方向等限制，因此不大于相应调度 Cmax。若现有 canonicalizer 会把相邻拆分兄弟合并为 WHOLE，必须先按合并后的块重新计算下界，避免错误剪枝。当前实现已处理该情况。

计算优化包括：端点方向转移与加工时间预计算、前缀／后缀两状态 DP、2×2 min-plus 区间矩阵、局部窗口与边界更新、Numba 编译，以及完全相同输入和路线结果的复用。5 条焊缝的 Factorial 改进结果已与 `5!×2^5=3840` 种完整顺序／方向枚举在测试容差内比较。

注意：处理 81 个逻辑组合不意味着必须重复调用原生调度器 81 次。相同路线复用和数学下界剪枝属于当前实现的一部分；没有经验性 top-3 跳过。

## 5. HGA 实现与修复

参考文献：`references/1-s2.0-S0957417425009212-main.pdf`，主要为第 3.1、3.2、4.1 节。

| 参数 | 正式设置 |
|---|---:|
| 种群规模 μ | 20 |
| 增广规模 λ | 10 |
| α 近邻数量 | 20 |
| 有效子代数 | 200,000 |
| VND 邻域数 | 6 |
| VND 最大轮数限制 | `None` |
| VND 候选数限制 | `None` |

保留一个四机区域种子和其余随机插入初解。初始化向当前最短合法路线的最小增量位置插入任务；交叉复制父代 A，再用父代 B 的一条路线替换，删除重复任务，将缺失任务插入其余合法路线，以周期 Cmax 为主目标、路线增量作为同值次序。X 子任务继续遵守其固定机器人。

| 邻域 | 当前实现的主要操作 |
|---|---|
| M1 | 从最长路线 Relocate 任务 u 到另一条路线 v 之后 |
| M2 | 两条路线交换 u、v，其中一条为最长路线 |
| M3 | 两条路线交换连续任务对，其中一条为最长路线 |
| M4 | 同路线按 `(u,x),(v,y) → (u,v),(x,y)` 反转 `x..v` |
| M5 | 跨路线连接两段前缀及两段后缀，对应 `(u,v),(x,y)` |
| M6 | 跨路线交换后缀，对应 `(u,y),(v,x)` |

接受改进后重新从 M1 开始，六个邻域都没有改进才终止。没有原先两轮或 256 个候选的提前返回。完整个体使用原生冲突评分，可行个体在锦标赛和存活选择中优先；μ+λ 管理、归一化路线边距离和目标／多样性 biased fitness 均保留。

一次迭代指一个子代成功构造并完成局部搜索，构造失败不消耗该预算，初始化的 20 个个体不计入 200,000。相同子代可以复用已经完成的确定性 VND 结果，因此 200,000 次有效子代不等于 200,000 个不同解，也不等于每次都重新执行全部邻域计算。

计算优化包括：一次排样中预计算各合法模式的块、加工、端点距离、资格与近邻排序；使用轻量数组和局部连接评价；只有候选可能改善时才构造新路线；完整方案对象只在必要时生成；路线边集合复用；热点循环用 Numba 编译。

## 6. 实际实验结果

以下 Cmax 均为 B32 结果。Cmax 均值／中位数只统计该方法的可行运行；运行时间统计所有对应运行，包括失败和首次编译开销。

| 方法与范围 | 完成次数 | 实际迭代 | FEASIBLE | DEADLOCK | Cmax 中位 / 均值 (s) | 平均 / 最长耗时 (s) |
|---|---:|---:|---:|---:|---:|---:|
| WAG VALIDATION | 30/30 | 每次 500 | 19/30 | 11 | 5,499.9 / 5,467.6 | 65.9 / 299.0 |
| WAG TEST | 60/60 | 每次 500 | 60/60 | 0 | 6,719.3 / 6,679.0 | 49.7 / 78.7 |
| HGA VALIDATION 代表样本 | 3/3 代表运行；完整计划为 30 次 | 每次 200,000 | 3/3 | 0 | 4,996.7 / 5,156.0 | 1,856.6 / 2,262.5 |
| HGA TEST | 0/60 | 未执行 | 无结果 | 无结果 | 无结果 | 无结果 |

HGA 三次完整预算运行均使用种子 20261110：

| 排样 | 父焊缝数 | B32 状态 | 原生 Cmax (s) | B32 Cmax (s) | 全程耗时 (s) |
|---|---:|---|---:|---:|---:|
| validation_03 | 82 | FEASIBLE | 4,996.7 | 4,996.7 | 1,636.1 |
| validation_04 | 91 | FEASIBLE | 4,842.2 | 4,842.2 | 1,671.2 |
| validation_07 | 115 | FEASIBLE | 5,629.2 | 5,629.2 | 2,262.5 |

这三次平均耗时 30.94 分钟、最长 37.71 分钟。按照任务中“预估仍需数小时则不启动大规模批次”的要求，没有继续启动其余 27 次 HGA VALIDATION 和 60 次 TEST。

以下为可行组的平均加工负载、MOVE、WAIT。HGA 仅为三个代表样本，与 WAG 的完整集合不同，不能直接凭不同集合均值判断算法优劣。

| 方法与范围 | P0 (s) | P1 (s) | P2 (s) | P3 (s) | MOVE (s) | WAIT (s) |
|---|---:|---:|---:|---:|---:|---:|
| WAG VALIDATION | 4,963.6 | 5,076.3 | 5,306.1 | 5,325.2 | 391.4 | 177.2 |
| WAG TEST | 6,402.7 | 6,400.6 | 6,334.0 | 6,352.1 | 422.2 | 8.5 |
| HGA 三个代表样本 | 4,771.4 | 4,936.8 | 5,072.9 | 5,051.4 | 411.4 | 0.0 |

P 与 MOVE 来自完整最终方案；死锁行的 WAIT 只能表示 B32 已完成部分时间线中的等待。正式可行记录中，原生和 B32 Cmax 的差不超过约 `1e-12 s`，但 WAG 的 11 次 VALIDATION 死锁说明：原生可行不能推出 B32 可行。

## 7. 与 RULE/BEAM 和 GNN 的配对结果

配对键为排样及种子，只在双方 B32 可行时比较。相对变化定义为：

\[
\Delta\%=100\times\frac{C_{\text{对照算法}}-C_{\text{参考方法}}}{C_{\text{参考方法}}}.
\]

负值表示本报告中的 WAG 或 HGA 的 Cmax 更小。

| 集合 | 对照算法 | 参考方法 | 共同可行 | 胜/平/负 | Cmax 差中位 (s) | 相对变化中位 |
|---|---|---|---:|---:|---:|---:|
| VALIDATION | WAG | RULE/BEAM | 11 | 11/0/0 | -988.2 | -15.016% |
| VALIDATION | WAG | GNN_WIDTH1 | 14 | 14/0/0 | -914.5 | -14.406% |
| TEST | WAG | RULE/BEAM | 48 | 48/0/0 | -121.1 | -1.820% |
| TEST | WAG | 当前负载感知 VNext | 48 | 48/0/0 | -1,539.9 | -18.851% |
| TEST | WAG | 联合 GNN | 48 | 48/0/0 | -527.4 | -7.060% |
| TEST | WAG | 联合 GNN + 瓶颈重构 | 48 | 48/0/0 | -403.9 | -5.852% |
| 代表性 VALIDATION | HGA | RULE/BEAM | 2 | 2/0/0 | -1,676.4 | -24.311% |
| 代表性 VALIDATION | HGA | GNN_WIDTH1 | 2 | 2/0/0 | -2,131.7 | -28.158% |

解释这些数字时必须保留以下限制：

- VALIDATION 的 RULE/BEAM、GNN_WIDTH1 引用现有每排样一次的确定性测量，并复用于种子标签；不是重新独立运行了 30 次参考方法。
- TEST 的 RULE/BEAM 也包含确定性结果的种子标签复用。20 个排样 × 3 个种子不是 60 个独立排样，统计推断应考虑排样内相关性。
- 联合 GNN 的 TEST 配对只使用 `COMPLETE_CONSTRUCTION` 来源，排除旧 60 秒实验来源。
- 这里的 GNN 配对对应上述历史冻结模型和构造流程，不包含之后另行开展的动作价值头实验，不能直接移用于新模型。
- HGA 只有三个代表样本，和参考方法仅两个共同可行配对；不能据此推断其完整 TEST 优于 GNN 或 WAG。
- 本报告没有做显著性检验，不能将配对胜数直接写成“统计显著优于”。

## 8. 性能分析与剩余瓶颈

旧实现工程诊断：`validation_03` 的 HGA 初始化加一个子代约 31 s，47,715 次完整路线评价累计约 21.4 s。WAG 单机器人候选构造约 1.14 s，其中 Factorial 搜索约 0.84 s。原实现存在完整路线重算、重复几何计算和对象创建的明显成本。

优化后最终 HGA 代码的暖机 1,000 子代诊断为 7.279 s，其中：

| 项目 | 诊断记录 |
|---|---:|
| 局部搜索与完整子代原生评分合计 | 6.122 s |
| 交叉 | 0.148 s |
| 种群管理 | 0.436 s |
| 轻量邻域候选计数 | 45,838,309 |

独立暖机 cProfile 在不调用 B32 的情况下，1,000 子代共约 8.002 s；`_vnd_arrays` 内部约 5.534 s，`_priority_arrays` 内部约 0.868 s。剩余开销包括 Numba 入参处理、交叉、种群管理等。

因此，HGA 的主要剩余开销是完整六邻域搜索，其次为子代原生调度和运行时边界处理。不能把半小时运行归因于搜索内重复调用 B32；正式搜索内没有 B32。

上述候选计数来自特定 1,000 次工程诊断。若线性外推到 200,000 次，工作量可达数十亿级，但这只是量级估算，不能冒充三个完整运行实际测得的候选总数。

运行时间包括预计算、初始化、全部规定搜索与最终 B32；读取数据和工作进程启动在单次 `baseline_job` 计时之外。首次使用 Numba 的编译计入对应记录，未从结果中扣除。WAG 批次使用过 4 个工作进程，HGA 代表测量使用 3 个工作进程。它们不是统一空闲状态下的单线程硬件消融。

旧 WAG 的约 7.26 小时记录只覆盖原先两个 TEST 排样的六次运行，与新全量实验的样本、实现细节及运行条件不同。本报告不据此给出严格的整算法加速倍数。

## 9. 结果来源与验证边界

当前两份优化 CSV 的构成为：

| 文件 | 正式完整预算记录 | 工程诊断记录 |
|---|---:|---:|
| `wag_vns_results_optimized.csv` | 90：VALIDATION 30、TEST 60 | 0 |
| `hga_results_optimized.csv` | 3：代表性 VALIDATION | 2：均为 1,000 子代 |

正式汇总必须筛选：

```python
row["source"] == "OPTIMIZED_ITERATIONS"
```

HGA 的 `PERFORMANCE_DIAGNOSTIC_*` 行不能计入完整 200,000 次实验，也不能与正式三例混算。第一条诊断在交叉插入周期目标修复前完成，记录保留，没有覆盖。

另有两焊缝微型算例实际完成 200,000 次有效子代，最终一次 B32 可行，Cmax 为 142.5926 s，VND 截断命中为 0。这只是预算计数和执行路径检查，不能替代真实排样。

一次开发运行沿用旧邻域连接定义，在已记录 120,000 次进度后因正确性修正而中止，未作为最终版本实验结果。

37 项算法／共同评价测试覆盖：局部代价与完整路线代价、反转矩阵、5!×2^5 枚举、81 组合、canonical 兄弟合并下界、编译等待调度与既有 ESS 一致、M4–M6 连接、六邻域代价、完整 VND 终止、交叉周期目标、覆盖与资格、失败子代不消耗有效迭代、内部不调用 B32／认证等。另有 194 项 GNN 相关测试通过。Numba reflected list 的弃用预告仍存在。

测试通过不等于完整算法已经获得文献等价性证明、全局最优证明或所有排样可行性保证。

## 10. 源码中需要 GPT 重点审查的适配细节

以下是本次读取代码可确认的实现事实，不能在报告中省略。它们是进一步审查的问题，不是已经证明错误的结论。

### 10.1 WAG 候选生成并非所有可能分配的穷举

主初始化的 `max_wag_variants=None` 取消了旧的数量上限，但 `generate_wag_assignments()` 仍从平衡边界附近枚举上下轨各 `-2..2` 的偏移，共至多 25 个边界组合，边界裁剪和去重后可能更少。

因此，“评价全部实际生成的 WAG”不等于“穷举所有合法 WAG”。这套四机器人边界生成规则与原论文 WAG 生成逻辑之间的关系，应独立解释。

### 10.2 模式切换与主初始化走不同路径

`transition_legal_pattern()` 内仍使用 `max_variants=1` 生成模式切换提案。V2 搜索还会直接评价该提案，然后继续邻域搜索；原始初始分配也有直接原生评价。

因此，不能把整个搜索描述成“所有进入最优解竞争的方案都必须先经过三种改进路线和 81 组合”。81 组合机制适用于相应的 routed-assignment 路径；直接提案属于项目扩展，需要审查其文献依据和命名。

### 10.3 Move／Swap 的 n 是尝试次数

代码按 n 次尝试生成邻域，固定 X 子任务、机器人资格等条件可能使提案被丢弃，重复提案也可能被去重。因此 n=10 不保证每轮得到 10 个不同的有效邻域。

需要审查这是否符合预期的论文搜索预算解释，不能只检查配置字段等于 10。

### 10.4 LNS 返回候选池的组成

当前 LNS 对四台机器人各破坏约 p 比例任务并修复一个基解；返回结果为固定模式下生成的 WAG 候选池，加这个破坏—修复基解。

保留部分对该基解有直接影响，但候选池本身还包含从固定模式重新生成的分配。需要核对其与原文 LNS 的“保留部分解，再按 WAG 生成规则修复”是否足够一致。

### 10.5 HGA 的局部目标与完整个体目标不同

正式快速路径调用 `_vnd_arrays(..., conflict_aware=False)`。VND 改善路线周期；完整子代再用共享优先级原生目标评分。

需要分析：这种分层目标是否会造成路线周期改善却原生 WAIT 或可行性变差，以及是否需要在不引入 B32、不破坏完整 VND 定义的前提下做更合适的轻量适配。不能将当前实现称为“对原生冲突感知 Cmax 的完整 VND”。

### 10.6 HGA 合法模式变异属于需明示的适配

代码保留 `pattern_mutation_probability=0.15` 的合法 WHOLE/Y_SPLIT/X_SPLIT 模式变异。它在项目原实现中已经存在，本轮没有另加新的进化模块；但仍应单独核对它的文献依据及四机器人拆分适配必要性。

### 10.7 原生评分与最终 B32 的策略差异

共享原生评分分别安排两条轨道的优先级，再检查跨轨干涉；B32 使用自己的派工与恢复逻辑。WAG 的 11 次 VALIDATION 死锁是两者不能互相替代的实测证据。

需要解释 VALIDATION 有死锁而 TEST 全部可行的差异。目前尚没有足够的几何分布与阻塞诊断支持一个确定的因果解释，不能直接归因于“泛化更好”或“数据泄漏”。

## 11. 可直接交给 GPT 的分析任务

请基于上述事实及附带源码，优先输出以下分析：

1. 分别判断 WAG、HGA 哪些部分是文献核心、哪些是四机器人共同模型适配、哪些尚缺充分依据。尤其检查第 10 节，而不是仅凭结果较好就认定忠实复现。
2. 判断 HGA 的路线周期 VND与冲突感知个体评分是否存在关键目标错位，并提出保持原算法结构的最小必要修正方向。
3. 根据 cProfile 和数据规模，分析完整 VND 的成本来源，区分可消除的重复计算与规定搜索预算带来的必要工作；不要把未实测的加速倍数当作结论。
4. 分析 WAG 在 VALIDATION 的 11 次 B32 死锁，以及为何三个 HGA 代表样本可行；需要哪些少量、具体的诊断才能区分分配、方向、顺序、等待和派工策略的影响。
5. 判断论文目前可以使用哪些结论、必须补充哪些实验。特别明确 HGA 没有完整 VALIDATION/TEST，三个代表样本不能替代完整实验。
6. 给出后续工作的优先级及可验证的成功条件，不重新设计 GNN 主方法，不用 TEST 调参。

后续分析必须遵守：不减少 WAG 500 轮或 HGA 200,000 有效子代；不恢复 VND 候选／轮数截断；不增加人工时间截止；不在搜索内调用 B32；不使用 GNN 提供初始解或邻域；不修改数据、资格、合法拆分和 B32；不新增无必要文件、复杂协议、manifest 或哈希身份系统。需要新增文件时先说明必要性并询问用户。

## 12. 配套证据与复现入口

报告本身不能替代源码审查。外部 GPT 无法仅凭本地路径读取文件，需要同时提供相应现有文件。

| 证据 | 项目内位置 |
|---|---|
| WAG 实现 | `src/mrta_baselines/wag_vns.py` |
| HGA 实现 | `src/mrta_baselines/hga.py` |
| 共同评价器 | `src/mrta_baselines/common.py` |
| 实验入口与汇总 | `scripts/run_gnn_weld_layout.py` |
| WAG 原始优化结果 | `results/new_dataset_final/wag_vns_results_optimized.csv` |
| HGA 原始优化结果 | `results/new_dataset_final/hga_results_optimized.csv` |
| 全部历史与本轮说明 | `docs/GNN_WELD_LAYOUT_EXPERIMENT.md` |
| 算法测试 | `tests/test_wag_baseline.py`、`tests/test_hga_baseline.py`、`tests/test_baseline_common.py` |
| WAG 论文 | `references/Novel method for welding gantry robot scheduling at shipyards.pdf` |
| HGA 论文 | `references/1-s2.0-S0957417425009212-main.pdf` |

只读汇总命令：

```powershell
Set-Location 'D:\pybullet_test\MRTA_GA\DRL'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py baseline-summary
```

该命令筛选正式来源并读取已有参考数据，不重新求解，不覆盖历史 CSV。

## 13. 对 `86e32c7` 的审查与修正（执行中）

基准完整提交号为 `86e32c7cb9a95deba71357a0b6aae0a2a00af2cb`。本地父工程的 Git HEAD 不是该提交，且 DRL 是父工程的未跟踪目录；没有切换或覆盖父工程。通过 GitHub 逐文件核对，HGA、WAG、共同评价器、实验入口及两份算法测试在修改前均与该提交一致（仅归一化行尾和文件尾空白后比较）。本轮只修改已有文件，不新增文件。

### 13.1 首轮审查结果与理论依据

下列行号是基准版本的位置，修正后的行号会移动。

| 问题 | 基准函数/位置 | 修正原则 | 搜索空间与可比性 |
|---|---|---|---|
| VND 优化独立路线周期，种群优化协调 Cmax | `hga.py:_vnd_arrays`；`hga_vnd` 约 737 行；`_run_hga_native.local` 约 1269 行传入 `False` | 局部搜索、锦标赛和生存评价统一为 `priority_schedule` 的原生协调 Cmax；独立路线周期用于下界、原论文 r_m 角色判定及种群平局辅助 | 接受轨迹改变；旧 HGA 结果不能与修正结果混用 |
| M4 错误地限制在最长路线 | `hga.py:_vnd_arrays` 约 602 行 `move in (1, 4)` | M4 是同一路线 2-opt，不限制其是否最长；保留 M1 最长路线要求 | 恢复遗漏的非最长路线 M4，扩大原实现的有效邻域 |
| 每次接受移动后重建四条路线全部区间和映射 | `hga.py:_vnd_arrays` 约 575–597 行 | 复用布局级工作区；只刷新实际变化路线的区间矩阵、处理时间前缀、资格前缀和任务位置 | 仅消除重复计算，不减少候选 |
| 缺少 VND 缓存和实际操作统计 | `hga.py:_run_hga_native.local`、`common.py:BaselineAccounting` | 增加 `vnd_cache_hits`、`vnd_unique_calls`、`vnd_candidate_operations` | 不改变搜索；操作计数不包括缓存重放 |
| 25 个偏移组合被误读成全部合法分配 | `wag_vns.py:generate_wag_assignments` 约 800–890 行 | 明确这是四机器人适配中的有限边界启发式，而非穷举，也非原文逐步生成规则的逐字复现 | 当前先如实说明范围，不擅自改为全分配枚举 |
| LNS 混入完整 WAG 重生成 | `wag_vns.py:wag_lns_candidates` 约 1043–1093 行 | 破坏后固定所有未破坏任务的归属，仅重插被破坏任务；不能与全局重生成混合 | 修复后 LNS 搜索集合改变，WAG 须使用新来源结果 |
| 对优胜路线重复原生调度 | `wag_vns.py:_evaluate_routed_assignment` 约 1219–1227 行 | 保留数值评价输出并交给共同评价器；规范化改变实际编码时回退重新评价 | 需证明指标、方向和时间线不变，才能认定为纯计算优化 |

HGA 论文 M1–M6 核对：M1 将最长路线的任务 u 移到另一条路线任务 v 后；M2 交换不同路线的单任务、至少一条为最长路线；M3 交换不同路线各两个连续任务、至少一条为最长路线；M4 在同一路线替换 `(u,x),(v,y)` 为 `(u,v),(x,y)`，反转 x 到 v；M5 在不同路线连接前缀与反转前缀、反转尾段与尾段；M6 交换不同路线的尾段且不反转。现有重连定义与这些定义一致，发现的限制错误在数值 VND 的 M4 筛选。

WAG 论文第 4.1 节先按三个机器人的专属区和中部任务形成分组，再从中部两端转移任务，随负载变化逐步生成 WAG；数量随实例变化。本项目是两条轨道各两个机器人，按轨道资格和负载分组，在两个平衡边界周围各偏移 −2…2，裁剪后去重，因此最多 25 个，通常更少。原文 LNS 清空中间机器人并随机破坏两端机器人 p 比例任务；本项目无同构的中间机器人，沿用每机器人按 p 破坏的四机器人适配，但保留任务必须固定。不能将这两处适配称为原文分配规则的严格复现。

### 13.2 HGA 已实施内容与验证边界

正式与独立调用的 VND 均使用协调 Cmax。路线代价、纯处理时间、未变化路线下界剪枝仍保留。另可复用完全未变化的同轨机器人对的原生完工时间作为下界；不能把与变化路线同轨的另一机器人原有 WAIT 后完工时间当成不变下界。变化机器人对始终按既有原生规则重新选择首方向和优先顺序，然后执行相同跨轨冲突检查。

M1–M3 的 r_m 仍按原实现和论文的路线周期长度确定，即方向自由的独立路线最短周期，平局取较小机器人编号；它用于判定邻域角色，不用于接受改进。含 WAIT 的原生完工最晚机器人可能与 r_m 不同。本轮保留这一角色定义，避免在修正目标和恢复 M4 之外，另行改变 M1–M3 的候选集合。每个符合角色和资格的候选均以协调 Cmax 决定是否接受。

`_native_pair` 复用一次计算的右路线固定首方向 DP，保持原有方向平局顺序；严格空间包围盒能证明同轨两路线全程分离时，省去禁入区间构造。`_native_update` 复用未变化的整个同轨对；`_refresh_vnd_route` 只刷新变化路线。工作区按合法任务数量分配容量，不是候选数上限。以上不调用 B32，不改变原生四机器人调度策略，不缩小六邻域或 alpha。

已通过的验证：首轮三个相关测试文件 38 项；添加非最长路线 M4 实例并加速原生对评价后，HGA 文件 14 项通过。后续共同评价器和 WAG 修正还需分别验证，并在最终记录统一测试总数。

后续阶段：三个相关文件 41 项通过；额外的 200 组几何、双优先方向共 400 组原始/空间加速跟随调度对比通过，WAIT 总量、有限/阻塞状态及逐操作数组完全一致。共同评价器预计算复用测试检查方向、指标和完整时间线一致；WAG 优胜候选调度调用计数测试通过，确认只调用一次原生数值调度。LNS 尚未修正，不将这些测试当作 LNS 已验证的证据。

统一复测结果为 **43 passed**（2026-11-11，三个算法/共同评价测试文件）。WAG `_evaluate_routed_assignment` 保存 `_priority_arrays` 的完整输出，在规范化后编码与模式均未变化时构建相同原生时间线，交给 `CommonBaselineEvaluator.evaluate(native_schedule=...)`；若规范化实际改变编码则重新评价。预计算接口只用于已有未认证原生评价器，不绕过 B32/正式认证路径。搜索内部仍不调用 B32。该改动仅消除重复评价；原生计时和调用数现在包括此前未计入的数值预评分，统计口径比旧 `reference_calls` 更完整，不能直接将此计数当作 B32 次数。

统一复测结果为 **43 passed**（2026-11-11，三个算法/共同评价测试文件）。WAG `_evaluate_routed_assignment` 保存 `_priority_arrays` 的完整输出，在规范化后编码与模式均未变化时构建相同原生时间线，交给 `CommonBaselineEvaluator.evaluate(native_schedule=...)`；若规范化实际改变编码则重新评价。预计算接口只用于已有未认证原生评价器，不绕过 B32/正式认证路径。搜索内部仍不调用 B32。该改动仅消除重复评价；原生计时和调用数现在包括此前未计入的数值预评分，统计口径比旧 `reference_calls` 更完整，不能直接将此计数当作 B32 次数。

性能诊断以 `validation_03` 的原区域种子、完整无截断 VND 为例：81255 次结构有效提案、71 次接受、终值 5119.837307 s。带组件观测的耗时 24.2616 s，其中 10953 次原生增量评价共 23.6506 s，约占 97.5%；45686 次路线区间候选代价共 0.1618 s、17356 次片段物化共 0.0546 s、11561 次首任务资格检查共 0.0417 s，88 次路线数据刷新共 0.0011 s。空间加速后同一个 VND 耗时 4.6630 s，提案、接受次数和终值相同。该对比包含组件观测开销差别，只说明该内核的主要瓶颈与优化效果，不能外推为完整 HGA 加速倍数。

空间加速不改变原生优先策略：原有严格空间分离/时间分离筛掉的固定操作不再重复进入最终冲突核对；保守包围盒已排除接近多边形时，省去该多边形投影，边界附近仍走原数值投影。原有 `_numeric_follow` 默认路径和 `_priority_arrays` 保留为对照，HGA 显式启用空间加速。一次精确实参缓存尝试即使在重复调用全部命中时也从 0.00655 s 变慢到 0.01675 s（100 次跟随调用），已完整撤回，没有留下该缓存或其额外统计。

首次三例长运行用于暴露协调评价成本，未完成且未写入 CSV；采用空间加速后曾于 19:59 重新从零启动。该批运行仍未完成，为采用后续验证的时间窗口加速，于 21:48 再次从零启动三个完整预算运行，未减少正式有效子代预算；当前状态见第 13.6 节。

空间加速版的短程工程诊断（只在控制台输出，没有加入正式 CSV）在 `validation_03` 跑完 100 次有效子代：312.4965 s，原生 Cmax 5045.7688 s；VND 唯一调用 119、缓存命中 1、实际候选 3331317，局部搜索累计 287.0906 s，交叉 0.0205 s，生存管理 0.6737 s，构造拒绝 0。这是性能诊断，不能替代每例 200000 次正式预算；也不能从前 100 次、尚未收敛的缓存命中率线性外推完整运行耗时。协调目标的计算成本仍是重大限制，完整三例运行尚待实测。

具体反例：较短独立路线的代价约 410.5343 s，最长独立路线约 506.7982 s。在较短路线执行 M4，将协调 Cmax 从 655.8006 s 改善到 650.5581 s，独立路线最大周期保持不变。测试隔离该同路线操作，验证数值 VND 接受它，并与完整原生评价核对结果；这不是只检查配置开关。

六种移动的增量评价均与未经改变的 `_priority_arrays` 比较，检查 Cmax、方向和逐操作时间线；无截断 VND 的终止测试检查最终六邻域都无原生 Cmax 改进。新增计数定义为：实际唯一 VND 执行一次计 unique；完整结果从确定性缓存取出计 hit；每个满足 M1–M6 路线角色与索引结构要求的 `(move,u,v)` 提案计一次 operation，之后即使被资格或有效下界拒绝仍计入，不计缓存复用为真实运算。

三例完整验证正在执行：`validation_03`、`validation_04`、`validation_07`，种子 `20261110`，μ=20、λ=10、α=20、200000 次有效子代、六邻域无候选/轮数上限；结束后各调用一次既有 B32。使用已有 `hga_results_optimized.csv` 追加来源 `COORDINATED_VND_V1`；原来源 `OPTIMIZED_ITERATIONS` 原样保留。**尚未完成的运行不得算作通过，也没有据此启动修正后的完整正式实验。**




### 13.3 WAG 历史 11 次死锁的逐例诊断

以下是原 WAG 完成 500 轮后的固定最优路线重放；诊断不反馈搜索。首次从完整 500 轮重建最优路线，随后从本报告保存的模式、路线和方向重放派工，避免 LNS 修正后拿新最优路线冒充旧实例。`search_rerun_iterations=0` 表示只重放已有旧方案。`reproduced` 核对原生 Cmax、最终状态和原搜索预算。B32 未找到时间线不构成数学不可行证明。阻塞边累计值是各派工状态 ESS 返回关联的累计次数，不能解释为独立冲突事件或唯一因果。

| 排样 | 种子 | 重现 | 原生 Cmax/s | B32 | 同轨/跨轨阻塞数 | 等待环 | 原生时间线冲突数 |
|---|---:|---|---:|---|---|---|---:|
| validation_01 | 20261111 | True | 5591.1279 | DEADLOCK | 18/0 | () | 0 |
| validation_02 | 20261111 | True | 5651.9824 | DEADLOCK | 7/0 | () | 0 |
| validation_03 | 20261110 | True | 5074.6182 | DEADLOCK | 82/0 | () | 0 |
| validation_03 | 20261111 | True | 5031.2890 | DEADLOCK | 86/0 | () | 0 |
| validation_03 | 20261112 | True | 5520.2745 | DEADLOCK | 33/0 | () | 0 |
| validation_07 | 20261110 | True | 6161.0848 | DEADLOCK | 72/0 | () | 0 |
| validation_07 | 20261111 | True | 6160.6548 | DEADLOCK | 72/0 | () | 0 |
| validation_07 | 20261112 | True | 6167.8285 | DEADLOCK | 72/0 | () | 0 |
| validation_10 | 20261110 | True | 6138.0833 | DEADLOCK | 4/0 | () | 0 |
| validation_10 | 20261111 | True | 5544.2747 | DEADLOCK | 20/0 | () | 0 |
| validation_10 | 20261112 | True | 6045.4907 | DEADLOCK | 4/0 | () | 0 |

逐例首任务、方向、首次阻塞、终端等待图和重放方案：

<!-- WAG_DEADLOCK_CASES_START -->
```json
[
  {
    "layout_id": "validation_01",
    "seed": 20261111,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5591.127885168832,
    "historical_native_cmax_s": 5591.127885168832,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 387,
    "next_index": [
      14,
      135,
      107,
      131
    ],
    "template_lengths": [
      19,
      135,
      107,
      131
    ],
    "completion_s": [
      3887.142111600733,
      5498.580883268584,
      5588.213113206971,
      5591.127885168832
    ],
    "wait_for_graph": [
      [
        0,
        [
          1
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 296,
      "robot": 0,
      "operation": "R0:B3:POST",
      "kind": "POST",
      "block_id": "P18-W05-S01::whole",
      "ready_s": 3887.142111600733,
      "start": [
        14.45326757812499,
        10.709610351562501
      ],
      "end": [
        14.45326757812499,
        10.709610351562501
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R1:B29:WELD"
      ],
      "graph": [
        [
          0,
          [
            1
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 0,
        "next_operation": "R0:B3:POST",
        "kind": "POST",
        "block_id": "P18-W05-S01::whole",
        "ready_s": 3887.142111600733,
        "start": [
          14.45326757812499,
          10.709610351562501
        ],
        "end": [
          14.45326757812499,
          10.709610351562501
        ],
        "same_rail_blockers": 18,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R1:B29:POST",
          "R1:B29:WELD",
          "R1:B30:MOVE",
          "R1:B30:POST"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 295,
      "robot": 1,
      "operation": "R1:B29:WELD",
      "ready_s": 3853.8751952948423,
      "earliest_start_s": 3884.377290515455
    },
    "blocked_edge_occurrences": {
      "same": 92,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 1,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 0,
        "blocker": 1,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P05-W01-S01::whole",
        "direction": 0,
        "start": [
          0.0,
          6.070024414062501
        ],
        "end": [
          12.5,
          6.070024414062512
        ]
      },
      {
        "robot": 1,
        "block": "P11-W01-S01::0",
        "direction": 1,
        "start": [
          12.972269531250001,
          6.0
        ],
        "end": [
          12.972269531250001,
          6.519296875
        ]
      },
      {
        "robot": 2,
        "block": "P05-W02-S01::whole",
        "direction": 0,
        "start": [
          0.0,
          5.320024414062501
        ],
        "end": [
          12.5,
          5.3200244140625115
        ]
      },
      {
        "robot": 3,
        "block": "P03-W05-S01::1",
        "direction": 0,
        "start": [
          17.498,
          6.0
        ],
        "end": [
          17.498,
          4.627024414062501
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 60.5020952236282,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P01-W07-S01:WHOLE",
      "P01-W08-S01:WHOLE",
      "P01-W09-S01:WHOLE",
      "P01-W10-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P02-W03-S01:WHOLE",
      "P02-W04-S01:WHOLE",
      "P02-W05-S01:WHOLE",
      "P02-W06-S01:WHOLE",
      "P02-W07-S01:WHOLE",
      "P02-W08-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P03-W03-S01:WHOLE",
      "P03-W04-S01:WHOLE",
      "P03-W05-S01:Y_SPLIT:BY_CENTER",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P04-W03-S01:WHOLE",
      "P04-W04-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:WHOLE",
      "P07-W05-S01:WHOLE",
      "P07-W06-S01:WHOLE",
      "P07-W07-S01:WHOLE",
      "P07-W08-S01:WHOLE",
      "P07-W09-S01:WHOLE",
      "P07-W10-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P09-W03-S01:WHOLE",
      "P09-W04-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P10-W03-S01:WHOLE",
      "P10-W04-S01:WHOLE",
      "P10-W05-S01:WHOLE",
      "P10-W06-S01:WHOLE",
      "P10-W07-S01:WHOLE",
      "P10-W08-S01:WHOLE",
      "P10-W09-S01:WHOLE",
      "P10-W10-S01:WHOLE",
      "P10-W11-S01:WHOLE",
      "P10-W12-S01:WHOLE",
      "P11-W01-S01:Y_SPLIT:BY_CENTER",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P12-W03-S01:WHOLE",
      "P12-W04-S01:WHOLE",
      "P12-W05-S01:WHOLE",
      "P12-W06-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:WHOLE",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P18-W01-S01:WHOLE",
      "P18-W02-S01:WHOLE",
      "P18-W03-S01:WHOLE",
      "P18-W04-S01:WHOLE",
      "P18-W05-S01:WHOLE",
      "P18-W06-S01:WHOLE",
      "P18-W07-S01:WHOLE",
      "P18-W08-S01:WHOLE",
      "P18-W09-S01:WHOLE",
      "P18-W10-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P05-W01-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P18-W05-S01::whole",
        "P15-W01-S01::whole"
      ],
      [
        "P11-W01-S01::0",
        "P18-W02-S01::whole",
        "P18-W03-S01::whole",
        "P18-W04-S01::whole",
        "P18-W09-S01::whole",
        "P18-W01-S01::whole",
        "P03-W02-S01::whole",
        "P03-W05-S01::0",
        "P03-W01-S01::whole",
        "P01-W08-S01::whole",
        "P01-W05-S01::whole",
        "P01-W01-S01::whole",
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W04-S01::whole",
        "P01-W09-S01::whole",
        "P01-W10-S01::whole",
        "P01-W07-S01::whole",
        "P01-W06-S01::whole",
        "P07-W10-S01::whole",
        "P07-W07-S01::whole",
        "P07-W06-S01::whole",
        "P07-W08-S01::whole",
        "P07-W05-S01::whole",
        "P07-W01-S01::whole",
        "P07-W02-S01::whole",
        "P07-W03-S01::whole",
        "P07-W04-S01::whole",
        "P07-W09-S01::whole",
        "P18-W08-S01::whole",
        "P18-W07-S01::whole",
        "P18-W06-S01::whole",
        "P18-W10-S01::whole",
        "P15-W02-S01::whole"
      ],
      [
        "P05-W02-S01::whole",
        "P20-W01-S01::whole",
        "P20-W02-S01::whole",
        "P12-W06-S01::whole",
        "P12-W04-S01::whole",
        "P12-W01-S01::whole",
        "P12-W02-S01::whole",
        "P12-W03-S01::whole",
        "P21-W02-S01::whole",
        "P21-W01-S01::whole",
        "P16-W01-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole",
        "P17-W02-S01::whole",
        "P17-W01-S01::whole",
        "P09-W03-S01::whole",
        "P09-W02-S01::whole",
        "P09-W01-S01::whole",
        "P09-W04-S01::whole",
        "P08-W01-S01::whole",
        "P08-W02-S01::whole",
        "P06-W02-S01::whole",
        "P06-W01-S01::whole",
        "P04-W03-S01::whole",
        "P04-W04-S01::whole",
        "P04-W01-S01::whole",
        "P04-W02-S01::whole"
      ],
      [
        "P03-W05-S01::1",
        "P02-W08-S01::whole",
        "P02-W06-S01::whole",
        "P02-W07-S01::whole",
        "P03-W03-S01::whole",
        "P03-W04-S01::whole",
        "P11-W01-S01::1",
        "P10-W12-S01::whole",
        "P10-W11-S01::whole",
        "P10-W10-S01::whole",
        "P10-W09-S01::whole",
        "P10-W08-S01::whole",
        "P10-W07-S01::whole",
        "P12-W05-S01::whole",
        "P10-W04-S01::whole",
        "P10-W05-S01::whole",
        "P10-W06-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P10-W03-S01::whole",
        "P13-W06-S01::whole",
        "P13-W07-S01::whole",
        "P13-W08-S01::whole",
        "P13-W05-S01::whole",
        "P02-W05-S01::whole",
        "P02-W04-S01::whole",
        "P02-W02-S01::whole",
        "P02-W01-S01::whole",
        "P02-W03-S01::whole",
        "P13-W03-S01::whole",
        "P13-W02-S01::whole",
        "P13-W01-S01::whole",
        "P13-W04-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        0,
        1,
        0,
        0
      ],
      [
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0
      ],
      [
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0
      ]
    ]
  },
  {
    "layout_id": "validation_02",
    "seed": 20261111,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5651.982401134807,
    "historical_native_cmax_s": 5651.982401134807,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 28,
    "rollout_budget": 32,
    "frontier_exhausted": true,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 353,
    "next_index": [
      127,
      141,
      75,
      10
    ],
    "template_lengths": [
      127,
      143,
      75,
      35
    ],
    "completion_s": [
      5580.00866342681,
      5446.747778195575,
      5637.894077469806,
      258.57571508240557
    ],
    "wait_for_graph": [
      [
        1,
        [
          0
        ]
      ],
      [
        3,
        [
          2
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 28,
      "robot": 3,
      "operation": "R3:B2:POST",
      "kind": "POST",
      "block_id": "P17-W01-S01::whole",
      "ready_s": 258.57571508240557,
      "start": [
        4.119296875,
        0.31
      ],
      "end": [
        4.119296875,
        0.31
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R2:B1:WELD"
      ],
      "graph": [
        [
          3,
          [
            2
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 1,
        "next_operation": "R1:B35:WELD",
        "kind": "WELD",
        "block_id": "P15-W04-S01::whole",
        "ready_s": 5446.747778195575,
        "start": [
          19.712999999999997,
          11.012198364257813
        ],
        "end": [
          19.712999999999997,
          10.312198364257814
        ],
        "same_rail_blockers": 2,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R0:B31:POST",
          "R0:B31:WELD"
        ],
        "cross_examples": []
      },
      {
        "robot": 3,
        "next_operation": "R3:B2:POST",
        "kind": "POST",
        "block_id": "P17-W01-S01::whole",
        "ready_s": 258.57571508240557,
        "start": [
          4.119296875,
          0.31
        ],
        "end": [
          4.119296875,
          0.31
        ],
        "same_rail_blockers": 5,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R2:B1:POST",
          "R2:B1:WELD",
          "R2:B2:MOVE",
          "R2:B2:SETUP"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": null,
    "blocked_edge_occurrences": {
      "same": 329,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 0,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 1,
        "blocker": 0,
        "finished": true
      },
      {
        "waiting": 3,
        "blocker": 2,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P01-W02-S01::whole",
        "direction": 0,
        "start": [
          0.3600000000000003,
          11.2885
        ],
        "end": [
          0.984,
          11.2885
        ]
      },
      {
        "robot": 1,
        "block": "P08-W01-S01::whole",
        "direction": 0,
        "start": [
          20.0,
          8.155
        ],
        "end": [
          7.499999999999999,
          8.154999999999989
        ]
      },
      {
        "robot": 2,
        "block": "P03-W04-S01::0",
        "direction": 1,
        "start": [
          2.122740749499278,
          5.5
        ],
        "end": [
          2.9029921874999998,
          4.0883593750000005
        ]
      },
      {
        "robot": 3,
        "block": "P09-W03-S01::whole",
        "direction": 1,
        "start": [
          5.722000000000001,
          6.271452148437501
        ],
        "end": [
          5.722000000000001,
          5.817524414062502
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 78.51262405539549,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P01-W07-S01:WHOLE",
      "P01-W08-S01:WHOLE",
      "P01-W09-S01:WHOLE",
      "P01-W10-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P03-W03-S01:WHOLE",
      "P03-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:X_SPLIT:LOWER:BX_OUTER_UPPER",
      "P05-W02-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P07-W01-S01:X_SPLIT:LOWER:BX_CENTER",
      "P07-W02-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P09-W03-S01:WHOLE",
      "P09-W04-S01:WHOLE",
      "P09-W05-S01:WHOLE",
      "P09-W06-S01:WHOLE",
      "P09-W07-S01:WHOLE",
      "P09-W08-S01:WHOLE",
      "P09-W09-S01:WHOLE",
      "P09-W10-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P12-W03-S01:WHOLE",
      "P12-W04-S01:WHOLE",
      "P12-W05-S01:WHOLE",
      "P12-W06-S01:WHOLE",
      "P12-W07-S01:WHOLE",
      "P12-W08-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P14-W03-S01:WHOLE",
      "P14-W04-S01:WHOLE",
      "P14-W05-S01:WHOLE",
      "P14-W06-S01:WHOLE",
      "P14-W07-S01:WHOLE",
      "P14-W08-S01:WHOLE",
      "P14-W09-S01:WHOLE",
      "P14-W10-S01:WHOLE",
      "P14-W11-S01:WHOLE",
      "P14-W12-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P15-W05-S01:WHOLE",
      "P15-W06-S01:WHOLE",
      "P15-W07-S01:WHOLE",
      "P15-W08-S01:WHOLE",
      "P15-W09-S01:WHOLE",
      "P15-W10-S01:WHOLE",
      "P15-W11-S01:WHOLE",
      "P15-W12-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P18-W01-S01:WHOLE",
      "P18-W02-S01:WHOLE",
      "P18-W03-S01:WHOLE",
      "P18-W04-S01:WHOLE",
      "P18-W05-S01:WHOLE",
      "P18-W06-S01:WHOLE",
      "P18-W07-S01:WHOLE",
      "P18-W08-S01:WHOLE",
      "P18-W09-S01:WHOLE",
      "P18-W10-S01:WHOLE",
      "P18-W11-S01:WHOLE",
      "P18-W12-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P20-W01-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W04-S01::whole",
        "P01-W09-S01::whole",
        "P01-W10-S01::whole",
        "P01-W07-S01::whole",
        "P01-W06-S01::whole",
        "P01-W08-S01::whole",
        "P01-W05-S01::whole",
        "P01-W01-S01::whole",
        "P03-W04-S01::1",
        "P03-W01-S01::whole",
        "P03-W02-S01::whole",
        "P03-W03-S01::whole",
        "P20-W01-S01::1",
        "P09-W04-S01::whole",
        "P09-W10-S01::whole",
        "P09-W06-S01::whole",
        "P09-W08-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P14-W11-S01::whole",
        "P14-W10-S01::whole",
        "P14-W03-S01::whole",
        "P14-W01-S01::whole",
        "P18-W09-S01::whole",
        "P18-W10-S01::whole",
        "P18-W12-S01::whole",
        "P14-W07-S01::whole",
        "P14-W08-S01::whole",
        "P14-W09-S01::whole",
        "P08-W02-S01::whole"
      ],
      [
        "P08-W01-S01::whole",
        "P09-W05-S01::whole",
        "P09-W09-S01::whole",
        "P09-W07-S01::whole",
        "P11-W02-S01::whole",
        "P11-W01-S01::whole",
        "P14-W12-S01::whole",
        "P14-W02-S01::whole",
        "P14-W06-S01::whole",
        "P14-W05-S01::whole",
        "P14-W04-S01::whole",
        "P18-W11-S01::whole",
        "P18-W03-S01::whole",
        "P18-W08-S01::whole",
        "P18-W07-S01::whole",
        "P18-W02-S01::whole",
        "P18-W01-S01::whole",
        "P18-W06-S01::whole",
        "P18-W05-S01::whole",
        "P18-W04-S01::whole",
        "P19-W02-S01::whole",
        "P19-W01-S01::whole",
        "P21-W01-S01::whole",
        "P21-W02-S01::whole",
        "P15-W12-S01::whole",
        "P15-W11-S01::whole",
        "P15-W10-S01::whole",
        "P15-W03-S01::whole",
        "P15-W09-S01::whole",
        "P15-W08-S01::whole",
        "P15-W07-S01::whole",
        "P15-W02-S01::whole",
        "P15-W01-S01::whole",
        "P15-W06-S01::whole",
        "P15-W05-S01::whole",
        "P15-W04-S01::whole"
      ],
      [
        "P03-W04-S01::0",
        "P12-W07-S01::whole",
        "P12-W08-S01::whole",
        "P12-W06-S01::whole",
        "P12-W05-S01::whole",
        "P12-W04-S01::whole",
        "P12-W01-S01::whole",
        "P12-W02-S01::whole",
        "P12-W03-S01::whole",
        "P06-W01-S01::whole",
        "P17-W02-S01::whole",
        "P04-W02-S01::whole",
        "P04-W01-S01::whole",
        "P16-W01-S01::whole",
        "P20-W01-S01::0",
        "P09-W01-S01::whole",
        "P07-W01-S01::0",
        "P05-W01-S01::0",
        "P05-W02-S01::whole"
      ],
      [
        "P09-W03-S01::whole",
        "P09-W02-S01::whole",
        "P17-W01-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::whole",
        "P13-W01-S01::whole",
        "P07-W02-S01::whole",
        "P07-W01-S01::1",
        "P05-W01-S01::1"
      ]
    ],
    "directions": [
      [
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1
      ],
      [
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0
      ],
      [
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0
      ],
      [
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0
      ]
    ]
  },
  {
    "layout_id": "validation_03",
    "seed": 20261110,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5074.618223159046,
    "historical_native_cmax_s": 5074.618223159046,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 259,
    "next_index": [
      67,
      83,
      14,
      95
    ],
    "template_lengths": [
      67,
      83,
      99,
      95
    ],
    "completion_s": [
      5009.905427938977,
      5062.91758627094,
      658.3635772025699,
      5034.703512712058
    ],
    "wait_for_graph": [
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 60,
      "robot": 2,
      "operation": "R2:B3:POST",
      "kind": "POST",
      "block_id": "P01-W04-S01::whole",
      "ready_s": 658.3635772025699,
      "start": [
        18.627849853515627,
        0.1299999999999999
      ],
      "end": [
        18.627849853515627,
        0.1299999999999999
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R3:B3:WELD"
      ],
      "graph": [
        [
          2,
          [
            3
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 2,
        "next_operation": "R2:B3:POST",
        "kind": "POST",
        "block_id": "P01-W04-S01::whole",
        "ready_s": 658.3635772025699,
        "start": [
          18.627849853515627,
          0.1299999999999999
        ],
        "end": [
          18.627849853515627,
          0.1299999999999999
        ],
        "same_rail_blockers": 82,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B10:MOVE",
          "R3:B10:POST",
          "R3:B10:SETUP",
          "R3:B10:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 55,
      "robot": 3,
      "operation": "R3:B3:WELD",
      "ready_s": 492.2514917236446,
      "earliest_start_s": 590.2973286999427
    },
    "blocked_edge_occurrences": {
      "same": 200,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 5,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P19-W06-S01::whole",
        "direction": 1,
        "start": [
          8.262796627220567,
          6.456038085937498
        ],
        "end": [
          7.065464235958514,
          6.157266601562499
        ]
      },
      {
        "robot": 1,
        "block": "P23-W01-S01::whole",
        "direction": 0,
        "start": [
          11.48301996512673,
          6.2384849229739245
        ],
        "end": [
          10.977692023232198,
          6.238484924921221
        ]
      },
      {
        "robot": 2,
        "block": "P25-W01-S01::whole",
        "direction": 0,
        "start": [
          14.583244140625002,
          3.1650463867187497
        ],
        "end": [
          16.17725390625,
          2.8839790039062496
        ]
      },
      {
        "robot": 3,
        "block": "P01-W01-S01::whole",
        "direction": 1,
        "start": [
          19.22181677246094,
          2.7365
        ],
        "end": [
          18.033882934570315,
          2.7365
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 128.04583697929803,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:Y_SPLIT:BY_CENTER",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P12-W03-S01:WHOLE",
      "P12-W04-S01:WHOLE",
      "P12-W05-S01:WHOLE",
      "P12-W06-S01:WHOLE",
      "P12-W07-S01:WHOLE",
      "P12-W08-S01:WHOLE",
      "P12-W09-S01:WHOLE",
      "P12-W10-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:Y_SPLIT:MIDPOINT",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P18-W01-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P19-W05-S01:WHOLE",
      "P19-W06-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P22-W03-S01:WHOLE",
      "P22-W04-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P23-W03-S01:WHOLE",
      "P23-W04-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P19-W06-S01::whole",
        "P13-W03-S01::0",
        "P08-W03-S01::whole",
        "P08-W02-S01::whole",
        "P08-W04-S01::1",
        "P16-W01-S01::whole",
        "P16-W02-S01::whole",
        "P07-W02-S01::whole",
        "P04-W01-S01::whole",
        "P15-W01-S01::whole",
        "P26-W02-S01::whole",
        "P26-W01-S01::whole",
        "P20-W01-S01::whole",
        "P20-W02-S01::whole",
        "P06-W01-S01::whole",
        "P11-W02-S01::0",
        "P11-W01-S01::whole"
      ],
      [
        "P23-W01-S01::whole",
        "P19-W05-S01::whole",
        "P19-W04-S01::1",
        "P13-W04-S01::whole",
        "P13-W01-S01::whole",
        "P13-W03-S01::1",
        "P13-W02-S01::whole",
        "P07-W01-S01::whole",
        "P12-W03-S01::whole",
        "P12-W02-S01::whole",
        "P17-W04-S01::1",
        "P12-W01-S01::whole",
        "P12-W05-S01::whole",
        "P12-W08-S01::whole",
        "P12-W06-S01::whole",
        "P12-W07-S01::whole",
        "P12-W10-S01::whole",
        "P12-W09-S01::whole",
        "P12-W04-S01::whole",
        "P04-W02-S01::whole",
        "P15-W02-S01::whole"
      ],
      [
        "P25-W01-S01::whole",
        "P25-W02-S01::whole",
        "P17-W03-S01::whole",
        "P01-W04-S01::whole",
        "P02-W01-S01::whole",
        "P05-W03-S01::whole",
        "P05-W04-S01::whole",
        "P05-W01-S01::whole",
        "P05-W02-S01::whole",
        "P14-W02-S01::whole",
        "P03-W02-S01::whole",
        "P03-W01-S01::whole",
        "P09-W02-S01::whole",
        "P10-W01-S01::whole",
        "P11-W02-S01::1",
        "P10-W02-S01::whole",
        "P09-W01-S01::whole",
        "P14-W01-S01::whole",
        "P08-W04-S01::0",
        "P08-W01-S01::whole",
        "P13-W05-S01::whole",
        "P13-W07-S01::whole",
        "P13-W06-S01::whole",
        "P19-W01-S01::whole",
        "P23-W04-S01::whole"
      ],
      [
        "P01-W01-S01::whole",
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W06-S01::whole",
        "P01-W05-S01::whole",
        "P02-W02-S01::whole",
        "P13-W08-S01::whole",
        "P19-W03-S01::whole",
        "P19-W02-S01::whole",
        "P19-W04-S01::0",
        "P18-W01-S01::whole",
        "P21-W01-S01::whole",
        "P21-W02-S01::whole",
        "P23-W03-S01::whole",
        "P23-W02-S01::whole",
        "P22-W02-S01::whole",
        "P22-W01-S01::whole",
        "P22-W04-S01::whole",
        "P22-W03-S01::whole",
        "P24-W01-S01::whole",
        "P24-W02-S01::whole",
        "P17-W04-S01::0",
        "P17-W02-S01::whole",
        "P17-W01-S01::whole"
      ]
    ],
    "directions": [
      [
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1
      ],
      [
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0
      ],
      [
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1
      ],
      [
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_03",
    "seed": 20261111,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5031.288984701484,
    "historical_native_cmax_s": 5031.288984701484,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 277,
    "next_index": [
      63,
      79,
      12,
      123
    ],
    "template_lengths": [
      63,
      79,
      75,
      123
    ],
    "completion_s": [
      4971.844643112453,
      5024.501481638732,
      1517.42828100035,
      5030.32614399818
    ],
    "wait_for_graph": [
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 92,
      "robot": 2,
      "operation": "R2:B3:SETUP",
      "kind": "SETUP",
      "block_id": "P02-W02-S01::whole",
      "ready_s": 1517.42828100035,
      "start": [
        16.95569970703125,
        1.459
      ],
      "end": [
        16.95569970703125,
        1.459
      ],
      "duration_s": 20.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R3:B9:WELD"
      ],
      "graph": [
        [
          2,
          [
            3
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 2,
        "next_operation": "R2:B3:SETUP",
        "kind": "SETUP",
        "block_id": "P02-W02-S01::whole",
        "ready_s": 1517.42828100035,
        "start": [
          16.95569970703125,
          1.459
        ],
        "end": [
          16.95569970703125,
          1.459
        ],
        "same_rail_blockers": 86,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B10:MOVE",
          "R3:B10:POST",
          "R3:B10:SETUP",
          "R3:B10:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": null,
    "blocked_edge_occurrences": {
      "same": 186,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 0,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P23-W01-S01::whole",
        "direction": 0,
        "start": [
          11.48301996512673,
          6.2384849229739245
        ],
        "end": [
          10.977692023232198,
          6.238484924921221
        ]
      },
      {
        "robot": 1,
        "block": "P07-W02-S01::whole",
        "direction": 0,
        "start": [
          16.95569970703125,
          8.795000000000002
        ],
        "end": [
          4.4556997070312505,
          8.794999999999991
        ]
      },
      {
        "robot": 2,
        "block": "P05-W01-S01::whole",
        "direction": 0,
        "start": [
          3.3858059809890637,
          2.4292690938847596
        ],
        "end": [
          2.8804780390945335,
          2.4292690958320566
        ]
      },
      {
        "robot": 3,
        "block": "P01-W05-S01::whole",
        "direction": 0,
        "start": [
          18.61193706083298,
          0.47795043945312576
        ],
        "end": [
          17.414588623046868,
          0.7766009521484527
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 0.9628407033039821,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:Y_SPLIT:BY_CENTER",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P12-W03-S01:WHOLE",
      "P12-W04-S01:WHOLE",
      "P12-W05-S01:WHOLE",
      "P12-W06-S01:WHOLE",
      "P12-W07-S01:WHOLE",
      "P12-W08-S01:WHOLE",
      "P12-W09-S01:WHOLE",
      "P12-W10-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:WHOLE",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P18-W01-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P19-W05-S01:WHOLE",
      "P19-W06-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P22-W03-S01:WHOLE",
      "P22-W04-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P23-W03-S01:WHOLE",
      "P23-W04-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P23-W01-S01::whole",
        "P13-W03-S01::whole",
        "P13-W01-S01::whole",
        "P13-W02-S01::whole",
        "P08-W04-S01::1",
        "P11-W02-S01::0",
        "P06-W01-S01::whole",
        "P20-W02-S01::whole",
        "P16-W02-S01::whole",
        "P16-W01-S01::whole",
        "P26-W02-S01::whole",
        "P26-W01-S01::whole",
        "P15-W01-S01::whole",
        "P04-W01-S01::whole",
        "P04-W02-S01::whole",
        "P15-W02-S01::whole"
      ],
      [
        "P07-W02-S01::whole",
        "P20-W01-S01::whole",
        "P08-W03-S01::whole",
        "P08-W02-S01::whole",
        "P13-W04-S01::whole",
        "P19-W06-S01::whole",
        "P19-W05-S01::whole",
        "P19-W04-S01::1",
        "P07-W01-S01::whole",
        "P12-W02-S01::whole",
        "P12-W03-S01::whole",
        "P12-W04-S01::whole",
        "P12-W09-S01::whole",
        "P12-W10-S01::whole",
        "P12-W07-S01::whole",
        "P12-W06-S01::whole",
        "P12-W08-S01::whole",
        "P12-W05-S01::whole",
        "P12-W01-S01::whole",
        "P17-W04-S01::1"
      ],
      [
        "P05-W01-S01::whole",
        "P05-W04-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::whole",
        "P13-W07-S01::whole",
        "P13-W08-S01::whole",
        "P13-W06-S01::whole",
        "P08-W04-S01::0",
        "P08-W01-S01::whole",
        "P11-W01-S01::whole",
        "P11-W02-S01::1",
        "P10-W02-S01::whole",
        "P10-W01-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P09-W01-S01::whole",
        "P09-W02-S01::whole",
        "P03-W01-S01::whole",
        "P03-W02-S01::whole"
      ],
      [
        "P01-W05-S01::whole",
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W06-S01::whole",
        "P01-W04-S01::whole",
        "P01-W01-S01::whole",
        "P17-W01-S01::whole",
        "P17-W02-S01::whole",
        "P17-W03-S01::whole",
        "P17-W04-S01::0",
        "P25-W02-S01::whole",
        "P25-W01-S01::whole",
        "P24-W02-S01::whole",
        "P24-W01-S01::whole",
        "P22-W03-S01::whole",
        "P22-W04-S01::whole",
        "P22-W01-S01::whole",
        "P22-W02-S01::whole",
        "P23-W02-S01::whole",
        "P23-W04-S01::whole",
        "P23-W03-S01::whole",
        "P21-W02-S01::whole",
        "P21-W01-S01::whole",
        "P18-W01-S01::whole",
        "P19-W01-S01::whole",
        "P19-W04-S01::0",
        "P19-W02-S01::whole",
        "P19-W03-S01::whole",
        "P13-W05-S01::whole",
        "P05-W02-S01::whole",
        "P05-W03-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0
      ],
      [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1
      ],
      [
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1
      ],
      [
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_03",
    "seed": 20261112,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5520.274547418261,
    "historical_native_cmax_s": 5520.274547418261,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 258,
    "next_index": [
      67,
      6,
      62,
      123
    ],
    "template_lengths": [
      67,
      87,
      75,
      123
    ],
    "completion_s": [
      5047.1122111588265,
      2388.564814814815,
      3421.2244449544364,
      5209.879101432442
    ],
    "wait_for_graph": [
      [
        1,
        [
          0
        ]
      ],
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 180,
      "robot": 1,
      "operation": "R1:B1:POST",
      "kind": "POST",
      "block_id": "P07-W02-S01::whole",
      "ready_s": 2388.564814814815,
      "start": [
        4.4556997070312505,
        8.794999999999991
      ],
      "end": [
        4.4556997070312505,
        8.794999999999991
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R0:B13:WELD"
      ],
      "graph": [
        [
          1,
          [
            0
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 1,
        "next_operation": "R1:B1:POST",
        "kind": "POST",
        "block_id": "P07-W02-S01::whole",
        "ready_s": 2388.564814814815,
        "start": [
          4.4556997070312505,
          8.794999999999991
        ],
        "end": [
          4.4556997070312505,
          8.794999999999991
        ],
        "same_rail_blockers": 14,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R0:B13:POST",
          "R0:B13:WELD",
          "R0:B14:MOVE",
          "R0:B14:POST"
        ],
        "cross_examples": []
      },
      {
        "robot": 2,
        "next_operation": "R2:B15:POST",
        "kind": "POST",
        "block_id": "P02-W01-S01::whole",
        "ready_s": 3421.2244449544364,
        "start": [
          16.95569970703125,
          0.709
        ],
        "end": [
          16.95569970703125,
          0.709
        ],
        "same_rail_blockers": 19,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B20:MOVE",
          "R3:B20:POST",
          "R3:B20:SETUP",
          "R3:B20:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 176,
      "robot": 0,
      "operation": "R0:B13:WELD",
      "ready_s": 2369.626954691307,
      "earliest_start_s": 2383.774595136528
    },
    "blocked_edge_occurrences": {
      "same": 128,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 5,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 1,
        "blocker": 0,
        "finished": true
      },
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P11-W01-S01::whole",
        "direction": 0,
        "start": [
          0.31,
          5.791220703125
        ],
        "end": [
          0.4580000000000109,
          5.791220703125
        ]
      },
      {
        "robot": 1,
        "block": "P07-W01-S01::whole",
        "direction": 1,
        "start": [
          4.4556997070312505,
          8.044999999999991
        ],
        "end": [
          16.95569970703125,
          8.045000000000002
        ]
      },
      {
        "robot": 2,
        "block": "P03-W01-S01::whole",
        "direction": 1,
        "start": [
          1.6248779296875002,
          0.3328251953125
        ],
        "end": [
          0.03086816406250037,
          0.613892578125
        ]
      },
      {
        "robot": 3,
        "block": "P13-W07-S01::whole",
        "direction": 1,
        "start": [
          5.807059511837419,
          2.7650000000000023
        ],
        "end": [
          5.807059511837419,
          4.0820000000000025
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 728.5359815719507,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:Y_SPLIT:BY_CENTER",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P12-W03-S01:WHOLE",
      "P12-W04-S01:WHOLE",
      "P12-W05-S01:WHOLE",
      "P12-W06-S01:WHOLE",
      "P12-W07-S01:WHOLE",
      "P12-W08-S01:WHOLE",
      "P12-W09-S01:WHOLE",
      "P12-W10-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:Y_SPLIT:MIDPOINT",
      "P13-W04-S01:Y_SPLIT:MIDPOINT",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:Y_SPLIT:BY_OUTER_LOWER",
      "P18-W01-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:Y_SPLIT:MIDPOINT",
      "P19-W05-S01:WHOLE",
      "P19-W06-S01:Y_SPLIT:MIDPOINT",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P22-W03-S01:WHOLE",
      "P22-W04-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P23-W03-S01:WHOLE",
      "P23-W04-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P11-W01-S01::whole",
        "P11-W02-S01::0",
        "P06-W01-S01::whole",
        "P08-W04-S01::1",
        "P16-W02-S01::whole",
        "P20-W02-S01::whole",
        "P20-W01-S01::whole",
        "P16-W01-S01::whole",
        "P08-W03-S01::whole",
        "P26-W02-S01::whole",
        "P26-W01-S01::whole",
        "P08-W02-S01::whole",
        "P15-W01-S01::whole",
        "P15-W02-S01::whole",
        "P13-W03-S01::0",
        "P04-W01-S01::whole",
        "P04-W02-S01::whole"
      ],
      [
        "P07-W01-S01::whole",
        "P07-W02-S01::whole",
        "P13-W04-S01::0",
        "P13-W01-S01::whole",
        "P13-W03-S01::1",
        "P13-W04-S01::1",
        "P13-W02-S01::whole",
        "P19-W06-S01::0",
        "P19-W04-S01::1",
        "P19-W05-S01::whole",
        "P23-W01-S01::whole",
        "P12-W04-S01::whole",
        "P17-W04-S01::1",
        "P12-W03-S01::whole",
        "P12-W02-S01::whole",
        "P12-W10-S01::whole",
        "P12-W07-S01::whole",
        "P12-W05-S01::whole",
        "P12-W01-S01::whole",
        "P12-W09-S01::whole",
        "P12-W08-S01::whole",
        "P12-W06-S01::whole"
      ],
      [
        "P03-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W01-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P11-W02-S01::1",
        "P03-W02-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P05-W04-S01::whole",
        "P05-W01-S01::whole",
        "P08-W04-S01::0",
        "P05-W02-S01::whole",
        "P05-W03-S01::whole",
        "P08-W01-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::whole",
        "P13-W05-S01::whole",
        "P13-W06-S01::whole"
      ],
      [
        "P13-W07-S01::whole",
        "P13-W08-S01::whole",
        "P18-W01-S01::whole",
        "P19-W03-S01::whole",
        "P19-W06-S01::1",
        "P19-W01-S01::whole",
        "P19-W04-S01::0",
        "P19-W02-S01::whole",
        "P21-W01-S01::whole",
        "P23-W04-S01::whole",
        "P23-W02-S01::whole",
        "P23-W03-S01::whole",
        "P21-W02-S01::whole",
        "P24-W02-S01::whole",
        "P22-W04-S01::whole",
        "P22-W01-S01::whole",
        "P22-W02-S01::whole",
        "P22-W03-S01::whole",
        "P24-W01-S01::whole",
        "P17-W04-S01::0",
        "P25-W01-S01::whole",
        "P17-W01-S01::whole",
        "P25-W02-S01::whole",
        "P17-W02-S01::whole",
        "P17-W03-S01::whole",
        "P01-W05-S01::whole",
        "P01-W02-S01::whole",
        "P01-W01-S01::whole",
        "P01-W04-S01::whole",
        "P01-W06-S01::whole",
        "P01-W03-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1
      ],
      [
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1
      ],
      [
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1
      ],
      [
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0
      ]
    ]
  },
  {
    "layout_id": "validation_07",
    "seed": 20261110,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 6161.084756775765,
    "historical_native_cmax_s": 6161.084756775765,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 439,
    "next_index": [
      123,
      179,
      18,
      119
    ],
    "template_lengths": [
      123,
      179,
      47,
      119
    ],
    "completion_s": [
      6161.084756775765,
      6159.636648473129,
      3021.709777675817,
      5164.324801065033
    ],
    "wait_for_graph": [
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 201,
      "robot": 2,
      "operation": "R2:B4:POST",
      "kind": "POST",
      "block_id": "P09-W02-S01::whole",
      "ready_s": 3021.709777675817,
      "start": [
        13.683336425781249,
        0.7230000000000107
      ],
      "end": [
        13.683336425781249,
        0.7230000000000107
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R3:B12:MOVE"
      ],
      "graph": [
        [
          2,
          [
            3
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 2,
        "next_operation": "R2:B4:POST",
        "kind": "POST",
        "block_id": "P09-W02-S01::whole",
        "ready_s": 3021.709777675817,
        "start": [
          13.683336425781249,
          0.7230000000000107
        ],
        "end": [
          13.683336425781249,
          0.7230000000000107
        ],
        "same_rail_blockers": 72,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B12:MOVE",
          "R3:B12:POST",
          "R3:B12:SETUP",
          "R3:B12:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 190,
      "robot": 3,
      "operation": "R3:B12:MOVE",
      "ready_s": 2951.506443966584,
      "earliest_start_s": 3017.0164124021962
    },
    "blocked_edge_occurrences": {
      "same": 239,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 11,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P02-W02-S01::1",
        "direction": 1,
        "start": [
          0.251,
          6.752323242187501
        ],
        "end": [
          0.251,
          5.963606445312501
        ]
      },
      {
        "robot": 1,
        "block": "P17-W01-S01::whole",
        "direction": 0,
        "start": [
          10.726102480587418,
          11.64
        ],
        "end": [
          10.726102480587418,
          11.016000000000002
        ]
      },
      {
        "robot": 2,
        "block": "P10-W01-S01::whole",
        "direction": 0,
        "start": [
          13.62,
          5.3130244140625
        ],
        "end": [
          1.1199999999999999,
          5.31302441406249
        ]
      },
      {
        "robot": 3,
        "block": "P04-W04-S01::whole",
        "direction": 1,
        "start": [
          19.266,
          0.01
        ],
        "end": [
          19.271,
          2.661
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 190.14196810042085,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P01-W07-S01:WHOLE",
      "P01-W08-S01:WHOLE",
      "P01-W09-S01:WHOLE",
      "P01-W10-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:Y_SPLIT:MIDPOINT",
      "P03-W01-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P04-W03-S01:WHOLE",
      "P04-W04-S01:WHOLE",
      "P04-W05-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P05-W05-S01:WHOLE",
      "P05-W06-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P06-W03-S01:WHOLE",
      "P06-W04-S01:Y_SPLIT:BY_CENTER",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:WHOLE",
      "P07-W05-S01:WHOLE",
      "P07-W06-S01:WHOLE",
      "P07-W07-S01:WHOLE",
      "P07-W08-S01:WHOLE",
      "P07-W09-S01:WHOLE",
      "P07-W10-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:X_SPLIT:UPPER:BX_OUTER_UPPER",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P11-W03-S01:WHOLE",
      "P11-W04-S01:WHOLE",
      "P11-W05-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P14-W03-S01:WHOLE",
      "P14-W04-S01:WHOLE",
      "P14-W05-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:WHOLE",
      "P17-W05-S01:WHOLE",
      "P17-W06-S01:WHOLE",
      "P17-W07-S01:WHOLE",
      "P17-W08-S01:WHOLE",
      "P17-W09-S01:WHOLE",
      "P17-W10-S01:WHOLE",
      "P18-W01-S01:WHOLE",
      "P18-W02-S01:WHOLE",
      "P18-W03-S01:WHOLE",
      "P18-W04-S01:WHOLE",
      "P18-W05-S01:WHOLE",
      "P18-W06-S01:WHOLE",
      "P18-W07-S01:WHOLE",
      "P18-W08-S01:WHOLE",
      "P18-W09-S01:WHOLE",
      "P18-W10-S01:WHOLE",
      "P18-W11-S01:WHOLE",
      "P18-W12-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P20-W03-S01:WHOLE",
      "P20-W04-S01:WHOLE",
      "P20-W05-S01:WHOLE",
      "P20-W06-S01:WHOLE",
      "P20-W07-S01:WHOLE",
      "P20-W08-S01:WHOLE",
      "P20-W09-S01:WHOLE",
      "P20-W10-S01:WHOLE",
      "P20-W11-S01:WHOLE",
      "P20-W12-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P02-W02-S01::1",
        "P10-W02-S01::0",
        "P11-W02-S01::whole",
        "P11-W01-S01::whole",
        "P11-W05-S01::whole",
        "P14-W05-S01::whole",
        "P14-W04-S01::whole",
        "P14-W03-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P17-W06-S01::whole",
        "P17-W07-S01::whole",
        "P17-W08-S01::whole",
        "P11-W04-S01::whole",
        "P11-W03-S01::whole",
        "P12-W02-S01::whole",
        "P12-W01-S01::whole",
        "P08-W03-S01::whole",
        "P08-W04-S01::whole",
        "P08-W01-S01::whole",
        "P08-W02-S01::whole",
        "P01-W08-S01::whole",
        "P01-W05-S01::whole",
        "P01-W01-S01::whole",
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W04-S01::whole",
        "P01-W09-S01::whole",
        "P01-W10-S01::whole",
        "P01-W07-S01::whole",
        "P01-W06-S01::whole"
      ],
      [
        "P17-W01-S01::whole",
        "P17-W05-S01::whole",
        "P17-W10-S01::whole",
        "P17-W09-S01::whole",
        "P17-W04-S01::whole",
        "P17-W03-S01::whole",
        "P17-W02-S01::whole",
        "P20-W12-S01::whole",
        "P20-W11-S01::whole",
        "P20-W10-S01::whole",
        "P20-W03-S01::whole",
        "P20-W09-S01::whole",
        "P20-W08-S01::whole",
        "P20-W07-S01::whole",
        "P20-W02-S01::whole",
        "P07-W09-S01::whole",
        "P07-W04-S01::whole",
        "P26-W02-S01::whole",
        "P24-W01-S01::whole",
        "P24-W02-S01::whole",
        "P23-W01-S01::whole",
        "P23-W02-S01::whole",
        "P26-W01-S01::whole",
        "P25-W01-S01::whole",
        "P25-W02-S01::whole",
        "P05-W01-S01::whole",
        "P05-W04-S01::whole",
        "P05-W03-S01::whole",
        "P05-W02-S01::whole",
        "P07-W07-S01::whole",
        "P07-W06-S01::whole",
        "P07-W10-S01::whole",
        "P07-W03-S01::whole",
        "P07-W02-S01::whole",
        "P07-W01-S01::whole",
        "P07-W05-S01::whole",
        "P07-W08-S01::whole",
        "P22-W02-S01::whole",
        "P20-W04-S01::whole",
        "P20-W05-S01::whole",
        "P20-W06-S01::whole",
        "P20-W01-S01::whole",
        "P22-W01-S01::whole",
        "P06-W04-S01::0",
        "P10-W02-S01::1"
      ],
      [
        "P10-W01-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::0",
        "P03-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W01-S01::whole",
        "P15-W02-S01::whole",
        "P15-W03-S01::whole",
        "P15-W04-S01::whole",
        "P15-W01-S01::whole",
        "P19-W03-S01::whole",
        "P19-W04-S01::whole"
      ],
      [
        "P04-W04-S01::whole",
        "P04-W05-S01::whole",
        "P04-W01-S01::whole",
        "P05-W05-S01::whole",
        "P05-W06-S01::whole",
        "P04-W03-S01::whole",
        "P04-W02-S01::whole",
        "P16-W01-S01::whole",
        "P06-W04-S01::1",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P06-W03-S01::whole",
        "P13-W01-S01::whole",
        "P13-W02-S01::whole",
        "P18-W04-S01::whole",
        "P18-W05-S01::whole",
        "P18-W06-S01::whole",
        "P18-W01-S01::whole",
        "P18-W02-S01::whole",
        "P18-W07-S01::whole",
        "P18-W08-S01::whole",
        "P18-W09-S01::whole",
        "P18-W03-S01::whole",
        "P18-W10-S01::whole",
        "P18-W11-S01::whole",
        "P18-W12-S01::whole",
        "P21-W02-S01::whole",
        "P21-W01-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole"
      ]
    ],
    "directions": [
      [
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1
      ],
      [
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_07",
    "seed": 20261111,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 6160.654835464882,
    "historical_native_cmax_s": 6160.654835464882,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 439,
    "next_index": [
      123,
      179,
      18,
      119
    ],
    "template_lengths": [
      123,
      179,
      47,
      119
    ],
    "completion_s": [
      6160.654835464882,
      6159.86730152539,
      3021.709777675817,
      5164.324801065033
    ],
    "wait_for_graph": [
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 202,
      "robot": 2,
      "operation": "R2:B4:POST",
      "kind": "POST",
      "block_id": "P09-W02-S01::whole",
      "ready_s": 3021.709777675817,
      "start": [
        13.683336425781249,
        0.7230000000000107
      ],
      "end": [
        13.683336425781249,
        0.7230000000000107
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R3:B12:MOVE"
      ],
      "graph": [
        [
          2,
          [
            3
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 2,
        "next_operation": "R2:B4:POST",
        "kind": "POST",
        "block_id": "P09-W02-S01::whole",
        "ready_s": 3021.709777675817,
        "start": [
          13.683336425781249,
          0.7230000000000107
        ],
        "end": [
          13.683336425781249,
          0.7230000000000107
        ],
        "same_rail_blockers": 72,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B12:MOVE",
          "R3:B12:POST",
          "R3:B12:SETUP",
          "R3:B12:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 194,
      "robot": 3,
      "operation": "R3:B12:MOVE",
      "ready_s": 2951.506443966584,
      "earliest_start_s": 3017.0164124021962
    },
    "blocked_edge_occurrences": {
      "same": 238,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 8,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P02-W02-S01::1",
        "direction": 1,
        "start": [
          0.251,
          6.752323242187501
        ],
        "end": [
          0.251,
          5.963606445312501
        ]
      },
      {
        "robot": 1,
        "block": "P17-W01-S01::whole",
        "direction": 0,
        "start": [
          10.726102480587418,
          11.64
        ],
        "end": [
          10.726102480587418,
          11.016000000000002
        ]
      },
      {
        "robot": 2,
        "block": "P10-W01-S01::whole",
        "direction": 0,
        "start": [
          13.62,
          5.3130244140625
        ],
        "end": [
          1.1199999999999999,
          5.31302441406249
        ]
      },
      {
        "robot": 3,
        "block": "P04-W04-S01::whole",
        "direction": 1,
        "start": [
          19.266,
          0.01
        ],
        "end": [
          19.271,
          2.661
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 190.14196810042085,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P01-W07-S01:WHOLE",
      "P01-W08-S01:WHOLE",
      "P01-W09-S01:WHOLE",
      "P01-W10-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:Y_SPLIT:MIDPOINT",
      "P03-W01-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P04-W03-S01:WHOLE",
      "P04-W04-S01:WHOLE",
      "P04-W05-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P05-W05-S01:WHOLE",
      "P05-W06-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P06-W03-S01:WHOLE",
      "P06-W04-S01:Y_SPLIT:BY_CENTER",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:WHOLE",
      "P07-W05-S01:WHOLE",
      "P07-W06-S01:WHOLE",
      "P07-W07-S01:WHOLE",
      "P07-W08-S01:WHOLE",
      "P07-W09-S01:WHOLE",
      "P07-W10-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:X_SPLIT:UPPER:BX_CENTER",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P11-W03-S01:WHOLE",
      "P11-W04-S01:WHOLE",
      "P11-W05-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P14-W03-S01:WHOLE",
      "P14-W04-S01:WHOLE",
      "P14-W05-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:WHOLE",
      "P17-W05-S01:WHOLE",
      "P17-W06-S01:WHOLE",
      "P17-W07-S01:WHOLE",
      "P17-W08-S01:WHOLE",
      "P17-W09-S01:WHOLE",
      "P17-W10-S01:WHOLE",
      "P18-W01-S01:WHOLE",
      "P18-W02-S01:WHOLE",
      "P18-W03-S01:WHOLE",
      "P18-W04-S01:WHOLE",
      "P18-W05-S01:WHOLE",
      "P18-W06-S01:WHOLE",
      "P18-W07-S01:WHOLE",
      "P18-W08-S01:WHOLE",
      "P18-W09-S01:WHOLE",
      "P18-W10-S01:WHOLE",
      "P18-W11-S01:WHOLE",
      "P18-W12-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P20-W03-S01:WHOLE",
      "P20-W04-S01:WHOLE",
      "P20-W05-S01:WHOLE",
      "P20-W06-S01:WHOLE",
      "P20-W07-S01:WHOLE",
      "P20-W08-S01:WHOLE",
      "P20-W09-S01:WHOLE",
      "P20-W10-S01:WHOLE",
      "P20-W11-S01:WHOLE",
      "P20-W12-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P02-W02-S01::1",
        "P10-W02-S01::0",
        "P11-W02-S01::whole",
        "P11-W01-S01::whole",
        "P11-W05-S01::whole",
        "P14-W05-S01::whole",
        "P14-W04-S01::whole",
        "P14-W03-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P17-W10-S01::whole",
        "P17-W07-S01::whole",
        "P17-W08-S01::whole",
        "P11-W04-S01::whole",
        "P11-W03-S01::whole",
        "P12-W02-S01::whole",
        "P12-W01-S01::whole",
        "P08-W03-S01::whole",
        "P08-W04-S01::whole",
        "P08-W01-S01::whole",
        "P08-W02-S01::whole",
        "P01-W08-S01::whole",
        "P01-W05-S01::whole",
        "P01-W01-S01::whole",
        "P01-W02-S01::whole",
        "P01-W03-S01::whole",
        "P01-W04-S01::whole",
        "P01-W09-S01::whole",
        "P01-W10-S01::whole",
        "P01-W07-S01::whole",
        "P01-W06-S01::whole"
      ],
      [
        "P17-W01-S01::whole",
        "P17-W05-S01::whole",
        "P17-W06-S01::whole",
        "P17-W09-S01::whole",
        "P17-W04-S01::whole",
        "P17-W03-S01::whole",
        "P17-W02-S01::whole",
        "P20-W12-S01::whole",
        "P20-W11-S01::whole",
        "P20-W10-S01::whole",
        "P20-W03-S01::whole",
        "P20-W09-S01::whole",
        "P20-W08-S01::whole",
        "P20-W07-S01::whole",
        "P20-W02-S01::whole",
        "P07-W09-S01::whole",
        "P07-W04-S01::whole",
        "P26-W02-S01::whole",
        "P24-W01-S01::whole",
        "P24-W02-S01::whole",
        "P23-W01-S01::whole",
        "P23-W02-S01::whole",
        "P26-W01-S01::whole",
        "P25-W01-S01::whole",
        "P25-W02-S01::whole",
        "P05-W01-S01::whole",
        "P05-W04-S01::whole",
        "P05-W03-S01::whole",
        "P05-W02-S01::whole",
        "P07-W07-S01::whole",
        "P07-W06-S01::whole",
        "P07-W10-S01::whole",
        "P07-W03-S01::whole",
        "P07-W02-S01::whole",
        "P07-W01-S01::whole",
        "P07-W05-S01::whole",
        "P07-W08-S01::whole",
        "P22-W02-S01::whole",
        "P20-W04-S01::whole",
        "P20-W05-S01::whole",
        "P20-W06-S01::whole",
        "P20-W01-S01::whole",
        "P22-W01-S01::whole",
        "P06-W04-S01::0",
        "P10-W02-S01::1"
      ],
      [
        "P10-W01-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::0",
        "P03-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W01-S01::whole",
        "P15-W02-S01::whole",
        "P15-W03-S01::whole",
        "P15-W04-S01::whole",
        "P15-W01-S01::whole",
        "P19-W03-S01::whole",
        "P19-W04-S01::whole"
      ],
      [
        "P04-W04-S01::whole",
        "P04-W05-S01::whole",
        "P04-W01-S01::whole",
        "P05-W05-S01::whole",
        "P05-W06-S01::whole",
        "P04-W03-S01::whole",
        "P04-W02-S01::whole",
        "P16-W01-S01::whole",
        "P06-W04-S01::1",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P06-W03-S01::whole",
        "P13-W01-S01::whole",
        "P13-W02-S01::whole",
        "P18-W04-S01::whole",
        "P18-W05-S01::whole",
        "P18-W06-S01::whole",
        "P18-W01-S01::whole",
        "P18-W02-S01::whole",
        "P18-W07-S01::whole",
        "P18-W08-S01::whole",
        "P18-W09-S01::whole",
        "P18-W03-S01::whole",
        "P18-W10-S01::whole",
        "P18-W11-S01::whole",
        "P18-W12-S01::whole",
        "P21-W02-S01::whole",
        "P21-W01-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole"
      ]
    ],
    "directions": [
      [
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1
      ],
      [
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_07",
    "seed": 20261112,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 6167.828488850058,
    "historical_native_cmax_s": 6167.828488850058,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 439,
    "next_index": [
      123,
      179,
      18,
      119
    ],
    "template_lengths": [
      123,
      179,
      47,
      119
    ],
    "completion_s": [
      6167.828488850058,
      6161.024167075806,
      3021.709777675817,
      5164.324801065033
    ],
    "wait_for_graph": [
      [
        2,
        [
          3
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 222,
      "robot": 2,
      "operation": "R2:B4:POST",
      "kind": "POST",
      "block_id": "P09-W02-S01::whole",
      "ready_s": 3021.709777675817,
      "start": [
        13.683336425781249,
        0.7230000000000107
      ],
      "end": [
        13.683336425781249,
        0.7230000000000107
      ],
      "duration_s": 30.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R3:B12:MOVE"
      ],
      "graph": [
        [
          2,
          [
            3
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 2,
        "next_operation": "R2:B4:POST",
        "kind": "POST",
        "block_id": "P09-W02-S01::whole",
        "ready_s": 3021.709777675817,
        "start": [
          13.683336425781249,
          0.7230000000000107
        ],
        "end": [
          13.683336425781249,
          0.7230000000000107
        ],
        "same_rail_blockers": 72,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R3:B12:MOVE",
          "R3:B12:POST",
          "R3:B12:SETUP",
          "R3:B12:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 216,
      "robot": 3,
      "operation": "R3:B12:MOVE",
      "ready_s": 2951.506443966584,
      "earliest_start_s": 3017.0164124021962
    },
    "blocked_edge_occurrences": {
      "same": 218,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 6,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 2,
        "blocker": 3,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P11-W05-S01::whole",
        "direction": 0,
        "start": [
          7.028102480587419,
          7.116024414062501
        ],
        "end": [
          7.028102480587419,
          9.2450244140625
        ]
      },
      {
        "robot": 1,
        "block": "P20-W12-S01::whole",
        "direction": 0,
        "start": [
          13.479398071289063,
          11.695
        ],
        "end": [
          12.779398071289055,
          11.69668017292349
        ]
      },
      {
        "robot": 2,
        "block": "P10-W01-S01::whole",
        "direction": 0,
        "start": [
          13.62,
          5.3130244140625
        ],
        "end": [
          1.1199999999999999,
          5.31302441406249
        ]
      },
      {
        "robot": 3,
        "block": "P04-W04-S01::whole",
        "direction": 1,
        "start": [
          19.266,
          0.01
        ],
        "end": [
          19.271,
          2.661
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 190.14196810042085,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:WHOLE",
      "P01-W03-S01:WHOLE",
      "P01-W04-S01:WHOLE",
      "P01-W05-S01:WHOLE",
      "P01-W06-S01:WHOLE",
      "P01-W07-S01:WHOLE",
      "P01-W08-S01:WHOLE",
      "P01-W09-S01:WHOLE",
      "P01-W10-S01:WHOLE",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:Y_SPLIT:MIDPOINT",
      "P03-W01-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P04-W03-S01:WHOLE",
      "P04-W04-S01:WHOLE",
      "P04-W05-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P05-W03-S01:WHOLE",
      "P05-W04-S01:WHOLE",
      "P05-W05-S01:WHOLE",
      "P05-W06-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P06-W03-S01:WHOLE",
      "P06-W04-S01:Y_SPLIT:BY_CENTER",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:WHOLE",
      "P07-W05-S01:WHOLE",
      "P07-W06-S01:WHOLE",
      "P07-W07-S01:WHOLE",
      "P07-W08-S01:WHOLE",
      "P07-W09-S01:WHOLE",
      "P07-W10-S01:WHOLE",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:X_SPLIT:UPPER:BX_OUTER_UPPER",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P11-W03-S01:WHOLE",
      "P11-W04-S01:WHOLE",
      "P11-W05-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P14-W03-S01:WHOLE",
      "P14-W04-S01:WHOLE",
      "P14-W05-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P17-W03-S01:WHOLE",
      "P17-W04-S01:WHOLE",
      "P17-W05-S01:WHOLE",
      "P17-W06-S01:WHOLE",
      "P17-W07-S01:WHOLE",
      "P17-W08-S01:WHOLE",
      "P17-W09-S01:WHOLE",
      "P17-W10-S01:WHOLE",
      "P18-W01-S01:WHOLE",
      "P18-W02-S01:WHOLE",
      "P18-W03-S01:WHOLE",
      "P18-W04-S01:WHOLE",
      "P18-W05-S01:WHOLE",
      "P18-W06-S01:WHOLE",
      "P18-W07-S01:WHOLE",
      "P18-W08-S01:WHOLE",
      "P18-W09-S01:WHOLE",
      "P18-W10-S01:WHOLE",
      "P18-W11-S01:WHOLE",
      "P18-W12-S01:WHOLE",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P19-W03-S01:WHOLE",
      "P19-W04-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE",
      "P20-W03-S01:WHOLE",
      "P20-W04-S01:WHOLE",
      "P20-W05-S01:WHOLE",
      "P20-W06-S01:WHOLE",
      "P20-W07-S01:WHOLE",
      "P20-W08-S01:WHOLE",
      "P20-W09-S01:WHOLE",
      "P20-W10-S01:WHOLE",
      "P20-W11-S01:WHOLE",
      "P20-W12-S01:WHOLE",
      "P21-W01-S01:WHOLE",
      "P21-W02-S01:WHOLE",
      "P22-W01-S01:WHOLE",
      "P22-W02-S01:WHOLE",
      "P23-W01-S01:WHOLE",
      "P23-W02-S01:WHOLE",
      "P24-W01-S01:WHOLE",
      "P24-W02-S01:WHOLE",
      "P25-W01-S01:WHOLE",
      "P25-W02-S01:WHOLE",
      "P26-W01-S01:WHOLE",
      "P26-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P11-W05-S01::whole",
        "P14-W05-S01::whole",
        "P14-W03-S01::whole",
        "P14-W04-S01::whole",
        "P14-W02-S01::whole",
        "P14-W01-S01::whole",
        "P08-W02-S01::whole",
        "P08-W01-S01::whole",
        "P08-W04-S01::whole",
        "P08-W03-S01::whole",
        "P12-W01-S01::whole",
        "P12-W02-S01::whole",
        "P01-W01-S01::whole",
        "P01-W05-S01::whole",
        "P01-W08-S01::whole",
        "P01-W06-S01::whole",
        "P01-W07-S01::whole",
        "P01-W10-S01::whole",
        "P01-W09-S01::whole",
        "P01-W04-S01::whole",
        "P01-W03-S01::whole",
        "P01-W02-S01::whole",
        "P02-W02-S01::1",
        "P10-W02-S01::0",
        "P11-W02-S01::whole",
        "P11-W01-S01::whole",
        "P11-W03-S01::whole",
        "P11-W04-S01::whole",
        "P17-W06-S01::whole",
        "P17-W09-S01::whole",
        "P20-W03-S01::whole"
      ],
      [
        "P20-W12-S01::whole",
        "P20-W11-S01::whole",
        "P20-W10-S01::whole",
        "P20-W09-S01::whole",
        "P20-W08-S01::whole",
        "P20-W07-S01::whole",
        "P20-W02-S01::whole",
        "P20-W01-S01::whole",
        "P20-W06-S01::whole",
        "P20-W05-S01::whole",
        "P20-W04-S01::whole",
        "P17-W10-S01::whole",
        "P17-W04-S01::whole",
        "P17-W03-S01::whole",
        "P17-W02-S01::whole",
        "P17-W01-S01::whole",
        "P17-W05-S01::whole",
        "P17-W08-S01::whole",
        "P17-W07-S01::whole",
        "P10-W02-S01::1",
        "P06-W04-S01::0",
        "P22-W01-S01::whole",
        "P22-W02-S01::whole",
        "P07-W08-S01::whole",
        "P07-W05-S01::whole",
        "P07-W01-S01::whole",
        "P07-W02-S01::whole",
        "P07-W03-S01::whole",
        "P07-W04-S01::whole",
        "P07-W09-S01::whole",
        "P07-W06-S01::whole",
        "P07-W07-S01::whole",
        "P07-W10-S01::whole",
        "P25-W02-S01::whole",
        "P25-W01-S01::whole",
        "P05-W04-S01::whole",
        "P05-W03-S01::whole",
        "P05-W02-S01::whole",
        "P05-W01-S01::whole",
        "P24-W02-S01::whole",
        "P24-W01-S01::whole",
        "P26-W02-S01::whole",
        "P26-W01-S01::whole",
        "P23-W02-S01::whole",
        "P23-W01-S01::whole"
      ],
      [
        "P10-W01-S01::whole",
        "P02-W01-S01::whole",
        "P02-W02-S01::0",
        "P03-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W01-S01::whole",
        "P15-W02-S01::whole",
        "P15-W03-S01::whole",
        "P15-W04-S01::whole",
        "P15-W01-S01::whole",
        "P19-W03-S01::whole",
        "P19-W04-S01::whole"
      ],
      [
        "P04-W04-S01::whole",
        "P04-W05-S01::whole",
        "P04-W01-S01::whole",
        "P05-W05-S01::whole",
        "P05-W06-S01::whole",
        "P04-W03-S01::whole",
        "P04-W02-S01::whole",
        "P16-W01-S01::whole",
        "P06-W04-S01::1",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P06-W03-S01::whole",
        "P13-W01-S01::whole",
        "P13-W02-S01::whole",
        "P18-W04-S01::whole",
        "P18-W05-S01::whole",
        "P18-W06-S01::whole",
        "P18-W01-S01::whole",
        "P18-W02-S01::whole",
        "P18-W07-S01::whole",
        "P18-W08-S01::whole",
        "P18-W09-S01::whole",
        "P18-W03-S01::whole",
        "P18-W10-S01::whole",
        "P18-W11-S01::whole",
        "P18-W12-S01::whole",
        "P21-W02-S01::whole",
        "P21-W01-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1
      ],
      [
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1
      ],
      [
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_10",
    "seed": 20261110,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 6138.083333819142,
    "historical_native_cmax_s": 6138.083333819142,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 14,
    "rollout_budget": 32,
    "frontier_exhausted": true,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 352,
    "next_index": [
      87,
      17,
      127,
      121
    ],
    "template_lengths": [
      87,
      19,
      127,
      131
    ],
    "completion_s": [
      5266.3161198980815,
      4874.37962962963,
      5675.072497097752,
      4753.591523513655
    ],
    "wait_for_graph": [
      [
        1,
        [
          0
        ]
      ],
      [
        3,
        [
          2
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 347,
      "robot": 3,
      "operation": "R3:B30:WELD",
      "kind": "WELD",
      "block_id": "P16-W05-S01::whole",
      "ready_s": 4753.591523513655,
      "start": [
        11.413784170910837,
        3.469801635742187
      ],
      "end": [
        11.413784170910837,
        4.1698016357421865
      ],
      "duration_s": 64.81481481481478,
      "blocker_count": 1,
      "blocker_examples": [
        "R2:B31:WELD"
      ],
      "graph": [
        [
          3,
          [
            2
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 1,
        "next_operation": "R1:B4:WELD",
        "kind": "WELD",
        "block_id": "P01-W02-S01::1",
        "ready_s": 4874.37962962963,
        "start": [
          20.0,
          11.277000000000001
        ],
        "end": [
          13.75,
          11.276999999999996
        ],
        "same_rail_blockers": 2,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R0:B21:POST",
          "R0:B21:WELD"
        ],
        "cross_examples": []
      },
      {
        "robot": 3,
        "next_operation": "R3:B30:WELD",
        "kind": "WELD",
        "block_id": "P16-W05-S01::whole",
        "ready_s": 4753.591523513655,
        "start": [
          11.413784170910837,
          3.469801635742187
        ],
        "end": [
          11.413784170910837,
          4.1698016357421865
        ],
        "same_rail_blockers": 2,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R2:B31:POST",
          "R2:B31:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": null,
    "blocked_edge_occurrences": {
      "same": 9,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 0,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 1,
        "blocker": 0,
        "finished": true
      },
      {
        "waiting": 3,
        "blocker": 2,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P01-W02-S01::0",
        "direction": 0,
        "start": [
          13.75,
          11.276999999999996
        ],
        "end": [
          7.5,
          11.27699999999999
        ]
      },
      {
        "robot": 1,
        "block": "P12-W01-S01::whole",
        "direction": 1,
        "start": [
          20.0,
          6.327000000000014
        ],
        "end": [
          7.5,
          6.327000000000003
        ]
      },
      {
        "robot": 2,
        "block": "P10-W07-S01::whole",
        "direction": 1,
        "start": [
          3.322,
          4.8889755859375
        ],
        "end": [
          3.322,
          3.5719755859375
        ]
      },
      {
        "robot": 3,
        "block": "P16-W09-S01::whole",
        "direction": 1,
        "start": [
          13.133784170910836,
          4.1698016357421865
        ],
        "end": [
          13.133784170910836,
          3.469801635742187
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 1097.8835379251063,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:X_SPLIT:UPPER:BX_CENTER",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P03-W03-S01:WHOLE",
      "P03-W04-S01:WHOLE",
      "P03-W05-S01:WHOLE",
      "P03-W06-S01:WHOLE",
      "P03-W07-S01:WHOLE",
      "P03-W08-S01:WHOLE",
      "P03-W09-S01:WHOLE",
      "P03-W10-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:Y_SPLIT:BY_CENTER",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P08-W05-S01:WHOLE",
      "P08-W06-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P09-W03-S01:WHOLE",
      "P09-W04-S01:WHOLE",
      "P09-W05-S01:WHOLE",
      "P09-W06-S01:WHOLE",
      "P09-W07-S01:WHOLE",
      "P09-W08-S01:WHOLE",
      "P09-W09-S01:WHOLE",
      "P09-W10-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P10-W03-S01:WHOLE",
      "P10-W04-S01:WHOLE",
      "P10-W05-S01:WHOLE",
      "P10-W06-S01:WHOLE",
      "P10-W07-S01:WHOLE",
      "P10-W08-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:WHOLE",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P13-W09-S01:WHOLE",
      "P13-W10-S01:WHOLE",
      "P13-W11-S01:WHOLE",
      "P13-W12-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P16-W03-S01:WHOLE",
      "P16-W04-S01:WHOLE",
      "P16-W05-S01:WHOLE",
      "P16-W06-S01:WHOLE",
      "P16-W07-S01:WHOLE",
      "P16-W08-S01:WHOLE",
      "P16-W09-S01:WHOLE",
      "P16-W10-S01:WHOLE",
      "P16-W11-S01:WHOLE",
      "P16-W12-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P18-W01-S01:Y_SPLIT:BY_CENTER",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P01-W02-S01::0",
        "P02-W01-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P02-W02-S01::whole",
        "P05-W02-S01::whole",
        "P05-W01-S01::whole",
        "P08-W06-S01::whole",
        "P08-W05-S01::whole",
        "P17-W01-S01::whole",
        "P17-W02-S01::whole",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P08-W02-S01::whole",
        "P08-W03-S01::whole",
        "P08-W01-S01::whole",
        "P08-W04-S01::whole",
        "P07-W03-S01::whole",
        "P07-W02-S01::whole",
        "P07-W04-S01::1",
        "P18-W01-S01::1",
        "P12-W02-S01::whole"
      ],
      [
        "P12-W01-S01::whole",
        "P04-W01-S01::whole",
        "P04-W02-S01::whole",
        "P01-W01-S01::whole",
        "P01-W02-S01::1"
      ],
      [
        "P10-W07-S01::whole",
        "P10-W06-S01::whole",
        "P10-W08-S01::whole",
        "P07-W01-S01::whole",
        "P07-W04-S01::0",
        "P09-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W03-S01::whole",
        "P09-W04-S01::whole",
        "P09-W05-S01::whole",
        "P09-W08-S01::whole",
        "P09-W07-S01::whole",
        "P09-W06-S01::whole",
        "P09-W09-S01::whole",
        "P09-W10-S01::whole",
        "P10-W05-S01::whole",
        "P10-W04-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P10-W03-S01::whole",
        "P03-W04-S01::whole",
        "P03-W05-S01::whole",
        "P18-W01-S01::0",
        "P03-W08-S01::whole",
        "P03-W06-S01::whole",
        "P03-W07-S01::whole",
        "P03-W10-S01::whole",
        "P03-W09-S01::whole",
        "P03-W03-S01::whole",
        "P03-W02-S01::whole",
        "P03-W01-S01::whole",
        "P11-W01-S01::whole"
      ],
      [
        "P16-W09-S01::whole",
        "P16-W08-S01::whole",
        "P16-W07-S01::whole",
        "P16-W02-S01::whole",
        "P16-W03-S01::whole",
        "P16-W10-S01::whole",
        "P16-W11-S01::whole",
        "P16-W12-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole",
        "P20-W02-S01::whole",
        "P20-W01-S01::whole",
        "P13-W12-S01::whole",
        "P13-W11-S01::whole",
        "P13-W10-S01::whole",
        "P13-W03-S01::whole",
        "P13-W09-S01::whole",
        "P13-W08-S01::whole",
        "P13-W07-S01::whole",
        "P13-W02-S01::whole",
        "P13-W01-S01::whole",
        "P13-W06-S01::whole",
        "P13-W05-S01::whole",
        "P13-W04-S01::whole",
        "P11-W02-S01::whole",
        "P15-W03-S01::whole",
        "P15-W04-S01::whole",
        "P15-W02-S01::whole",
        "P15-W01-S01::whole",
        "P16-W04-S01::whole",
        "P16-W05-S01::whole",
        "P16-W06-S01::whole",
        "P16-W01-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0
      ],
      [
        1,
        1,
        0,
        1,
        0
      ],
      [
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_10",
    "seed": 20261111,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 5544.274746701505,
    "historical_native_cmax_s": 5544.274746701505,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 32,
    "rollout_budget": 32,
    "frontier_exhausted": false,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 352,
    "next_index": [
      79,
      20,
      123,
      130
    ],
    "template_lengths": [
      79,
      31,
      123,
      135
    ],
    "completion_s": [
      5491.334528240328,
      2938.241396811875,
      5422.75549442495,
      4099.753588021585
    ],
    "wait_for_graph": [
      [
        1,
        [
          0
        ]
      ],
      [
        3,
        [
          2
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 264,
      "robot": 1,
      "operation": "R1:B5:SETUP",
      "kind": "SETUP",
      "block_id": "P12-W02-S01::whole",
      "ready_s": 2938.241396811875,
      "start": [
        7.5,
        5.577000000000003
      ],
      "end": [
        7.5,
        5.577000000000003
      ],
      "duration_s": 20.0,
      "blocker_count": 1,
      "blocker_examples": [
        "R0:B17:MOVE"
      ],
      "graph": [
        [
          1,
          [
            0
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 1,
        "next_operation": "R1:B5:SETUP",
        "kind": "SETUP",
        "block_id": "P12-W02-S01::whole",
        "ready_s": 2938.241396811875,
        "start": [
          7.5,
          5.577000000000003
        ],
        "end": [
          7.5,
          5.577000000000003
        ],
        "same_rail_blockers": 12,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R0:B17:MOVE",
          "R0:B17:POST",
          "R0:B17:SETUP",
          "R0:B17:WELD"
        ],
        "cross_examples": []
      },
      {
        "robot": 3,
        "next_operation": "R3:B32:POST",
        "kind": "POST",
        "block_id": "P03-W05-S01::whole",
        "ready_s": 4099.753588021585,
        "start": [
          6.4728369140624995,
          1.2153261718750001
        ],
        "end": [
          6.4728369140624995,
          1.2153261718750001
        ],
        "same_rail_blockers": 8,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R2:B29:MOVE",
          "R2:B29:POST",
          "R2:B29:SETUP",
          "R2:B29:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": {
      "depth": 339,
      "robot": 2,
      "operation": "R2:B29:MOVE",
      "ready_s": 4079.369249066504,
      "earliest_start_s": 4098.504403860622
    },
    "blocked_edge_occurrences": {
      "same": 102,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 1,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 1,
        "blocker": 0,
        "finished": true
      },
      {
        "waiting": 3,
        "blocker": 2,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P17-W01-S01::whole",
        "direction": 0,
        "start": [
          0.15192382812500005,
          8.517207031249999
        ],
        "end": [
          0.43299121093750004,
          10.111216796874999
        ]
      },
      {
        "robot": 1,
        "block": "P08-W05-S01::whole",
        "direction": 1,
        "start": [
          2.100023681640617,
          9.459630737304703
        ],
        "end": [
          3.2973721194267274,
          9.160980224609377
        ]
      },
      {
        "robot": 2,
        "block": "P07-W01-S01::whole",
        "direction": 1,
        "start": [
          2.1626982421875,
          5.275833984375
        ],
        "end": [
          1.5473383789062503,
          6.096865234375
        ]
      },
      {
        "robot": 3,
        "block": "P13-W04-S01::whole",
        "direction": 1,
        "start": [
          19.712999999999997,
          2.8661983642578157
        ],
        "end": [
          19.712999999999997,
          3.566198364257816
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 102.1447805925036,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:X_SPLIT:UPPER:BX_OUTER_UPPER",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P03-W03-S01:WHOLE",
      "P03-W04-S01:WHOLE",
      "P03-W05-S01:WHOLE",
      "P03-W06-S01:WHOLE",
      "P03-W07-S01:WHOLE",
      "P03-W08-S01:WHOLE",
      "P03-W09-S01:WHOLE",
      "P03-W10-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:Y_SPLIT:BY_CENTER",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P08-W05-S01:WHOLE",
      "P08-W06-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P09-W03-S01:WHOLE",
      "P09-W04-S01:WHOLE",
      "P09-W05-S01:WHOLE",
      "P09-W06-S01:WHOLE",
      "P09-W07-S01:WHOLE",
      "P09-W08-S01:WHOLE",
      "P09-W09-S01:WHOLE",
      "P09-W10-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P10-W03-S01:WHOLE",
      "P10-W04-S01:WHOLE",
      "P10-W05-S01:WHOLE",
      "P10-W06-S01:WHOLE",
      "P10-W07-S01:WHOLE",
      "P10-W08-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P12-W01-S01:Y_SPLIT:MIDPOINT",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:WHOLE",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P13-W09-S01:WHOLE",
      "P13-W10-S01:WHOLE",
      "P13-W11-S01:WHOLE",
      "P13-W12-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P16-W03-S01:WHOLE",
      "P16-W04-S01:WHOLE",
      "P16-W05-S01:WHOLE",
      "P16-W06-S01:WHOLE",
      "P16-W07-S01:WHOLE",
      "P16-W08-S01:WHOLE",
      "P16-W09-S01:WHOLE",
      "P16-W10-S01:WHOLE",
      "P16-W11-S01:WHOLE",
      "P16-W12-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P18-W01-S01:Y_SPLIT:BY_CENTER",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P17-W01-S01::whole",
        "P17-W02-S01::whole",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P08-W02-S01::whole",
        "P08-W03-S01::whole",
        "P08-W01-S01::whole",
        "P08-W04-S01::whole",
        "P07-W03-S01::whole",
        "P07-W02-S01::whole",
        "P07-W04-S01::1",
        "P08-W06-S01::whole",
        "P05-W01-S01::whole",
        "P05-W02-S01::whole",
        "P02-W02-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P12-W01-S01::0",
        "P01-W02-S01::0",
        "P01-W01-S01::whole"
      ],
      [
        "P08-W05-S01::whole",
        "P02-W01-S01::whole",
        "P04-W02-S01::whole",
        "P04-W01-S01::whole",
        "P18-W01-S01::1",
        "P12-W02-S01::whole",
        "P12-W01-S01::1",
        "P01-W02-S01::1"
      ],
      [
        "P07-W01-S01::whole",
        "P07-W04-S01::0",
        "P09-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W03-S01::whole",
        "P09-W09-S01::whole",
        "P09-W10-S01::whole",
        "P09-W07-S01::whole",
        "P09-W06-S01::whole",
        "P09-W08-S01::whole",
        "P09-W05-S01::whole",
        "P09-W04-S01::whole",
        "P10-W07-S01::whole",
        "P10-W08-S01::whole",
        "P10-W06-S01::whole",
        "P10-W05-S01::whole",
        "P10-W04-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P10-W03-S01::whole",
        "P03-W04-S01::whole",
        "P03-W09-S01::whole",
        "P03-W07-S01::whole",
        "P18-W01-S01::0",
        "P03-W08-S01::whole",
        "P03-W06-S01::whole",
        "P03-W10-S01::whole",
        "P03-W03-S01::whole",
        "P03-W02-S01::whole",
        "P03-W01-S01::whole",
        "P11-W01-S01::whole"
      ],
      [
        "P13-W04-S01::whole",
        "P13-W05-S01::whole",
        "P13-W06-S01::whole",
        "P13-W01-S01::whole",
        "P13-W02-S01::whole",
        "P13-W07-S01::whole",
        "P13-W08-S01::whole",
        "P13-W09-S01::whole",
        "P13-W03-S01::whole",
        "P13-W10-S01::whole",
        "P13-W11-S01::whole",
        "P13-W12-S01::whole",
        "P20-W01-S01::whole",
        "P20-W02-S01::whole",
        "P19-W02-S01::whole",
        "P19-W01-S01::whole",
        "P16-W12-S01::whole",
        "P16-W11-S01::whole",
        "P16-W10-S01::whole",
        "P16-W03-S01::whole",
        "P16-W09-S01::whole",
        "P16-W08-S01::whole",
        "P16-W07-S01::whole",
        "P16-W02-S01::whole",
        "P16-W01-S01::whole",
        "P16-W06-S01::whole",
        "P16-W05-S01::whole",
        "P16-W04-S01::whole",
        "P15-W01-S01::whole",
        "P15-W02-S01::whole",
        "P15-W04-S01::whole",
        "P15-W03-S01::whole",
        "P03-W05-S01::whole",
        "P11-W02-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1
      ],
      [
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1
      ],
      [
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1
      ]
    ]
  },
  {
    "layout_id": "validation_10",
    "seed": 20261112,
    "reproduced": true,
    "iterations": 500,
    "search_rerun_iterations": 0,
    "replay_mode": "SAVED_INCUMBENT",
    "native_cmax_s": 6045.490741217289,
    "historical_native_cmax_s": 6045.490741217289,
    "b32_status": "DEADLOCK",
    "b32_cmax_s": null,
    "recovery_rollouts": 14,
    "rollout_budget": 32,
    "frontier_exhausted": true,
    "baseline_status": "DEADLOCK",
    "baseline_matches_b32_failure": true,
    "terminal_depth": 352,
    "next_index": [
      87,
      17,
      127,
      121
    ],
    "template_lengths": [
      87,
      19,
      127,
      131
    ],
    "completion_s": [
      5312.612416194377,
      4874.37962962963,
      5675.072497097752,
      4753.591523513655
    ],
    "wait_for_graph": [
      [
        1,
        [
          0
        ]
      ],
      [
        3,
        [
          2
        ]
      ]
    ],
    "wait_for_cycles": [],
    "first_block": {
      "depth": 347,
      "robot": 3,
      "operation": "R3:B30:WELD",
      "kind": "WELD",
      "block_id": "P16-W05-S01::whole",
      "ready_s": 4753.591523513655,
      "start": [
        11.413784170910837,
        3.469801635742187
      ],
      "end": [
        11.413784170910837,
        4.1698016357421865
      ],
      "duration_s": 64.81481481481478,
      "blocker_count": 1,
      "blocker_examples": [
        "R2:B31:WELD"
      ],
      "graph": [
        [
          3,
          [
            2
          ]
        ]
      ]
    },
    "terminal_blockers": [
      {
        "robot": 1,
        "next_operation": "R1:B4:WELD",
        "kind": "WELD",
        "block_id": "P01-W02-S01::1",
        "ready_s": 4874.37962962963,
        "start": [
          20.0,
          11.277000000000001
        ],
        "end": [
          14.25,
          11.276999999999996
        ],
        "same_rail_blockers": 2,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R0:B21:POST",
          "R0:B21:WELD"
        ],
        "cross_examples": []
      },
      {
        "robot": 3,
        "next_operation": "R3:B30:WELD",
        "kind": "WELD",
        "block_id": "P16-W05-S01::whole",
        "ready_s": 4753.591523513655,
        "start": [
          11.413784170910837,
          3.469801635742187
        ],
        "end": [
          11.413784170910837,
          4.1698016357421865
        ],
        "same_rail_blockers": 2,
        "cross_rail_blockers": 0,
        "same_examples": [
          "R2:B31:POST",
          "R2:B31:WELD"
        ],
        "cross_examples": []
      }
    ],
    "first_delayed_choice": null,
    "blocked_edge_occurrences": {
      "same": 9,
      "cross": 0
    },
    "intermediate_wait_cycles": [],
    "delayed_edge_occurrences": {
      "same": 0,
      "cross": 0
    },
    "first_cross_delay": null,
    "all_reported_wait_cycles": [],
    "blocker_finished": [
      {
        "waiting": 1,
        "blocker": 0,
        "finished": true
      },
      {
        "waiting": 3,
        "blocker": 2,
        "finished": true
      }
    ],
    "first_tasks": [
      {
        "robot": 0,
        "block": "P01-W02-S01::0",
        "direction": 0,
        "start": [
          14.25,
          11.276999999999996
        ],
        "end": [
          7.5,
          11.27699999999999
        ]
      },
      {
        "robot": 1,
        "block": "P12-W01-S01::whole",
        "direction": 1,
        "start": [
          20.0,
          6.327000000000014
        ],
        "end": [
          7.5,
          6.327000000000003
        ]
      },
      {
        "robot": 2,
        "block": "P10-W07-S01::whole",
        "direction": 1,
        "start": [
          3.322,
          4.8889755859375
        ],
        "end": [
          3.322,
          3.5719755859375
        ]
      },
      {
        "robot": 3,
        "block": "P16-W09-S01::whole",
        "direction": 1,
        "start": [
          13.133784170910836,
          4.1698016357421865
        ],
        "end": [
          13.133784170910836,
          3.469801635742187
        ]
      }
    ],
    "native_coverage": true,
    "native_inter_robot_conflicts": 0,
    "native_chain_continuous": true,
    "native_weld_order": true,
    "native_wait_s": 958.9946490269576,
    "patterns": [
      "P01-W01-S01:WHOLE",
      "P01-W02-S01:X_SPLIT:UPPER:BX_OUTER_UPPER",
      "P02-W01-S01:WHOLE",
      "P02-W02-S01:WHOLE",
      "P03-W01-S01:WHOLE",
      "P03-W02-S01:WHOLE",
      "P03-W03-S01:WHOLE",
      "P03-W04-S01:WHOLE",
      "P03-W05-S01:WHOLE",
      "P03-W06-S01:WHOLE",
      "P03-W07-S01:WHOLE",
      "P03-W08-S01:WHOLE",
      "P03-W09-S01:WHOLE",
      "P03-W10-S01:WHOLE",
      "P04-W01-S01:WHOLE",
      "P04-W02-S01:WHOLE",
      "P05-W01-S01:WHOLE",
      "P05-W02-S01:WHOLE",
      "P06-W01-S01:WHOLE",
      "P06-W02-S01:WHOLE",
      "P07-W01-S01:WHOLE",
      "P07-W02-S01:WHOLE",
      "P07-W03-S01:WHOLE",
      "P07-W04-S01:Y_SPLIT:BY_CENTER",
      "P08-W01-S01:WHOLE",
      "P08-W02-S01:WHOLE",
      "P08-W03-S01:WHOLE",
      "P08-W04-S01:WHOLE",
      "P08-W05-S01:WHOLE",
      "P08-W06-S01:WHOLE",
      "P09-W01-S01:WHOLE",
      "P09-W02-S01:WHOLE",
      "P09-W03-S01:WHOLE",
      "P09-W04-S01:WHOLE",
      "P09-W05-S01:WHOLE",
      "P09-W06-S01:WHOLE",
      "P09-W07-S01:WHOLE",
      "P09-W08-S01:WHOLE",
      "P09-W09-S01:WHOLE",
      "P09-W10-S01:WHOLE",
      "P10-W01-S01:WHOLE",
      "P10-W02-S01:WHOLE",
      "P10-W03-S01:WHOLE",
      "P10-W04-S01:WHOLE",
      "P10-W05-S01:WHOLE",
      "P10-W06-S01:WHOLE",
      "P10-W07-S01:WHOLE",
      "P10-W08-S01:WHOLE",
      "P11-W01-S01:WHOLE",
      "P11-W02-S01:WHOLE",
      "P12-W01-S01:WHOLE",
      "P12-W02-S01:WHOLE",
      "P13-W01-S01:WHOLE",
      "P13-W02-S01:WHOLE",
      "P13-W03-S01:WHOLE",
      "P13-W04-S01:WHOLE",
      "P13-W05-S01:WHOLE",
      "P13-W06-S01:WHOLE",
      "P13-W07-S01:WHOLE",
      "P13-W08-S01:WHOLE",
      "P13-W09-S01:WHOLE",
      "P13-W10-S01:WHOLE",
      "P13-W11-S01:WHOLE",
      "P13-W12-S01:WHOLE",
      "P14-W01-S01:WHOLE",
      "P14-W02-S01:WHOLE",
      "P15-W01-S01:WHOLE",
      "P15-W02-S01:WHOLE",
      "P15-W03-S01:WHOLE",
      "P15-W04-S01:WHOLE",
      "P16-W01-S01:WHOLE",
      "P16-W02-S01:WHOLE",
      "P16-W03-S01:WHOLE",
      "P16-W04-S01:WHOLE",
      "P16-W05-S01:WHOLE",
      "P16-W06-S01:WHOLE",
      "P16-W07-S01:WHOLE",
      "P16-W08-S01:WHOLE",
      "P16-W09-S01:WHOLE",
      "P16-W10-S01:WHOLE",
      "P16-W11-S01:WHOLE",
      "P16-W12-S01:WHOLE",
      "P17-W01-S01:WHOLE",
      "P17-W02-S01:WHOLE",
      "P18-W01-S01:Y_SPLIT:BY_CENTER",
      "P19-W01-S01:WHOLE",
      "P19-W02-S01:WHOLE",
      "P20-W01-S01:WHOLE",
      "P20-W02-S01:WHOLE"
    ],
    "routes": [
      [
        "P01-W02-S01::0",
        "P02-W01-S01::whole",
        "P14-W01-S01::whole",
        "P14-W02-S01::whole",
        "P02-W02-S01::whole",
        "P05-W02-S01::whole",
        "P05-W01-S01::whole",
        "P08-W06-S01::whole",
        "P08-W05-S01::whole",
        "P17-W01-S01::whole",
        "P17-W02-S01::whole",
        "P06-W01-S01::whole",
        "P06-W02-S01::whole",
        "P08-W02-S01::whole",
        "P08-W03-S01::whole",
        "P08-W01-S01::whole",
        "P08-W04-S01::whole",
        "P07-W03-S01::whole",
        "P07-W02-S01::whole",
        "P07-W04-S01::1",
        "P18-W01-S01::1",
        "P12-W02-S01::whole"
      ],
      [
        "P12-W01-S01::whole",
        "P04-W01-S01::whole",
        "P04-W02-S01::whole",
        "P01-W01-S01::whole",
        "P01-W02-S01::1"
      ],
      [
        "P10-W07-S01::whole",
        "P10-W06-S01::whole",
        "P10-W08-S01::whole",
        "P07-W01-S01::whole",
        "P07-W04-S01::0",
        "P09-W01-S01::whole",
        "P09-W02-S01::whole",
        "P09-W03-S01::whole",
        "P09-W04-S01::whole",
        "P09-W05-S01::whole",
        "P09-W08-S01::whole",
        "P09-W07-S01::whole",
        "P09-W06-S01::whole",
        "P09-W09-S01::whole",
        "P09-W10-S01::whole",
        "P10-W05-S01::whole",
        "P10-W04-S01::whole",
        "P10-W01-S01::whole",
        "P10-W02-S01::whole",
        "P10-W03-S01::whole",
        "P03-W04-S01::whole",
        "P03-W05-S01::whole",
        "P18-W01-S01::0",
        "P03-W08-S01::whole",
        "P03-W06-S01::whole",
        "P03-W07-S01::whole",
        "P03-W10-S01::whole",
        "P03-W09-S01::whole",
        "P03-W03-S01::whole",
        "P03-W02-S01::whole",
        "P03-W01-S01::whole",
        "P11-W01-S01::whole"
      ],
      [
        "P16-W09-S01::whole",
        "P16-W08-S01::whole",
        "P16-W07-S01::whole",
        "P16-W02-S01::whole",
        "P16-W03-S01::whole",
        "P16-W10-S01::whole",
        "P16-W11-S01::whole",
        "P16-W12-S01::whole",
        "P19-W01-S01::whole",
        "P19-W02-S01::whole",
        "P20-W02-S01::whole",
        "P20-W01-S01::whole",
        "P13-W12-S01::whole",
        "P13-W11-S01::whole",
        "P13-W10-S01::whole",
        "P13-W03-S01::whole",
        "P13-W09-S01::whole",
        "P13-W08-S01::whole",
        "P13-W07-S01::whole",
        "P13-W02-S01::whole",
        "P13-W01-S01::whole",
        "P13-W06-S01::whole",
        "P13-W05-S01::whole",
        "P13-W04-S01::whole",
        "P11-W02-S01::whole",
        "P15-W03-S01::whole",
        "P15-W04-S01::whole",
        "P15-W02-S01::whole",
        "P15-W01-S01::whole",
        "P16-W04-S01::whole",
        "P16-W05-S01::whole",
        "P16-W06-S01::whole",
        "P16-W01-S01::whole"
      ]
    ],
    "directions": [
      [
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0
      ],
      [
        1,
        1,
        0,
        1,
        0
      ],
      [
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
      ],
      [
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1
      ]
    ]
  }
]
```
<!-- WAG_DEADLOCK_CASES_END -->



### 13.4 死锁诊断结论与 WAG 修正

首次诊断逐例完整重跑原 WAG 500 轮，11/11 的原生 Cmax 与历史 CSV 一致，既有 B32 再次全部 DEADLOCK，且失败的部分时间线与被观察的原基线派工分支逐操作一致。保存旧方案后，在 LNS 修正的代码环境中再次从保存的模式/路线/方向重放，11/11 仍完全重现；第二次 `search_rerun_iterations=0` 是固定方案的诊断重放，不是把零次搜索冒充 500 轮。

所有 11 条原生时间线的焊缝覆盖、焊缝顺序和机器人自身操作链连续性均通过检查；用现有 `operations_conflict` 对不同机器人操作逐对检查，冲突数均为 0。这些固定方案因此具有已有原生调度器给出的可行时间线证据，不能把 B32 的失败称为该任务分配在共同模型下数学不可行。B32 的 Cmax 在失败时为空，原生 Cmax 与 B32 Cmax 的数值差未定义，不用无穷大或人工惩罚值伪造差值。

首任务与方向已逐例保存在上节 JSON。22 对同轨首位置的最小 x 间隔为 1.94810 m，大于模型 0.5 m 安全间隔；跨轨首位置接近条件触发数为 0。因此这里没有发现“首任务初始位置已经非法”的解释。这不排除后续路线方向与顺序影响派工行为。

全部原基线派工状态中记录到 1750 次无法取得有限 ESS 的同轨阻塞边关联，另有 37 次有限 ESS 候选需要等待的同轨关联，两类跨轨关联均为 0；这是逐状态累计，不能当作独立物理冲突个数。终端重查不重复计数。不可启动子图、包含有限等待候选的全部 ESS 报告图及终端图均没有观察到等待环，所有终端阻塞者在相应派工分支中均已完成。首次没有有限 ESS 的操作中，7 例是 POST，2 例是 SETUP，2 例是 WELD，并非仅 MOVE 会导致阻塞。固定操作已经先后提交后，等待位置/同轨顺序约束可使剩余操作没有可接受 ESS；等待图无环也不保证这套派工策略必能继续。这里的逐状态关系来自原基线分支，没有声称逐一记录了全部恢复分支的等待图。

恢复终止方式不能混淆：8 例实际执行满 32 次恢复 rollout 后仍有未探索前沿；`validation_02/20261111` 在 28 次、`validation_10/20261110` 与 `/20261112` 在 14 次后该恢复策略的前沿耗尽。后者也只说明现有受限偏差派工恢复过程没有找到解，并不穷尽全部连续时间调度。对应的原生可行时间线进一步排除了“前沿耗尽即数学不可行”的解释。

具体例：`validation_03/20261110` 的原生 Cmax 为 5074.618223 s。原 B32 分支在派工深度 60 首次无法提交 `R2:B3:POST`（ready=658.363577 s），被已提交的 `R3:B3:WELD` 阻塞；此前深度 55 的 R3 焊接已被推迟到 590.297329 s。最终 R3 虽已完成，R2 仍被该分支的固定时间线阻塞，等待图为 R2→R3、没有环。原生时间线通过约 128.045837 s 等待安排完成全部任务且无机器人间冲突。这支持“派工顺序与既有承诺造成策略失败”，不能据此声称任务分配不可行，也不据此擅自修改固定 B32。

据此决定：**保留既有原生四机器人优先调度策略和最终 B32，修正具有独立语义错误的 LNS。** 此修正是恢复论文的部分解保留机制，不承诺自动消除所有 B32 死锁。

`wag_lns_candidates` 仍按每机器人 `ceil(p×任务数)` 破坏，p=0.5，X 子任务仍只能回到规定机器人；记录所有未破坏任务的原归属。`generate_wag_assignments(..., retained_assignments=...)` 把保留任务和固定 X 子任务的现有处理负载计入各机器人，仅对破坏任务做剩余负载平衡与边界偏移。返回的边界候选和原贪心候选均为部分修复，删除此前混入的整组 WAG 重生成。初始化调用不传保留分配，其原有有限边界生成规则不变。

该修正改变 LNS 的实际候选集合与后续随机搜索轨迹，修正前后 WAG 必须分开报告。Move、Swap、LNS 三操作、500 轮、n=10、p=0.5、5 任务精确 Factorial Edge Combination 和原全部路线组合均保留；没有减少 HGA 邻域、alpha 或有效子代。

统一复测 **45 passed**。新增 WHOLE 与 WHOLE/Y/X 混合两类测试、每类 10 个随机种子，逐候选核对未破坏物理区间仍属于原机器人，兼容合法子任务合并，并核对 X 子任务固定归属。新来源为 `PARTIAL_LNS_V1`，使用既有 WAG CSV 追加结果；旧 `OPTIMIZED_ITERATIONS` 记录保留。修正后 WAG 的三个既有代表排样已启动完整 500 轮验证，各结束后调用一次既有 B32。HGA 的三个完整 200000 有效子代验证仍在执行；本次并行推进经用户明确同意。

### 13.5 修正后的完整预算验证与正式重跑进度

WAG 三个代表样本均已完成 500 轮和最终一次 B32：

| 排样（种子均 20261110） | 原生 Cmax/s | 最终 B32 | B32 Cmax/s | 总耗时/s |
|---|---:|---|---:|---:|
| validation_03 | 5026.139171 | FEASIBLE | 5026.139171 | 129.8724 |
| validation_04 | 4983.647422 | FEASIBLE | 4983.647422 | 117.7954 |
| validation_07 | 6161.084757 | DEADLOCK | — | 141.9106 |

相对旧版同排样同种子，03 的原生 Cmax 改善 48.479052 s，最终状态从 DEADLOCK 变为 FEASIBLE；04 原生 Cmax 与最终可行状态相同；07 仍是同一原生 Cmax 和 B32 死锁。这是三例观测，不能代替完整数据集结论，也不能证明 LNS 修正普遍提高可行率。

在既有预算和 45 项相关测试通过后，另一个测试禁止 B32/认证器调用，实际执行 WAG 原生搜索的完整默认 500 轮并通过。运行环境仍为指定的 Python、DRL 项目目录。机器报告 8 个逻辑 CPU；WAG 正式 VALIDATION 使用 4 个 worker，每个配置单 CPU 线程，另有 3 个 HGA 验证 worker 持续运行。运行时间包含冷编译与并发资源竞争，不据此把算法定义改变前后差异全归为计算加速。

新版 WAG 正式 VALIDATION 已开始按 10 排样×3 种子重跑；保留三例完整预算记录并补足其余运行。来源仅筛选 `PARTIAL_LNS_V1`，不混入旧 30 次 VALIDATION/60 次 TEST。HGA 仍须先完成三个代表排样的 200000 有效子代验证，尚未启动 HGA 新版全量 VALIDATION 或 TEST；现有旧 HGA 三例不计作修正后的验证通过。

WAG 新版 VALIDATION 已全部完成：30 次均为 500 轮，每次最终调用一次 B32，共 30 次。最终可行 22/30（旧版 19/30），可行结果 Cmax 平均 5389.672876 s、中位 5466.123862 s；全部运行耗时平均 114.5412 s、中位 103.5916 s、最大 316.0831 s。以上为本机并发实测，不是统一硬件独占计时。

| 旧版状态 → 新版状态 | 次数 |
|---|---:|
| FEASIBLE → FEASIBLE | 17 |
| DEADLOCK → FEASIBLE | 5 |
| FEASIBLE → DEADLOCK | 2 |
| DEADLOCK → DEADLOCK | 6 |

共同可行 17 次的 B32 Cmax：新版 7 胜、8 平、2 负，配对差中位数 0 s、变化比例中位数 0%。这既有改善也有退化，不能只报告可行数净增 3 而省略旧可行变死锁的 2 次。TEST 60 次已在固定代码与参数下启动；不依据 TEST 修改算法。

HGA 的另一次性能探索仅在独立进程内存中启用精确机器人对成本缓存，没有编辑仓库实现：同一 100 有效子代诊断耗时 305.9865 s（此前 312.4965 s），唯一 VND=119、VND 缓存命中=1、提案=3331317、终值=5045.768777 s 均相同；缓存拒绝了 84617 次重复机器人对计算，仍完整计算 213074 次，终态缓存 95017 条。两次并发负载不同，约 2% 的墙钟差不能证明可靠加速，不足以支撑引入额外缓存复杂度，因此没有采用。本轮未为此重启或缩减正在执行的三个完整 HGA。

WAG 的 TEST 已全部完成，未依据 TEST 调参：

| 新版数据集 | 运行数 | B32 可行 | 可行 Cmax 均值/s | 可行 Cmax 中位/s | 耗时均值/s | 耗时最大/s | 每次迭代 | 最终 B32 调用总数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VALIDATION | 30 | 22 | 5389.672876 | 5466.123862 | 114.5412 | 316.0831 | 500 | 30 |
| TEST | 60 | 60 | 6671.902844 | 6717.024993 | 100.7024 | 155.3539 | 500 | 60 |

旧版 TEST 同样为 60/60 可行；逐排样逐种子配对，新版 32 胜、26 平、2 负，Cmax 差中位 −0.546611 s、变化比例中位 −0.008051%。改善幅度的中位数很小，不能写成普遍大幅提升。两版的算法定义不同，旧版仅用于修正前后对照，不能合并成新版多次重复。新版 WAG 共完成 90 个正式运行、45000 轮 VNS 和 90 次最终 B32；旧方案的诊断重放调用不计入这 90 次正式运行。

只读 `baseline-summary` 已成功执行。与已有参考方法比较时，TEST 的共同可行运行均为 48 次：WAG 相对 RULE/BEAM、GNN_VNEXT、GNN_JOINT、GNN_JOINT_RECONSTRUCT 分别均为 48 胜，Cmax 配对变化中位数分别为 −1.9566%、−18.8513%、−7.0939%、−5.8532%。这些只涉及共同可行子集；HGA 新版尚没有完整验证结果，不能据此形成新版 WAG/HGA 的完整联合对照结论。

### 13.6 时间窗口加速、完整性边界及当前 HGA 状态

进一步诊断发现，原跟随计算会先对整条固定路线的未来操作计算禁入区间，许多操作在当前候选时间段之后，不可能影响当前开始时刻。`_numeric_follow(..., time_window=True)` 仍按固定时间线的原顺序处理操作，只先纳入可能影响 `[start, start+duration]` 的操作；若 WAIT 推进 start，则继续纳入后续操作，并沿用原 `_numeric_forbidden` 公式及相同区间合并和等待推进规则。

这个跳过的依据是时间投影约束：若某禁入区间包含当前 start，对应固定操作必须与候选操作在时间上接近或相交。实现用比 EPS 松弛更宽的保守读取边界，边界附近仍计算原投影，不改变 `ScientificConfig.numeric_epsilon` 或任何物理判定。每次推进严格越过至少一个区间终点；最多有 `2×固定操作数` 个区间，因此单操作的 `2N+2` 次循环边界足以完成整个区间并集搜索，是有限性上界，不是 VND 候选或迭代截断。候选操作和完整 WAIT 的最终冲突检查保留；尚未读取的更晚操作不可能与候选或 WAIT 的时间段相交。

WAG 使用默认完整扫描路径，HGA 显式启用空间与时间窗口跳过。首方向选择、正反优先方案、跨轨检查、Cmax、路线下界、M1–M6 和外层遗传过程均保持原定义。没有引入新调度策略、B32 搜索反馈或 GNN。

验证包括 400 组原始/加速时间线逐项相等；补充 EPS 前后边界、零时长 MOVE、正反优先、等待与阻塞测试后，三份相关测试文件统一为 **47 passed**。同一真实 `validation_03`、同一随机种子的 100 有效子代诊断，从 312.4965 s 变为 89.9355 s，局部搜索累计从 287.0906 s 变为 74.6918 s；唯一 VND=119、缓存命中=1、候选提案=3331317、构造拒绝=0、终值=5045.768777 s 均一致。该诊断含编译/预热与并发负载差别，不能将其比值直接当作完整 HGA 的加速倍数。

完整性边界也需明确：VND 在当前选定模式的活动块集合内执行六邻域，模式变化由既有外层合法模式变异处理。可达资格、X 子任务固定归属，以及数值编码避免相邻兄弟块在 VND 内触发模式合并的过滤均沿用 `86e32c7`；这些属于共同模型/固定活动任务集适配，原论文没有 Y/X 模式字段。“完整 VND”指当前活动任务集和 alpha=20 下六邻域无额外候选数、轮数或人工时间截断，不代表穷举全部合法模式及任务分配，也不能把共同模型约束称为原论文额外限制。

为采用经验证的时间窗口实现，19:59 启动但仍未完成的三例已停止，没有新增完成结果；最终实现于 **2026-11-11 21:48（Asia/Shanghai）** 从零启动三例，均保持 μ=20、λ=10、α=20、200000 次有效子代和完整 VND。新增每分钟进度输出只用于观测，不构成时间截止条件。

截至 22:07 读取到的最近日志如下，**均非完整实验结果，也没有最终 B32 结果**：

| 排样 | 已完成有效子代 | 唯一 VND | VND 缓存命中 | 实际结构有效候选提案 |
|---|---:|---:|---:|---:|
| validation_03（82 母焊缝） | 2440 | 2429 | 31 | 74174836 |
| validation_04（91 母焊缝） | 1803 | 1784 | 39 | 56315619 |
| validation_07（115 母焊缝） | 653 | 664 | 9 | 33319244 |

上述是当时的进度记录。用户随后明确要求先参考 HGA 文献修正效率，21:48 启动的这一批三例已停止，未向 CSV 写入完成结果。最新状态见 13.7；不得将这些部分预算运行或旧版三例写作本轮验证通过。

### 13.7 按文献增量评价原则继续修正 HGA 效率（执行中）

当前任务已由等待长跑转为修正计算实现。三个低效完整预算进程已停止，保留既有 WAG 90 次正式结果，未减少 HGA 的正式预算，也未把中止记录记为完成。新版 HGA 全量 VALIDATION/TEST 尚未启动。

文献依据为 Liu 等《An effective hybrid genetic algorithm for the multi-robot task allocation problem with limited span》，第 3.2.3 节及第 4.1 节：六种邻域、alpha=20、mu=20、lambda=10，每个有效迭代均完成局部搜索，200000 次停止条件保留。论文引用的 He 与 Hao (2023)《Memetic search for the minmax multiple traveling salesman problem with single and multiple depots》，第 3.3.2 节使用移除/插入代价辅助数组 A1/A2，接受操作后只更新相关数据。这里参考其增量评价原则，不能把该文章中不同定义的六邻域、交叉或后优化直接替换当前焊接 HGA，也不能把独立路线代价冒充四机器人协调 Cmax。

来源：[HGA 原论文](https://doi.org/10.1016/j.eswa.2025.127299)、[He 与 Hao 的作者版全文](https://leria-info.univ-angers.fr/~jinkao.hao/papers/HeHaoEJOR2022.pdf)、[作者公开代码仓库](https://github.com/pengfeihe-angers/minmax-mTSP)。公开代码为 minmax mTSP 的实现，不是焊接 HGA 原代码；本轮未在项目中新增或引入这些代码文件。

已修改的计算路径：

1. `_segment_head_costs` 从已有 2x2 区间摘要计算候选路线固定首方向的代价；`_pair_head_bound` 按原生最小合计路线代价的合法首方向规则，保留所有数值近似并列选择，用其 Cmax 下界排除必不改善的候选。对未变化、无 WAIT 且严格连续的路线，可直接复用相同首方向的精确名义完成时刻。原自由方向路线下界仍保留。
2. `_numeric_follow` 对已确定的调度前缀使用“当前时刻 + 剩余名义处理/移动时间”下界。只有有限 incumbent 且保守下界足以排除改善时才结束该优先方向；另一优先方向仍可求解。初始化的无限 incumbent 必须完整保留两个机器人对的状态，不能复用尚未计算的对。
3. `_forbidden_may_contain` 对原禁入投影多边形的宽松半平面计算当前开始时刻的截面。原枚举接受的全部顶点及其凸包都包含在这个更宽松的集合中；截面空时原禁入区间不可能包含当前时刻。可能生效时仍调用原 `_numeric_forbidden_buffer` 顶点枚举，WAIT 推进后重查尚未求解的区间。它只延后冗余计算，不用近似区间替换原公式。
4. `_direction_array` 改用两个标量代价和两个标量词典序秩，保留原加法顺序及方向并列规则；`_numeric_templates` 直接写操作单元格；`_numeric_follow` 在同次调用中复用区间、待求解区间、约束及候选操作缓冲区。消除原先每个任务/操作循环中的反复小数组分配。

以上不改变 M1—M6、alpha、候选枚举顺序、接受目标、合法模式、原生优先调度策略和最终 B32，不改变理论搜索空间或有效子代定义。搜索计数仍在结构合法提案进入安全剪枝前累计。相对本轮已修正目标/M4 的 HGA，属于计算等价优化；相对 86e32c7 的目标/M4 修正影响及旧结果不可直接混用的结论不变。

单进程、VND 主核预热后的真实 `validation_03`、种子 20261110、100 有效子代诊断：本阶段起点 36.2452 s；固定首方向下界和前缀完成下界后 28.1281 s；再加入禁入投影截面延迟求解后 23.2724 s。三次终值均为 5045.768776731937 s，唯一 VND=119、缓存命中=1、结构有效候选提案=3331317。此前并发环境的 89.9355 s、312.4965 s 不用于计算本阶段加速倍数。

后续保留的增量计算与数据结构修正如下，全部修改已有文件：

| 函数位置 | 原问题与理论依据 | 保留的修改 | 搜索空间和可比性 |
|---|---|---|---|
| `hga.py:_native_pair`、`_vnd_arrays` | 邻域可能只改变一条路线，却重复计算未变化伙伴的两种固定首方向 DP 和操作模板 | 只刷新变化路线的方向选项及模板；未变化机器人对和未变化伙伴均复用精确结果 | 当前修正版 HGA 的定义与接受顺序不变 |
| `wag_vns.py:_follow_reusable_prefix` | 固定时间线和跟随操作均有相同前缀时，仍重放已经确定的早期操作 | 分别缓存两种优先方向；只复用严格早于固定时间线首次变化的完整操作及其 WAIT，保留 EPS 边界余量 | 不从最终胜出的优先方向推断另一方向的结果；原优先策略不变 |
| `wag_vns.py:_numeric_follow` | 每个后续操作都从第一个固定操作重新扫描，重复访问已结束前缀 | ready 单调，因此用单调游标让每个已结束固定操作只退出一次，原时窗和冲突判据不变 | 等价计算，不改变邻域 |
| `hga.py:_remember_vnd` | 原缓存每次最多淘汰一个键，却可能插入输入/终态两个新键，声明的 4096 项上限实际上失效 | 在两个键写入后淘汰到 4096 项；命中时刷新访问顺序 | 只影响缓存命中与内存，重新计算仍是同一确定性完整 VND |
| `hga.py:_vnd_arrays` | 同一未变化轮次中，M2/M3/M4/M6 的 `(u,v)` 与 `(v,u)` 可能生成完全相同的四条路线 | 按活动任务索引记忆该轮已判定的同一候选；M1/M5 有方向性，均不合并。新统计 `vnd_move_cache_hits` 单列复用次数 | alpha 列表、提案访问顺序及首改善位置不变，不删除任何不同候选 |
| `hga.py:_unique_pair_heads` | 已有固定首方向区间代价足以证明唯一最小合计代价的合法首方向时，仍对两种方向都完整回溯 | 只在宽松误差带外确证唯一选择时回溯选中方向；近似并列完整回退原 DP 和原词典序规则 | 不用近似方向结果接受候选，不减少方向选择 |

精确恒等操作也直接复用：M4 反转单任务、M6 交换两个空尾段，所得解完全不变。`vnd_candidate_operations` 仍记录剪枝及复用前的结构合法提案访问数，`vnd_move_cache_hits` 记录其中反向参数复用的次数；不能把两者都当成独立原生调度调用次数。

已保留代码的阶段计时：36.2452 → 28.1281 → 23.2724 → 10.5280 → 8.7104 → 7.6467 → **6.1828 s**，均为同一 100 子代诊断，并非完整 HGA 的运行时间。最新局部搜索累计 5.6422 s；终值、119 次唯一 VND、1 次完整 VND 缓存命中、3331317 次提案与本阶段起点完全相同，另精确复用 945599 次反向参数候选。首个 100 子代计时仍包含少量首次种群管理编译，不把单次比值视作跨机器保证。

2026-11-12 稳定代码的持续吞吐诊断如下。均为同一指定 Python、同一 DRL 目录、种子 20261110、mu20/lambda10/alpha20、完整六邻域；诊断显式运行 1000 有效子代，不写正式 CSV、不调用最终 B32，不能冒充 200000 次验证：

| 排样 | 诊断有效子代 | 耗时/s | 原生 Cmax/s | 唯一 VND | 完整 VND 命中 | 结构合法提案 | 反向参数复用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation_03 | 1000 | 55.0140 | 5021.342027480 | 1013 | 7 | 30107864 | 8393775 |
| validation_04 | 1000 | 61.2798 | 4855.489701566 | 1004 | 16 | 30382235 | 8640899 |
| validation_07 | 1000 | 144.1318 | 5639.087197666 | 1006 | 14 | 50954117 | 14796164 |

相对加入最后两项复用前的同设置诊断，耗时分别为 67.1724/76.5437/181.3920 s；三例的终值、唯一 VND、完整 VND 命中与提案数全部相同，构造拒绝均为 0。仍没有充分证据证明完整预算已可在分钟级完成。不能把约 5.86 倍的 100 子代阶段加速写成已经解决完整实验耗时。

已撤回或未引入的试验：

- 更强的调度前缀下界原型仅少量排除候选，100 子代耗时 7.6565 s，对照 7.6467 s，无可确认收益，已撤回对应代码及仅服务该原型的测试。
- 静态投影顶点缓存仅在进程内存验证了 89907 次精确一致查询；孤立组件可加速，但没有端到端收益证据，未引入项目。
- 把方向/模板/跟随计算的全部大缓冲区在 VND 中共享的原型虽通过功能检查，同设置三个 1000 子代却耗时 59.2359/64.8357/163.4755 s，对照 56.1476/63.2458/150.8795 s，出现退化；整套工作池及其附带候选缓冲区修改已撤回。保留的是前述循环内的小数组消除和确定有效的复用。
- 完整终态机器人对复用的潜在命中上界诊断也仅约 10%–22%，尚未加入复杂的机器人对局部最优证明缓存。

最终稳定代码三份相关测试为 **54 passed，28 条 Numba 现有类型反射弃用提示，67.04 s**。覆盖六邻域条件代价、方向 DP 穷举与并列、有限 Cmax 剪枝、双机器人对初始化、四路线与跨轨检查、400 组正反优先时间线、2500 组投影端点/中点、EPS/零 MOVE、前缀复用、对称候选精确等价与双键缓存容量。共享内核修改后还实际重跑了 `validation_03/20261110` 的 WAG 完整 500 轮，仍为 5026.139171078455 s；该检查没有重复写正式 CSV。

### 13.8 尚需确认的方向编码定义

原 HGA 第 3.2.1 节明确讨论无向焊缝插入的两种方向，但没有描述当前适配采用的“每个邻域候选对整条路线重新优化所有方向”的 DP。当前 `_native_pair` 会这样做，再依据两个方向选项的路线合计代价选择合法首方向。这使后段任务移动可能改变前段方向，甚至伙伴路线任务顺序未变化时也可能改变其方向，削弱局部调度前缀的可复用性。

需要区分两种定义：保留当前全路线方向 DP，只进行计算等价优化；或者将方向显式纳入 HGA 个体，交叉继承有向路线，邻域只修改相关任务/反转片段，并保持种群与 VND 都按同一原生协调 Cmax 评价。后一种需要明确记录方向搜索定义及可比性的变化，不能声称论文已经唯一规定这种实现，也不能未经确认悄悄替换。

已向用户提出这一具体定义问题，当前生产代码仍保留全路线方向 DP。mu=20、lambda=10、alpha=20、200000 有效子代、六完整邻域、共同物理模型与最终 B32 均未改变。本轮未新增文件；三个修正版 HGA 完整预算验证和后续全量 HGA 正式实验仍未完成。

上述为提出定义问题时的记录。用户现已确认采用显式方向编码，后续实现不再以全路线 DP 作为当前正式 HGA 定义。

补充了只在内存中观测的方向扰动诊断。对两个既有区域初始解完整执行同一 VND，在真正进入原生机器人对评价的候选中，比较候选与当前解共同任务前缀的方向：

| 排样 | 可比较路线次数 | 有共同任务前缀 | 共同前缀发生方向变化 | 首方向相同且共同前缀超过一任务 | 其中内部方向变化 | 完全未改任务顺序的伙伴路线方向变化 |
|---|---:|---:|---:|---:|---:|---:|
| validation_03 | 13474 | 12698 | 1944（15.31%） | 11317 | 1642（14.51%） | 0 |
| validation_07 | 1514 | 1460 | 385（26.37%） | 1214 | 174（14.33%） | 0 |

完整 VND 终值和提案次数分别仍为 5119.837307023396 s/81255、5884.736782002155 s/13627；接受次数仍为 71/8。观测改变了运行开销，因此不把该诊断耗时用作求解性能。表中为路线评价访问次数，不是独立解数量、全部结构提案数量或数据集发生率；这两例未观察到未改伙伴的方向变化。结果确证了一部分非局部方向扰动，并不证明显式方向编码一定能实现分钟级完整预算，需要在定义确认后独立验证。

### 13.9 用户确认后的显式方向 HGA（实施与验证中）

用户明确授权采用显式方向编码并记录定义变化。只修改现有代码、测试和说明文件，不新增文件。正式运行来源改为 `ORIENTED_COORDINATED_VND_V1`，此前 `COORDINATED_VND_V1`、`OPTIMIZED_ITERATIONS` 的记录不得作为这一版本完成结果。

该定义把每个个体表示为四条 `(任务, 方向)` 序列；完整 VND 缓存及候选缓存均须区分方向。初始构造和交叉缺失任务的插入比较两种方向；交叉复制路线时同时复制方向。M1、M2、M3 和 M6 随任务/片段保留方向；M4、M5 在反转片段时同时翻转其中焊缝的方向。M4 反转单任务现在是有效的方向翻转，不能沿用旧版本“单任务反转恒等”的跳过。M1 的最长路线限制、M2/M3 至少一条为最长路线、M4 不限最长路线等条件保留；最长路线按个体实际方向的名义路线周期确定。

固定方向个体仍使用同一原生正反优先跟随调度和跨轨冲突检查。改变的是方向搜索定义，不是最终 B32 或共同物理约束。种群比较和局部搜索都按该固定方向个体的原生协调 Cmax；在记录优胜解时用完整原生评分再次核对同一方向，禁止用无方向评分替代。共同评价缓存也把预计算时间线的方向纳入键，防止同任务顺序、不同方向被误认为同一候选。

还需处理一个具体搜索失败情形：若两个独立机器人对均被原生策略拒绝，单次只修复其中一对时，全局 Cmax 仍非有限。显式 VND 只在双方 Cmax 均非有限时允许以“被原生策略拒绝的机器人对数量减少”作为严格次目标；一旦 Cmax 有限，仍只接受严格的协调 Cmax 改善。该次目标不把策略失败解释为数学不可行，也不添加新邻域或额外评价器。

全路线方向 DP 的旧实现保留为显式命名的 `LEGACY_ROUTE_DP` 审查参照。新定义会改变方向候选及搜索轨迹，扩大可表达的方向个体集合；不能要求其终值或提案数与旧定义相等，也不能将两者合并作重复实验。mu20、lambda10、alpha20、200000 有效子代和完整六邻域、数据/几何/资格/拆分规则/B32 均保持原要求。真实排样持续吞吐和三个完整预算验证仍待完成。

本次定义变更及等价加速分别记录如下，避免把两者混为一谈：

| 修改函数 | 原问题与依据 | 修改及验证 | 是否改变搜索定义/既有可比性 |
|---|---|---|---|
| `hga.py:_run_hga_oriented_native`、`_oriented_identity`、`_repair_oriented_numeric` | 原方向 DP 可因局部任务变更而重算整条路线；论文明确比较插入的两个方向，未唯一规定全路线 DP | 个体和交叉同时继承任务/方向；插入比较两个方向；完整 VND 缓存及多样性区分方向；覆盖继承和三种合法模式家族的测试 | 用户明确批准的定义变化；来源与旧版分开，不合并旧结果 |
| `hga.py:_materialize_directions`、`_oriented_segment_cost` | 有向片段反转若不翻转焊缝方向，不能沿反向几何路径复用内部代价；单任务反转也不再是恒等 | 反转同时互补方向位；固定方向区间前缀、资格前缀和位置映射仅更新变化路线；六邻域片段与完整评分核对 | 显式编码定义的一部分；区间代价计算本身等价，不缩减邻域 |
| `hga.py:_fixed_native_pair`、`_fixed_native_update` | 候选重复构建未变化伙伴和机器人对；两对均拒绝时不能只留下一个已计算的状态 | 复用未变化对、模板及分别对应两种优先方向的前缀；无限 incumbent 时完整计算两对；有完整评分复核和双对逐步修复测试 | 复用不改策略；非有限目标下的拒绝对数次目标已明确记录 |
| `hga.py:_fixed_follow_progress`、`_fixed_prefix_reject`；`wag_vns.py:_numeric_follow` | 相同有向前缀已积累的等待或已发生的原生策略拒绝仍被重算 | 仅使用严格早于固定路线首次变化的完整操作前缀；记录真实冲突失败的位置及时间；Cmax 提前截止标为未知，不能当作冲突失败复用。完整 VND 有/无该剪枝逐项一致，并用实际末方向修改验证失败前缀可复用 | 精确排除必不改善候选，不减少 alpha 或设置候选上限；最终 B32 不变 |
| `common.py:CommonBaselineEvaluator.evaluate`；`wag_vns.py:priority_schedule` | 同任务顺序但不同显式方向可能被共同缓存错误合并，或记录优胜解时重选方向 | 固定方向评分使用传入方向且验证其合法位；预计算评分缓存区分方向；记录每个改进解时独立完整评分复核 Cmax | 修复显式定义下的评价一致性；默认 WAG 方向选择策略保持原样 |

2026-11-12 使用指定 Python 执行三份现有相关测试：`tests/test_hga_baseline.py`、`tests/test_wag_baseline.py`、`tests/test_baseline_common.py`，结果为 **62 passed，65 warnings，103.37 s**。警告为现有 Numba 类型反射弃用提示。该结果证明目前覆盖的功能与等价计算一致，不代表完整 200000 子代验证已通过。为避免生成额外文件，运行使用 `-B`、`PYTHONDONTWRITEBYTECODE=1` 和 pytest `-p no:cacheprovider`；未新增文件。

显式定义初版的真实持续吞吐诊断（单进程，Numba/OMP 单线程，种子20261110，先小例预热；仍非正式完整预算）：

| 排样 | 有效子代 | 总耗时/s | 初始化/s | 搜索/s | 原生 Cmax/s | 唯一 VND | VND 命中 | 结构合法提案 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation_03 | 1000 | 67.1625 | 3.2169 | 63.9377 | 5035.252692826 | 1008 | 12 | 43432968 |
| validation_04 | 1000 | 103.6616 | 4.5714 | 99.0825 | 4857.208099382 | 1002 | 18 | 47771513 |
| validation_07 | 1000 | 196.2721 | 5.4764 | 190.7877 | 5657.522305150 | 1006 | 14 | 81248402 |

三例构造拒绝均为0，VND轮次/候选数上限命中均为0，搜索内B32调用均为0。新增前缀证明分别排除43479/44085/148708个完整候选，但100子代相同轨迹耗时为8.9180/12.0220/27.6884s，对照未加该证明的8.5027/11.8034/26.7893s，没有确认端到端收益，不能以剪枝计数代替提速证据。显式定义还增加了可搜索的方向状态及提案，不能与旧DP版的同预算时间直接计算等价加速比。目前瓶颈仍在VND，不能归因为初始化或用上述1000次结果宣称完整预算已解决。下一步直接测量VND内部各阶段，未盲目启动新的200000次长跑。

随后用内存中的CPU周期插桩定位瓶颈，未创建性能文件，也未把插桩计时作为求解性能：100子代中，`validation_03/07` 分别实际进入原生候选更新344584/863490次；在区间代价、前缀证明、候选物化、原生更新这四个被计时块的累计周期中，原生更新约占84%/91%。进一步观测投影查询分别为12436087/52766111次；嵌套计时块不能直接相加。两例Cmax和逻辑提案数与无插桩版本一致。实测机器为11th Gen Intel Core i5-1135G7（环境报告8逻辑处理器）；原论文使用i5-12490F桌面CPU，不能将硬件不同的绝对时间直接当作算法等价加速比。

以下无收益试验已撤回，生产代码保留串行原生计算，没有为它们保留新配置、并行模块或测试：

- 精简零系数约束虽然20000组投影端点精确一致，三例100子代却耗时13.3669/20.1087/46.9699s，未写入生产代码。
- 展开固定长度约束检查同样通过20000组端点核对，但耗时8.7553/12.0339/27.5955s，没有确认整体收益，未写入生产代码。
- 串行首改善顺序不变的分批并行在2/4线程小例中逐项一致，三个真实排样也保持Cmax、方向和逻辑提案数一致；但串行9.0555/12.3580/27.6981s，4线程反而13.2025/14.7651/31.0997s。困难例扩大批次为16/64源任务仍为31.2228/45.0167s；复制线程私有调度数组为17.5636/40.5002s（03/07），也没有收益。整套并行代码、配置和配套测试已撤回；批次从未作为候选上限使用。

保留新增的 `vnd_native_candidate_evaluations` 统计，以区分真正进入原生候选评分的次数和结构提案数；它不改变搜索或评分。完整原生记录检查次数 `reference_calls` 只统计优胜解记录等外层调用，不能拿它代替全部搜索内部的原生评分工作量。其余加速仍在验证，不能将原型通过或短测改善写成完整预算已通过。

已保留的新加速为 `wag_vns.py:_numeric_follow(..., early_reject=False)` 的单区间提前拒绝，且仅在 `hga.py:_fixed_native_pair` 的显式方向路径启用，WAG 默认路径不变。原实现会先求齐当前时窗中相关投影，再统一推进开始时刻、检查WAIT和Cmax。新实现每得到一个**原公式的精确禁入区间**，若其包含当前候选开始时刻，则已经能证明最终开始时刻至少超过该区间右端点：剩余名义工时已超过incumbent时可立即按下界拒绝；从ready到该右端点的必需静止WAIT已与同一固定操作冲突时也可立即拒绝。增加其余禁入区间只能延长WAIT，不能消除已有冲突。冲突证明时间戳保守覆盖该完整固定操作，不把Cmax截止记作冲突证据。该修改不替换投影公式、不近似接受候选、不改变两种优先方向选择、候选访问顺序或搜索空间，也不将某优先策略失败解释为数学不可行。

三个既有代表排样、相同种子和全部算法参数、预热后100有效子代的阶段结果如下。每例终值、唯一VND、完整VND命中和逻辑结构提案数均与显式初版一致；方向和完整时间线的逐项核对由相关单元测试覆盖：

| 排样 | 显式初版/s | 单区间提前拒绝/s | 原生Cmax/s | 逻辑结构提案 |
|---|---:|---:|---:|---:|
| validation_03 | 8.9180 | 6.6580 | 5055.003333789 | 5535709 |
| validation_04 | 12.0220 | 8.9639 | 4927.542092108 | 5278913 |
| validation_07 | 27.6884 | 17.5471 | 5671.721576255 | 10744628 |

该阶段的三份相关现有测试再次执行，**62 passed，65 warnings，105.83s**。额外扩展现有400组正反优先时间线、EPS/零MOVE和前缀复用测试，逐项比较开启/关闭提前拒绝的完整输出与原完整评分；六邻域、有限incumbent及双机器人对修复继续验证。没有新增文件。完整200000次验证仍未通过，不能按短测表宣称已解决完整预算耗时。

相同设置的1000有效子代持续吞吐复测已完成：

| 排样 | 显式初版/s | 提前拒绝/s | 初始化/s | 搜索/s | 原生候选评分次数 |
|---|---:|---:|---:|---:|---:|
| validation_03 | 67.1625 | 52.3979 | 2.4135 | 49.9767 | 2628663 |
| validation_04 | 103.6616 | 78.4154 | 3.9204 | 74.4862 | 4085540 |
| validation_07 | 196.2721 | 136.1843 | 3.5774 | 132.5979 | 6528324 |

耗时分别下降约22%/24%/31%；每例Cmax、唯一VND、VND命中、反向参数复用、逻辑结构提案数、构造拒绝及完成后原生非有限候选数均与显式初版一致。三例仍为0次搜索内B32、0次候选/轮次上限命中、0次构造拒绝。上述为各1000子代的完整诊断，不是各200000次正式验证；不得把耗时简单线性外推值写成已完成运行时间，效率目标仍未解决。

本阶段最终保留的另一项计算优化为 `wag_vns.py:_temporal_vertex_valid`、`_fixed_temporal_projection`、`_numeric_forbidden_specialized`。原问题是每次投影均构造通用约束矩阵，再对全部约束对重复求交和遍历验证；六条时间约束的形式其实固定。理论依据是：这六条约束有12个非平行约束对，其中三个产生相同的 `(end,end)` 交点，可以直接枚举10个不同时间交点，再按原行列式、交点公式和EPS判据枚举时间约束与几何约束、几何约束之间的全部交点，仍取原有效顶点的最小/最大开始时刻。顺序冲突对应1条几何约束，距离冲突对应4条几何约束。它不使用近似投影、不改变数值精度或fastmath设置，也不缩减任何不同候选。

`_numeric_forbidden_buffer` 保留原通用实现，新增的可选专用路径默认关闭；只有显式HGA调用 `_numeric_follow(..., early_reject=True)` 时启用。WAG默认计算及最终B32不变。独立内存诊断20000组投影端点精确一致；现有测试又增加10000次通用/专用端点精确比较，包含零时长和正反顺序，同时继续对完整时间线、EPS、前缀复用、六邻域及有限incumbent核对。该优化相对同一定义的显式HGA仅改变计算，不改变搜索空间或既有数值结果的可比性。

最终数值内核的三个1000子代结果如下，种子仍为20261110，单进程单线程预热，搜索内B32调用为0：

| 排样 | 显式初版/s | 最终内核/s | 降幅 | 初始化/s | 搜索/s | 原生Cmax/s |
|---|---:|---:|---:|---:|---:|---:|
| validation_03 | 67.1625 | 47.4721 | 29.3% | 2.2281 | 45.2372 | 5035.252692826 |
| validation_04 | 103.6616 | 68.9105 | 33.5% | 2.9343 | 65.9688 | 4857.208099382 |
| validation_07 | 196.2721 | 121.9410 | 37.9% | 3.2278 | 118.7029 | 5657.522305150 |

每例的终值、逻辑提案、唯一VND/命中、反向参数复用、构造拒绝、原生非有限候选和原生候选评分次数均与前一项提前拒绝版本一致；专用投影只降低同一工作的开销。相对最初显式编码版本，终值与主要搜索计数也保持一致。仍不能把该表称作完整200000次验证或已经达到本轮效率目标。

还修正了一项实际接口回归：仓库 `scripts/run_gnn_multi_initial.py` 和 `src/mrta_ranker/ranking.py` 中存在共5处使用有限预算的HGA调用；新显式模式原先会因 `time_limit` 有限而直接抛错。`hga.py:_run_hga_oriented_native`、`run_adapted_hga` 现支持显式传入的有限预算，在完整VND之间检查时限，正确处理初始化前已超时的空种群，并保留 `TIME_LIMIT`/`ITERATION_LIMIT` 区分。未修改这些GNN/排序调用处，也不在搜索内调用B32。正式 `baseline_job` 仍传 `math.inf`，mu20/lambda10/alpha20/200000及完整VND均不受该兼容修复影响；有限预算调用不得冒充正式固定预算实验。

最终三份现有测试结果为 **63 passed，65条Numba类型反射弃用提示，117.79s**。新增的有限预算回归测试还将内部B32/认证调用替换为直接抛错，验证正常预算与初始化前超时路径都不调用它们。没有新增文件，也没有向正式CSV添加本版本的未完成结果。

后续未保留的原型还包括：投影内点省略（收益约0%–3%波动）、对原投影/冲突函数强制内联（6.6831/9.4257/17.9087s，没有收益）、后缀调度汇合复用（6.3292/8.2305/15.7482s，相对专用投影6.1148/7.9256/15.8650s未形成稳定收益）。焊接顺序乐观下界也仅在内存原型中验证，未加入生产代码。当前实现没有遗留这些额外缓存、并行配置或独立评价模块。

供GPT进一步分析的主要结论：显式编码定义已经贯通，数值等价加速已经验证，但每1000有效子代仍进入约263万/409万/653万次原生候选评分，瓶颈依然是协调等待求值。论文第2节假设避障时间可忽略，其路由代价增量技巧不能自动等同于当前逐候选精确几何协调评分的计算复杂度。这是模型适配开销的分析，不能据此降低当前要求的协调Cmax、六邻域或200000预算；下一步应继续减少等价重复调度工作。当前没有证据可宣称完整预算已在可接受时间内完成，正式HGA验证仍待解决。

最终测试复现命令（在指定DRL目录执行，不创建pytest或字节码缓存文件）：

```powershell
Set-Location -LiteralPath 'D:\pybullet_test\MRTA_GA\DRL'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest tests/test_hga_baseline.py tests/test_wag_baseline.py tests/test_baseline_common.py -q -p no:cacheprovider --disable-warnings
```

### 13.10 HGA α-nearness 单种子参数筛选（18/18完成）

实验结果于2026-11-12落盘，本节于2026-11-14补充。18组均完成1000个有效子代，未启动200000子代正式长跑。本轮建议以 **α=18** 作为下一轮较长预算验证的主候选，与α=20配对比较；α=16保留为速度优先的备选。正式HGA默认α=20和200000有效子代保持不变。

**源码与实验条件。** 本轮以GitHub提交 [47f9030](https://github.com/luckyfishanddog/DRL/commit/47f90300497382c2c5d4807be7e767b92d3c24f0) 为源码参照。核对本地 `hga.py`、`wag_vns.py`、`common.py` 与该提交一致，实验入口之外的 `run_gnn_weld_layout.py` 也一致。本轮只增加已有脚本的 `hga-alpha-sensitivity` 分支，没有修改HGA实现。

直接读取既有VALIDATION中的validation_03/04/07（82/91/115条父焊缝），依排样顺序分别运行α=16、18、20、22、24、26。每组重新调用HGA，重置seed=20261110，独立初始化种群及VND缓存。使用 `ORIENTED_COORDINATED_VND_V1`、EXPLICIT方向、μ=20、λ=10、默认模式变异概率0.15、完整M1～M6、`vnd_max_passes=None`、`vnd_candidate_cap=None`、`time_limit=math.inf`、`ScientificConfig()` 和既有 `FORMAL_SCOPE_V2`。只通过 `replace(AdaptedHGAConfig(), alpha=..., max_iterations=1000)` 改变筛选参数。

运行采用一个进程、一个计算线程，18组串行。先用validation_03、α=20完成一次10子代Numba预热，耗时64.6476秒，预热种群和结果丢弃。正式 `runtime_s` 从调用HGA前计至返回后，包含预计算、初始化、交叉、完整VND和种群管理，排除该预热和最终B32时间。有效子代沿用现有HGA计数：取得完整VND结果后计数，缓存只复用本次运行已完成的VND结果，不以构造尝试数替代。搜索内不调用B32；每组结束后沿用 `baseline_job` 的 `schedule_explicit(best.solution, best.directions, CONFIG)` 路径调用一次最终B32，另记 `b32_runtime_s`。

**原始数据。** 仅新增用户批准的 `D:\pybullet_test\MRTA_GA\DRL\results\new_dataset_final\hga_alpha_sensitivity.csv`，共18行、13列。包含全部11项指定指标，另含用于区分完整运行的 `run_status` 和单独计时的 `b32_runtime_s`。18行均为 `iterations=1000`、`run_status=COMPLETED`；各行 `vnd_unique_calls+vnd_cache_hits=1020`，对应20次初始化与1000次子代VND请求。初始Cmax、VND唯一调用、命中和实际原生候选评价次数均保留在CSV中，没有覆盖两个既有 `_optimized.csv`。

下面时间和Cmax单位均为秒；Δ为相对同排样、本轮α=20的 `(当前值/基准值-1)×100%`，负值表示降低。所有行均为1000个有效子代。除标注DEADLOCK的一行外，B32状态均为FEASIBLE。

| 排样 | α | 搜索耗时/s | 最终原生Cmax/s | 原生Δ | 最终B32 Cmax/s或状态 | B32 Δ |
|---|---:|---:|---:|---:|---:|---:|
| validation_03 | 16 | 60.94 | 5038.305783 | +0.061% | 5038.305783 | +0.061% |
| validation_03 | 18 | 58.93 | 5026.670267 | -0.170% | 5026.670267 | -0.170% |
| validation_03 | 20 | 47.82 | 5035.252693 | +0.000% | 5035.252693 | +0.000% |
| validation_03 | 22 | 53.54 | 5023.854268 | -0.226% | 5023.854268 | -0.226% |
| validation_03 | 24 | 58.24 | 5022.758746 | -0.248% | 5022.758746 | -0.248% |
| validation_03 | 26 | 60.77 | 5018.224590 | -0.338% | 5018.224590 | -0.338% |
| validation_04 | 16 | 59.37 | 4849.229308 | -0.164% | 4849.229308 | -0.164% |
| validation_04 | 18 | 64.97 | 4853.653233 | -0.073% | 4853.653233 | -0.073% |
| validation_04 | 20 | 75.91 | 4857.208099 | +0.000% | 4857.208099 | +0.000% |
| validation_04 | 22 | 84.72 | 4850.633060 | -0.135% | 4850.633060 | -0.135% |
| validation_04 | 24 | 75.47 | 4852.282551 | -0.101% | DEADLOCK | — |
| validation_04 | 26 | 83.93 | 4846.216353 | -0.226% | 4846.216353 | -0.226% |
| validation_07 | 16 | 99.64 | 5658.345089 | +0.015% | 5658.345089 | +0.015% |
| validation_07 | 18 | 109.42 | 5647.756543 | -0.173% | 5647.756543 | -0.173% |
| validation_07 | 20 | 121.19 | 5657.522305 | +0.000% | 5657.522305 | +0.000% |
| validation_07 | 22 | 129.88 | 5646.478318 | -0.195% | 5646.478318 | -0.195% |
| validation_07 | 24 | 143.31 | 5629.855400 | -0.489% | 5629.855400 | -0.489% |
| validation_07 | 26 | 145.95 | 5638.162122 | -0.342% | 5638.162122 | -0.342% |

validation_04、α=24的最终原生Cmax为4852.282551秒，但B32返回DEADLOCK，因此CSV中B32 Cmax为空，表中不计算其B32质量变化，也不把未完成时间线当作Cmax。其余17组原生与B32 Cmax在1e-8秒内一致。这一例说明本轮仍需分别记录原生目标与最终B32状态；单次死锁既不能证明数学不可行，也不能证明α=24必然导致死锁。本轮未扩展死锁诊断或修改调度策略。

**按α汇总的速度。** 均值及中位数各基于三个排样。有效子代/秒采用 `3000/三次搜索耗时之和`；速度倍数采用 `α=20平均耗时/当前α平均耗时`，大于1表示更快。耗时Δ与速度变化的分母不同，二者不混用。

| α | 平均耗时/s | 中位耗时/s | 有效子代/秒 | 耗时Δ | 相对α=20速度 | B32可行数 |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 73.31 | 60.94 | 13.640 | -10.20% | 1.114× | 3/3 |
| 18 | 77.77 | 64.97 | 12.858 | -4.74% | 1.050× | 3/3 |
| 20 | 81.64 | 75.91 | 12.249 | +0.00% | 1.000× | 3/3 |
| 22 | 89.38 | 84.72 | 11.189 | +9.47% | 0.913× | 3/3 |
| 24 | 92.34 | 75.47 | 10.829 | +13.11% | 0.884× | 2/3 |
| 26 | 96.88 | 83.93 | 10.322 | +18.67% | 0.843× | 3/3 |

18组HGA搜索耗时合计1533.99秒，最终B32合计5.83秒，均不包含64.65秒预热。相对本轮α=20，α=16平均耗时减少10.20%（吞吐提高11.36%），α=18平均耗时减少4.74%（吞吐提高4.97%）。这些均为本次实测比例，没有将1000子代耗时外推成已完成的200000子代耗时。

**规模与解质量。** 下表分别列出低α的耗时和原生Cmax变化；这六组B32均可行，其B32质量变化与原生质量变化一致。

| 排样 | 父焊缝数 | α=16耗时Δ | α=16 Cmax Δ | α=18耗时Δ | α=18 Cmax Δ |
|---|---:|---:|---:|---:|---:|
| validation_03 | 82 | +27.45% | +0.061% | +23.24% | -0.170% |
| validation_04 | 91 | -21.80% | -0.164% | -14.41% | -0.073% |
| validation_07 | 115 | -17.79% | +0.015% | -9.71% | -0.173% |

低α的影响并不随规模一致。在82条焊缝上，α=16/18分别比α=20慢27.45%/23.24%；在91条上分别减少21.80%/14.41%，在115条上分别减少17.79%/9.71%。因此不能把三个排样的平均收益表述成所有规模都加速。

α=16在82条和115条上分别增加Cmax 3.0531秒（0.0606%）和0.8228秒（0.0145%），在91条上降低7.9788秒（0.1643%）。当前退化幅度较小，值得作为速度优先候选继续验证，但本轮没有预设可接受退化阈值，也不能据此宣称质量等价。α=18在三个排样上分别降低Cmax 8.5824、3.5549、9.7658秒（0.1704%、0.0732%、0.1726%），兼顾本轮平均耗时和逐例质量。

原生候选评价次数支持降低α可减少协调求值工作量：α=16/18每例平均3737992.00/4008598.33次，α=20为4414175.67次，分别减少15.32%/9.19%；115条排样分别为5400596/5966494次，对比α=20的6528324次。82条排样虽然候选评价次数也减少，墙钟耗时却增加，说明评价次数不能直接代替运行时间。不同α会改变初始化和后续VND轨迹，单次候选的协调计算工作也可能不同；本轮没有额外剖析这些因素，固定运行顺序及机器负载波动的影响也未被重复试验分离。

α=22/24/26的平均耗时分别比α=20增加9.47%/13.11%/18.67%。α=22在三个排样均比α=20有小幅质量改善，但本轮提速验证优先级低于α=18。α=26取得82条和91条的最低原生/B32 Cmax，α=24取得115条的最低原生/B32 Cmax（5629.855400秒，比α=20低0.4890%），说明较大α仍可能带来质量收益；α=24同时有一例B32死锁。它们的结果应保留作质量参照，本轮不列为下一轮提速验证的主候选，不据此删除任何参数选项。

论文Table 1提供α=16～26、默认20，其关于较大α增加时间、α>15后质量变化较小的结论，在本轮仅得到有限的描述性支持：平均耗时随α递增，逐排样耗时不严格单调；相对α=20的原生Cmax变化均小于0.5%，但存在B32死锁。论文模型与本项目四机器人几何协调适配不同，不能仅凭这18组就认定结论已普遍迁移。

**下一步参数。** 推荐以α=18为下一轮较长预算的主候选，保留α=20配对基准，重点确认115条排样的加速及质量能否持续；若优先追求吞吐，可再加入α=16，核对本轮两例轻微退化是否持续。α=20仍是当前正式默认值。本轮只做3个排样、1个种子、1000有效子代筛选，不能声称统计显著、跨规模普遍有效或已证明200000子代最终质量；也未证明仅调整α即可解决完整预算的耗时问题。未自动扩大重复次数或启动下一轮长跑。

**复现与核对。** 实际运行命令如下，工作目录和解释器均为用户指定路径。入口用独占新建模式写CSV；该文件已经存在时，不应直接重跑同一命令覆盖原始数据。

```powershell
Set-Location -LiteralPath 'D:\pybullet_test\MRTA_GA\DRL'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -W ignore 'scripts/run_gnn_weld_layout.py' hga-alpha-sensitivity --workers 1
```

实验前以内存模拟核对了预热加18组的调用顺序、相同种子和固定参数、18次最终B32、死锁Cmax留空及999子代标为INCOMPLETE，未将模拟结果写入CSV。实测后独立读取CSV，核对18个排样/α组合唯一、全部1000次完成、VND请求数、原生Cmax不劣于初始化及B32空值含义；已有脚本语法检查通过。本轮没有改动算法实现，因此未重跑13.9节的63项算法测试，也不将该历史测试数当成本轮新增测试结果。

本轮修改范围：`scripts/run_gnn_weld_layout.py` 增加实验函数和CLI分支；本报告仅追加13.10节；新增上述一个获批CSV。没有新增辅助文件、实验管理模块、评价器、缓存机制、指纹或协议，没有修改正式HGA默认参数、WAG、GNN、原始数据、几何、ScientificConfig、FORMAL_SCOPE_V2或最终B32。


### 13.11 新版PARTIAL_LNS_V1的8次VALIDATION死锁诊断

使用当前代码、原始种子对CSV中PARTIAL_LNS_V1的8条失败记录逐条重新执行完整500轮搜索；不读取13.3节的旧路线快照，不修改搜索或B32。只有预算、原生Cmax（容差1e-6）和最终B32状态全部复现后才诊断；未复现者跳过归因。诊断只观察既有派工内核，不反馈搜索。

| 排样 | 种子 | 完成轮数 | 重现 | 原生 Cmax/s | 与CSV差值/s | B32 |
|---|---:|---:|---|---:|---:|---|
| validation_02 | 20261110 | 500 | True | 6480.711522171011 | 0.0 | DEADLOCK |
| validation_02 | 20261112 | 500 | True | 6423.460396311804 | 0.0 | DEADLOCK |
| validation_03 | 20261112 | 500 | True | 5012.5553338942445 | 0.0 | DEADLOCK |
| validation_07 | 20261110 | 500 | True | 6161.084756775765 | 0.0 | DEADLOCK |
| validation_07 | 20261111 | 500 | True | 6166.432940777202 | 0.0 | DEADLOCK |
| validation_07 | 20261112 | 500 | True | 6167.828488850058 | 0.0 | DEADLOCK |
| validation_10 | 20261110 | 500 | True | 6138.083333819142 | 0.0 | DEADLOCK |
| validation_10 | 20261112 | 500 | True | 6045.490741217289 | 0.0 | DEADLOCK |

逐例诊断（机器人与操作编号沿用代码的0起始编号）：

**validation_02 / 20261110**：搜索208.183s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 310, "robot": 3, "operation": "R3:B3:WELD", "kind": "WELD", "block_id": "P13-W01-S01::whole", "ready_s": 3657.4414858688106, "start": [7.02773046875, 4.259368164062502], "end": [7.02773046875, 1.3002275390625015], "duration_s": 273.9945023148148, "blocker_count": 1, "blocker_examples": ["R2:B21:WELD"], "graph": [[3, [2]]]}`。
终端阻塞：`[{"robot": 1, "operation": "R1:B37:WELD", "ready_s": 4459.931544823048, "same": 5, "cross": 0, "same_examples": ["R0:B22:POST", "R0:B22:WELD", "R0:B23:MOVE", "R0:B23:SETUP"], "cross_examples": []}, {"robot": 3, "operation": "R3:B3:WELD", "ready_s": 3657.4414858688106, "same": 10, "cross": 0, "same_examples": ["R2:B21:POST", "R2:B21:WELD", "R2:B22:MOVE", "R2:B22:POST"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=78/0；延迟边累计同轨/跨轨=1/0；终端等待图=((1, (0,)), (3, (2,)))，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 1, 'blocker': 0, 'finished': True}, {'waiting': 3, 'blocker': 2, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_02 / 20261112**：搜索236.571s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 279, "robot": 3, "operation": "R3:B3:WELD", "kind": "WELD", "block_id": "P07-W01-S01::whole", "ready_s": 3670.862222222221, "start": [7.500000000000002, 6.436999999999999], "end": [20.0, 6.43700000000001], "duration_s": 1157.407407407407, "blocker_count": 1, "blocker_examples": ["R2:B21:MOVE"], "graph": [[3, [2]]]}`。
终端阻塞：`[{"robot": 1, "operation": "R1:B25:POST", "ready_s": 4086.638444067021, "same": 2, "cross": 0, "same_examples": ["R0:B23:POST", "R0:B23:WELD"], "cross_examples": []}, {"robot": 3, "operation": "R3:B3:WELD", "ready_s": 3670.862222222221, "same": 12, "cross": 0, "same_examples": ["R2:B21:MOVE", "R2:B21:POST", "R2:B21:SETUP", "R2:B21:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=34/0；延迟边累计同轨/跨轨=2/0；终端等待图=((1, (0,)), (3, (2,)))，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 1, 'blocker': 0, 'finished': True}, {'waiting': 3, 'blocker': 2, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_03 / 20261112**：搜索165.154s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 93, "robot": 2, "operation": "R2:B3:SETUP", "kind": "SETUP", "block_id": "P02-W02-S01::whole", "ready_s": 1517.42828100035, "start": [16.95569970703125, 1.459], "end": [16.95569970703125, 1.459], "duration_s": 20.0, "blocker_count": 1, "blocker_examples": ["R3:B9:WELD"], "graph": [[2, [3]]]}`。
终端阻塞：`[{"robot": 2, "operation": "R2:B3:SETUP", "ready_s": 1517.42828100035, "same": 86, "cross": 0, "same_examples": ["R3:B10:MOVE", "R3:B10:POST", "R3:B10:SETUP", "R3:B10:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=189/0；延迟边累计同轨/跨轨=0/0；终端等待图=((2, (3,)),)，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 2, 'blocker': 3, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_07 / 20261110**：搜索174.590s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 201, "robot": 2, "operation": "R2:B4:POST", "kind": "POST", "block_id": "P09-W02-S01::whole", "ready_s": 3021.709777675817, "start": [13.683336425781249, 0.7230000000000107], "end": [13.683336425781249, 0.7230000000000107], "duration_s": 30.0, "blocker_count": 1, "blocker_examples": ["R3:B12:MOVE"], "graph": [[2, [3]]]}`。
终端阻塞：`[{"robot": 2, "operation": "R2:B4:POST", "ready_s": 3021.709777675817, "same": 72, "cross": 0, "same_examples": ["R3:B12:MOVE", "R3:B12:POST", "R3:B12:SETUP", "R3:B12:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=239/0；延迟边累计同轨/跨轨=11/0；终端等待图=((2, (3,)),)，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 2, 'blocker': 3, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_07 / 20261111**：搜索184.722s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 190, "robot": 2, "operation": "R2:B4:POST", "kind": "POST", "block_id": "P09-W02-S01::whole", "ready_s": 3021.709777675817, "start": [13.683336425781249, 0.7230000000000107], "end": [13.683336425781249, 0.7230000000000107], "duration_s": 30.0, "blocker_count": 1, "blocker_examples": ["R3:B12:MOVE"], "graph": [[2, [3]]]}`。
终端阻塞：`[{"robot": 2, "operation": "R2:B4:POST", "ready_s": 3021.709777675817, "same": 72, "cross": 0, "same_examples": ["R3:B12:MOVE", "R3:B12:POST", "R3:B12:SETUP", "R3:B12:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=250/0；延迟边累计同轨/跨轨=8/0；终端等待图=((2, (3,)),)，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 2, 'blocker': 3, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_07 / 20261112**：搜索148.939s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 222, "robot": 2, "operation": "R2:B4:POST", "kind": "POST", "block_id": "P09-W02-S01::whole", "ready_s": 3021.709777675817, "start": [13.683336425781249, 0.7230000000000107], "end": [13.683336425781249, 0.7230000000000107], "duration_s": 30.0, "blocker_count": 1, "blocker_examples": ["R3:B12:MOVE"], "graph": [[2, [3]]]}`。
终端阻塞：`[{"robot": 2, "operation": "R2:B4:POST", "ready_s": 3021.709777675817, "same": 72, "cross": 0, "same_examples": ["R3:B12:MOVE", "R3:B12:POST", "R3:B12:SETUP", "R3:B12:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=218/0；延迟边累计同轨/跨轨=6/0；终端等待图=((2, (3,)),)，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 2, 'blocker': 3, 'finished': True}]；B32恢复32/32，前沿耗尽=False；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_10 / 20261110**：搜索126.063s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 347, "robot": 3, "operation": "R3:B30:WELD", "kind": "WELD", "block_id": "P16-W05-S01::whole", "ready_s": 4753.591523513655, "start": [11.413784170910837, 3.469801635742187], "end": [11.413784170910837, 4.1698016357421865], "duration_s": 64.81481481481478, "blocker_count": 1, "blocker_examples": ["R2:B31:WELD"], "graph": [[3, [2]]]}`。
终端阻塞：`[{"robot": 1, "operation": "R1:B4:WELD", "ready_s": 4874.37962962963, "same": 2, "cross": 0, "same_examples": ["R0:B21:POST", "R0:B21:WELD"], "cross_examples": []}, {"robot": 3, "operation": "R3:B30:WELD", "ready_s": 4753.591523513655, "same": 2, "cross": 0, "same_examples": ["R2:B31:POST", "R2:B31:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=9/0；延迟边累计同轨/跨轨=0/0；终端等待图=((1, (0,)), (3, (2,)))，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 1, 'blocker': 0, 'finished': True}, {'waiting': 3, 'blocker': 2, 'finished': True}]；B32恢复14/32，前沿耗尽=True；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

**validation_10 / 20261112**：搜索101.516s。
覆盖完整=True，操作链连续=True，焊接顺序一致=True，原生机器人间实际冲突=0。
首次阻塞：`{"depth": 347, "robot": 3, "operation": "R3:B30:WELD", "kind": "WELD", "block_id": "P16-W05-S01::whole", "ready_s": 4753.591523513655, "start": [11.413784170910837, 3.469801635742187], "end": [11.413784170910837, 4.1698016357421865], "duration_s": 64.81481481481478, "blocker_count": 1, "blocker_examples": ["R2:B31:WELD"], "graph": [[3, [2]]]}`。
终端阻塞：`[{"robot": 1, "operation": "R1:B4:WELD", "ready_s": 4874.37962962963, "same": 2, "cross": 0, "same_examples": ["R0:B21:POST", "R0:B21:WELD"], "cross_examples": []}, {"robot": 3, "operation": "R3:B30:WELD", "ready_s": 4753.591523513655, "same": 2, "cross": 0, "same_examples": ["R2:B31:POST", "R2:B31:WELD"], "cross_examples": []}]`。
阻塞边累计同轨/跨轨=9/0；延迟边累计同轨/跨轨=0/0；终端等待图=((1, (0,)), (3, (2,)))，等待环=()，中间完全阻塞等待环=[]，含可延迟执行操作的关联环=[]。
阻塞机器人是否完成=[{'waiting': 1, 'blocker': 0, 'finished': True}, {'waiting': 3, 'blocker': 2, 'finished': True}]；B32恢复14/32，前沿耗尽=True；诊断原始派工状态=DEADLOCK，失败操作时间线与最终B32一致=True。

阻塞边累计是各派工状态返回的关联次数，不是独立冲突数或唯一因果。含可延迟执行操作的关联环不等同于真正等待环。B32失败仅表示该有限恢复策略未找到时间线，不构成数学不可行证明。完整路线快照未保存；13.3节历史11例及其旧快照保持原样。

**本次诊断结论（2026-11-14实测）。** 8/8均完成当前版本500轮，native Cmax与原CSV差值全部为0，最终B32状态全部重现。8条原生时间线均完整覆盖焊缝、操作链连续、焊接顺序一致，机器人间实际冲突均为0。诊断派工重放的失败操作时间线也全部与最终B32一致。

逐例首次阻塞分别为：validation_02/20261110的R3:B3:WELD被R2:B21:WELD阻塞；validation_02/20261112的R3:B3:WELD被R2:B21:MOVE阻塞；validation_03/20261112的R2:B3:SETUP被R3:B9:WELD阻塞；validation_07三个种子均为R2:B4:POST被R3:B12:MOVE阻塞；validation_10两个种子均为R3:B30:WELD被R2:B31:WELD阻塞。终端另有validation_02的R1→R0，以及validation_10的R1→R0和R3→R2阻塞，详见上列操作编号。

这些失败均表现为B32既有派工承诺下的同轨固定操作阻塞；终端阻塞者都已完成自身路线，其已固定操作时窗仍约束被阻塞机器人的较早ready时刻。全部8例的跨轨阻塞/延迟关联、终端等待环、中间完全阻塞环及含延迟选择的关联环均为0。validation_02/03/07共6例用尽32次恢复且前沿未耗尽；validation_10两例各执行14次恢复后前沿耗尽。原生优先排程与B32派工及有限恢复策略的差异足以解释本次可复现的失败现象；本轮仅诊断并保留失败，没有修改搜索、物理约束或恢复策略。

复现命令为指定解释器在DRL目录执行`-B -X utf8 -W ignore scripts/run_gnn_weld_layout.py wag-deadlock-diagnosis --workers 4`，同时设置`PYTHONDONTWRITEBYTECODE=1`以及Numba/OMP/MKL/OpenBLAS/NumExpr线程数为1。只重跑上述8例，未重跑90次正式WAG实验。

完成8例诊断后，使用同一解释器和单线程环境执行既有`tests/test_wag_baseline.py`、`tests/test_hga_baseline.py`、`tests/test_baseline_common.py`，结果为**63 passed，65 warnings，166.54s**；警告为既有Numba类型反射弃用提示。命令附加`-B -X utf8 -m pytest -q -p no:cacheprovider --disable-warnings`，未运行无关GNN测试。另在内存中核对诊断入口对native Cmax超容差和不足500轮的返回值均跳过归因；HGA入口按预热→α20/200000→α18/200000顺序调用，追加保留旧行、DEADLOCK的B32 Cmax留空、199999有效子代标为INCOMPLETE。这些模拟结果未写入实验CSV。

### 13.12 HGA α=20与α=18的200000有效子代完整预算配对实验（2/2完成）

本轮于2026-11-14至2026-11-15实际完成两组长预算运行。固定`validation_07`（115条父焊缝）、seed=20261110，使用当前`ORIENTED_COORDINATED_VND_V1`和EXPLICIT方向编码。两组均为μ=20、λ=10、200000有效子代、完整M1～M6、`vnd_max_passes=None`、`vnd_candidate_cap=None`、模式变异概率0.15、`time_limit=math.inf`、`ScientificConfig()`及`FORMAL_SCOPE_V2`；只改变α。没有修改WAG/HGA算法主体、数据、物理约束、拆分规则或最终B32，`AdaptedHGAConfig`默认α仍为20。

使用指定解释器，在DRL目录执行既有脚本新增的简单入口`hga-alpha-full-budget --workers 1`。Numba、OMP、MKL、OpenBLAS、NumExpr计算线程均设为1，先用10个有效子代预热，再依次运行α=20和α=18。每次正式调用均重新设置同一种子，新建种群、工作区和VND缓存，没有注入预热或另一组的搜索结果。首次准备阶段在预热66.830159s后因现有CSV写句柄权限停止，尚未开始正式搜索；解决写入权限后，同一正式进程重新预热60.520375s，随后两组各完整执行一次。上述预热均不计入正式结果，未因B32结果挑选种子或重跑。

`runtime_s`为调用HGA至返回的实测墙钟总耗时，包含预计算、初始化、交叉、完整VND及种群管理，排除预热和最终B32；B32另行计时。初始native Cmax沿用13.10节既有字段定义，为20个初始个体完成VND后的当前最优值，α已影响这一阶段。搜索内不调用B32或独立认证；每组结束后仅对最终最优固定方向方案调用一次既有`schedule_explicit()`。

两组主要结果如下，时间及Cmax单位均为秒：

| α | 实际有效子代 | HGA总耗时 | 初始native Cmax | 最终native Cmax | 最终B32状态 | 最终B32 Cmax | B32耗时 | 完成状态 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 20 | 200000 | 24677.210502 | 5702.410599243 | 5628.571419059 | FEASIBLE | 5628.571419059 | 0.315199500 | COMPLETED |
| 18 | 200000 | 21671.872442 | 5751.467420702 | 5630.628466249 | FEASIBLE | 5630.628466249 | 0.112084700 | COMPLETED |

| α | VND唯一调用 | VND缓存命中 | 原生候选评价次数 |
|---|---:|---:|---:|
| 20 | 197066 | 2954 | 1222088730 |
| 18 | 197187 | 2833 | 1178765872 |

结果直接追加到既有`results/new_dataset_final/hga_alpha_sensitivity.csv`，当前共20条数据：原18条1000子代记录逐字段、按原顺序完整保留，最后两条依次为α=20和α=18的200000子代记录。两组均为`run_status=COMPLETED`，各自`vnd_unique_calls+vnd_cache_hits=200020`，对应20次初始化和200000次完整VND子代请求。未将1000子代记录混入完整预算结果，也没有将未完成运行标为正式完成。程序正常结束，退出码为0。

**完整预算分析。**

1. **本次α=18确实减少完整预算墙钟时间。** α=20实测24677.210502s（约6小时51分17秒），α=18为21671.872442s（约6小时1分12秒），减少3005.338059s，约50分5秒，降幅**12.18%**。这是两次真实200000有效子代的结果，没有使用短预算线性外推。
2. **短预算的速度方向在本配对样本中保留。** 13.10节同排样/种子的1000子代耗时为α=20的121.194535s、α=18的109.421881s，α=18减少9.71%；完整预算下减少12.18%。固定运行顺序、机器状态等影响未通过重复试验分离，不能据此宣称跨排样或跨种子的普遍提速幅度。
3. **α=18没有改善最终native Cmax。** 完整预算下α=18比α=20高2.057047s，即**+0.03655%**。1000子代时α=18为5647.756543s、α=20为5657.522305s，α=18低0.17262%；该质量排序在完整预算下反转。两组均改善各自初始化结果，但不能把这点解释为α=18相对α=20的最终质量改善。
4. **最终B32均可行。** 两次均返回FEASIBLE，且本次各自B32 Cmax与native Cmax相同。只进行了每组一次最终B32，未用B32筛选搜索候选，也未追加独立认证。
5. **评价次数只能部分解释时间差。** α=18原生候选评价减少43322858次，降幅3.545%，与耗时下降方向一致，但明显小于12.18%的耗时降幅。其VND唯一调用反而增加121次、缓存命中减少121次，提速不能归因于更多完整VND缓存命中。α改变搜索轨迹及每次协调调度的计算代价，其他VND工作和机器状态也影响墙钟时间；仅凭这些计数不足以唯一分解提速原因，本轮未追加性能插桩或优化模块。
6. **下一步正式主参数建议继续固定α=20。** α=18在这个完整预算样本中确认了速度收益，但最终native和B32 Cmax均略高，尚无充分依据用单排样、单种子结果替换正式主参数。α=18保留为有完整预算数据支持的提速备选；默认α=20未改。本轮不声称统计显著，也不宣称任一α普遍优于另一α。

本轮任务A和任务B均已完成。其他代表排样（validation_03/04）的新版EXPLICIT完整预算验证，以及α结论的跨排样/多种子验证仍未完成；本轮没有启动这些运行，也没有启动全量30次VALIDATION和60次TEST的HGA实验。

正式长预算入口复现命令如下；其行为是向已有CSV追加结果：

```powershell
Set-Location -LiteralPath 'D:\pybullet_test\MRTA_GA\DRL'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:NUMBA_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -W ignore 'scripts/run_gnn_weld_layout.py' hga-alpha-full-budget --workers 1
```

本轮只修改现有实验入口、本报告和上述既有CSV，没有新增文件。算法模块`wag_vns.py`、`hga.py`、`common.py`及HGA默认α保持原样；保留工作区其他既有修改。已核对脚本语法、旧18行保留、两组实际200000子代、VND请求计数和最终结果字段；相关三份单元测试的本轮实测结果见13.11节，未重复运行无关GNN测试。

### 13.13 路线级增量VND与原生协调选择：20000子代试验未通过时间门槛

本轮于2026-11-15实际修改既有HGA并完成性能试验。新来源为`ORIENTED_ROUTE_VND_NATIVE_SELECT_V1`，它是新的HGA适配定义，**不是**13.12节`ORIENTED_COORDINATED_VND_V1`的等价提速。旧入口、旧结果和默认`vnd_objective='COORDINATED'`保留；本轮入口显式选择`ROUTE_CYCLE`，默认α仍为20。

**评价方式。** 四条路线继续保存任务ID和显式方向。VND使用现有processing/travel、`_oriented_route_cost()`和`_oriented_segment_cost()`，只增量计算受影响路线的周期，目标为四条路线周期的最大值；可能改善时才物化路线，并用完整受影响路线成本消除浮点近似带来的接受歧义。完整M1～M6、α-nearness、最长路线限制、M4双重反转、机器人资格和严格改善后从M1重启的规则保留，不设置轮数或候选上限。

每个完整子代结束VND后只调用一次`_priority_arrays()`，种群锦标赛、生存选择和最优记录使用这个固定方向的原生协调Cmax。`record()`直接复用返回数组，通过既有`_schedule_from_arrays()`生成最优时间线，不重复调度。原有VND缓存继续使用，命中后也只做该子代的一次完整原生评价；没有新增缓存体系、评价器或认证器。保留原始初始化方案的原生最优记录，因此本次20000子代的完整原生调用数为20040：20个VND前初始化方案、20个VND后初始个体、20000个完整子代。

`_move_segments()`改用固定三个区间的元组，零长区间由既有消费者跳过，避免逐候选创建两份动态列表；已有区间代数和六邻域定义不变。原有工作区缓冲复用为processing前缀，先用非负空行程成立的处理时间下界排除不可能改善的候选，再进行区间连接评价。这些剪枝不截断VND，候选计数在剪枝前递增。WAG、GNN、共同评价模块、数据、拆分规则、`ScientificConfig()`、`FORMAL_SCOPE_V2`及B32均未修改。

**实测条件和全部保留记录。** 固定validation_07（115条父焊缝）、seed=20261110、μ=20、λ=10、α=20、模式变异概率0.15、EXPLICIT、完整六邻域、无VND上限、`time_limit=math.inf`、`certify=False`。使用指定解释器，单进程，Numba/OMP/MKL/OpenBLAS/NumExpr线程均为1。各次先用10个子代预热，预热不计入表中时间或CSV；每次正式试验重新初始化种群、工作区和缓存，搜索中不调用B32。

| 20000子代性能试验 | 实际有效子代 | 总运行时间/s | 最终native Cmax/s | 时间门槛 | 处理 |
|---|---:|---:|---:|---|---|
| 路线VND初版，动态区间列表 | 20000 | 435.970468 | 5642.149363711 | 超过180s | 保留原始失败记录 |
| 固定区间元组 | 20000 | 256.295397 | 5642.149363711 | 超过180s | 保留记录 |
| 加入既有处理时间下界，当前保留实现 | 20000 | 236.561027 | 5642.149363711 | 超过180s | 保留实现及记录 |
| 改传路线长度的尝试 | 20000 | 388.964712 | 5642.149363711 | 超过180s | 未显示整体收益，撤回该改动；保留记录 |
| Python入口强制Numba字面量的尝试 | 未完成，最终计数未知 | 未知 | 未记录最终值 | INCOMPLETE | 最后日志39子代/60.646s，反复类型处理导致严重额外开销，停止并撤回 |

上述四次完成试验均遍历587841067个逻辑候选，VND唯一调用19237、缓存命中783，合计20020次VND请求；VND内原生候选评价次数为0，轮数/候选截断次数均为0。四次最终native Cmax和上述计数完全一致，未减少邻域或有效子代数。所有记录追加到既有`results/new_dataset_final/hga_results_optimized.csv`：完整短预算记录的状态为`NATIVE_ONLY`、phase为`PERFORMANCE_PILOT`，B32调用数为0；INCOMPLETE记录只保留最后观察进度，实际最终预算、总耗时和最终Cmax留空，没有把39当作精确最终预算。没有删除失败或修改种子。

当前保留实现的搜索时间为236.561014s，初始化时间0.690739s，VND时间202.512355s，完整个体原生协调评价27.365192s，交叉4.245611s，种群管理1.243803s。初始化中包含自身VND和原生评价，其时间不能再次加到上述阶段总和。VND占搜索时间约85.6%，已经消除了逐邻域原生协调评价，但587841067次候选遍历及区间评价仍使本次试验超过150/180秒要求。

另在内存中用相同2000子代、相同种子做两组四次交替计时定位，未写成正式实验结果：将未改动最大路线判断改为四位掩码，原实现43.607632/43.305388s，对应尝试42.982538/43.752355s；编译调用入口固定路线目标，原实现43.555543/42.581446s，对应尝试42.463075/43.468050s。每组搜索的候选数64554752、native Cmax 5717.841225825和缓存计数一致，但均无稳定整体收益，故没有把这些改动写入项目。不同进程的墙钟和各阶段时间存在变化，不能将所有时间差唯一归因于某一项修改；这里不声称统计提速幅度，也没有保留强制内联、并行或复杂缓存优化。

**完整预算结果及验收。** 由于20000子代未通过门槛，本轮遵守执行顺序，没有启动任何新版本200000子代实验，最终B32也未调用。

| 排样 | 父焊缝数 | 新版本200000子代实验 | 完整预算运行时间 | 完整预算native Cmax | 最终B32/Cmax |
|---|---:|---|---|---|---|
| validation_07 | 115 | 未启动 | 无实测结果 | 无实测结果 | 未调用/无结果 |
| validation_03 | 82 | 前置条件未满足，未启动 | 无实测结果 | 无实测结果 | 未调用/无结果 |
| validation_04 | 91 | 前置条件未满足，未启动 | 无实测结果 | 无实测结果 | 未调用/无结果 |

13.12节当前协调VND完整预算基准仍为validation_07/seed=20261110：α=20，24677.210502s、Cmax 5628.571419059、B32 FEASIBLE；α=18，21671.872442s、Cmax 5630.628466249、B32 FEASIBLE。新版本只有20000子代的native 5642.149363711s，预算不相等，不能据此报告200000子代的提速或最终质量退化幅度，更不能把耗时乘十当作完整预算实测。旧完整预算记录未被本轮数据覆盖。

**本轮结论：尚未达到30分钟验收目标，尚不具备作为正式HGA对照组的完整预算证据。** 已完成路线级增量VND、每个完整子代一次原生评价及结果复用、相关短预算实测和失败记录保存。仍需使实际20000子代通过门槛，之后才有条件依次验证validation_07及另外两个代表排样的200000子代运行时间、最终B32可行性和同预算质量；本轮没有启动全量30次VALIDATION和60次TEST，也没有用短预算冒充完整预算。

复现入口为在DRL目录用指定解释器执行`-B -X utf8 -W ignore scripts/run_gnn_weld_layout.py hga-route-vnd --workers 1`，并设置`PYTHONDONTWRITEBYTECODE=1`及上述计算线程为1。入口先预热，再实际运行20000子代；超过180秒或不完整即返回，150～180秒返回供谨慎评估，只有不超过150秒才自动执行validation_07完整预算；仅当该完整预算实际不超过1800秒且B32 FEASIBLE，才继续validation_03/04。

最终保留代码使用指定解释器执行`tests/test_hga_baseline.py`和`tests/test_baseline_common.py`，结果为**42 passed，63 warnings，167.05s**；警告为既有Numba类型反射弃用提示。测试覆盖六邻域显式方向、区间成本与物化成本一致、严格局部最优终止、下界开关结果一致、路线VND禁止调用原生协调，以及每个完整子代恰好一次原生评价并复用最优时间线。未运行无关GNN/WAG测试。另已核对三个修改的Python文件语法、原有HGA结果5行按顺序逐字段保留、α实验CSV仍为原20行、所有短预算/INCOMPLETE字段及历史13.3、13.9～13.12节保留。本轮仅修改既有`hga.py`、实验入口、HGA测试、本报告和HGA结果CSV，未新增文件。

### 13.14 有序候选遍历提速与200000有效子代验证

本轮以GitHub提交`70d64ccc9cecba07096b4aef542d95a15f0eeab1`为参照；修改前的本地`hga.py`和HGA测试已与该提交逐字核对一致。保留本地实验脚本的既有额外修改，不切换、回退或整理父目录工作区。来源继续使用`ORIENTED_ROUTE_VND_NATIVE_SELECT_V1`，本轮只优化计算实现，没有改变13.13节的路线VND及原生协调种群选择定义。

**本次唯一性能改动。** 在`_vnd_oriented_arrays()`中每轮VND只扫描一次当前`all_ids × near[u]`；新增同文件私有函数`_collect_oriented_candidates()`，按既有资格条件分别向预分配的六组NumPy整数缓冲写入有序候选。缓冲在一次VND调用开始时分配，接受改善后刷新机器人归属、位置、长度和最长路线，再重新填充；仍按M1→M6处理，并在首次严格改善后立即从M1重启。每组容量为任务数×α，不产生候选截断。

每一邻域的数组都是原有有向遍历序列经过相同资格条件的稳定过滤，因此候选先后顺序不变。数组保留全部对称重复，原有计数、`seen`判断、对称候选首次遇到及M6空变化判断仍在原评价位置执行；没有使用`index[u] < index[v]`删边。生成数组不改变路线或随机数状态，后续成本、剪枝和接受规则均沿用原代码。这避免六遍重复近邻扫描及逐候选重复资格判断，同时保留逻辑候选统计语义。

`_move_segments()`、`_oriented_segment_cost()`、`_materialize_segments()`和`_materialize_directions()`本轮均未改写。已有加工时间下界、受影响路线成本和未改动路线下界全部保留，没有增加剪枝。显式方向、完整六邻域、α=20、μ=20、λ=10、模式变异概率0.15、无VND轮数/候选上限、每个完整子代一次原生协调评价及既有VND缓存和种群管理均保持原样。

**正确性验证。** HGA与共同评价测试为**43 passed，63 warnings，158.80s**。新增候选序列测试核对六邻域原始顺序，覆盖仅单向存在且起点在当前任务序列中更靠后的近邻、对称首次遇到、相邻任务对末端限制及四种最长路线归属；既有显式方向、增量成本、局部最优终止、资格、原生调用数和缓存测试通过。警告为既有Numba类型反射弃用提示，未运行无关GNN/WAG测试。

另在内存中使用指定提交的原VND函数，对validation_07分别以seed=20261110及731执行100子代对比。两个种子的新旧最优方案、显式方向、完整原生时间线及候选/缓存/去重计数全部一致，每次原生完整调用均为140。对应逻辑候选数分别为5306095和4378943，VND唯一/命中分别为115/5及117/3。这些是实现正确性检查，不作为完整预算结果写入CSV，没有保存另一套评价代码或新文件。

**20000子代实测已通过150秒门槛。** 固定validation_07（115父焊缝）、seed=20261110、`vnd_objective='ROUTE_CYCLE'`、`certify=False`、`time_limit=math.inf`及上述全部参数，使用指定Python解释器，单进程及单Numba/OMP/MKL/OpenBLAS/NumExpr线程。正确性检查之后再用10个子代预热（0.626259s），检查及预热均不计入正式计时。实际计时对比如下：

| 实现 | 实际有效子代 | 总运行时间/s | VND时间/s | 原生协调评价时间/s | 最终native Cmax/s |
|---|---:|---:|---:|---:|---:|
| 13.13节保留实现 | 20000 | 236.561027 | 202.512355 | 27.365192 | 5642.149363711 |
| 本轮有序候选数组 | 20000 | 107.794275 | 74.060937 | 26.990151 | 5642.149363711 |

本次总墙钟减少54.43%，约为原记录的2.195倍速度；VND时间减少63.43%。新旧最终native Cmax完全相同，逻辑候选均为587841067，VND唯一调用19237、缓存命中783，完整原生调用20040，VND内部原生候选评价及轮数/候选截断均为0。新记录已追加至原HGA结果CSV，phase为PERFORMANCE_PILOT、状态NATIVE_ONLY、B32调用0，不混作正式完整预算。该时间对比是同排样、同种子的实际记录，不表示跨排样或跨种子的统计结论。

通过门槛后，现有入口按validation_07→validation_03→validation_04顺序实际完成三次200000有效子代运行。各组重新初始化种群、工作区及缓存，不继承短预算或其他排样的搜索结果；均固定seed=20261110、ScientificConfig()及FORMAL_SCOPE_V2。validation_07搜索1346.294013s且B32 FEASIBLE后才开始另外两个代表排样。每个完整搜索结束只调用一次既有B32；以下为实际结果，没有用短预算耗时外推替代实测。搜索时间包含本次工作区准备、初始化和全部遗传搜索，CSV总时间另包含最终B32。

| 排样 | 实际有效子代 | 搜索时间/s | 含B32总时间/s | native Cmax/s | B32状态 | B32 Cmax/s | B32时间/s |
|---|---:|---:|---:|---:|---|---:|---:|
| validation_07 | 200000 | 1346.294013 | 1346.575278 | 5631.192830637 | FEASIBLE | 5631.192830637 | 0.281255 |
| validation_03 | 200000 | 966.541063 | 966.654897 | 5003.849097384 | FEASIBLE | 5003.849097384 | 0.113819 |
| validation_04 | 200000 | 922.811360 | 922.911627 | 4844.711022732 | FEASIBLE | 4844.711022732 | 0.100257 |

validation_07完整预算与13.12节旧协调VND比较：相对α=20的24677.210502s，本次搜索1346.294013s，减少94.544%，约18.330倍速度；最终native/B32 Cmax高2.621412s（+0.046573%）。相对α=18的21671.872442s，减少93.788%，约16.097倍速度，Cmax高0.564364s（+0.010023%）。这些是完整200000预算的实测比较，但局部搜索目标已在13.13节发生适配变化，不能把全部跨版本提速归因于本轮候选数组；本轮数组优化本身的同算法时间对照为上表20000子代的236.561027→107.794275s。没有原路线VND的200000子代时间对照，也未用外推补造该值。validation_03/04缺少当前EXPLICIT协调VND同条件完整预算对照，本轮只报告其实际完整结果，不用历史OPTIMIZED_ITERATIONS值代替。

完整预算的实际候选与缓存统计如下：

| 排样 | VND逻辑候选 | VND唯一调用 | VND缓存命中 |
|---|---:|---:|---:|
| validation_07 | 6267110274 | 192969 | 7051 |
| validation_03 | 4143338326 | 187142 | 12878 |
| validation_04 | 3901038206 | 187427 | 12593 |

每组唯一调用加缓存命中均为200020（20个初始化个体和200000个子代）；每组完整原生协调调用均为200040（另含20个原始初始化方案），没有为记录最优时间线再做一次原生评价。三组VND内部原生候选评价数、轮数截断和候选截断均为0，最终B32均恰好调用一次。全部新记录的run_status=COMPLETED；运行进程正常退出，退出码0。

**本轮验收通过。** 20000子代实际107.794275s，不超过150秒；三个代表排样均真正完成200000有效子代，搜索时间及含B32总时间均不超过1800秒，最终B32全部FEASIBLE。validation_07约22分26秒、validation_03约16分7秒、validation_04约15分23秒。本轮只做候选遍历这一项性能优化即达到目标，沿用现有精确区间成本计算，不再追加无必要的成本公式、Numba编译技巧或缓存改造。该实现已具备三个代表排样、单种子完整预算的执行证据，可作为后续正式HGA对照实现；不据此宣称跨种子统计优越性，也不掩盖相对旧协调VND的小幅质量退化。α继续固定20，没有自动修改默认参数。

结果继续追加到既有`results/new_dataset_final/hga_results_optimized.csv`，当前14行：原有10行按原顺序完整保留，新增1条20000子代性能记录及3条200000子代正式完整预算记录，来源均为ORIENTED_ROUTE_VND_NATIVE_SELECT_V1。原有失败/INCOMPLETE未删除，`hga_alpha_sensitivity.csv`原20行完全保留。已核对每组实际预算、阶段、耗时、B32调用和状态及计数；代码与已通过43项相关测试的版本一致。本轮仅修改既有`src/mrta_baselines/hga.py`、`tests/test_hga_baseline.py`、本报告及HGA结果CSV，未新增文件，现有实验入口及其本地额外修改完整保留。WAG、GNN、B32、科学配置、数据与拆分规则均未修改，未启动全量30次VALIDATION和60次TEST。其他排样/种子的统计验证仍未执行，不属于本轮已完成结果。

### 13.15 新版HGA正式VALIDATION/TEST实验与WAG配对比较

本轮使用`ORIENTED_ROUTE_VND_NATIVE_SELECT_V1`补齐正式90次HGA运行。开始时仅有13.14节validation_03/04/07、seed=20261110的3条新版200000有效子代完整结果；其余旧版、20000子代性能记录和INCOMPLETE均保留但不纳入正式统计。固定validation.jsonl的10个排样、test.jsonl的20个排样以及种子20261110/20261111/20261112，先补齐VALIDATION剩余27次，再执行TEST的60次。WAG只读取`PARTIAL_LNS_V1`、500轮的既有90条记录，不重新计算，不使用OPTIMIZED_ITERATIONS。

**必要入口修改与验证。** 仅在既有实验脚本增加简单的正式来源映射，保留原BASELINE_SOURCES供历史实验使用。正式baseline_job显式选择ROUTE_CYCLE，仍为EXPLICIT、μ=20、λ=10、α=20、模式变异概率0.15、完整M1～M6、无VND轮数/候选上限、200000有效子代、time_limit=math.inf、certify=False、ScientificConfig()及FORMAL_SCOPE_V2。HGA/WAG算法主体和最终B32均未修改。完整子代继续使用原生协调Cmax选择，每次完整搜索结束只调用一次既有B32，不调用独立认证。

恢复和汇总共用完成记录判定：来源、方法、种子、完整预算、实际正数耗时及真实最终B32状态/一次调用须匹配；HGA还要求既有diagnostic的FULL/COMPLETED标记。NATIVE_ONLY、INCOMPLETE、短预算和旧来源均不能跳过或混入统计。主进程在future完成后立即向原CSV追加一行并关闭写句柄，子进程不写CSV；任务异常保存为INCOMPLETE，未知预算/时间不补造，完成记录保留供补跑缺失任务。

指定解释器下先运行HGA与共同评价两份既有测试，**43 passed，63 warnings，109.76s**。另外在内存中核对正式/历史目标分离、3条已有结果恰好保留、VALIDATION仅27个缺失任务、主进程单写、B32一次及各类错误完成标记被排除；模拟数据未写入CSV。没有运行无关GNN/WAG测试。

**执行条件。** 机器8个逻辑CPU、总内存15.80 GiB，启动检查可用约5.47 GiB，选择3个工作进程。Numba/OMP/MKL/OpenBLAS/NumExpr计算线程均为1。每进程先在validation_07用10个子代进行Numba预热，不调用B32，预热不计入正式任务时间；每个正式任务重新构造种群、工作区与VND缓存并设置原始种子。原3次代表结果沿用单进程实测，不重新计算。补齐任务的运行时间为并行环境中的真实墙钟，CSV runtime_s包含该次搜索和最终B32，没有做时间归一化或外推。TEST仅在VALIDATION全部完成后开始，期间不根据TEST结果调整参数、算法或种子。

正式执行命令为DRL目录下的指定Python调用`scripts/run_gnn_weld_layout.py baselines --method HGA --split validation --workers 3`，成功完成后再用同入口`--split test --workers 3`。运行前设置PYTHONDONTWRITEBYTECODE=1，并使用-B -X utf8 -W ignore。恢复仍使用上述入口，依赖已有CSV跳过完成任务，不新增管理脚本、结果文件或manifest。

**执行已全部完成（2026-11-15至2026-11-16实测）。** 新版HGA VALIDATION为30/30次完成、30次B32可行，TEST为60/60次完成、59次B32可行。原3条代表排样结果沿用，本轮实际补齐27次VALIDATION及60次TEST，共新增87条完整预算记录；每条均真正完成200000有效子代，实际正数运行时间和一次真实最终B32已保存。先完成VALIDATION再运行TEST，TEST期间没有修改算法、参数、物理约束或随机种子。进程与最终baseline-summary均正常退出，退出码0。

**正式汇总。** HGA仅统计ORIENTED_ROUTE_VND_NATIVE_SELECT_V1/200000有效子代，WAG仅统计PARTIAL_LNS_V1/500轮。下表Cmax只取最终B32 FEASIBLE记录；运行时间取全部正式记录（含DEADLOCK），单位均为秒。报告显示六位小数，CSV保留原始精度。

| 集合 | 算法 | 实际完成/应完成 | B32可行 | 可行率 | 可行Cmax均值/s | 可行Cmax中位数/s | 运行时间均值/s | 运行时间中位数/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VALIDATION | WAG+VNS | 30/30 | 22/30 | 73.33% | 5389.672876 | 5466.123862 | 114.541215 | 103.591607 |
| VALIDATION | HGA | 30/30 | 30/30 | 100.00% | 5374.413953 | 5445.501277 | 1993.324817 | 1095.432327 |
| TEST | WAG+VNS | 60/60 | 60/60 | 100.00% | 6671.902844 | 6717.024993 | 100.702366 | 94.060588 |
| TEST | HGA | 60/60 | 59/60 | 98.33% | 6623.893949 | 6637.949903 | 1190.647852 | 1134.253504 |

**同排样、同随机种子的配对比较。** 定义ΔCmax=HGA最终B32 Cmax−WAG最终B32 Cmax，负值表示HGA更短；胜/平/负均以HGA为视角，平局容差1e-6秒。胜平负与Cmax差值只计算双方B32均可行的样本，不将DEADLOCK转换为惩罚值或native Cmax。两算法可行集合不同，因此共同可行配对结果是相同实例质量比较的直接依据。

| 集合 | 同ID/seed配对数 | 共同可行数 | HGA胜/平/负 | ΔCmax中位数/s | 相对差值中位数 | 仅HGA可行 | 仅WAG可行 | 双方均失败 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| VALIDATION | 30 | 22 | 21/0/1 | -51.716959 | -0.924983% | 8 | 0 | 0 |
| TEST | 60 | 59 | 59/0/0 | -21.837687 | -0.306610% | 0 | 1 | 0 |

VALIDATION唯一的共同可行失利为validation_05/20261110：HGA 5483.312793563s、WAG 5479.538663817s，HGA高3.774129746s。其余21个共同可行VALIDATION样本和59个共同可行TEST样本，HGA Cmax均更短。VALIDATION另有8例仅HGA可行；TEST有1例仅WAG可行。以上为本数据集和既定种子的实际观测，不据此宣称全面或跨分布优势。

**保留的新版HGA最终失败。** 唯一失败是test_07/20261112；它仍是完成200000子代且完成一次B32的正式运行，恢复入口会跳过，避免为提高可行率重跑。最终Cmax留空，不用native Cmax替代。

| 排样 | 种子 | 实际有效子代 | native Cmax/s | 最终B32 | 最终B32 Cmax | 总运行时间/s | B32调用 |
|---|---:|---:|---:|---|---|---:|---:|
| test_07 | 20261112 | 200000 | 6120.415393703 | DEADLOCK | — | 986.092620 | 1 |

该次搜索984.746694s、最终B32 1.345773s；原返回诊断为wait-for cycles=()，R3被R2:B13:MOVE/SETUP/WELD阻塞。本轮只保留已有返回信息，未额外重放B32、调用独立认证或修改恢复策略，也不把这次有限策略失败解释为物理不可行证明。

**运行时间解释。** runtime_s为脚本perf_counter起止差值的实测墙钟，包含该次搜索和最终B32；不是每任务CPU时间。下列五条最长原始记录全部保留并纳入上表统计，均已完成相同200000预算及最终评价。没有按耗时长短删选、重跑、扣减暂停/竞争时间或推算替代值。未记录每任务CPU秒，无法确认较长墙钟的具体来源，主机暂停等仅是未核实可能性。原3次代表结果为单进程，其余87次为3进程并行环境；WAG使用既有正式墙钟。故不将全部时间差归因于纯算法计算效率。

| HGA排样 | 种子 | 原始总运行时间/s |
|---|---:|---:|
| validation_10 | 20261111 | 14179.488676 |
| validation_10 | 20261112 | 14145.375746 |
| test_17 | 20261110 | 3039.379796 |
| test_17 | 20261111 | 2998.629438 |
| test_16 | 20261112 | 2693.235891 |

**保存与完整性核对。** 既有hga_results_optimized.csv现有101条数据：90条新版正式完整预算记录、11条保留的历史/短预算/INCOMPLETE记录；启动时14条按原顺序逐字段完整保留，其中3条代表正式结果没有重算。wag_vns_results_optimized.csv的180条内容与启动时完全一致，其中90条PARTIAL_LNS_V1用于本次统计，另90条OPTIMIZED_ITERATIONS保持原样且排除。α结果文件原20条、HGA/WAG/共同评价模块及HGA测试文件亦与启动时一致。

已核对90个预期(layout_id, seed)键完整且唯一、每条200000子代与一次B32、FULL/COMPLETED标记及真实耗时；每组VND唯一调用+缓存命中=200020，完整原生评价=200040（含20个原始初始化方案及20个VND后初始个体），VND内部原生候选评价、轮数截断和候选截断全部为0。90条最终B32共调用90次，没有独立认证流程。再次以同一正式入口分别恢复VALIDATION与TEST，输出completed=30/60、remaining=0，未创建工作进程或重复计算，包括DEADLOCK和较长耗时记录。

本轮完成要求已达到：新版HGA VALIDATION 30/30次、TEST 60/60次均实际完成，87次缺失运行补齐，WAG冻结结果保持原样，既有baseline-summary可直接复现上述统计与配对比较。仅修改现有scripts/run_gnn_weld_layout.py、本报告及HGA结果CSV；未新增文件、脚本、结果CSV或报告，未改动HGA/WAG算法主体、GNN主算法、B32、科学配置、资格/可达性、数据或拆分规则。相关43项测试结果及入口检查见本节前文。

### 13.16 WAG+VNS与HGA等时间预算对照实验

本轮补充固定120秒搜索墙钟预算的TEST对照，不重跑13.15节已有90+90次完整迭代预算实验。120秒在查看本轮TEST结果前确定，依据是WAG VALIDATION完整预算平均114.541215s、中位103.591607s。两算法分别运行20个TEST排样×种子20261110/20261111/20261112，共各60次，预定总数120次；不根据TEST结果改变预算、算法或种子。

WAG沿用PARTIAL_LNS_V1、n=10、p=0.50、500轮上限、max_wag_variants=None及Move/Swap/LNS；HGA沿用ORIENTED_ROUTE_VND_NATIVE_SELECT_V1、ROUTE_CYCLE、EXPLICIT、μ=20、λ=10、α=20、模式变异概率0.15、完整M1～M6、200000有效子代上限，不限制VND轮数或候选数。均为ScientificConfig()、FORMAL_SCOPE_V2、原数据/资格/拆分/物理约束及certify=False。每个任务重新初始化随机数及算法工作区，HGA重新初始化种群和VND缓存；不承接预热或其他任务的最优方案。

预算从各自既有CommonBaselineEvaluator启动后计算，包含工作区准备、算法初始化及搜索；使用result.actual_runtime记录这段实际搜索墙钟，并单列外层搜索调用时间、初始化时间、实际迭代数、result.overshoot和终止原因。现有HGA可在一个完整子代结束后检查超时，WAG也可能完成正在进行的原生评价后退出；不强制中断VND、不减少候选。最终只选result.timely_best，其completed_at必须不超过120秒；超时后更新的无时限最优结果不参与最终B32。若没有及时候选，状态为NO_TIMELY_CANDIDATE，B32调用0次；否则搜索结束后仅调用一次既有B32，时间单列且不计入120秒。DEADLOCK保留，最终Cmax留空，不以native Cmax替代，不额外认证或重放。

两算法分开按WAG→HGA顺序执行，各3个进程，每进程Numba/OMP/MKL/OpenBLAS/NumExpr线程为1。每进程先用validation_07/seed=20261110预热：WAG20轮覆盖Move/Swap/LNS热路径，HGA10子代复用既有预热；预热不调用B32、不写CSV、不计入正式任务时间。使用D:\pybullet_test\.venv\Scripts\python.exe，工作目录D:\pybullet_test\MRTA_GA\DRL，设置PYTHONDONTWRITEBYTECODE=1及上述线程环境变量为1。

既有baselines入口分别增加--method "WAG+VNS"或--method HGA、--split test --workers 3 --time-limit 120；完整预算不传--time-limit，原入口与恢复判定保持独立。等时间记录只追加到各自原优化CSV，source=EQUAL_TIME_120S，diagnostic保留底层算法来源、预算、实际搜索/B32耗时、及时候选完成时间、超时、迭代及最终状态。主进程每完成任务追加一行，正常返回的DEADLOCK及无及时候选亦计入完成；异常记录INCOMPLETE，恢复只补缺失任务。baseline-summary --time-limit 120读取等时间记录，不传参数仍只汇总原完整预算。

配对统计先按(layout_id, seed)匹配双方B32可行记录，再在同一排样的共同可行种子上分别计算两算法Cmax中位数；若某种子有一方失败，则两方均不把该种子用于Cmax配对，并单独报告可行次数和共同可行种子数。以20个排样作为分组比较单位，另外报告运行层面的描述性胜平负；不把60个种子运行解释为60个独立排样，不给失败赋惩罚值。ΔCmax=HGA−WAG，负值表示HGA更短，平局容差1e-6秒。

指定解释器下运行tests/test_wag_baseline.py、tests/test_hga_baseline.py和tests/test_baseline_common.py，结果为68 passed、63 warnings、118.94s；警告为既有Numba反射列表弃用提示。另在内存中验证超时更优候选被排除、及时/无及时候选分别一次/零次B32、完成记录及来源隔离、失败不赋Cmax和共同可行种子的排样中位数汇总；模拟数据未写CSV。未运行无关GNN测试。

**120次实测已全部完成（2026-11-16）。** WAG和HGA各60/60次；每个(layout_id, method, seed)唯一，正常执行进程、两种预算汇总和恢复检查均退出0。没有INCOMPLETE或缺失任务，也没有为获得可行结果重复运行TEST种子。以下数字由现有CSV直接读取，报告显示六位小数，CSV保留原精度。

**可行性与最终Cmax。** 只使用source=EQUAL_TIME_120S的TEST记录；Cmax仅取最终B32 FEASIBLE样本，搜索/运行时间则包含全部完成运行。DEADLOCK的最终Cmax留空，不赋惩罚值，不用native Cmax代替。

| 算法 | 完成/应完成 | B32可行次数 | 可行率 | DEADLOCK | 无及时候选 | 可行Cmax均值/s | 可行Cmax中位数/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| WAG+VNS | 60/60 | 60 | 100.00% | 0 | 0 | 6671.902844 | 6717.024993 |
| HGA | 60/60 | 59 | 98.33% | 1 | 0 | 6619.936004 | 6642.496748 |

**实际时间及迭代数量。** 下表搜索时间以评价器的result.actual_runtime为准，包含该次初始化和搜索；B32在搜索结束后执行。runtime_s是包含外层搜索调用、最终B32及包装开销的实际总墙钟；这里没有把预算120直接填作运行时间。

| 算法 | 搜索均值/s | 搜索中位数/s | 搜索最大/s | B32均值/s | B32中位数/s | 总时间均值/s | 总时间中位数/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| WAG+VNS | 63.583226 | 58.395433 | 96.796383 | 0.309718 | 0.135548 | 63.916873 | 59.607160 |
| HGA | 120.004296 | 120.002586 | 120.020205 | 0.587558 | 0.134024 | 120.621457 | 120.167180 |

外层baseline_run调用还包含参数构造及finish结果整理。WAG外层调用均值/中位数为63.606993/58.418655s，HGA为120.033737/120.032481s；原值均单独保存在diagnostic的search_call_runtime_s。B32最大耗时分别为3.201703s和7.386657s，均在搜索预算外。

| 算法 | 实际迭代最小值 | 均值 | 中位数 | 最大值 | 达迭代上限次数 | 达时间上限次数 |
|---|---:|---:|---:|---:|---:|---:|
| WAG+VNS | 500 | 500.0 | 500.0 | 500 | 60 | 0 |
| HGA | 18188 | 23102.1 | 22630.5 | 30097 | 0 | 60 |

WAG单位为VNS轮，HGA单位为有效子代。WAG实际累计30000轮，60次均在120秒前完成500轮；HGA实际累计1386126个有效子代，每次18188～30097，均因时间预算结束，未达到200000子代上限。这些完整完成的120秒实验不属于新的200000子代完整预算结果。HGA计数取算法返回的实际完成值，可能包含到时正在处理、随后才完整结束的最后一个子代；最终候选仍以其评价完成时间筛选。

WAG搜索超时均为0；HGA超时均值0.004296s、中位数0.002586s、最大0.020205s。HGA外层调用最大120.046480s，包含上述结果整理。120个送入B32的候选都由timely_best取得，评价完成时间均≤120秒；WAG/HGA所选候选完成时间最大值分别为83.346983/118.475018s。本轮迟到全局最优更新数均为0，迟到更新不回填最终方案的情形已由前述测试验证。搜索过程中独立认证调用数为0，最终B32实际共调用120次。

**同排样、同种子及排样分组配对。** 60对运行中共同B32可行59对，HGA胜/平/负为56/0/3；ΔCmax中位数-12.533988s，相对变化中位数-0.191903%。共同59对的WAG/HGA平均Cmax为6667.931985/6619.936004s，相同样本平均Cmax之比对应HGA改善0.719803%。上面各算法单独可行均值的样本集合不同，不能直接拿它们的比值作为配对改善。仅WAG可行1对，仅HGA可行0对，双方均失败0对。3个运行层面的HGA失利为test_08/20261111（+0.901364s）、test_14/20261112（+7.476818s）、test_16/20261110（+2.626230s）。

下面先在每个排样的共同可行种子集合上，分别取两算法最终Cmax中位数，再计算Δ和相对变化。19个排样有3个共同可行种子；test_19只使用20261110与20261112这两个匹配种子，双方均排除20261111，因此没有用不同种子集合制造质量差异。

| 排样 | 共同可行种子数 | WAG Cmax中位数/s | HGA Cmax中位数/s | ΔCmax/s | 相对变化 |
|---|---:|---:|---:|---:|---:|
| test_01 | 3 | 6373.253818 | 6362.743947 | -10.509871 | -0.164906% |
| test_02 | 3 | 6777.342092 | 6768.418870 | -8.923222 | -0.131663% |
| test_03 | 3 | 6849.851354 | 6639.922919 | -209.928435 | -3.064715% |
| test_04 | 3 | 7035.863538 | 6999.762057 | -36.101481 | -0.513107% |
| test_05 | 3 | 6042.534759 | 6034.616966 | -7.917793 | -0.131034% |
| test_06 | 3 | 7274.034086 | 7270.798082 | -3.236004 | -0.044487% |
| test_07 | 3 | 6364.484331 | 6125.230640 | -239.253690 | -3.759200% |
| test_08 | 3 | 6641.595384 | 6641.154957 | -0.440427 | -0.006631% |
| test_09 | 3 | 7137.933916 | 6873.795395 | -264.138521 | -3.700490% |
| test_10 | 3 | 6126.261043 | 6105.008342 | -21.252701 | -0.346911% |
| test_11 | 3 | 6397.927295 | 6384.052089 | -13.875206 | -0.216870% |
| test_12 | 3 | 6782.701768 | 6766.235388 | -16.466380 | -0.242770% |
| test_13 | 3 | 6659.158399 | 6643.326761 | -15.831639 | -0.237742% |
| test_14 | 3 | 6814.421497 | 6814.354637 | -0.066860 | -0.000981% |
| test_15 | 3 | 6356.986819 | 6347.829524 | -9.157295 | -0.144051% |
| test_16 | 3 | 6367.828974 | 6344.446449 | -23.382524 | -0.367198% |
| test_17 | 3 | 6546.207863 | 6535.304205 | -10.903658 | -0.166564% |
| test_18 | 3 | 7100.507410 | 7083.919262 | -16.588148 | -0.233619% |
| test_19 | 2 | 6901.097053 | 6888.069644 | -13.027409 | -0.188773% |
| test_20 | 3 | 6850.517445 | 6839.234536 | -11.282908 | -0.164702% |

20个排样均可形成上述配对，HGA排样层面胜/平/负为20/0/0，ΔCmax中位数-13.451307s，相对变化中位数-0.202822%，即改善中位数0.202822%。这是一组20排样、每排样3种子（其中一例质量配对仅2种子）的描述性结果；没有把60次运行当作60个独立排样，也没有做额外显著性宣称。排样中位数均更短，同时仍存在上述3个种子失利和1次B32失败。

**原样保留的HGA最终失败。** 唯一DEADLOCK为test_19/20261111，它是已完成时间预算并完成一次最终B32的正常记录，恢复时应跳过。

| 排样 | 种子 | 实际有效子代 | native Cmax/s | 最终B32 | 最终B32 Cmax | 搜索时间/s | B32时间/s |
|---|---:|---:|---:|---|---|---:|---:|
| test_19 | 20261111 | 20229 | 6880.465980 | DEADLOCK | — | 120.001673 | 1.941398 |

该候选在90.921793s完成原生评价；一次B32返回的已有摘要为wait-for cycles=()，R1被R0的B2～B8相关操作阻塞。只记录原返回，不做额外重放或独立认证。本轮test_07/20261112为FEASIBLE、Cmax 6125.230640s，原完整预算同一键的DEADLOCK/native 6120.415394s仍原样保留；两个预算选出的方案不同，不能将本轮可行解释为旧失败已经被修复。完整预算与本轮的共同可行集合也不同，跨预算的均值或改善幅度不能直接作为同样本配对结果相减。

**完整性核对与解释。** 原WAG CSV的180条、原HGA CSV的101条按原字节前缀完整保留，各追加60条；现WAG共240条，HGA共161条。90+90条完整预算正式记录及其原汇总逐项一致，默认baseline-summary未混入等时间记录。两算法源码、共同评价、scheduler/model/scope、三份相关测试、validation/test数据和α CSV共12个文件已逐字核对不变。HGA每次完整原生评价次数=实际子代+40，VND唯一调用+命中=实际子代+20，VND内部原生候选评价及轮数/候选截断全部为0。60条新WAG均完成500轮，native与最终B32 Cmax/状态与原正式同键记录一致，这些仍是本轮重新计时执行的120秒预算实验。

两种算法的等时间TEST恢复和原完整预算TEST恢复共四次检查，均为completed=60、remaining=0，直接返回，没有新建工作进程、预热或重复B32，包括DEADLOCK记录。本轮仅修改既有实验入口、本报告和两份既有优化结果CSV；未新增文件、脚本、CSV、评价器、认证器、配置哈希、manifest或管理框架。

120秒是相同的最大允许搜索墙钟。WAG先达到500轮上限，实际搜索中位数58.395433s；HGA用满时间预算，中位数120.002586s。在这个预先确定的双重上限设置下，HGA的共同可行Cmax整体更短，但实际搜索更久、最终可行率为98.33%，WAG为100%。本轮预热及3进程环境下的墙钟是新的真实观测，不把它与历史完整预算跨轮耗时差解释为本轮又做了算法提速，也不把20排样的观察外推为普遍优势。

**本轮对照组工作结束并冻结。** Adapted WAG+VNS固定PARTIAL_LNS_V1及既定500轮/原参数，Adapted HGA固定ORIENTED_ROUTE_VND_NATIVE_SELECT_V1、ROUTE_CYCLE、EXPLICIT及μ=20/λ=10/α=20/完整M1～M6/200000有效子代上限。保留完整预算与120秒预算两套独立统计及所有失败记录。没有待补跑实验；不继续投入对照算法主体优化，不为消除DEADLOCK修改B32、物理约束或种子。下一阶段研究重点转向调度反馈驱动的冲突感知联合构造与瓶颈重构。

## 14. 2026-11-17：4R-SAEAX 实现、一次 TRAIN 预实验与完整 VALIDATION

**结论：最终B32可行30/30，完整耗时中位22.589484秒；对WAG/HGA的共同可行Cmax配对中位变化为+0.444664%/+1.202004%，分别6胜16负、0胜30负。质量未达标，目前不适合作为SCI主算法，未启动TEST。**

### 14.1 版本、共同模型与修改范围

DRL没有独立.git，外层HEAD为0935ad8a723057084be190e31c6be4733ce3cb62，整个DRL在外层为未跟踪目录；无法确认任务中的f15242d14d75a302895159ee8dbe8419b0743cc4是此目录HEAD。以现有本地文件为准，没有回退或清理外层已有改动。已阅读HGA/WAG/common、lns/gnn_constructor、原实验入口和两份指定报告。HGA使用显式方向、整路线替换交叉、完整M1–M6路线VND及原生种群选择；WAG使用几何初始化、MOVE/SWAP/LNS和500轮停止。

继续使用ScientificConfig、FORMAL_SCOPE_V2、现有WHOLE/Y_SPLIT/X_SPLIT域、资格、加工及运动模型和B32。model/scheduler文件时间为11月4日/5日，TRAIN/VALIDATION数据为11月10日，既有报告明确保留共同模型；没有发现近期科学规则变化的证据，未开展历史大回放。原生评价按同轨固定优先级协调，跨轨冲突拒绝；B32具有不同的分派与恢复机制，两者不等价。

经用户逐项确认必要后，只新增以下三个文件：

- src/mrta_search/saeax.py：SAEAXConfig、enhanced_adjacency、adjacency_fragments、assemble_fragments、adjacency_crossover、mutate_child、validate_individual、select_population、final_shortlist、run_saeax。
- scripts/run_saeax.py：预实验/验证执行、已有CSV恢复、正式对照筛选和配对汇总。
- results/saeax_validation.csv：3个pilot与30个validation行，642491字节；包含完整有向方案、进度点、实际计数/计时及每个最终B32候选的状态。

修改现有tests/test_hga_baseline.py、本报告和README。HGA/WAG主体、common、lns、GNN、科学规则、调度器、数据和历史实验均未改动。WAG/HGA优化CSV仍为240行/105107字节、161行/128721字节。没有新增模型、hash协议、manifest、独立认证器、管理框架、VND、ALNS或方向优化阶段。

### 14.2 算子及与HGA的区别

个体保存逐父模式索引、四条实际子块路线与同步0/1方向。复用_HGAWorkspace的合法模式、端点、加工时间、旅行矩阵和资格掩码，包括X子块固定机器人。先从双亲选择模式，再按0.15概率变异一个合法模式，之后才能提取当前有效子块。不同拆分几何即使打印ID相同，也不会混用数值节点。Y的两个子块分别保留归属与转移能力。

增强邻接表包含双亲有向后继与反向前驱（翻转两端方向），以及来源机器人、原生质量、路线周期和局部MOVE。共享边优先，部分独有边随机断开，质量接近的双亲交错贡献；按方向和端点一致性连接成最长6块片段，禁止环、相邻兄弟及没有共同合法机器人的片段。从空四路线联合枚举机器人、插入位置和正向/反向有向片段，用现有旅行矩阵计价。峰值含未插入片段在继承机器人的预留加工量，首点合法性也在动作中判断。随后仅允许轻量单子块relocate或携带方向的swap。没有全路线方向DP或搜索后方向优化。初始化只复用未搜索的几何种子及既有首任务合法性修复，其余初始个体由新算子构造。

HGA复制父代A，再以B的一条整路线替换并修复。SAEAX从双亲邻接重新组装全部路线，不替换完整路线，不调用workspace.repair或HGA交叉修复；测试禁止这些调用并验证同一子代保留双亲各自独有的边。

完整子代先检查覆盖、资格/X归属、方向对齐及兄弟分离，再用已有_priority_arrays固定方向评价。普通tuple键包含模式和有向路线，已见方案在评分前跳过；不建立新哈希系统。种群可行优先、原生Cmax优先，兼顾路线边多样性，并显式保留全局精英。结束时从原生排序前12个不同完整方案中选出最佳及两个尽量不同的邻接方案，最多3次调用原reference_schedule_formal，取实际B32可行且Cmax最小者。所有调用计入总时间；全失败时DEADLOCK且Cmax为空，失败方案的WAIT明确标为部分时间线。

参考 Gao 等（2026）[Multi-objective collaborative optimization for multi-robot systems with high coupling characteristics](https://www.sciencedirect.com/science/article/abs/pii/S2210650226001070)，DOI 10.1016/j.swevo.2026.102387。本次仅取得出版方摘要与章节概述，未取得全文，因此是按本项目要求设计的四机器人适配方法，不声称严格复现该文算子或TSP交替环EAX。

### 14.3 停止、测试与运行口径

固定采用12个体、每代4个新合法且已评价子代、最多400代、至少80代、连续120代无超过1e-8秒的全局最优原生Cmax改善则停止。每代至多48次构造尝试；重复不重评，无效有限重试，凑不齐4个新子代则如实记录重试耗尽和未完成代。实验未发生重试耗尽。没有90/120秒截止。三个TRAIN实例350–400代仍有改善且完整耗时21–37秒，因此保留400/120起始参数，之后不按VALIDATION修改。

预实验前修复了独有边严格排序导致单亲占满邻接的问题。首次入口因Numba线程池上限设置过晚而未完成任何个体，随后受限进程无法创建已获批准的CSV；修正导入前配置并通过获准执行后，才开始下面三条真实pilot轨迹，没有丢弃或重跑已完成搜索。预实验后仅补充失败方案的统计与停止边界测试，搜索规则及参数未变。

136项相关回归通过（215.52秒）；补充边界后12项SAEAX专项再次通过（11.92秒）。覆盖父子完整性、全部模式、独立Y归属、X固定资格、方向同步、重复/遗漏拒绝、双亲邻接混合、精英、停滞/代数/重试计数及原生/B32分离。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py tests/test_wag_baseline.py tests/test_baseline_common.py tests/test_canonical_candidate.py tests/test_scheduler_equivalence.py -q --disable-warnings
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_saeax.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_saeax.py validation --workers 3
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_saeax.py summary
```

全部在DRL目录使用指定解释器。单次runtime包含预计算、初始化、首次编译、全部搜索和所有最终B32调用；与原baseline_job一致，数据读取及进程启动在单次计时之外。TRAIN单进程，VALIDATION三进程，每进程Numba活动线程1；没有扣除长耗时或编译。验证批次墙钟245.151475秒，30任务墙钟之和711.873055秒，两者不同。对照直接读取正式500轮/200000有效子代结果，排除等时间、短测、NATIVE_ONLY和INCOMPLETE，没有新跑对照。跨轮主机负载和编译状态不是严格统一的计时消融。

### 14.4 唯一一次TRAIN预实验

按父焊缝数/layout_id排序，取40个TRAIN索引0、20、39，种子均为20261110。每例只运行一次完整轨迹。

| 排样 | 父数 | 最终B32 | 最终Cmax/s | 原生最优/s | 完整耗时/s | 代数/子代 | B32调用 |
|---|---:|---|---:|---:|---:|---:|---:|
| train_15 | 56 | FEASIBLE | 4546.654207 | 4478.987604 | 36.643134 | 400/1600 | 3 |
| train_34 | 90 | FEASIBLE | 5730.042884 | 5730.042884 | 21.320295 | 400/1600 | 3 |
| train_28 | 132 | FEASIBLE | 6043.935396 | 6043.935396 | 31.392276 | 400/1600 | 3 |

| 排样 | 代数 | 原生最优/s | 累计时间/s | 完整有效子代 | 邻接多样性 |
|---|---:|---:|---:|---:|---:|
| train_15 | 100 | 4603.365717 | 25.913958 | 400 | 0.854820 |
| train_15 | 200 | 4602.663799 | 28.933831 | 800 | 0.778580 |
| train_15 | 400 | 4478.987604 | 35.351922 | 1600 | 0.703985 |
| train_34 | 100 | 5774.422548 | 6.118812 | 400 | 0.584163 |
| train_34 | 200 | 5753.122311 | 10.914658 | 800 | 0.795296 |
| train_34 | 400 | 5730.042884 | 20.898404 | 1600 | 0.752829 |
| train_28 | 100 | 6069.571520 | 8.202964 | 400 | 0.864614 |
| train_28 | 200 | 6064.074969 | 15.532201 | 800 | 0.847232 |
| train_28 | 400 | 6043.935396 | 30.486953 | 1600 | 0.785742 |

进度点来自同一轨迹，不为100/200代额外调用B32。train_15初始12个体均原生不可行，不能计算虚假初始Cmax降幅；搜索原生最优改善15次，但原生第一名B32失败，由第二候选取得可行结果。最终加工峰值4386.353→4313.943秒，MOVE533.873→443.799秒。train_34原生Cmax6471.752→5730.043秒、峰值5698.825→5636.700、MOVE528.641→467.109；train_28相应6417.704→6043.935、6035.162→5944.526、1248.213→582.015。

共4800个不同有效子代、4836次原生评价、9次B32、45次原生最优改善，只有1次重复尝试；1772/4800子代原生可行，7/9最终候选B32可行。中/大例交叉15.449/22.208秒，原生评价3.926/6.447秒，主要成本是邻接/片段构造。能够生成新方案并改善路线与分配，运行时间没有明显异常，故保留参数进入完整验证。

### 14.5 完整VALIDATION逐例结果

时间均为秒。Pmax为完整最终方案最大单机加工负载。S=连续120代停滞，G=400代上限。每条均有3次B32和12次初始化原生评价；有效子代=4×代数，原生评价=有效子代+12。

| 排样 | 种子 | B32 | Cmax | 完整耗时 | 代数 | Pmax | MOVE | WAIT | 停止 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| validation_01 | 20261110 | FEASIBLE | 5595.077043 | 51.401185 | 400 | 5484.256546 | 396.620102 | 128.696964 | G |
| validation_01 | 20261111 | FEASIBLE | 5592.010427 | 50.271030 | 400 | 5484.256546 | 419.718073 | 0.000000 | G |
| validation_01 | 20261112 | FEASIBLE | 5625.422885 | 50.287842 | 400 | 5504.629630 | 429.833434 | 0.000000 | G |
| validation_02 | 20261110 | FEASIBLE | 5713.408190 | 25.990340 | 366 | 5646.547852 | 406.827239 | 0.000000 | S |
| validation_02 | 20261111 | FEASIBLE | 7284.334444 | 8.212991 | 122 | 7244.444444 | 344.748632 | 0.000000 | S |
| validation_02 | 20261112 | FEASIBLE | 6109.554081 | 34.167838 | 400 | 6037.037037 | 425.611894 | 5.862501 | G |
| validation_03 | 20261110 | FEASIBLE | 5053.545677 | 8.887410 | 146 | 4948.285866 | 441.600893 | 0.000000 | S |
| validation_03 | 20261111 | FEASIBLE | 5121.744105 | 15.971155 | 247 | 4948.285866 | 462.967826 | 63.492178 | S |
| validation_03 | 20261112 | FEASIBLE | 5031.508540 | 22.194211 | 400 | 4948.285866 | 422.063474 | 51.174480 | G |
| validation_04 | 20261110 | FEASIBLE | 4953.221327 | 23.536069 | 400 | 4871.823180 | 439.631551 | 0.000000 | G |
| validation_04 | 20261111 | FEASIBLE | 4892.820430 | 24.012662 | 400 | 4807.218056 | 448.031601 | 0.000000 | G |
| validation_04 | 20261112 | FEASIBLE | 4947.878323 | 22.984757 | 400 | 4874.104720 | 407.710734 | 0.000000 | G |
| validation_05 | 20261110 | FEASIBLE | 5538.873693 | 17.087635 | 249 | 5462.412689 | 444.355483 | 11.496349 | S |
| validation_05 | 20261111 | FEASIBLE | 5540.835191 | 21.147516 | 321 | 5462.412689 | 419.317447 | 0.000000 | S |
| validation_05 | 20261112 | FEASIBLE | 5544.559769 | 21.837619 | 400 | 5462.412689 | 393.470901 | 25.104623 | G |
| validation_06 | 20261110 | FEASIBLE | 5419.569511 | 26.626553 | 400 | 5332.019954 | 395.807076 | 12.514664 | G |
| validation_06 | 20261111 | FEASIBLE | 5411.432936 | 24.116485 | 400 | 5332.019954 | 382.922967 | 13.524670 | G |
| validation_06 | 20261112 | FEASIBLE | 5429.198904 | 16.398646 | 244 | 5332.019954 | 432.311094 | 0.000000 | S |
| validation_07 | 20261110 | FEASIBLE | 5769.507035 | 26.984716 | 225 | 5659.387584 | 571.043842 | 15.162372 | S |
| validation_07 | 20261111 | FEASIBLE | 5822.445508 | 19.095287 | 157 | 5659.387584 | 591.909819 | 122.458461 | S |
| validation_07 | 20261112 | FEASIBLE | 5783.133923 | 25.209613 | 301 | 5659.387584 | 580.452241 | 13.788877 | S |
| validation_08 | 20261110 | FEASIBLE | 5478.197763 | 29.884519 | 400 | 5384.622431 | 529.793823 | 0.000000 | G |
| validation_08 | 20261111 | FEASIBLE | 5443.990469 | 29.852345 | 400 | 5380.810725 | 518.543428 | 0.000000 | G |
| validation_08 | 20261112 | FEASIBLE | 5465.752901 | 28.377885 | 400 | 5337.339983 | 542.032963 | 0.000000 | G |
| validation_09 | 20261110 | FEASIBLE | 5535.343972 | 13.779618 | 164 | 5461.805287 | 331.403721 | 195.489758 | S |
| validation_09 | 20261111 | FEASIBLE | 5517.028647 | 16.705683 | 283 | 5422.700366 | 328.752941 | 0.000000 | S |
| validation_09 | 20261112 | FEASIBLE | 5515.694544 | 17.471383 | 303 | 5422.700366 | 322.699607 | 184.614890 | S |
| validation_10 | 20261110 | FEASIBLE | 6312.551596 | 15.549724 | 291 | 6291.029279 | 405.393151 | 0.000000 | S |
| validation_10 | 20261111 | FEASIBLE | 6005.630092 | 11.081527 | 287 | 5926.226183 | 375.277042 | 0.000000 | S |
| validation_10 | 20261112 | FEASIBLE | 5553.071619 | 12.748813 | 400 | 5508.333333 | 352.152759 | 20.000000 | G |

### 14.6 对照比较及完整计数

| 方法 | B32可行 | 可行Cmax中位/s | 可行Cmax均值/s | 全部耗时中位/s | 全部耗时均值/s | 可行Pmax均值/s | 可行MOVE均值/s | 可行WAIT均值/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4R-SAEAX | 30/30 | 5537.108832 | 5566.911452 | 22.589484 | 23.729102 | 5476.540141 | 432.100192 | 28.779360 |
| WAG+VNS | 22/30 | 5466.123862 | 5389.672876 | 103.591607 | 114.541215 | 5296.378099 | 387.491124 | 47.394398 |
| HGA | 30/30 | 5445.501277 | 5374.413953 | 1095.432327 | 1993.324817 | 5304.615579 | 420.856519 | 4.838371 |

WAG可行集不同，不以单独方法的Cmax均值比值代替配对。时间统计包含全部30次，包括WAG的8次DEADLOCK及HGA的历史长耗时。

| 对照 | 共同可行 | SAEAX胜/平/负 | Cmax差中位/s | 相对变化中位 | 相对变化均值 |
|---|---:|---:|---:|---:|---:|
| WAG+VNS | 22 | 6/0/16 | 24.066835 | 0.444664% | 1.895471% |
| HGA | 30 | 0/0/30 | 65.800485 | 1.202004% | 3.527099% |

相对变化=100×(SAEAX B32 Cmax−对照B32 Cmax)/对照B32 Cmax，负数更优，平局容差1e-6秒。WAG失败的8键不算Cmax胜利。新算法时间中位约为WAG的21.806%，但质量条件未达到。

验证共9706代、38824个不同有效子代、39184次原生评价（含360次初始化）与90次B32。9次重复构造在评价前跳过；19518/38824个子代原生可行，全局原生最优改善405次。15次停滞停止，15次达到400代，最少122代、代数中位383，没有重试耗尽。90个最终候选77可行、13死锁，死锁Cmax为空；3次运行原生第一名B32失败后由另一个候选取得可行结果。运行层面0次DEADLOCK、0次ERROR。

只读核对33个(stage,layout_id,seed)键唯一、参数相同、原生计数=初始化+有效子代、每条3次B32、最终Cmax为三个实际可行B32值的最小值，以及总时间包含搜索和B32。没有为核对重新求解或调度。

### 14.7 明确失败原因与研究决定

1. **加工分配停滞。** 11/30次最终Pmax单独就超过同键HGA的完整Cmax，即使消除MOVE/WAIT也无法击败HGA。validation_02/20261111初始低峰值方案原生不可行，进入可行区只改善1次，在122代停滞；四机加工7097.473/3726.925/3677.700/7244.444秒，最终Cmax7284.334秒，HGA为5571.938秒。WAIT为0，不能将该失败归因于等待。
2. **分配与协调可行性仍耦合。** 该例25至122代的最佳Pmax一直为7244.444秒；validation_10/20261110最终Pmax6291.029秒也超过HGA完整5418.993秒。只读资格/加工量分析发现，两例瓶颈块仍有其他合法接收机器人，单块转移可将加工峰值下界降至7097.473/6054.323秒；没有评价这些动作的B32，故不称其为可行改进。多块片段保留父代归属、独立relocate仅轻量触发，可能难以脱离可行但失衡的分配；这是依据轨迹和结构作出的解释，没有额外消融证明因果。
3. **路线与协调质量不足。** 相同30键下SAEAX/HGA的MOVE均值432.100/420.857秒，WAIT均值28.779/4.838秒；其余19次Pmax未超过HGA Cmax的运行仍全部落后。单纯减少加工峰值不够，有向片段插入也未取得HGA完整路线搜索的质量。没有为掩盖负结果增加VND等模块。
4. **原生与B32差异存在。** 3次原生第一名B32死锁，最多3候选选优避免了运行级失败，但没有把原生搜索质量自动变为最终优势；未改B32、未重复调度失败方案。
5. **没有大量重复空转，也不能笼统归因于代数不足。** 已生成38824个不同合法子代，只有9次重复，15例达到400代。主要成本为交叉构造（验证中位16.141秒），原生评分中位2.449秒，最终B32中位2.433秒；原生计时包含初始化，不能将这些重叠时段简单相加。没有定位到需要扩大/缩小代数修复的实质代码错误。

| 主算法目标 | 实际 | 结果 |
|---|---|---|
| 可行率不低于两对照 | SAEAX30/30；WAG22/30，HGA30/30 | 达到 |
| 对两者配对Cmax中位均为负 | +0.444664% / +1.202004% | 未达到 |
| 胜数大于负数 | 6<16；0<30 | 未达到 |
| 完整时间中位不超过或接近WAG | 22.589484<103.591607秒 | 达到 |

**到此结束本轮研究：保留实现、一次预实验和全部30次验证，未启动TEST；不重复调参、不重训GNN、不拼接VNS/ALNS/VND。真正剩余问题是邻接重组对负载分配与协调可行性的耦合处理，以及原生优质候选向B32最终质量的转化。当前方法速度快、可行率高，但尚不适合作为SCI主算法。**


## 15. 2026-11-17：BR-SAEAX 多父焊缝瓶颈联合重插入——TRAIN未通过

### 15.1 实际工作区、已有机制与修改范围

只读核实远程main为[d278b373868c38b2c2b56f6cb275824b8b9a7ef7](https://github.com/luckyfishanddog/DRL/commit/d278b373868c38b2c2b56f6cb275824b8b9a7ef7)。实际本地DRL没有独立.git，属于外层MRTA工作区中的未跟踪目录；外层HEAD为0935ad8a723057084be190e31c6be4733ce3cb62、origin指向MRTA。没有把外层HEAD当作DRL版本，也没有拉取覆盖、reset、clean或清理已有未提交工作。

先检查了saeax.py、run_saeax.py、lns.py、HGA/WAG原生内核、两份现有实验报告，以及已有run_joint_oriented_repair_pilot.py、run_bottleneck_neighborhood_pilot.py、run_gnn_joint_repair_pilot.py和MODE_NEW_ALGORITHM_TECHNICAL_NOTE.md。旧路径已具备父级整体移除、Regret-2联合插入及方向状态，旧MODE_NEW使用RANDOM/TRAVEL/LOAD轮换、4/8父、4096次插入配额和逐完整候选B32；后续GNN父级500次与子焊缝反馈也已实测但没有达到对照质量。旧pilot脚本所指的单独历史报告/CSV在当前目录中并不齐全，本轮没有重建或重跑这些文件。可用技术说明和汇总报告足以确认：联合repair是已有原语，不是本轮重新发明的LNS框架。

仅修改现有src/mrta_search/saeax.py、scripts/run_saeax.py、tests/test_hga_baseline.py、本报告及README。实验追加到原results/saeax_validation.csv，来源为BR_SAEAX_BOTTLENECK_REINSERT_V1；原4R_SAEAX_ADJACENCY_V1的33条、642491字节前缀逐字节保留，现在共36条。没有新增文件，没有修改lns.py、HGA/WAG主体、旧模型、对照结果、ScientificConfig、FORMAL_SCOPE_V2、资格、拆分或B32。

### 15.2 唯一新增算子及固定预算

保留原双亲模式继承/模式变异和有向邻接片段组装。每代第一个有效子代槽位改用瓶颈联合重插入，其余三个继续原轻量变异。初始化仍走完全相同的原路径；三个实例与原pilot的初始化评价数、尝试数、初始原生Cmax、Pmax及MOVE全部一致。

对尚未评分的交叉子代计算与Individual.processing/costs相同的精确加工量P和路线周期P+MOVE，不调用调度器。按P/Pmax与路线周期/最大周期之和排序瓶颈机器人，优先选其加工贡献大的可转移父；预留另一条路线上的一个可转移父，优先选潜在接收机器人上还可转向第三机器人的父，再从重载路线补足。本轮每次实际均选择4个父；不足三个不同可转移父或没有其他来源路线则回退。带种子的±10%贡献扰动用于集合多样性，不改变加工时间或资格。

通过workspace.solution/encode做必要适配，调用原_remove_parent_set移除这些父的全部子焊缝，再只调用一次原repair_partial_state_oriented，固定：
- insertion_limit=4；
- max_insertion_evaluations=256；
- coordination_proxy=True（实际为P+MOVE构造代理，不是真实WAIT或Cmax）；
- direction_versions=2；
- allow_pattern_changes=False，保持既有WHOLE/Y_SPLIT/X_SPLIT模式与切点。

WHOLE/Y子段遵守原资格，Y两个子段可以分别归属，X子段保持固定左右归属。适配前后验证父子唯一完整覆盖、模式、方向对齐、兄弟不相邻合并及原首点约束。方向在既有插入动作中选择，只使用返回的一个完整方案，不额外评价两种方向状态。无法构造合法变化则回到原轻量变异。

新完整子代仍在普通tuple去重后只调用一次原_priority_arrays。没有在子代生成、父选择或任何插入动作中调用B32；未调用依赖ScheduleResult的select_bottleneck_parents/build_crrr_candidate。保留原种群选择、全局精英、最终三候选及最多3次B32规则，没有独立第二阶段、VND、ALNS、GNN或方向后处理。

停止规则原样保留：12个体、每代4个新有效子代、最多400代、至少80代、连续120代没有超过1e-8秒的全局原生Cmax改善则停止；初始化重试12次/个体、每代最多48次构造尝试及重复处理不变。没有墙钟截止。运行前将TRAIN的“明显不合理耗时增长”具体化为任一配对不超过旧SAEAX耗时3倍且三例中位不超过103.591607秒；该条件只用于运行后的研究判断，不终止搜索。

### 15.3 三个TRAIN完整运行结果

在DRL目录使用D:\pybullet_test\.venv\Scripts\python.exe，单进程、Numba活动线程1，种子均为20261110。原SAEAX仅读取历史pilot，不重新求解。Cmax均为最终实际B32可行值，负变化更优。

| 排样 | 父数 | SAEAX Cmax/s | BR-SAEAX Cmax/s | 配对变化 | SAEAX Pmax/s | BR Pmax/s | Pmax差/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 56 | 4546.654207 | 4432.852321 | -2.502981% | 4313.942962 | 4400.132740 | +86.189779 |
| train_34 | 90 | 5730.042884 | 5707.054804 | -0.401185% | 5636.700414 | 5636.700414 | 0.000000 |
| train_28 | 132 | 6043.935396 | 6100.693704 | +0.939095% | 5944.526346 | 5977.462397 | +32.936051 |

train_34的Pmax差为-9.09e-13秒，是浮点求和误差，按1e-8容差计为持平，不能记成负载改善。

| 排样 | B32状态 | BR MOVE/s | BR WAIT/s | 原SAEAX完整耗时/s | BR完整耗时/s | 代数 | 有效子代 | 停止原因 | 停滞计数 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| train_15 | FEASIBLE | 354.108640 | 98.501825 | 36.643134 | 26.898155 | 364 | 1456 | STAGNATION | 120 |
| train_34 | FEASIBLE | 410.950333 | 0.000000 | 21.320295 | 15.127201 | 313 | 1252 | STAGNATION | 120 |
| train_28 | FEASIBLE | 647.297063 | 39.234323 | 31.392276 | 29.268052 | 400 | 1600 | MAX_GENERATIONS | 9 |

完整时间从工作区初始化前到所有最终B32返回后，包含预计算、首次编译、种群初始化、交叉、联合重插、原生评价和最终B32，不扣除失败。train_15初始化含首次编译15.832579秒。BR三例耗时中位26.898155秒、均值23.764469秒，任务时间之和71.293408秒，单进程整批71.344054秒。历史SAEAX三例耗时中位31.392276秒；两例按原停滞规则提前停止，编译/系统状态也非统一计时消融，因此这些时间只报告真实完整运行记录，不声称算子本身带来等轨迹加速。

### 15.4 算子执行、评价及正确性核查

“成功”指相对当前未评分交叉子代产生合法且有变化的完整方案，不等于改善最终目标。Pmax下降同样是与重插前的交叉子代相比，不是与原SAEAX最终结果相比。

| 排样 | 重插尝试 | 合法成功/新有效子代 | 回退 | 插入评价次数 | 重插耗时/s | 转移子块数 | 多源转移次数 | 三机链式转移次数 | 重插后Pmax下降次数 | 重插子代原生可行 | 重插子代改善全局最优 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 364 | 364 | 0 | 93146 | 3.771076 | 326 | 58 | 13 | 54 | 186 | 4 |
| train_34 | 313 | 307 | 6 | 79192 | 3.854672 | 190 | 52 | 16 | 25 | 92 | 1 |
| train_28 | 400 | 400 | 0 | 102396 | 6.078535 | 692 | 261 | 3 | 62 | 20 | 0 |
| 合计 | 1077 | 1071 | 6 | 274734 | 13.704283 | 1208 | 371 | 32 | 141 | 298 | 5 |

每次尝试均恰好选择4父，单次插入评价不超过256；6次回退均为既有修复器返回NO_LEGAL_REPAIR_INSERTION。多源指至少两个原机器人发生子块转出；三机链式指存在R_a→R_b及R_b→R_c且a≠c，交换不计为三机链。这证明实际产生了联合归属变化，不是仅换路线顺序或只做单任务高到低转移。

实际4308个有效子代、36次初始化评价、4344次原生调用、9次最终B32；每条均满足native_calls=valid_children+12、valid_children=4×generations。最终9候选中7个可行、2个DEADLOCK，运行层面3/3可行。train_28原生第一名6094.525623秒的B32死锁，最终选择第三候选6100.693704秒；失败记录及空Cmax完整保留。重插计时包含在crossover_s中，native_s也含初始化阶段，不能把重叠计时简单相加。

215项HGA/SAEAX、现有LNS、公共基线、canonical及调度等价回归通过（139.33秒）；随后补充第三机器人接力选择、已完成pilot不重跑不写入、TRAIN失败不触碰VALIDATION三个用例，SAEAX/BR专项20项再次通过。覆盖固定模式/X归属、父级完整移除、Y资格、方向、实际修复复用且禁止调度/时间线选择器、失败回退、每代一个有效算子槽位、一次原生评分及DEADLOCK不参加Cmax配对。

另对三个保存的最终方案只做数值/结构复核，没有新调用原生或B32：重新映射当前合法模式目录，父/子块数量分别56/58、90/91、132/133，覆盖、资格、固定模式、canonical不改写均通过；从实际路线重新求P与MOVE，逐机器人加工及CSV的Pmax/MOVE全部匹配。原生/B32计数、最终Cmax取实际可行候选最小值、完整时间包含搜索+B32也均一致。

### 15.5 通过条件、负结果及后续决定

| TRAIN要求 | 实际 | 结论 |
|---|---|---|
| 三例B32可行 | 3/3 | 通过 |
| 至少两例Cmax优于原SAEAX | 2胜0平1负，配对变化中位-0.401185% | 通过 |
| 至少两例Pmax下降 | 0降、1平、2升 | **未通过** |
| 没有明显不合理耗时增长 | 配对耗时比0.734057/0.709521/0.932333，中位26.898155秒 | 通过 |

已检查固定模式适配、父集合多来源、合法迁移资格、完整覆盖、初始化一致性、真实评价计数和终止规则，没有发现需要通过重跑修正的实现错误。中间141次局部加工峰值下降没有转化为三个终解中任意一个低于原SAEAX的Pmax。train_15的最终MOVE/WAIT分别比原结果低89.690807/105.735268秒，但加工峰值反升86.189779秒；train_34峰值持平、MOVE降低56.158341秒，收益主要表现为路线质量变化。train_28的加工峰值、MOVE、WAIT全部高于原SAEAX，400代中该算子400个子代仅20个原生可行，且没有贡献一次全局原生最优改善。该例的原生最优本身也比原SAEAX最终6043.935396秒差，不能只归因于最后的B32差异。

当前证据支持：算子能实现多机器人联合转移，并能改善部分实例Cmax，但在256次有界P+MOVE修复与现有种群选择下，尚不能稳定解除最终加工负载失衡。协调可行性限制及代理排序与完整原生Cmax不一致是与轨迹相符的解释，没有额外消融证明各自因果贡献，也不能将局部成功次数当作终局质量优势。

**TRAIN未通过，停止本轮方向；没有根据结果改参数、增加模块或重跑pilot。BR-SAEAX VALIDATION为0/30、TEST为0/60，均未启动。** 正式WAG/HGA文件没有这三个TRAIN键，因此BR与WAG/HGA的共同可行配对数均为0，不能报告它们的BR质量胜负。原SAEAX的30次VALIDATION及其对WAG/HGA的+0.444664%/+1.202004%历史比较仍见第14节，不能借用为BR结果。没有证据宣称BR-SAEAX已超过两个正式对照或适合作为主算法。

复现与只读汇总（已有三条BR结果时pilot只跳过已完成键，不重跑）：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_saeax.py summary --method br
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py pilot --method br --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -k 'saeax or br_' -q --disable-warnings
```

实验入口保留原方法默认值与历史来源；BR验证入口仅在TRAIN全部条件满足时运行，TEST仅在完整VALIDATION全部成功标准满足时运行。这是任务指定的分阶段研究规则，没有新增算法管理器、哈希、manifest、独立认证器或发布门禁。


## 16. 2026-11-17：LA-4R-EAX-NSGA-II——全文核对、实现及TRAIN筛查负结果

### 16.1 来源、工作区与论文适配边界

本轮开始时远程main为[c0ed6dc1eb6868edaff807450347c7a251267f0e](https://github.com/luckyfishanddog/DRL/commit/c0ed6dc1eb6868edaff807450347c7a251267f0e)。本地saeax.py、run_saeax.py、lns.py已与该提交逐字核对一致。DRL仍没有独立.git，外层MRTA工作区HEAD为0935ad8a723057084be190e31c6be4733ce3cb62，DRL显示为未跟踪目录；没有切换、回退、覆盖或清理工作区。

用户随后提供原文[C:/Users/Renjl/Desktop/main.pdf](C:/Users/Renjl/Desktop/main.pdf)，已核对PDF元数据DOI为[10.1016/j.swevo.2026.102387](https://doi.org/10.1016/j.swevo.2026.102387)，共17页。实际阅读第4节（PDF第7–9页）、第6节（第12–15页），并在内存中渲染检查第10页图9，未复制PDF或创建提取/渲染文件。正式实验在全文规则核对及测试通过后才启动。

第4.4节和图9的EAX是增强邻接表交叉—变异：阈值比较发生在双亲前驱/后继之间，一侧短一侧长时保留短侧，两侧都长时保留两侧；之后以pc向其他节点的邻接表插入节点。后代按剩余邻居少者优先，无邻居则最近邻。它不是AB-cycle EAX。原文未明确两侧均短、阈值相等及开放路线端点的全部分支，本实现分别保留两条短边、将相等视为短边、保留端点唯一可用邻居。没有给任意非父代近邻补边。

本轮没有把原双机器人点焊模型直接当作本项目：用selected+四条routes+四条directions表达父模式、机器人归属、线段顺序与方向，不建立WAP式重复归属编码；保留所有选定模式的子焊缝，不按双亲路线长度随机截断任务。线段距离使用既有方向端点MOVE距离，方向嵌入合法插入，而非独立姿态阶段。当前目标为用户指定的native Cmax、Pmax、MOVE，与论文原目标有明确区别。第6节的2000代TSP、120秒对比及147点工业算例是论文验证设置，本项目没有据此改变自己的停止规则或追加TSP研究。因此这是有来源说明的四机器人适配，不宣称原论文完整模型、RRT或姿态规划的严格复现。

### 16.2 新算子与NSGA-II的实际实现

仅新增用户逐项批准的src/mrta_search/eax_nsga2.py和scripts/run_eax_nsga2.py。测试复用tests/test_hga_baseline.py，结果复用results/saeax_validation.csv，报告和README只修改已有文件。SAEAX/BR-SAEAX、HGA、WAG、LNS主体、GNN及旧模型、正式数据、ScientificConfig、FORMAL_SCOPE_V2、拆分目录/0.50m规则、资格与B32均未修改。

双亲所有路线的有向前驱/后继参与同一邻接表，表中不保存机器人归属。活动节点按当前selected映射，替代几何即使有相同打印child ID也不会混入。原文第4.4节没有给出本平台可直接采用的数值阈值，本轮预先固定为双亲当前活动有向邻接距离的75%分位数；统计规则固定，阈值随双亲自然变化，没有扫描或按结果调整。邻接表变异概率固定0.10，插入随机其他父节点，并补充方向反向互补映射；原合法模式继承与0.15模式变异保持原语义。

联合构造从空四路线开始，每次优先选择当前节点邻接表中剩余有向邻居数少的活动节点，无邻居则按既有有向MOVE选最近未用节点。对该节点枚举合法机器人、全部插入位置和两个方向，按插入后的Pmax、最大路线周期、邻接支持及新增MOVE顺序选择；剩余精确平局使用带种子的随机机器人次序和稳定位置次序。初始节点为带种子随机选择。若该节点当前无合法位置，才尝试后续候选；全部无法安放则本次构造失败并按12次上限重试。每个节点只提交一次，兄弟不相邻合并、X固定左右归属、Y子段独立资格及原首点约束在构造中检查，没有片段组装、继承机器人偏好、LNS修复、VND或方向后处理。

原未搜索的REGION方案只提供初始化邻接模板；其方向初值为0，不先做方向优化，不评价旧种子。新种群的20个成员全部由新联合算子产生并各做一次原生评价。它没有读取SAEAX、WAG/HGA终解或GNN权重作为初始种群。

环境选择使用三个原始目标(native Cmax,Pmax,MOVE)的非支配排序及归一化拥挤距离，常量轴贡献0。原生可行个体约束支配所有原生不可行个体；原生只给出可行性而无定量违约量，故不可行个体互不作有限目标竞争、拥挤距离为0，不以Pmax/MOVE冒充可行Cmax。父代使用rank/crowding二元锦标赛，父子合并后按标准环境选择保留20个体，完整方案用已有普通tuple去重。

最终在已有seen候选中按可行性、native Cmax、Pmax、MOVE取字典序最小者；该点必然是全部已评价目标向量中的非支配点，无须另设档案或重复评价。只有这一个方案调用一次原B32，DEADLOCK不再取其他候选回退，也不把原生值当作最终Cmax。

### 16.3 固定停止规则、测试与运行口径

正式入口固定20个体、每代10个不同且已完成原生评价的有效子代、最多500代、至少100代、连续120代没有超过1e-8秒的全局最优native Cmax改善则停止。每个子代槽位最多12次构造，不使用跨槽位的120次总池替代该限制；凑不齐一代不会增加代数，初始化或槽位重试耗尽保留实际计数并标为未完成搜索。本轮三例均完整按停滞规则结束，没有重试耗尽。没有90秒、120秒或103.59秒墙钟截止，500代也不等于500轮WAG或200000个HGA子代。

正式实验前修复一处Numba不支持float(bool)的编译兼容问题，随后233项HGA/SAEAX、现有LNS、公共基线、canonical及调度等价回归全部通过（140.55秒）。试验后为核查负载异常，补充单节点插入的加工/空走增量与独立完整路线重算比较，16项新算法专项再次通过（13.13秒），未修改算法或参数、未重跑实验。

测试覆盖论文一短一长/两长/两短/端点阈值分支、跨机器人双亲邻接、实际邻接插入变异、活动模式筛选、WHOLE/Y/X完整覆盖、Y独立合法归属、X固定归属、方向同步、新算子禁止委托SAEAX片段/LNS/调度、NSGA-II已知Pareto层及拥挤距离、不可行约束支配、去重、自然停止和逐槽12次重试、仅最终一次B32及其DEADLOCK保留、TRAIN条件失败不访问下一数据集。

在DRL目录用D:\pybullet_test\.venv\Scripts\python.exe实际执行，TRAIN单进程、Numba活动线程1，三个种子均为20261110。运行时间从工作区建立前开始，包含必要预计算、种群初始化、首次编译、全部交叉/变异、NSGA-II选择、原生评价和最终一次B32。模块导入、文件读取和进程启动不在单次计时内，与既有实验入口口径一致。首次编译没有扣除，三例跨轮历史耗时不是严格等轨迹加速消融。

### 16.4 三个TRAIN逐例结果

Cmax只来自最终B32；正的配对变化表示变差，DEADLOCK不计算变化。所有时间单位为秒。

| 排样 | 父数 | B32状态 | 原SAEAX Cmax | LA-4R-EAX Cmax | 配对变化 | Pmax | MOVE | WAIT |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| train_15 | 56 | FEASIBLE | 4546.654207 | 5348.591980 | +17.637976% | 4482.058360 | 546.802763 | 796.155562 |
| train_34 | 90 | FEASIBLE | 5730.042884 | 8744.267157 | +52.603869% | 8585.080976 | 561.057650 | 0.000000 |
| train_28 | 132 | DEADLOCK | 6043.935396 | — | — | 5924.948469 | 745.402889 | 0.000000（仅部分时间线） |

train_28的native Cmax为8514.291591秒，但最终B32返回DEADLOCK，CSV的cmax_s为空。该原生值不进入Cmax均值、中位数或配对胜负；部分WAIT为0也不能解释为完整无等待调度。

| 排样 | 完整耗时 | 搜索耗时 | B32耗时 | 代数 | 有效子代 | 初始化/原生总调用 | 最终B32调用 | 停止原因 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train_15 | 30.164902 | 29.767853 | 0.396986 | 427 | 4270 | 20/4290 | 1 | STAGNATION，120代 |
| train_34 | 4.756109 | 4.703218 | 0.052821 | 133 | 1330 | 20/1350 | 1 | STAGNATION，120代 |
| train_28 | 22.516469 | 21.707476 | 0.808920 | 330 | 3300 | 20/3320 | 1 | STAGNATION，120代 |

三例B32可行率为2/3。仅两例可行Cmax的中位数和均值均为7046.429568秒，必须与2/3可行率同时阅读。对SAEAX共同可行2对为0胜/0平/2负，相对变化中位数+35.120923%、Cmax差中位+1908.081023秒；第三例失败不作为Cmax胜负。

全部三例完整时间中位22.516469秒、均值19.145827秒，搜索中位21.707476秒、均值18.726182秒，B32中位0.396986秒、均值0.419576秒。三任务时间之和57.437480秒，单进程整批57.491087秒。通过了事后的103.59秒时间目标，但质量与可行性未通过，不能因运行快而宣称主算法成功。

### 16.5 实际遗传行为与失败定位

下表统计每个已评价的完整新方案，包含各例20个初始化个体。重新归属只比较该父代中存在的同一活动数值节点，替代模式的新节点不制造虚假的“跨机转移”。

| 排样 | 结构拒绝 | 重复 | 子代原生可行/有效子代 | 邻接变异新增条目 | 对第一/第二亲本重新归属节点数 | 少邻居选择/最近邻回退 | 全局native严格改善 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 14 | 0 | 32/4270 | 48220 | 127677/127348 | 145085/96759 | 9 |
| train_34 | 5 | 0 | 19/1330 | 24230 | 60429/60805 | 75263/46420 | 1 |
| train_28 | 3 | 0 | 22/3300 | 86982 | 219240/219332 | 272271/163177 | 3 |

合计8900个不同有效子代、60次初始化、8960次原生评价和3次最终B32。全局native严格改善13次，结构无效22次、重复0次。父代邻接阈值累计排除514131个条目，变异实际新增159432条；实际继承父代有向边372644条、新建边399712条。对双亲分别发生407346/407485次可比较节点重新归属，说明新构造没有固定继承机器人，也不是把SAEAX片段拼接换名。三例阈值均值分别2.322441/1.359200/1.440034米，是固定分位数规则的观测结果而非选参。最终可行Pareto种群前沿大小分别为5/5/4。

但子代原生可行率仅73/8900=0.820225%；三个初始种群的原生可行数为0/2/0。大量覆盖及资格合法的构造仍无法形成原生协调可行调度，选择只能从很小的可行池中保留目标折中。此事实说明局部负载/距离构造代理不足以控制完整协调可行性；不是重复子代空转，也没有证据以继续扩大代数代替问题分析。

- train_15：Pmax比原SAEAX高168.115398秒、MOVE高103.003315秒，实际WAIT为796.155562秒（原204.237093秒），最终Cmax高801.937772秒。方向/任务完整并不等于协调质量良好。
- train_34：Pmax高2948.380562秒，四机加工为2350.776542/8492.069159/8585.080976/2907.578364秒，WAIT为0，失衡本身已足以解释很大的Cmax劣势。不能把该例归咎于等待或B32耗时。
- train_28：Pmax比原SAEAX低19.577878秒，真实路线周期最大值6065.642419秒，但native Cmax为8514.291591秒，最终B32又死锁。这同时说明加工峰值改善不足以保证协调质量，以及原生最优向单次最终B32可行性的转化仍有失败风险。未增加候选回退、独立认证或额外B32。

以上是实际观测及与轨迹一致的解释，没有额外消融来区分初始化、邻接筛选、负载决策和可行性选择各自的因果贡献。不能因为子代确实重新分配了任务，就断言其重新分配有效。

另对三个保存终解只做结构/数值核查，没有新增原生或B32调用：父/子节点数量为56/57、90/91、132/132；覆盖、资格、固定X归属、模式与canonical不改写均通过。由实际路线重新计算的逐机加工、Pmax和MOVE与CSV逐项一致；每条native_calls=20+10×generations、B32调用=1、停滞计数=120、完整时间包含搜索+B32，均已核对。单节点负载/空走增量测试与完整重算也通过，没有发现可据以修正后重跑的代码错误。

### 16.6 筛查决定、结果保存与命令

| TRAIN筛查要求 | 实际 | 结论 |
|---|---|---|
| 三例全部B32可行 | 2/3 | 未通过 |
| 至少两例B32 Cmax优于同键SAEAX | 共同可行0胜2负，另1例DEADLOCK | 未通过 |
| 完整时间中位不超过103.59秒 | 22.516469秒 | 通过 |

**本轮停止：未通过TRAIN，不执行VALIDATION或TEST，不根据负结果修改参数、重跑pilot、恢复旧机器人偏好或叠加局部搜索/GNN。** WAG/HGA只有既有正式VALIDATION/TEST对照，这三个TRAIN键无匹配正式记录，因此本轮对两者的Cmax配对数均为0。没有新算法的VALIDATION比较证据，更不能借用旧SAEAX的30次结果宣称本轮超过对照。

新3行以source=LA_4R_EAX_NSGA2_V1追加到原CSV。原36行、710231字节前缀逐字节保留，现在共39行、766874字节；旧SAEAX、BR、GNN、WAG和HGA记录及失败均未覆盖。

在D:\pybullet_test\MRTA_GA\DRL运行：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_eax_nsga2.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_eax_nsga2.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -k 'eax and not saeax' -q --disable-warnings
```

已有三行时pilot会跳过完成键，summary只读，不重跑算法。入口保留通过TRAIN后才允许validation、通过完整VALIDATION后才允许test的任务指定研究规则；当前不会触发后续求解。没有新增hash、manifest、配置指纹、实验注册框架、独立认证器或额外调度器。

## 17. 2026-11-18：BC-ECS 实现、TRAIN通过及完整VALIDATION负结果

### 17.1 结论与实际修改范围

BC-ECS已完成三个TRAIN及30次VALIDATION，不是研究计划。TRAIN为3/3 B32可行、对同键SAEAX 2胜1平，完整时间中位30.489505秒，通过指定筛查。固定同一搜索实现和参数后，VALIDATION为30/30 B32可行、完整时间中位31.649188秒，但对WAG共同可行22对仅7胜15负、Cmax变化中位+0.096921%；对HGA 30对全部落后、中位+1.082869%。质量条件失败，本轮停止，不启动TEST、不反复调参、不叠加其他算法。

首先检查了实际本地工作区。外层MRTA仓库HEAD为0935ad8a723057084be190e31c6be4733ce3cb62，DRL在外层作为未跟踪目录；不能将该HEAD当作DRL版本。本轮开始读取的DRL远程main为718a4a24b3cd4510b869e7f13f97ae7a915c18fa，本地saeax.py、lns.py和run_saeax.py文本与该版本一致。没有reset、clean、pull覆盖或丢弃本地工作。

新增用户逐项批准的src/mrta_search/bc_ecs.py和scripts/run_bc_ecs.py。实验完成后收到运行脚本的单独批准，补充薄入口，直接复用同一命令解析和实验函数，不复制运行逻辑、不修改搜索实现或参数；原算法文件的直接入口继续可用。必要测试追加到tests/test_hga_baseline.py，33行结果追加到既有results/saeax_validation.csv，本报告和README只更新已有文件。SAEAX/BR-SAEAX、EAX-NSGA-II、GNN、HGA、WAG、LNS、物理模型、资格、拆分规则、ScientificConfig、FORMAL_SCOPE_V2、B32、数据集及旧模型均未修改。WAG/HGA正式对照直接读取，没有重新运行。

### 17.2 核心算子、下界及B32使用

每次BC-ECS从头实际调用未修改的4R-SAEAX，完整运行12个体、每代4有效子代、最多400代、至少80代、120代无严格改善停止的原规则；不读取任何历史最优方案作为初解。初始化返回的方案、方向和已执行的B32结果直接在内存复用。33次实际生成的初解与历史同键SAEAX方案及Cmax一致，这只是完成后的复核，不是复用历史答案；初始化搜索、其全部原生评价、其3次实际B32以及后续所有操作均计入新算法完整时间。

瓶颈根机器人取加工负载最大者和P+有向MOVE路线周期最大者的并集。可转移子块按实际已观测WAIT涉及父焊缝、加工贡献和固定种子平局顺序排列。交错生成单转移、两机器人交换、接收机器人继续向第三机器人转移、三机器人闭环。一次链不重复移除同一父焊缝；无资格替代的X子块不制造转移边。

复用_remove_parent_set整体移除链上父焊缝的全部子块；模式和几何保持初解不变。强制链上指定子块的合法接收机器人，其他Y子块独立枚举自己的原资格。已有数值插入内核在两种固定构造次序中联合选择位置和方向；未插入任务的精确加工量仅用于构造排序，不作安全下界。完成链骨架后，原joint_block_candidates在同一个尚未评价的联合候选中枚举根子块全部合法插入位置、两个方向和其支持的邻居方向组合。它不是另加方向优化或精修阶段。完整覆盖、X固定归属、Y独立资格、首点约束及同父相邻不会被canonical合并均有检查。

先用完整目标归属的精确Pmax剪枝，再对已经完整确定路线和方向的候选用max_r(P_r+MOVE_r)下界剪枝。没有把可能因后续插入而降低的部分路线MOVE当下界。只在下界严格超过目标并越过保守浮点保护量时剪枝；目标为当前最佳native与本次可行初解B32 Cmax的较小者。完整候选经普通tuple去重后只调用一次原生协调评分，按超过1e-8秒的真实native改善接受。原生非有限值只记录为原生不通过，不标记成B32必然死锁；WAIT来自实际初始B32或后续已执行原生时间线，不是静态几何冲突代理。

搜索中不调用B32。始终保留最佳native方案及本次已B32可行的初解。结束时只有存在native改善才对所选一个完整方案执行一次B32；实际B32改善且可行才返回，否则直接返回原初解及同一个已保存的调度结果，不再调度初解。全部失败最终候选及其完整方案也保存在诊断中。

### 17.3 固定预算、停止口径和测试

BC-ECS参数只有max_candidates=6000、stagnation_candidates=1500，按实际完成的新原生评价计数；剪枝、重复、构造失败不充数，不设墙钟截止。另一实际结束状态为CHAIN_PLANS_EXHAUSTED：在当前接受方案下，有限的已定义链、父级归属、两种构造次序和根插入组合已经遍历，没有剩余未评价候选。不能以重复评分凑满6000。该状态不是对全部合法四机器人方案空间的穷尽证明，其他链子块的非贪心插入组合没有被完整枚举；它是本实现的搜索范围限制，必须与停滞结束区别报告。

本轮实际没有一次达到6000次上限。TRAIN为1次STAGNATION、2次CHAIN_PLANS_EXHAUSTED；VALIDATION为11次STAGNATION（均1500次未改善实际评价）、19次CHAIN_PLANS_EXHAUSTED，新候选评价数范围0至4999、中位486.5。没有用时间到期或事后回填评价数冒充完成预算。BC-ECS不是遗传算法，CSV的generations留空，既有valid_children通用列保存BC新原生候选数；SAEAX自身的代数、有效子代和调用数在initial_saeax诊断中另存。

在DRL目录、D:\pybullet_test\.venv\Scripts\python.exe下执行；TRAIN单进程，VALIDATION三个工作进程，每个Numba活动线程1，主进程按完成键追加CSV。单次完整时间从SAEAX初始化前开始，到最终返回后结束，包括工作区、必要编译、原SAEAX全部搜索和B32、新链构造/剪枝/原生评价、最终一次B32。模块导入、数据文件读取和进程启动沿用旧入口口径，不在单次计时内。没有扣除首次编译。两轮历史时间比较没有做等轨迹加速消融。

正式实验前，HGA/SAEAX、LNS、公共基线、canonical及调度等价测试248项通过（148.50秒）。补充命令入口测试后，BC专项17项通过（15.31秒）。覆盖精确加工增量与完整路线重算、浮点下界保护、剪枝先于原生/B32、完整有向下界不超过有限native、两/三机链、Y/X与父级完整性、真实WAIT、一次实际初解与至多一次最终B32、原生非有限值、失败/更差最终方案返回原调度对象、实际评价停滞计数及已完成键跳过。TRAIN结束后未修改搜索算子或参数，仅补独立入口和测试。

实验后只读复核了13个排样的89个已保存初始/返回/最终候选方案：覆盖、资格、模式、方向、canonical不改写及按实际路线重算的逐机加工、Pmax、MOVE均通过，没有增加原生或B32调用。33次调用计数、严格接受、阶段计时、缓存回退及去重关系也核对通过。真实入口再次执行pilot/validation均报告remaining 0，未运行算法或追加记录，CSV字节保持不变。

### 17.4 TRAIN三个实例

种子均为20261110，全部B32 FEASIBLE。时间单位为秒；MOVE和WAIT为各机器人之和，不是与Pmax相加即可得到Cmax的分解。配对变化为100×(BC/SAEAX−1)，正值更差。

| 排样 | 同键SAEAX Cmax | BC-ECS B32 Cmax | 配对变化 | Pmax | MOVE | WAIT | 完整时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 4546.654207 | 4493.303351 | -1.173409% | 4303.790443 | 438.483954 | 102.850874 | 30.489505 |
| train_34 | 5730.042884 | 5715.971434 | -0.245573% | 5637.427843 | 451.833029 | 0.000000 | 37.916611 |
| train_28 | 6043.935396 | 6043.935396 | 0.000000% | 5944.526346 | 582.014908 | 0.000000 | 24.287275 |

| 排样 | 本次SAEAX代数/子代 | 初始化+BC=原生总调用 | 真实B32总调用 | 新逐出链时间 | 含初始化的搜索时间 | 全部B32时间 | BC停止 |
|---|---:|---:|---:|---:|---:|---:|---|
| train_15 | 400/1600 | 1612+3628=5240 | 4 | 5.746496 | 29.660574 | 0.809659 | STAGNATION（1500次） |
| train_34 | 400/1600 | 1612+10=1622 | 4 | 25.874147 | 37.579184 | 0.314494 | CHAIN_PLANS_EXHAUSTED |
| train_28 | 400/1600 | 1612+0=1612 | 3 | 5.768432 | 23.758026 | 0.501273 | CHAIN_PLANS_EXHAUSTED |

train_15实际接受4次单转移，train_34接受7次两机器人交换，train_28无改善。三例完成3638次新原生评价，初始化4836次，合计8474次；真实B32为9次初始化加2次最终，共11次。三个初解均实际运行400代、1600有效子代。train_28的0次新增评价表示全部已构造候选被安全下界等筛选掉，5.768432秒链枚举时间仍完整计入，没有把0次误写成1500次停滞。

TRAIN共同可行3对为2胜1平0负，配对变化中位−0.245573%、差值中位−14.071450秒；完整时间中位30.489505秒、均值30.897797秒。三任务时间之和92.693392秒。3/3可行、至少2例实际Cmax改善、时间中位不超过103.59秒三个条件均通过，因此按事先规则继续VALIDATION。此三个TRAIN键无WAG/HGA正式匹配记录，不构造伪配对。

### 17.5 完整VALIDATION配对与判定

10个原有排样×种子20261110/20261111/20261112均完成，B32 FEASIBLE为30/30。BC-ECS可行Cmax中位5522.927328秒、均值5552.572559秒。配对只使用相同layout_id和seed且双方最终B32可行的记录。

| 正式对照 | 共同可行配对数 | BC胜/平/负 | Cmax变化中位 | Cmax差中位（秒） | 对照可行率 | 对照完整时间中位 |
|---|---:|---:|---:|---:|---:|---:|
| 原SAEAX | 30 | 19/11/0 | −0.033148% | −1.770296 | 30/30 | 22.589484 |
| WAG+VNS，PARTIAL_LNS_V1，500轮 | 22 | 7/0/15 | +0.096921% | +5.155780 | 22/30 | 103.591607 |
| HGA，ORIENTED_ROUTE_VND_NATIVE_SELECT_V1，200000有效子代、完整VND | 30 | 0/0/30 | +1.082869% | +58.866194 | 30/30 | 1095.432327 |

WAG的8次DEADLOCK只计入其可行率，不作为BC的Cmax胜负；没有用任何native Cmax替代B32值。HGA仅取原正式FULL/COMPLETED记录，WAG/HGA均未重算。各算法可行集合不同，不以独立Cmax中位数之比代替逐例配对变化中位数。

| 指定正式成功条件 | 实际 | 结论 |
|---|---|---|
| 新算法B32可行30/30 | 30/30 | 通过 |
| 对HGA配对中位变化<0且胜数>负数 | +1.082869%，0胜30负 | 未通过 |
| 对WAG共同可行配对中位变化<0且胜数>负数 | +0.096921%，7胜15负 | 未通过 |
| 完整时间中位≤103.59秒 | 31.649188秒 | 通过 |

完整时间均值53.631391秒，范围11.402912至267.931497秒，30任务时间之和1608.941729秒（不是并行整批墙钟时间）。本次初始化时间中位17.804575秒、新逐出链中位11.851887秒；包含初始化搜索的总搜索时间中位30.173696秒，全部B32时间中位2.058751秒。各部分中位数不要求可加。103.59秒仅用于事后性能判断，超过者没有截断。

### 17.6 VALIDATION逐例结果

时间均为秒，全部返回状态为FEASIBLE。SA代数指本次算法内实际执行的SAEAX初解；BC评价为新完成原生评价数，原生总数含该初解。S=1500次无改善评价停止，E=有限链计划耗尽。

| 排样 | seed | 最终B32 Cmax | Pmax | MOVE | WAIT | 完整时间 | SA代数 | BC评价 | 原生总数 | B32总数 | 停止 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| validation_01 | 20261110 | 5595.077043 | 5484.256546 | 396.620102 | 128.696964 | 51.152493 | 400 | 0 | 1612 | 3 | E |
| validation_01 | 20261111 | 5592.010427 | 5484.256546 | 419.718073 | 0.000000 | 49.310221 | 400 | 0 | 1612 | 3 | E |
| validation_01 | 20261112 | 5612.856270 | 5504.629630 | 433.196453 | 0.000000 | 60.915448 | 400 | 14 | 1626 | 4 | E |
| validation_02 | 20261110 | 5641.740813 | 5538.770074 | 466.517477 | 2.043596 | 134.713375 | 366 | 22 | 1498 | 4 | E |
| validation_02 | 20261111 | 7284.334444 | 7244.444444 | 344.748632 | 0.000000 | 15.541402 | 122 | 2963 | 3463 | 4 | S |
| validation_02 | 20261112 | 6034.810461 | 5909.071408 | 623.110377 | 295.784386 | 175.655065 | 400 | 2638 | 4250 | 4 | E |
| validation_03 | 20261110 | 5030.067506 | 4925.397921 | 443.711516 | 11.377265 | 34.107245 | 146 | 855 | 1451 | 4 | E |
| validation_03 | 20261111 | 5074.789587 | 4941.617447 | 562.444896 | 24.632483 | 50.361735 | 247 | 4448 | 5448 | 4 | S |
| validation_03 | 20261112 | 5031.508540 | 4948.285866 | 422.063474 | 51.174480 | 23.834951 | 400 | 118 | 1730 | 3 | E |
| validation_04 | 20261110 | 4951.470102 | 4883.937423 | 427.220182 | 0.000000 | 28.884151 | 400 | 1 | 1613 | 4 | E |
| validation_04 | 20261111 | 4889.544306 | 4819.332298 | 447.888081 | 0.000000 | 31.322105 | 400 | 1 | 1613 | 4 | E |
| validation_04 | 20261112 | 4947.878323 | 4874.104720 | 407.710734 | 0.000000 | 24.904736 | 400 | 0 | 1612 | 3 | E |
| validation_05 | 20261110 | 5528.826010 | 5399.673296 | 520.908929 | 11.496349 | 19.409466 | 249 | 1795 | 2803 | 4 | S |
| validation_05 | 20261111 | 5540.835191 | 5462.412689 | 419.317447 | 0.000000 | 22.120554 | 321 | 0 | 1296 | 3 | E |
| validation_05 | 20261112 | 5506.150036 | 5424.542319 | 462.292256 | 5.774969 | 31.864075 | 400 | 1003 | 2615 | 4 | E |
| validation_06 | 20261110 | 5418.244498 | 5331.292525 | 420.173899 | 2.900998 | 38.273840 | 400 | 2114 | 3726 | 4 | S |
| validation_06 | 20261111 | 5386.895656 | 5303.769571 | 403.054078 | 11.072941 | 86.587877 | 400 | 1816 | 3428 | 4 | E |
| validation_06 | 20261112 | 5418.360904 | 5314.455420 | 503.448201 | 8.958522 | 69.937015 | 244 | 4999 | 5987 | 4 | S |
| validation_07 | 20261110 | 5769.507035 | 5659.387584 | 571.043842 | 15.162372 | 40.152507 | 225 | 3427 | 4339 | 4 | S |
| validation_07 | 20261111 | 5750.959946 | 5692.418454 | 790.026584 | 0.000000 | 26.611651 | 157 | 2985 | 3625 | 4 | S |
| validation_07 | 20261112 | 5781.344557 | 5659.387584 | 578.662875 | 7.953458 | 31.434300 | 301 | 2639 | 3855 | 4 | S |
| validation_08 | 20261110 | 5458.224992 | 5351.042595 | 581.914760 | 0.000000 | 137.074956 | 400 | 52 | 1664 | 4 | E |
| validation_08 | 20261111 | 5433.039733 | 5354.084786 | 541.633706 | 0.000000 | 267.931497 | 400 | 16 | 1628 | 4 | E |
| validation_08 | 20261112 | 5464.630757 | 5338.067412 | 545.816873 | 0.000000 | 54.969194 | 400 | 2188 | 3800 | 4 | S |
| validation_09 | 20261110 | 5531.343030 | 5454.040822 | 336.668970 | 195.489758 | 17.714475 | 164 | 1 | 669 | 4 | E |
| validation_09 | 20261111 | 5517.028647 | 5422.700366 | 328.752941 | 0.000000 | 12.286572 | 283 | 0 | 1144 | 3 | E |
| validation_09 | 20261112 | 5515.694544 | 5422.700366 | 322.699607 | 184.614890 | 21.094922 | 303 | 0 | 1224 | 3 | E |
| validation_10 | 20261110 | 6311.301694 | 6054.322718 | 486.637283 | 1827.022693 | 21.644301 | 291 | 2083 | 3259 | 4 | S |
| validation_10 | 20261111 | 6005.630092 | 5926.226183 | 375.277042 | 0.000000 | 11.402912 | 287 | 1500 | 2660 | 3 | S |
| validation_10 | 20261112 | 5553.071619 | 5508.333333 | 352.152759 | 20.000000 | 17.728685 | 400 | 0 | 1612 | 3 | E |

完整initial_saeax轨迹、阶段计时、候选类型、实际接受转移、最终尝试及返回方案均在CSV diagnostic中保留，未再建结果文件。不能把该表的BC评价数当作初始化SAEAX有效子代数。

### 17.7 失败原因与本轮边界

30次运行累计接受123次真实native改善：80次两机器人交换、20次三机器人链、23次单转移；累计评价9339个单转移、14377个两机器人交换、13962个三机器人链候选，3062次涉及已观测WAIT父焊缝。跨机器人联合调整确实发生，不是仅对同机路线做重排。但实际Pmax相对SAEAX只有13降、13平、4升；10个返回方案的Pmax单独就已超过同键HGA最终Cmax，因此这些归属即使没有MOVE/WAIT也不能胜过HGA。native严格改善也允许加工负载略升而总协调时间下降，不能把该目标描述成Pmax单调下降。

新搜索完成37678次原生评价，其中17870次有限、19808次非有限（有限率47.43%）；加初始SAEAX的39184次，共76862次。新原生不通过不能转写为B32 DEADLOCK数。实际最终一次B32在21次运行发生：19次可行且改善，2次失败；另9次没有native改善，未额外B32。真实B32合计111次=初解90次+最终21次。两次失败均返回本次已可行的原初解，不重复调度，因而30/30是包含明确初解回退机制的返回可行率，不代表21个最终链方案全部可行。

下界处理50168个加工量超界组合、34539908个完整路线超界状态。完整插入状态访问34613034次=路线下界剪枝34539908+重复35448+实际原生37678；这些访问不是不同已评分方案，更不计入候选预算。新原生评分只占74.699147秒，而链搜索累计1027.848448秒，约7.27%用于评分，其余包含构造、枚举、下界、缓存和Python开销。剪枝数量很大不等于已证实同比例加速。

validation_08/20261111只有16次新原生评价，但链枚举246.225825秒、完整267.931497秒；所有时间均保留。表明大量未评分组合的生成本身有成本，并非把原生预算扩大就能解决。没有将这一长尾作为自动缩短预算的理由。

本轮能够小幅改进SAEAX，但没有实验证明它已解决多机器人负载失衡：对SAEAX变化中位只有−0.033148%，仍无法超过正式HGA和WAG，且存在枚举时间长尾。可行初解回退解决返回可行性保护，不能补足搜索质量。没有额外消融来区分固定模式限制、非根子块贪心插入、链范围和协调接受各自的因果贡献；这些属于实现边界，不将推测写成已证明的唯一原因。依据预先条件，停止本方向的本轮实验，不启动TEST，也不以扩展模块掩盖负结果。

### 17.8 保存和复核命令

本轮以source=BC_ECS_V1追加3行pilot和30行validation。原39行、766874字节前缀逐字节保留，当前共72行、2332972字节；旧SAEAX、BR、EAX-NSGA-II、GNN、WAG及HGA历史结果和失败记录保留。没有新增hash、manifest、独立认证器、评分器或实验管理框架。

在D:\pybullet_test\MRTA_GA\DRL运行：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_bc_ecs.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_bc_ecs.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_bc_ecs.py validation --workers 3
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -k bc -q --disable-warnings
```

已有33行时两个运行命令均跳过已完成键，summary只读。入口在TRAIN失败时不会访问VALIDATION，没有TEST入口；本轮VALIDATION失败，不追加任何后续求解。

运行脚本获批后，实际用scripts/run_bc_ecs.py复核summary、pilot和validation：新旧入口summary输出完全一致，3个pilot和30个validation均remaining 0，没有重新运行实验；CSV仍为2332972字节，逐字节未变。相同环境再次运行BC专项测试17项通过（31.89秒）。本次只补充入口，没有修改算法参数或实验结果。

## 18. 2026-11-19：RC-IG 父焊缝模式联合重构实验

### 18.1 工作区和修改范围

开始时外层MRTA仓库HEAD为0935ad8a723057084be190e31c6be4733ce3cb62；DRL没有独立.git，在父仓库中为未跟踪目录。父仓库已有大量暂存/未暂存修改及删除，没有reset、clean、切换版本或覆盖旧工作。使用当前本地实现；没有把父仓库HEAD当作DRL版本，也没有用远程内容替换本地文件。

用户明确批准仅新增src/mrta_search/rc_ig.py和scripts/run_rc_ig.py。必要测试追加到tests/test_hga_baseline.py，结果追加到既有results/saeax_validation.csv，实验记录追加到本节。未修改SAEAX、BC-ECS、WAG、HGA、GNN、LNS、模型、几何、资格、合法模式目录、FORMAL_SCOPE_V2、B32或数据集。未增加求解器依赖、hash、manifest、配置指纹、独立认证器、额外调度器或实验管理模块。

三个TRAIN写入后，汇总遇到原公共CSV读取函数的128KiB字段上限。为使新旧入口都能读取完整修复轨迹，只在scripts/run_saeax.py的read_rows增加一行csv.field_size_limit(2**31-1)，并补充内存CSV测试；没有修改该文件中的SAEAX算法调用、参数、结果列、配对规则或搜索实现，也没有重跑这三个已完成记录。

### 18.2 实际算法及固定设置

每个键从头调用未修改的SAEAX，保留其返回的完整方案、方向、native值和已有B32调度对象。历史CSV不参与初始化或候选构造。搜索以当前实际方案的Pmax机器人、P+有向MOVE最大机器人及P+MOVE+已观测WAIT最大机器人为瓶颈集合；父级加工贡献和实际WAIT作为加权抽样依据，并保留所有父焊缝被抽到的正概率。每轮选择round(0.06×父数)，限制3至8条且不超过父数；三个TRAIN分别移除3/5/8个父焊缝的全部子段。

修复每轮重新比较每个待修父焊缝在原目录中的全部合法模式，几何节点和资格直接复用_HGAWorkspace。对每个子段遍历合法机器人、全部路线位置、正反两个方向及紧邻前后焊缝方向组合；X_SPLIT仍按原空间左右子段固定到同轨R0/R1或R2/R3，Y子段按各自资格分配。数值增量只重算受影响的相邻弧，复用_route_statistics、_put_oriented、_heads_legal及validate_individual。测试把数值选择与原joint_block_candidates/oriented_route_state动作和完整行程重算逐项比较；没有复制SAEAX、WAG或HGA算法主体。

每个(robot,position)保留该位置的最佳方向组合，单子段保留两个最佳不同位置/机器人结构；双子段尝试两种插入顺序，保留首段两个选择并各自尝试次段两个选择。完成父焊缝后，跨合法模式保留两个不同的模式/归属/路线结构，方向不同但结构相同不充作第二选择。regret-2优先提交最大“次优−最优”代理差的父焊缝，唯一可插方案优先；同主差时比较总周期差。该宽度2构造不是全部双子段组合的穷举，尤其可能遗漏首段即时代理较差而联合结果较好的组合，属于明确实现边界。

构造代理为(max_r(P_r+MOVE_r), sum_r(P_r+MOVE_r))，只用于排序，不作部分路线下界剪枝。完整候选用已有普通tuple去重，最多一次_priority_arrays原生协调评价。原生非有限值拒绝；对当前native改善接受，变差按exp(−Δ/T)接受，相等接受；始终单独保存全局最佳完整native方案。温度T0=0.01×本次SAEAX返回的B32 Cmax，每次外层尝试后乘0.99，固定500次外层迭代，不设90秒或其他墙钟截止，不以停滞、重复或评价数减少预算。

搜索中不调用B32。只有全局最佳native严格优于初始native时，在搜索结束后对这一候选调用一次原B32；该B32可行且优于本次初始B32才返回，否则复用初始已可行方案和原调度对象。未再次调度初解，也不尝试第二个最终候选。native与B32值分列保存。TRAIN通过条件仅为三个返回B32可行、至少两个实际B32改善、完整搜索正常结束，没有TRAIN时间条件。VALIDATION的时间条件从现有同键WAG正式记录计算，用于事后判定；三个算法的迭代单位不等价。

### 18.3 验证与运行环境

在D:\pybullet_test\MRTA_GA\DRL用D:\pybullet_test\.venv\Scripts\python.exe执行，TRAIN为单进程，Numba活动线程1。新算法专项13项通过（12.85秒）；随后HGA/SAEAX/BC、LNS、公共基线、canonical及调度等价相关回归共264项通过（242.49秒）。检查覆盖WHOLE/Y/X合法模式、父级全覆盖、资格及X固定归属、插入与相邻方向同步、增量与原动作集合/完整路线一致、500次实际尝试及失败/重复不冒充评价、无90秒停止、退火接受较差但拒绝不可行、返回全局最佳、原生与B32分离、仅一次最终B32及失败/不改善回退、初始化失败和TRAIN条件。

单次计时从本次SAEAX调用前到RC-IG最终返回，包含工作区建立、必要编译、SAEAX全部搜索及其B32、500轮构造和原生评价、必要最终B32。沿用旧实验口径，模块导入、数据读取、进程启动及CSV追加不在单次算法计时内。

首次TRAIN批次完成train_15后，沙箱内追加原CSV被PermissionError拒绝，批次中断，另外两例尚未启动。该次已实际完成500轮、124次新原生评价，初解原生1612次、合计1736次，B32合计4次；完整算法耗时67.041859秒，返回初解Cmax=4546.654207秒。其完整诊断在写入前随进程退出而未保存，不能伪造为一条完整CSV记录。只读检查确认CSV仍为原72行、2332972字节，文件非只读且无Excel进程；获批的非沙箱追加打开检查成功，没有写入字节。因此随后在获批执行环境中以完全相同代码/参数重新运行三个TRAIN，以保存完整诊断。首次未落盘执行单列披露，不能从实际计算成本中隐藏，也不将其视为参数试验或第四个独立TRAIN样本。

公共CSV读取兼容修正后，新算法专项14项再次通过（9.33秒），新增的一项只用内存中200000字符的诊断字段验证既有reader，不创建测试文件。summary随即正常读取全部三行。随后只读复核三个实例保存的初解、返回方案和最终尝试共9个方案：合法模式、完整父子覆盖、四机资格、X固定归属、方向同步、canonical不改写、逐机加工及有向MOVE重算、调用计数和完整计时等式均通过，没有新增原生或B32评价。

### 18.4 TRAIN完整结果与进入VALIDATION的决定

种子均为20261110；Cmax仅来自返回的B32调度，单位秒。配对变化为100×(RC-IG/本次初解−1)。三个初解均实际运行400代、1600有效子代、1612次原生和3次B32；没有读取历史方案。

| 排样 | 父数/每轮破坏数 | 本次SAEAX B32 Cmax | RC-IG返回B32 Cmax | 配对变化 | Pmax | MOVE | WAIT | 完整耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 56/3 | 4546.654207 | 4546.654207 | 0.000000% | 4313.942962 | 443.799448 | 204.237093 | 62.180254 |
| train_34 | 90/5 | 5730.042884 | 5728.337251 | −0.029766% | 5636.700414 | 464.784915 | 0.000000 | 75.245201 |
| train_28 | 132/8 | 6043.935396 | 6027.991785 | −0.263795% | 5938.402938 | 551.459574 | 25.233495 | 114.851456 |

| 排样 | 外层迭代 | 构造失败/重复 | 新原生评价（有限/非有限） | 初解+新搜索=原生总数 | 初解耗时 | 新搜索耗时 | 最终B32耗时/状态 | 返回原因 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| train_15 | 500 | 1/375 | 124（80/44） | 1612+124=1736 | 36.467547 | 25.274128 | 0.435348 / DEADLOCK | 复用原初解 |
| train_34 | 500 | 0/91 | 409（297/112） | 1612+409=2021 | 20.732624 | 54.392456 | 0.113565 / FEASIBLE | B32实际改善 |
| train_28 | 500 | 0/111 | 389（187/202） | 1612+389=2001 | 31.648446 | 82.872003 | 0.324980 / FEASIBLE | B32实际改善 |

三例均以ITERATION_LIMIT结束，search_complete=True；1500次外层尝试=1次构造失败+577次重复+922次真正的新原生评价。新原生有限564次、非有限358次，不把非有限值计作B32死锁。原生总数5758=4836初解+922新搜索，真实B32为12=9次初解+3次最终。完整时间中位75.245201秒、均值84.092304秒、三次之和252.276911秒；单进程整批算法及追加耗时252.314025秒。另有首次未落盘train_15的67.041859秒，已执行算法总成本为319.318770秒、原生总数7494、B32总数16；这不是三个正式样本的运行时间中位数。

搜索中实际评价381个模式变化候选，接受26次含模式变化的转移、累计改变34个父模式；接受当前native严格改善53次、相等101次、较差32次，全局最佳严格改善6次。并非固定BC-ECS模式换名。train_28返回的全局最佳仍全为WHOLE，不能把搜索中的模式变化次数等同于最终采用拆分数量。

train_15最佳native为4487.857025秒（比初始native低58.797182秒），但唯一最终B32 DEADLOCK；返回的4546.654207秒来自初始已可行B32。该失败候选的263.585170秒WAIT仅为部分时间线，不是完整调度WAIT。train_34和train_28末轮当前native分别5749.171361及6058.765483秒，返回的全局最佳分别5728.337251及6027.991785秒，证明没有错误返回退火的最后状态。

三例B32返回可行率3/3，对本次初解为2胜1平0负，中位变化−0.029766%、中位差−1.705633秒。指定TRAIN三个条件全部通过，故按原设置进入10个VALIDATION×3个原种子；未把train_28超过90秒或超过WAG历史中位数当作停止理由。这三个TRAIN键没有WAG/HGA正式同键结果，对两者的TRAIN配对数均为0，不构造不匹配的质量比较。TRAIN通过不代表已经达到最终主算法标准。


### 18.5 完整VALIDATION比较与正式判定

10个现有VALIDATION排样×种子20261110/20261111/20261112全部完成，每次实际500次外层迭代、search_complete=True。返回B32可行率30/30，Cmax中位5513.182253秒、均值5479.859000秒。对照直接读取既有正式CSV，按同layout_id和seed配对，正的Cmax相对变化表示更差。

| 正式对照 | 共同可行配对 | RC-IG胜/平/负 | Cmax相对变化中位 | Cmax差中位（秒） | 对照可行率 | 对照完整时间中位（秒） |
|---|---:|---:|---:|---:|---:|---:|
| 本次运行内SAEAX初解 | 30 | 22/8/0 | −0.194858% | −10.974079 | 30/30 | 21.520654 |
| WAG+VNS，PARTIAL_LNS_V1，500轮 | 22 | 10/0/12 | +0.043209% | +2.415850 | 22/30 | 103.591607 |
| HGA，ORIENTED_ROUTE_VND_NATIVE_SELECT_V1，200000有效子代及完整VND | 30 | 0/0/30 | +0.882012% | +47.538440 | 30/30 | 1095.432327 |

WAG的8次DEADLOCK计入其可行率，不制造Cmax配对胜负。HGA仅取原FULL/COMPLETED、一次最终B32的正式结果。SAEAX配对来自各次新运行自己保存的initial_saeax，不使用历史SAEAX的Cmax或运行时间代替，亦不以两个独立Cmax中位数之比代替配对变化中位数。

| 指定成功条件 | 本轮实际 | 判定 |
|---|---|---|
| B32返回可行30/30 | 30/30 | 通过 |
| 对WAG配对变化中位<0且胜数>负数 | +0.043209%，10胜12负 | 未通过 |
| 对HGA配对变化中位<0且胜数>负数 | +0.882012%，0胜30负 | 未通过 |
| 完整时间中位≤WAG正式中位 | 70.137177≤103.591607秒 | 通过 |

**VALIDATION未通过，不启动TEST，不调参后重跑，不追加新的搜索阶段。** 这是一次完整的固定设置负结果，不是待执行的实验计划。相比历史BC-ECS，对HGA的配对变化中位从+1.082869%缩小到+0.882012%，但两轮实际初解与计时轨迹不同，该差值不是等初解消融或统计显著性证据，更不表示已达到用户的成功条件。

### 18.6 VALIDATION逐例结果

所有行均返回FEASIBLE，均完成500次外层迭代。所有时间单位为秒；“原生总数”含本次SAEAX初始化，“新原生”只计本次RC-IG完成的不同完整候选评价。B32总数包含SAEAX实际3次调度；I=最终B32实际改善，F=最终B32 DEADLOCK后复用初解，N=没有native改善且不再调用B32。“改变父模式”只比较返回方案与本次初解，包含同类模式的合法切点变化。

| 排样 | 种子 | 本次初始B32 Cmax | 返回B32 Cmax | 对初解变化 | 完整耗时 | 新原生 | 原生总数 | B32总数 | 返回 | 改变父模式 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| validation_01 | 20261110 | 5595.077043 | 5595.077043 | +0.000000% | 116.314721 | 379 | 1991 | 3 | N | 0 |
| validation_01 | 20261111 | 5592.010427 | 5592.010427 | +0.000000% | 114.634050 | 369 | 1981 | 4 | F | 0 |
| validation_01 | 20261112 | 5625.422885 | 5558.599477 | -1.187882% | 111.783028 | 390 | 2002 | 4 | I | 3 |
| validation_02 | 20261110 | 5713.408190 | 5603.463262 | -1.924332% | 125.299239 | 422 | 1898 | 4 | I | 5 |
| validation_02 | 20261111 | 7284.334444 | 5642.894551 | -22.533835% | 122.959233 | 434 | 934 | 4 | I | 4 |
| validation_02 | 20261112 | 6109.554081 | 6109.554081 | +0.000000% | 107.779798 | 383 | 1995 | 4 | F | 0 |
| validation_03 | 20261110 | 5053.545677 | 5021.361020 | -0.636873% | 67.992747 | 266 | 862 | 4 | I | 2 |
| validation_03 | 20261111 | 5121.744105 | 5055.819966 | -1.287142% | 69.834757 | 308 | 1308 | 4 | I | 4 |
| validation_03 | 20261112 | 5031.508540 | 5025.601031 | -0.117410% | 70.439597 | 326 | 1938 | 4 | I | 0 |
| validation_04 | 20261110 | 4953.221327 | 4866.728563 | -1.746192% | 51.285001 | 347 | 1959 | 4 | I | 3 |
| validation_04 | 20261111 | 4892.820430 | 4892.820430 | +0.000000% | 65.661515 | 291 | 1903 | 3 | N | 0 |
| validation_04 | 20261112 | 4947.878323 | 4938.120816 | -0.197206% | 47.812498 | 326 | 1938 | 4 | I | 2 |
| validation_05 | 20261110 | 5538.873693 | 5510.934120 | -0.504427% | 64.944092 | 201 | 1209 | 4 | I | 0 |
| validation_05 | 20261111 | 5540.835191 | 5515.430385 | -0.458501% | 62.723593 | 236 | 1532 | 4 | I | 1 |
| validation_05 | 20261112 | 5544.559769 | 5538.047926 | -0.117446% | 65.011423 | 316 | 1928 | 4 | I | 0 |
| validation_06 | 20261110 | 5419.569511 | 5413.721421 | -0.107907% | 62.937265 | 322 | 1934 | 4 | I | 1 |
| validation_06 | 20261111 | 5411.432936 | 5411.432936 | +0.000000% | 67.246061 | 342 | 1954 | 3 | N | 0 |
| validation_06 | 20261112 | 5429.198904 | 5424.239953 | -0.091339% | 55.341983 | 354 | 1342 | 4 | I | 0 |
| validation_07 | 20261110 | 5769.507035 | 5697.252454 | -1.252353% | 127.674663 | 432 | 1344 | 4 | I | 2 |
| validation_07 | 20261111 | 5822.445508 | 5682.347689 | -2.406168% | 124.584953 | 441 | 1081 | 4 | I | 4 |
| validation_07 | 20261112 | 5783.133923 | 5708.087869 | -1.297671% | 106.950313 | 440 | 1656 | 4 | I | 4 |
| validation_08 | 20261110 | 5478.197763 | 5449.558573 | -0.522785% | 81.921671 | 393 | 2005 | 4 | I | 3 |
| validation_08 | 20261111 | 5443.990469 | 5433.510198 | -0.192511% | 92.210818 | 395 | 2007 | 4 | I | 1 |
| validation_08 | 20261112 | 5465.752901 | 5458.118420 | -0.139678% | 91.756382 | 339 | 1951 | 4 | I | 0 |
| validation_09 | 20261110 | 5535.343972 | 5491.566821 | -0.790866% | 58.003074 | 378 | 1046 | 4 | I | 3 |
| validation_09 | 20261111 | 5517.028647 | 5517.028647 | +0.000000% | 58.169419 | 338 | 1482 | 4 | F | 0 |
| validation_09 | 20261112 | 5515.694544 | 5515.694544 | +0.000000% | 77.153590 | 420 | 1644 | 3 | N | 0 |
| validation_10 | 20261110 | 6312.551596 | 6312.551596 | +0.000000% | 64.045496 | 290 | 1466 | 4 | F | 0 |
| validation_10 | 20261111 | 6005.630092 | 5994.162204 | -0.190952% | 61.582450 | 386 | 1546 | 4 | I | 0 |
| validation_10 | 20261112 | 5553.071619 | 5420.033592 | -2.395756% | 106.674402 | 353 | 1965 | 4 | I | 6 |

完整方向向量、初解B32状态/完成时间、三次初始B32候选、最终B32尝试及失败诊断、接受的父级模式/机器人/位置/方向/相邻翻转动作、各阶段计时均在同一CSV的diagnostic保留；未新增结果文件。

### 18.7 完整时间、迭代及实际调用数

VALIDATION完整时间中位70.137177秒、均值83.357594秒、范围47.812498至127.674663秒，比WAG正式中位低32.294537%。30任务时间之和2500.727832秒；3工作进程、各Numba线程1的整批墙钟887.695748秒，二者不同。初始化中位21.534279秒，新搜索中位53.461408秒，全部B32中位3.524083秒；各部分中位数不可直接相加。首次三个工作进程各自的编译都计入各次完整时间。超过90秒或WAG中位的实例全部保留，没有截断、扣除初始化或缩短500轮。

15000次外层尝试=5次构造失败+4378次完整候选重复+10617次真正的新原生评价。新评价范围201至441次、中位353.5次；有限5857次（55.166243%）、非有限4760次、NaN为0。初始SAEAX原生39184次，总原生49801次；逐次总原生范围862至2007、中位1900.5次。不能把15000次外层尝试或634101628次插入位置/方向访问写成原生评价数。

真实B32共116=90次初始SAEAX+26次最终候选；26次最终尝试中22次可行且实际改善、4次DEADLOCK，其余4次没有native改善而不调用最终B32。四次失败分别为validation_01/20261111、validation_02/20261112、validation_09/20261111、validation_10/20261110；全部返回同一已保存的初始调度，未再调度初解。30/30是带有明确回退的返回可行率，不能改写成26个最终候选全部可行。搜索期间没有B32。

对照的迭代单位保持原义：WAG为500轮，HGA为200000有效子代及完整VND，RC-IG为500次外层尝试。既有HGA正式diagnostic逐次记录native_calls=200040，30次合计6001200，包含其初始化；VND内部原生候选评价另列为0。WAG正式CSV没有保存原生调用数，保持缺失，不由500轮反推、不补造调用数，也不为补计数重跑对照。不能宣称三者“迭代数相同”或计算量相同。

### 18.8 主要失败原因、证据及复核命令

本轮确实改变了父级结构：4678个新评分候选含模式变化，接受386次含模式变化的转移、累计改变500个父模式；16/30返回方案相对本次初解改变合计48个父模式。最终2808个父模式为WHOLE 2724、Y_SPLIT 66、X_SPLIT 18，所有切点仍来自原合法目录。搜索接受当前native严格改善752次、相等1113次、较差325次，保存全局最佳严格改善213次。真实退火和模式重选都发生，没有固定模式换名或只重排同机路线。

最直接的质量瓶颈仍是分配与有向路线周期。返回方案Pmax相对本次初解为16降、13平、1升；其中3个方案仅Pmax已超过同键HGA最终Cmax。进一步按保存的完整路线及方向重新计算，30/30的max_r(P_r+MOVE_r)均高于同键HGA最终B32 Cmax，且30个返回native Cmax均与自身该周期相等（浮点误差1e-7内）。返回B32 Cmax与该周期的差中位为0，最大约9.1e-13秒。因而即使消除这些固定方案的全部WAIT，也不能超过HGA；主要剩余差距在机器人归属、拆分和有向路线本身。22个实际B32改善方案仍全部落后于HGA，4次B32回退只解释部分改善损失。

单次B32转化风险仍存在，最优native向B32的转化为22/26；不能以native最优替代B32可行及Cmax。构造主要用部分路线P+MOVE代理、每步保留两个位置结构，父级联合也只做宽度2的两顺序组合。此构造具有明确的有限搜索边界，没有对所有父模式/归属/位置组合做穷举或VND；本轮没有额外消融来证明其中某一项是唯一原因，因此不把推测写成已证明的因果结论。

效率方面，新搜索累计1786.304407秒，其中实际原生评价27.981098秒，仅1.566424%；其余包含662037次合法父模式试放、634101628次位置/方向枚举访问、路线复制、regret重算、去重和Python/数值内核交互开销。枚举访问包含最终未提交的部分方案和首点不通过方向，既不是不同完整候选数，也不是安全剪枝收益证明。运行时间达标，但大量构造开销没有转化为超过正式对照的质量；没有据此追加优化或重新跑一轮。

实验后只读复核33次运行的95个已保存初解/返回/最终尝试方案，合法模式、父子覆盖、资格、X固定归属、方向、canonical不改写、逐机加工与有向MOVE重算均通过。全部500轮计数等式、有限/非有限数、接受转移与模式变化数、原生/最终B32调用数、初解回退、完整计时包含各阶段均通过，没有额外原生或B32调用。CSV现为105行、8391581字节，其中72行历史记录保留，新RC-IG为3行pilot+30行validation；source=RC_IG_V1。含首次未落盘执行，所有实际算法耗时合计2820.046602秒，原生57295次、B32 132次；仅33条完整保存的正式记录为55559次原生和128次B32。

在D:\pybullet_test\MRTA_GA\DRL执行：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_rc_ig.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_rc_ig.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_rc_ig.py validation --workers 3
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -k rcig -q --disable-warnings
```

已有33条结果时，pilot/validation按完成键跳过，summary只读；本轮VALIDATION未达到正式条件，不运行test入口。涉及CSV追加的实际实验使用获批执行环境解决已确认的沙箱写入限制；没有调整文件ACL或清除其他工作。

实际再次执行pilot和validation入口，分别报告completed 3/remaining 0及completed 30/remaining 0，没有重新初始化、搜索、B32或追加结果。旧SAEAX及BC-ECS的summary也正常读取并保留原结论，CSV仍为105行、8391581字节，RC-IG TEST记录为0。SAEAX/BC/LNS/WAG/HGA及模型、几何、scope、B32共9个核心文件的长度与修改时间同开始记录一致，外层Git HEAD保持原值。

## 19. 2026-11-24：CRF-DP 关键路线片段联合动态规划

### 19.1 调用链和本轮实现范围

本轮开始时外层仓库 `D:/pybullet_test/MRTA_GA` HEAD 为 `0935ad8a723057084be190e31c6be4733ce3cb62`，DRL没有独立.git；用户给出的远程审查参照为 `bf39716a854a29da166db8b6b04771f22a5ad44b`，两者不混作同一版本。已有工作区内容均保留。

仅新增用户已批准的 `src/mrta_search/crf_dp.py`、`scripts/run_crf_dp.py`。测试追加到原 `tests/test_hga_baseline.py`；数据只追加到原 `results/saeax_validation.csv`，source=`CRF_DP_V1`。README及本节记录结果。GNN、HGA、WAG、BC-ECS、RC-IG、SAEAX、LNS、B32和正式模型均未修改。

实际调用链：`scripts/run_crf_dp.py → run_crf_dp → run_saeax（本次重新运行）→ _HGAWorkspace / _route_statistics → reconstruct_fragment → _subset_routes / _partition_subsets → _priority_arrays → reference_schedule_formal（至多一次最终尝试）`。初始化B32是原SAEAX的已有行为，完整计入本方法成本；搜索阶段不调用B32。历史同键记录仅用于事后比较，未用于初始化。

### 19.2 联合DP、搜索和精确性范围

取 `argmax_r(P_r+MOVE_r)` 的关键路线，从保留首任务的连续窗口中选择5、6、7、8子焊缝：优先保留各长度的最佳贡献窗口，再按移除的加工及有向空走贡献补足，至多6个。四机器人当前父模式固定。关键机器人使用移除后原槽位；其他机器人按合法子任务的有向插入代价选择至多两个槽位。

每个机器人、每个槽位计算带显式方向的Held–Karp `DP_r(S,i,d)`。起始、内部和闭合转移计入加工、有向MOVE和固定边界；资格及X_SPLIT固定归属直接读取workspace.eligible；左右边界和内部均禁止同父兄弟子段相邻。然后以 `F(r,S)=min_{T⊆S} max(F(r−1,S\T),L_r(T))` 同时划分任务集合给四机器人。完整回溯恢复归属、整段排列、方向和插入位置，再用原完整性、首任务和canonicalize检查核对。

本轮采用**保留四条原路线首任务及其方向**的局部邻域，空路线保持空，因此四机器人子集代价可独立计算，首任务约束不会在划分后引入遗漏的耦合。每个接收机器人仅插入一个连续块，未移除的子任务及方向保持原顺序。精确性只限于所选任务集合、这些槽位及保留首任务的固定模式邻域；不是整个MRTA的全局最优，也不覆盖任意多槽插入。

最多8轮、每轮Best Improvement、最多40次新增完整原生评价；严格native Cmax改善才接受，接受后重新识别关键机器人。完整方案先用普通方案元组去重，再用 `max_r(P_r+MOVE_r)≥当前最佳native Cmax` 安全剪枝。原生非有限值单独计数，不称为B32死锁。无退火、方向后处理、逐条Regret修复或逐出链。

最终仅对严格改善的最佳完整方案尝试一次原B32；若不可行或实际B32 Cmax未改善，则直接返回本次SAEAX已可行的初始方案及原调度对象，不重复B32。完整运行时间包含实际初始化、workspace、DP、原生及最终B32；不按103.59秒或任意墙钟截止。103.59秒仅用于完成后的运行成本判定。

### 19.3 正确性测试与实验

新增测试将小型人工实例的每个机器人全部子集代价逐项与独立穷举比较，再穷举四机器人分配比较min-max最优值；包含资格、固定归属、方向及兄弟子段边界。另验证回溯、固定模式/首任务/未改片段、重新识别瓶颈、去重、下界、40次预算、原生非有限值及唯一最终B32与复用初解。

指定环境 `D:/pybullet_test/.venv/Scripts/python.exe`、DRL工作目录下执行：新增CRF测试 **11 passed**（6.39秒）；整个 `tests/test_hga_baseline.py` **120 passed**（148.62秒）。Numba DP使用 `cache=False`，Python使用 `-B`，pytest禁用cacheprovider，未新建测试/缓存文件。

### 19.4 三个TRAIN逐例B32结果

均为seed=20261110；历史三方法只读取原CSV，不重跑、不作为初解。四列Cmax均为B32 FEASIBLE，单位秒；CRF完整耗时含本次SAEAX初始化和全部实际评价。

| 排样 | 父数 | SAEAX历史Cmax | BC-ECS历史Cmax | RC-IG历史Cmax | CRF-DP Cmax | CRF完整耗时 | 相对同键最强变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train_15 | 56 | 4546.654207 | 4493.303351 | 4546.654207 | 4546.654207 | 26.288337 | +1.187342% |
| train_34 | 90 | 5730.042884 | 5715.971434 | 5728.337251 | 5729.119901 | 30.878340 | +0.230030% |
| train_28 | 132 | 6043.935396 | 6043.935396 | 6027.991785 | 6033.274308 | 19.861475 | +0.087633% |

| 配对参照 | 胜/负/平 | Cmax变化中位数 | 秒差中位数 |
|---|---|---:|---:|
| SAEAX | 2/0/1 | −0.016108% | −0.922983 |
| BC-ECS | 1/2/0 | +0.230030% | +13.148467 |
| RC-IG | 0/2/1 | +0.013663% | +0.782650 |
| 同键最强（前两例BC-ECS，第三例RC-IG） | 0/3/0 | **+0.230030%** | +13.148467 |

本次三个SAEAX均重新运行400代、1600有效子代，各1612次原生和3次B32；其返回Cmax与原SAEAX历史同键一致。CRF相对本次初解的B32改善分别0、0.922983、10.661088秒。train_15未有native改善，不额外调用B32；另两例最终唯一B32均可行且实际改善。

### 19.5 搜索计数、耗时和最终任务转移

“唯一新候选”不含初解，包含随后被路线下界剪掉的方案；新增原生只计完整四机器人方案，不把DP内部子状态计作调度评价。

| 排样 | 轮数/窗口 | 重复/下界剪枝 | 唯一新候选 | 新原生（有限/非有限） | 接受次数 | 真实子任务转移（生成/评价/接受/最终） | 原生总调用 | B32（初解+最终） |
|---|---|---|---:|---|---:|---|---:|---|
| train_15 | 1/6 | 0/0 | 6 | 6（3/3） | 0 | 8/8/0/0 | 1618 | 3+0 |
| train_34 | 2/12 | 11/0 | 1 | 1（1/0） | 1 | 0/0/0/0 | 1613 | 3+1 |
| train_28 | 3/18 | 15/1 | 3 | 2（2/0） | 2 | 0/0/0/0 | 1614 | 3+1 |
| 合计 | 6/36 | 26/1 | 10 | 9（6/3） | 3 | 8/8/0/0 | 4845 | 9+2 |

全部36个窗口均得到合法DP重构；没有构造失败。40次原生是上限，没有要求耗完：三个案例均在没有严格改善时以 `NO_IMPROVEMENT` 停止。train_28接受第一次后，关键机器人实际从R2变为R3；第二次接受后再检查一轮，证明没有固定优化同一条路线。

train_34初始化29.120421秒、DP搜索1.667960秒（其中DP1.630408秒、原生0.001279秒）、最终B32 0.084799秒；该进程首次DP编译成本也包含其中。train_28初始化19.674311秒、DP搜索0.070737秒（其中DP0.022980秒、原生0.004137秒）、最终B32 0.109984秒。完整时间中位 **26.288337秒**、三次合计 **77.028152秒**；后两例批处理实际50.749911秒，含终端输出及追加开销。三个实际案例分两次进程执行，均保留自身冷编译/初始化成本，未扣除任何初始化以制造时间优势。

train_15首次完成后的CSV追加被沙箱拒绝；该进程已结束，未重跑搜索。随后使用获批执行环境将终端可观察的Cmax、完整耗时、窗口结果和调用计数追加原CSV，再按已有完成键仅运行另外两例。首例的初始化/DP分项计时及完整方案对象未留存，记录中明确缺省，并用 `logging_note` 标注来源；未根据历史方案补造本次解或计时。其最终仍为本次初解、无接受/额外B32这一事实有终端进度和完整结果支持。

### 19.6 失败环节与停止决定

**TRAIN未通过，不进入VALIDATION，不启动TEST，不继续调参。** 可行率3/3、耗时中位26.288337≤103.59秒两项通过；但对最强同键历史为0胜3负，未达到至少2胜，配对中位+0.230030%也没有小于0。CRF的VALIDATION及TEST记录均为0。没有重新运行HGA、WAG或历史三个方法。

具体失败不是DP回溯或原生评价预算耗尽：

1. **路线周期改善未转化为协调Cmax改善。** train_15六个DP解均把最大路线周期从4424.539635降至4389.859246秒，但3个原生非有限；另3个为4622.659666、4655.774208、4622.659666秒，均高于当前4546.654207秒，故全部拒绝。非有限原生不等同于B32 DEADLOCK，也未为其伪造Cmax。
2. **被接受的联合划分没有跨机器人重新分配。** train_34和train_28的最优DP候选全部保留原机器人归属，仅改善窗口内顺序/方向。train_28两个被接受窗口的R0/R1合法接收槽位为空，合法性限制来自原workspace资格，未放宽。三例最终跨机器人转移数均为0；不能以8次被拒候选中的子任务转移宣称完成了有效重分配。
3. **固定邻域内的大量窗口折叠为同一个完整解。** 36个窗口中26个是已见方案、1个下界无法改善，实际仅9个新原生评价。第二、第三例B32分别只降低0.922983、10.661088秒，仍落后同键最强13.148467、5.282523秒。仅保留每窗min-max DP最优解，没有穷举其所有同值但冲突结构不同的路线，因此也不能据此断言这些实例不存在其他更好的协调方案。

已实现并测试的是固定窗口/槽位内的联合精确路线周期优化，以及原生严格接受和最终真实B32判定。实测证明本轮设置成本较低、保持三例可行，但未得到足以击败现有最强结果的归属结构。未验证其对正式HGA/WAG的质量优势；TRAIN失败后没有使用VALIDATION标签、增加模块或扩大预算。

### 19.7 文件与复核命令

新增算法和入口共2个文件；修改原测试、README、本报告和原CSV，共6个文件。CSV从105条历史记录增加到108条、8,510,335字节；历史105条保留，新增source只有CRF_DP_V1三例。全DRL文件检查无超过100,000,000字节文件；上述两个新文件均不足25KB。SAEAX、LNS、HGA、WAG、BC-ECS及RC-IG文件的长度及修改时间均与开始审查一致。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -k crf -q --disable-warnings
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 -m pytest -p no:cacheprovider tests/test_hga_baseline.py -q --disable-warnings
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_crf_dp.py pilot
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_crf_dp.py summary
```

已有三例时pilot按source/stage/实例/seed完成键跳过；summary只读。入口没有TEST命令；VALIDATION在TRAIN未通过时拒绝执行。无需新增结果文件或额外运行管理系统。



