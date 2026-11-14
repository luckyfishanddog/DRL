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

