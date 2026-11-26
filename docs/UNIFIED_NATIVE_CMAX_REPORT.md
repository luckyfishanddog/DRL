# 原生调度统一 Cmax 重构记录

日期：2026-11-26。目录：`D:\pybullet_test\MRTA_GA\DRL`。解释器：`D:\pybullet_test\.venv\Scripts\python.exe`。

本轮删除 B32，实现当前算法搜索与最终输出共用已有 Numba 原生调度内核。未训练模型、未生成排样、未运行完整算法实验、未提交或推送 GitHub。固定原子任务切分、资格、连续合并、准备/处理时间和 GNN 图及网络保持原有实现。旧 ALNS 后续不使用，按用户最新要求停止相关工作，撤回为其额外调整的测试时限，不计入下述验证结果。

## 删除范围

- `scheduler.py`：删除 `FormalReferenceEvaluator`、`reference_schedule_formal()`、`reference_schedule_from_templates_formal()`、`_bounded_dispatch_recovery()`、slow/optimized limited-discrepancy 恢复及其别名。
- `dispatch_recovery.py`：删除 `_FrontierItem`、`RecoverySummary`、`limited_discrepancy_recovery()` 和恢复前沿/去重搜索；保留通用 Python 参考工具仍依赖的派工状态、单次追踪及 `run_dispatch_rollout()`。文件名保留，不再包含 bounded 或 limited-discrepancy 恢复引擎。
- `scope.py`、`model.py`、`SchedulerProfile`、公共 accounting/stats：删除恢复政策常量、状态/rollout 预算、恢复计次、expanded/frontier/exhaustion/discrepancy 等专属字段。现有 Scope 几何校验与认证检查保留；现有政策标记改为 `NATIVE_PRIORITY_FOUR_ROBOTS`，不回写历史记录的 Scope 或政策字段。
- WAG、HGA、GNN-VNS-BR、SAEAX、EAX-NSGA2、BC-ECS、RC-IG、CRF-DP、CEM、Joint-MIP、Route-Columns，以及公共构造、排序器、实验脚本的 B32 引用均已迁移。候选评价次数限制仍是候选预算，未将其误删为派工恢复预算；相关计数改名为原生协调评价计数。
- 删除 multi-initial 的 sparse-recovery-budget、recovery-stratified、recovery-summary CLI 分支及其恢复试验函数、未使用的恢复预算和辅助认证函数；保留与构造、路线下界等其他功能共用的代码。
- 删除旧恢复预算选择/采样测试；原混合测试文件中的几何、合并、内核、缓存与认证检查继续保留，部分恢复专属测试改为原生政策和通用 Python 工具回归。
- 平台读取和几何校验继续由现有测试覆盖；删除只包含旧B32二次筛选补丁的 `joint4_native_gnn_diversity.patch`。未整包删除共用模块。

最终检索：`src/`、`scripts/`、`tests/`、`weld_data/` 的活动 Python 代码没有 B32 名称或已删除入口的调用。历史 CSV、checkpoint、旧实验报告中的 B32 文本和原始值保留。

## 统一调用关系与返回结果

```text
WAG/HGA 原生协调评价 -> _priority_arrays -> ScheduleResult -> 最终直接复用
GNN FixedWorkspace.evaluate -> _priority_arrays -> State 缓存数组
    -> FixedWorkspace.schedule_result -> 最终封装（不调度）
其他当前正式入口 -> native_schedule -> priority_schedule -> _priority_arrays
```

`native_schedule()` 位于现有 `src/mrta_reference/scheduler.py`，内部只调用一次已有 `priority_schedule()`，不调用 Python `reference_schedule_optimized()`。公共调度结果包含 `ScheduleStatus`、Cmax、四机完工时间、操作时间线、方向、等待时间和诊断；`waiting_time`、`robot_waiting_time` 直接由所选原生时间线计算。

可行状态统一为 `Cmax = max(robot_completion)`。MOVE、SETUP、WELD、POST、WAIT 均来自同一次原生计算；没有以路线周期或加工负载替代正式 Cmax，也没有比较不同调度器后择优。

| 状态 | 处理 |
|---|---|
| FEASIBLE | 输出原生 Cmax；搜索接受与全局最优更新比较这个值 |
| DEADLOCK | 当前固定双优先级规则未完成派工；Cmax 为空，可能保留原生未完成时间线及数组诊断，不作为完整可行调度或物理不可行证明 |
| INFEASIBLE | 完整方案结构、合并规范或方向等输入不合法；Cmax 为空 |
| NUMERIC_FAILURE | 数值异常或 NaN；Cmax 为空，单独记录，不当作 DEADLOCK |

GNN 监督入口仍走 `FixedWorkspace.evaluate()`。教师路线仅按已保存的有限原生值选择，恢复后使用同一内核产生候选标签；不再按历史 B32 可行状态、调用次数或终评 Cmax 筛选教师。有限 Cmax 优于 DEADLOCK；两个 DEADLOCK 不构造 Cmax 优劣；NaN 不参与 Pairwise 对。本轮仅在构造的小型测试实例中验证标签逻辑，没有对真实 TRAIN/VALIDATION/TEST 构建新标签。

通用 `reference_schedule()`、`reference_schedule_optimized()` 和其单次派工追踪仍供独立几何、exact/oracle 工具和测试使用。活动算法及实验脚本中没有对这两个 Python 调度入口的直接调用；它们不参与当前正式结果的第二次评价。现有独立认证检查仍保留，它检查给定时间线，不生成另一条时间线或另一套 Cmax。

## 算法与调用次数

| 位置 | 原流程中的末端调用 | 当前流程 |
|---|---|---|
| 固定任务集 WAG/HGA `baseline_job()` | 搜索后再调用一次 B32 | 复用 `result.schedule`，末端新增调度调用为 0 |
| 旧焊缝排样 WAG/HGA 实验入口 | 对 timely best 再调用一次 B32 | 复用 `best.schedule`，末端新增调度调用为 0 |
| GNN-VNS-BR `run_one()` | 对最终方案调用一次 B32 | 封装 State 中已有数组，末端新增调度调用为 0 |
| `native_schedule()` | 原正式入口可能执行基准派工和最多 32 次替代派工 | 每个待评价完整方案只调用一次原生内核，无替代派工 |
| HGA ROUTE_CYCLE VND | 邻域内不调用完整协调调度 | 保持 0；完整个体仍按原分工原生评价一次 |
| WAG 原生候选评价 | 原有快速数组评价及获胜时间线复用 | 保持，没有增加候选重复调度 |

结果中的 `coordination_calls` 表示实际累计原生协调评价次数，不再固定为 1；`final_scheduler_extra_calls` 为 0。GNN 断点续跑据完整记录和实际正调用数识别完成项，完整 DEADLOCK 记录同样跳过，不因调用数大于 1 而重跑。

与本轮开始时的源码快照比较，以下 7 个关键函数的计算代码未改变：`_numeric_templates`、`_numeric_follow`、`_priority_arrays`、`_priority_pair_directions`、`priority_schedule`、`_merge_fixed_arrays`、`_vnd_oriented_arrays`。仅更新了 `_priority_arrays` 的说明文字。WAG Move/Swap/LNS、2-opt、factorial combination 以及 HGA VND/M1–M6、交叉和种群逻辑没有重写。

GNN 的 `candidate_actions()`、`local_graph()`、`BottleneckGAT`、`ActionMLP`、`make_ranker()` 保留既有实现。输入仍为 16 维节点、3 维边、18 维动作；仍是合并作业局部图、两层 64 维四头 GAT，没有改成固定原子异构图或增加预测头。候选、Top-3 和搜索预算未改。

准备/处理公式仍是 `20 + L / 0.0108 + 30` 秒，空走速度仍为 `0.20 m/s`。同父焊缝、同机器人连续原子先合并，再对实际作业计算一次 SETUP/WELD/POST。`fixed_taskset.py` 未修改。

## 实际验证

指定解释器下分批执行了 **55 个不同的当前范围针对性用例，最终均通过**，重复执行的用例不重复计数。没有运行完整历史测试集、WAG 500 轮正式实验、HGA 200000 子代实验或真实 GNN TRAIN/VALIDATION/TEST。

| 验证内容 | 结果 |
|---|---|
| 公共入口只调用一次现有内核、禁止转入 Python 优化调度器 | 通过 |
| GNN 数组接口与公共接口的状态、Cmax、四机完工、方向和时间线一致 | 通过 |
| 最终封装及 WAG/HGA 两类实验入口复用已有结果，不再调度 | 通过 |
| WAIT 进入四机实际完工时间，Cmax 为四机最大值 | 通过 |
| 准备 20 秒、处理 30 秒；长父焊缝合并后不重复计费 | 通过 |
| DEADLOCK、结构 INFEASIBLE、NUMERIC_FAILURE 分离，无恢复调度 | 通过 |
| HGA ROUTE_CYCLE VND 零完整协调调用、完整个体一次原生评价 | 通过 |
| WAG 原生获胜方案只评价一次、Numba WAIT 与现有几何 ESS 一致 | 通过 |
| GNN 六邻域、全局最优保留、16/3/18 图、前向和单次反向梯度 | 通过；未进行优化器训练 |
| TRAIN-only 小型监督测试、NaN/DEADLOCK Pairwise 语义 | 通过 |
| 原生完成记录跳过、完整失败记录跳过、CSV 中断尾部处理 | 通过；使用内存文件，无新实验 CSV |
| 原生观察器、政策标记、原有认证身份检查及通用 Python 工具回归 | 通过 |

另对 94 个活动 Python 文件进行了语法编译检查：0 错误；已删除调度入口调用为 0；当前算法/脚本调用通用 Python 正式调度路径为 0。旧 ALNS 的额外进程冷启动诊断不计入这 55 项，也不据此修改现用算法预算。

首批验证曾发现原生“显式方向”标记与现有认证身份检查不一致，已在公共接口中统一政策标记后通过复跑，没有关闭认证。新增输出目录后，内存 CSV 测试替身也补齐了 Path 的 parent/mkdir 接口。Numba 存在既有 reflected-list 弃用提示，本轮未修改其依赖或计算内核。

## 历史产物和输出管理

对本轮开始时记录的 **75 个历史模型、结果和实验文档**复核文件大小、修改时间，均未改变。包括旧 23 维/40 TRAIN/scale1000 checkpoint，WAG/HGA CSV、旧 60 条验证、3 条 Smoke、190 条 scale1000 验证，以及 `docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md`。

旧数据保留原 B32 终评口径，不更新字段，不与新原生结果混合配对。原始固定任务集仍从历史 `tasksets.json` 读取，没有生成重复文件。后续原生结果路径改为 `results/native_priority/` 下的分类目录；本轮该目录和其中 CSV 未创建。现用两个主实验入口的 summary 只打印汇总，不重写历史报告或本报告。

本轮只新增本文件，已获用户确认，大小远低于 100 MB。未新增模型、结果 CSV、训练样本副本、hash、manifest、冻结协议或独立调度/认证模块。README 仅更新当前口径和历史标识，旧实测数值保留。

## 修改文件

核心：

```text
src/mrta_reference/{scheduler,dispatch_recovery,scope,model,__init__}.py
src/mrta_baselines/{common,wag_vns,hga}.py
src/mrta_search/{gnn_vns_br,stats,pipeline,initialization,lns,vns}.py
src/mrta_search/{bc_ecs,crf_dp,eax_nsga2,gnn_cem,gnn_constructor,gnn_joint_mip,
                gnn_route_columns,rc_ig,saeax}.py
src/mrta_ranker/{dataset,ranking}.py
```

实验入口、历史诊断与字段迁移：

```text
scripts/run_fixed_taskset_experiment.py
scripts/run_gnn_vns_br.py
scripts/run_gnn_weld_layout.py
scripts/run_joint_oriented_repair_pilot.py
scripts/run_gnn_multi_initial.py
scripts/run_gnn_{cem,cmax_finetune,construct_vns_pilot,initial_mode_new,joint_mip,
                 joint_repair_pilot,route_columns,shared_construction}.py
scripts/run_{crf_dp,eax_nsga2,eg_gnn_pvnd,rc_ig,rule_guided_gnn,saeax}.py
scripts/train_mlp_ranker.py
```

现有测试文件：`test_baseline_common`、`test_gnn_construct_vns`、`test_gnn_schedule_blocker_pilot`、`test_hga_baseline`、`test_interference_scheduler`、`test_model_geometry`、`test_phase3_validation`、`test_scheduler_equivalence`、`test_search_direction_refinement`、`test_search_initialization`、`test_search_lns`、`test_wag_baseline`。另修改 `README.md`，新增本报告；上述删除文件单独列于删除范围。没有新建测试文件。

## 适用范围与尚存问题

当前 Cmax 衡量的是现有固定同轨双优先级与跨轨检查规则下的调度结果，不是所有物理派工顺序中的全局最优值。删除 B32 后不再恢复原生 DEADLOCK，历史 B32 曾可行的方案可能在当前政策下没有可行派工；这属于评价口径变化，不能据旧终评可行率推断当前可行率。

Numba 首次 JIT 仍有成本；稳态计算内核没有被更慢的 Python 调度器替换。没有进行新的正式性能基准，因此本报告不声称新口径的 Cmax 改善率、模型有效性或整体加速比。下一步若需算法性能结论，应另行明确新口径实验范围；本轮不启动这些实验，也不处理已停用的旧 ALNS。
