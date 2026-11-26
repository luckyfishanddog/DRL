# 双滑轨四机器人焊缝分配与排序

2026-11-25：WAG按文献使用未合并`W_k`分配/VNS、合并`W'_k`排序评价，保留原生数值快速路径；`train_03`（93父、167固定子段）、种子20261111完整500轮含冷编译及最终B32实测73.015秒，B32可行。用户已同意4个工作进程后台批量续跑70排样×3种子。HGA慢速原因已查明：旧正式入口覆盖为`ROUTE_CYCLE`，新固定入口误用`COORDINATED`，使大量邻域候选调用协调器，确实改变了实际搜索路径。现在恢复旧正式入口的路线代价VND、每完整个体一次原生协调选择及最终B32；mu=20、lambda=10、alpha=20、完整六邻域及200000有效子代不变。HGA完整回归155项通过；单例完整结束后，后台自动加入HGA批量（4个工作进程），两组对照各210条完成后自动BC预训练8轮、PPO微调80次更新及最终评价，全程由本机进程执行，无在线模型轮询。旧诊断原样保留并通过普通input_adapter字段区分，实际进度与结果见[报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py baselines --method wag_vns --workers 4
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py benchmark --method hga --layout train_03 --seed 20261111
```

2026-11-25：正在执行新的固定欧氏拆分任务域。70个原始排样保持不变，TRAIN/VALIDATION/TEST生成6,446/1,671/3,201个固定原子任务；Y责任判定后才进行欧氏长拆分，同机器人承担的任意连续段在排序前强制合并。WAG每例500轮、HGA每例200,000有效子代，最终使用原B32。新逻辑专项9项和核心回归313项通过（有重叠），共享GNN的BC+PPO实现及微型执行测试已完成，正式训练等待本轮完整教师结果；**当前不能宣称三算法正式实验已经完成或模型已经训练。** 新任务集及逐例结果独立保存在[固定任务域目录](results/fixed_euclidean_split/)，进度和复现命令见[新报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。旧动态WHOLE/Y/X实验保留，仅下文记录其历史情况，不混入本次比较。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_fixed_taskset_experiment.py all --workers 4
```

本项目联合决定父焊缝的合法拆分、四机器人归属、路线顺序和焊接方向，以 B32 调度返回的 Cmax 为优化目标。当前数据来自真实小组立排样，TRAIN、VALIDATION、TEST 分别包含 40、10、20 个排样和 3,749、936、2,060 条父焊缝。所有最终父焊缝的平面长度至少为 50 mm，TEST 的构件型号独立于训练侧。

2026-11-25 **GNN-CEM已完成全局联合概率模型和一次固定TRAIN实验，未通过1%中位改善门槛，停止，未进入VALIDATION/TEST。** 两层64维/4头GAT复用几何编码，新模式头先采样合法拆分，再在实际激活的动态任务图上从空路线采样任务、机器人和方向；每代重放完整精英序列进行真实参数更新。train_12/05/07各运行8个体、3代，最终Cmax为5066.242573/5844.684790/5804.536919秒，均保留本次公共RULE/BEAM初解，改善率均0%。67个新完整候选全部B32 DEADLOCK，另5次构造未完成；三例含初解的B32分别23/23/24次，完整时间77.73/103.81/150.58秒，中位103.81秒。9次更新全部使用保留的初解精英，编码器和两个新头确有变化，但没有获得新的可行精英。新增仅获批的[算法](src/mrta_search/gnn_cem.py)、[入口](scripts/run_gnn_cem.py)和[结果CSV](results/new_dataset_final/gnn_cem.csv)，CSV保留完整最终方案；11项新测试及1项相关回归通过。无同键正式HGA/WAG结果，不能宣称超过对照；失败证据和复核命令见[报告第22节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#22-2026-11-25gnn-cem-全局联合采样与一次train筛查未通过)。

2026-11-24 **GNN-JN-MIP已完成联合弧MILP、集合评分头及三个TRAIN配对实测，未通过0.5%改善阈值，停止，未进入VALIDATION/TEST。** GNN选择4～6父焊缝集合，同一MILP联合决定全部合法模式、归属、有向排序和全部插入间隙，原B32严格改善才接受。train_20/21/09均B32可行且严格改善本次初解，对Rule-JN-MIP为2胜1平；但初解配对中位改善仅0.052710%，对Rule中位优势仅0.028041秒，完整耗时中位32.34秒。新头只用实际MILP+B32偏好固定训练6轮；拟合train_35初解失败、未重跑或替换，实际仅1个正标签和2个偏好对，不能以训练排序100%宣称泛化。15项新增测试通过，仅新增获批的[算法](src/mrta_search/gnn_joint_mip.py)和[入口](scripts/run_gnn_joint_mip.py)；结果继续追加原CSV。完整数学模型、逐例结果、超时状态及首进程失败诊断缺失说明见[报告第21节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#21-2026-11-24gnn-jn-mip-联合弧整数规划train筛查)。

2026-11-24 **GNN-RCG已完成实现、固定定价头训练及三个TRAIN配对实测，未通过筛查，停止，未进入VALIDATION/TEST。** 经用户接受历史暴露限制，使用train_30/13/36（71/90/121父），三例新方法与Greedy-RCG均B32返回可行，但全部保留本次SAEAX初解，彼此及相对初解均0胜3平，Cmax中位改善0%。GNN完整时间中位19.68秒；实际评分591,922个追加动作、加入183条新路线列，仍没有改善整数路线组合，不能宣称GNN有效。新头只用旧TRAIN的288条完整路线约化成本标签固定训练6轮，未用B32标签。仅新增获批的[算法](src/mrta_search/gnn_route_columns.py)和[入口](scripts/run_gnn_route_columns.py)，12项新增测试通过；结果追加原CSV，未新增模型或日志文件。三例无同键正式HGA/WAG参照，完整结果及失败证据见[报告第20节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#20-2026-11-24gnn-rcg-实现与历史train暴露核查)。

2026-11-24 **CRF-DP已完成实现和三个TRAIN实测，未通过筛查，停止，未进入VALIDATION/TEST。** 三例均B32可行，完整耗时中位26.29秒；相对SAEAX为2胜1平，但相对SAEAX/BC-ECS/RC-IG中的同键最强为0胜3负，Cmax变化中位+0.2300%。DP接受的改善均为路线内重排/方向，最终跨机器人任务转移为0。120项测试通过。逐例结果、调用数、精确性范围和首例日志恢复说明见[报告第19节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#19-2026-11-24crf-dp-关键路线片段联合动态规划)。

2026-11-24 **GNN构造策略B32直接监督已完成，开发筛查未通过，停止，未进入VALIDATION/TEST。** 修复了原ChildJoint没有直接训练assignment policy的缺口：既定20个TRAIN自生成40组、236条构造反事实标签，新增233次B32；assignment和insertion两头均获得直接梯度，固定三轮训练后保存独立[权重](models/gnn_child_joint_construct.pt)。但构造训练内可行Cmax排序仅49.33%，六个开发案例新方法1/6可行（历史ChildJoint 4/6），仅一个共同可行配对改善11.17%，另3例由可行变为DEADLOCK；在线时间中位47.36秒（旧30.27秒）。**梯度缺口修复不代表性能达标，未追加调参。** 数据继续追加原CSV，完整调用链、监督定义、梯度、逐例实测及失败分析见[报告第12节](docs/GNN_JOINT4_REASSIGN_REPORT.md#12-2026-11-24构造策略的自生成反事实b32直接监督)。

## 正式模型

实验共用 `ScientificConfig()`、`FORMAL_SCOPE_V2` 和原 B32。每个 canonical 焊接块的加工时间为 `20 + L/0.0108 + 30` 秒，空走速度为 0.20 m/s，合法拆分的子焊缝最短 0.50 m。机器人的完成时间包括加工、相邻任务间空走和调度等待。

## 方法

- RULE/BEAM 用原单提案构造，作为快速基线。
- 正式 WAG+VNS 使用 `n=10`、`p=0.50`、500 轮；正式 HGA 使用显式方向、完整路线 M1–M6 VND、`μ=20/λ=10/α=20`、200,000 个有效子代。两者使用原生协调评价选择完整方案，最终各调用一次 B32。正式结果与历史短测、120 秒预算结果分开统计。
- 4R-SAEAX 从双亲有向邻接组装路线片段，联合选择机器人、插入位置和方向，搜索使用既有原生协调内核，结束后最多 3 次 B32 选优。采用 12 个体、每代 4 子代、最多 400 代、至少 80 代、连续 120 代停滞停止，没有墙钟硬截止。
- CRF-DP 从本次重新运行的SAEAX开始，对关键路线5～8条子焊缝用带方向Held–Karp子集DP及四机器人集合划分，联合决定归属、顺序和方向。固定父模式及首任务，每状态至多6窗、8轮、40次新原生评价，仅接受严格native改善，最后至多一次额外B32；不改GNN或原调度器。见[算法](src/mrta_search/crf_dp.py)及[入口](scripts/run_crf_dp.py)。
- EG-GNN-PVND 第一阶段只学习 HGA 已访问动作的优先级与真实原生评价成本。20 个 TRAIN 轨迹中，15 个排样拟合、5 个排样留出；评分器为两层 hidden32/2头，共 14,938 参数，未重新训练原焊缝分配 GNN。
- GNN-BVNS 用原两层64维/4头GAT对父级Relocate、Swap、瓶颈路线2-opt及LNS修复候选批量评分；每个当前状态编码一次，搜索复用原priority_schedule，最终只调用一次B32。加入有向转移连续属性与来源/目标路线增量，使用TRAIN真实B32 Cmax配对续训；没有新增独立GNN或Cmax预测器。旧GNN联合构造及反馈实验继续保留。
- GNN-ChildJoint-B32Rank 从原Cmax-Conflict冻结权重初始化，用同一GNN构造Beam2初解、选择子焊缝转移关系并解码完整插入/方向，支持单子段、双子段及三/四机器人循环。整体动作只经一个原评分头，真实B32训练的无动作相对分数控制接受和状态保留；原生Cmax只作快速信息，最终一次B32。未使用HGA/WAG轨迹或v2微调权重。
- GNN-ChildJoint-ConstructB32 直接重放实际构造的assignment＋insertion联合评分，用同一部分状态、共同种子续构造的真实B32结果拟合相对no-op归一化Cmax改善。仍为原两层64维/4头/11关系GNN，沿用原ChildJoint局部动作；初始Beam4，500原生/80状态/8扰动，每个最终方案一次B32。

此前ChildJoint方法完成59项相关回归、20个TRAIN排样自产候选标注（520标签、519次新增B32）、固定3轮学习及六例同预算消融。新GNN可行4/6，无GNN可行5/6；4个共同可行Cmax配对为0胜4负、中位增加37.012%，在线墙钟中位30.27s/9.29s。学习接受已实际生效，但训练内无动作判断仅45.41%，六例关键路线周期均继续上升。**本轮未通过TRAIN，停止于阶段2，未运行VALIDATION/TEST，也未追加调参。** 数学定义、真实调用链、动作计数与负结果见[报告第11节](docs/GNN_JOINT4_REASSIGN_REPORT.md)，独立[权重](models/gnn_child_joint.pt)与[实验CSV](results/new_dataset_final/gnn_child_joint.csv)保留；第10节BVNS历史结果不覆盖。

2026-11-24 GNN-BVNS完成调用链/耗时分析、三臂TRAIN比较、固定20排样/624个真实B32标签/3轮训练及官方30次VALIDATION。独立六例为6/6可行，相对历史Cmax-Conflict中位改善1.988%；相对无GNN仅3胜2负1平、中位改善0.0093%。正式VALIDATION为23/30可行，共同可行Cmax相对HGA/WAG中位分别增加3.110%/1.285%，为0胜23负/1胜19负；墙钟中位33.90s，每例一次最终B32。**质量未达标，停止于阶段4，未运行TEST或继续调参。** 完整六例、30例精确种子配对、各邻域计数及失败分析见[报告第10节](docs/GNN_JOINT4_REASSIGN_REPORT.md)，新权重为[gnn_bvns.pt](models/gnn_bvns.pt)，唯一新实验表为[gnn_bvns.csv](results/new_dataset_final/gnn_bvns.csv)。历史Cmax-Conflict参照使用旧多B32协议，不能当成同预算正式对照。

EG-GNN-PVND 已完成教师采集、固定 12 轮训练和离线检验。留出观察池总体排序准确率 59.49%，首次原生改善评价数均值 15.33→6.72，但两个排样退化；真正优化已有可行 Cmax 的子集在计入决策开销后净变慢。**当前不值得进入阶段二，未接入 HGA 搜索，未跑 VALIDATION/TEST。** 详见[第一阶段报告](docs/GNN_WELD_LAYOUT_EXPERIMENT.md)，数据与模型分别保存在 [TRAIN教师](data/weld_layout/hga_vnd_teacher.jsonl.gz)和[评分器 checkpoint](models/eg_gnn_pvnd.pt)。

2026-11-17 已完成 3 个 TRAIN 预实验与 30 次 SAEAX VALIDATION：B32 可行 30/30，完整耗时中位 22.59 s；对 WAG/HGA 的共同可行 Cmax 配对中位变化为 **+0.445% / +1.202%**（正值更差），分别 6 胜 16 负与 0 胜 30 负。质量未达主算法要求，未启动 TEST。完整逐例结果、停止计数与失败分析见[报告第 14 节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#14-2026-11-174r-saeax-实现一次-train-预实验与完整-validation)及[唯一实验 CSV](results/saeax_validation.csv)。

BR-SAEAX 随后完成同种子的三个 TRAIN：最终 Cmax 为 4432.852/5707.055/6100.694 s，相对 SAEAX 为 2 胜 1 负、配对中位变化 -0.401%；B32 可行 3/3，完整时间中位 26.90 s，实际停止于 364/313/400 代。但最终 Pmax 为 0 降、1 平、2 升，**未通过 TRAIN，不继续 VALIDATION/TEST**。1071 次合法联合重插及 32 次三机器人接力没有建立稳定负载改善，不能宣称超过 WAG/HGA。三条结果追加到原 CSV，详见[报告第 15 节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#15-2026-11-17br-saeax-多父焊缝瓶颈联合重插入train未通过)。

LA-4R-EAX-NSGA-II 已按用户提供的 Gao 论文第4.4节及图9实现独立增强邻接表交叉、四机器人联合归属/顺序/方向构造和三目标 NSGA-II。20个体、每代10子代、最多500代、120代停滞、最终一次B32的三个 TRAIN 已完成：B32可行2/3，两个共同可行配对均落后于SAEAX，Cmax变化中位 **+35.121%**；完整时间中位22.52 s，停止于427/133/330代。8900个有效子代仅73个原生可行，**TRAIN未通过，未进入VALIDATION/TEST**。实现见 [eax_nsga2.py](src/mrta_search/eax_nsga2.py)，入口见 [run_eax_nsga2.py](scripts/run_eax_nsga2.py)，详细负结果及命令见[报告第16节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#16-2026-11-17la-4r-eax-nsga-ii全文核对实现及train筛查负结果)。三条结果继续追加到原CSV，全部历史记录保留。

BC-ECS 已完成本次运行内实际生成 SAEAX 初解的3个 TRAIN及30次 VALIDATION，未读取历史最优解。TRAIN为3/3可行、对SAEAX 2胜1平，时间中位30.49 s，通过筛查；VALIDATION为30/30可行、对SAEAX 19胜11平，但对WAG共同可行22对为7胜15负、Cmax变化中位 **+0.0969%**，对HGA为0胜30负、**+1.0829%**。完整时间中位31.65 s，包含初始化、逐出链和全部B32。**质量未达标，本轮停止，未启动TEST或继续调参。** 实现见 [bc_ecs.py](src/mrta_search/bc_ecs.py)，独立入口见 [run_bc_ecs.py](scripts/run_bc_ecs.py)，33条结果追加到原CSV，调用计数、停止原因及全部逐例数据见[报告第17节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#17-2026-11-18bc-ecs-实现train通过及完整validation负结果)。

## 运行

在 `D:\pybullet_test\MRTA_GA\DRL` 中执行。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py child-joint-summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py child-construct-summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py bvns-summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py bvns-trained-summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_gnn_weld_layout.py bvns-formal-summary --split validation
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_eg_gnn_pvnd.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_eg_gnn_pvnd.py collect --workers 4
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_eg_gnn_pvnd.py fit
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py validation --workers 3
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_crf_dp.py summary
```

EG 入口在已有20个教师排样及模型时直接读取，不重新采集或训练。SAEAX 入口在已有3次预实验及30次验证时直接读取，不重复搜索或B32。旧GNN与对照入口保留在原脚本中。

## 新数据实验（早期记录）

以下为历史阶段结果；当前正式对照及 SAEAX 结论以上述报告为准。

上一轮新 TEST 上，RULE/BEAM、原始 GNN_VNEXT 和负载感知 GNN_VNEXT 的单提案 B32 可行数分别为 16/20、17/20、16/20。可行 Cmax 中位分别为 6,901.0、8,733.4、8,053.3 秒。详细结果与本轮实验进度见 [新排样实验报告](docs/GNN_WELD_LAYOUT_EXPERIMENT.md)。

完整流程设置下，两种联合 GNN 各完成 20 个 TEST 排样和三个种子的 60 次运行，均为 48/60 B32 可行。可行 Cmax 中位分别为 7,319.3 s、7,224.6 s，总耗时中位分别为 9.14 s、9.84 s。重构版相对 RULE/BEAM 的配对 Cmax 中位高 4.34%。当前汇总使用完整流程结果。

此前 60 秒多方案搜索中，两种联合 GNN 各完成 60 次 TEST，均为 58/60 B32 可行，可行 Cmax 中位分别为 7,172.3 s 和 7,143.1 s。带瓶颈重构版本在共同可行样本上相对负载感知 VNext 的配对 Cmax 中位下降 10.89%，相对 RULE/BEAM 高 3.49%。这些记录保留在 `gnn_results.csv` 中。HGA 论文评分版完成前四个排样的 12 次运行，最终 B32 可行 0/12。WAG+VNS 已暂停，保留两个排样各三个种子的 500 轮结果，B32 可行 5/6。[本轮已完成结果](results/new_dataset_final/final_comparison.csv)
