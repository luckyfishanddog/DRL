# 双滑轨四机器人焊缝分配与排序

当前研究路线：**固定拆分 → 四机器人任务分配 → 连续原子任务合并 → 焊缝排序及方向 → 调度评价**。

| 方法 | 定位 |
|---|---|
| Adapted-WAG+VNS | 独立文献对照组，保留既有210次旧口径正式结果 |
| Adapted-HGA | 独立文献对照组，保留既有210次旧口径正式结果 |
| GNN-VNS-BR | 本项目方法：监督学习的GNN候选排序与变邻域瓶颈重构 |

主方法使用GNN监督学习排序，结合固定任务域上的瓶颈重构与原生调度评价。

## 固定任务域与调度

[src/mrta_reference/fixed_taskset.py](src/mrta_reference/fixed_taskset.py)是唯一切分来源。先按整条父焊缝的轨道可达性确定责任区域，无法由单侧承担时在共享带固定切分；区域段欧氏长度L>5m时使用n=ceil(L)-1等分，新生成子段严格大于1m。优化过程中不增删或移动切分点。

以固定原子任务确定机器人归属；同父焊缝中由同机器人承担的连续原子任务，在排序与评价前强制合并为实际作业。排序和方向作用于这些合并作业。原始TRAIN/VALIDATION/TEST分别有40/10/20个排样，共6446/1671/3201个固定原子任务。

当前正式评价统一使用原有Numba `_priority_arrays()` 内核。公共接口为 `native_schedule()` → `priority_schedule()` → `_priority_arrays()`；GNN搜索直接调用数组接口。WAG/HGA复用已选中的原生ScheduleResult，GNN最终封装已保存的数组，不再重复调度。可行方案的Cmax为四台机器人实际完工时间的最大值，等待时间计入完工时间。

B32正式入口、bounded/limited-discrepancy恢复及其预算和状态字段已删除。同轨双优先级、方向组合、跨轨干涉、准备/处理时间和连续原子合并规则保持不变。DEADLOCK只表示当前固定调度规则未完成派工，不证明物理上绝对不可行。通用Python参考调度器仅供独立几何工具、exact/oracle诊断和测试，不是当前实验的第二条正式评价路径。

## GNN-VNS-BR

[算法](src/mrta_search/gnn_vns_br.py)独立执行贪心初始化和搜索，不调用WAG/HGA求解器、种群进化或完整VND。由实际快速调度的四机器人完工时间确定瓶颈；尚未找到原生可行派工的初始状态仅使用路线代价辅助寻求可行状态。

图包含`FixedTaskset.tasks`中的全部固定原子任务和四个机器人节点。搜索和调度仍作用于强制合并后的实际作业，通过固定原子映射将源作业、目标作业及M1被迁移的端部原子映射到图中，动作评分对相应原子嵌入作均值聚合。

| 节点类型 | 12维输入，按列顺序 |
|---|---|
| 焊缝任务 | is_task_node、is_robot_node、start_x、start_y、end_x、end_y、length、u_start、u_end、is_assigned、route_position、direction |
| 机器人 | is_task_node、is_robot_node、R1、R2、R3、R4、tcp_x、tcp_y、processing_load、empty_time、completion、waiting_time |

两类节点分别使用12→64输入投影，共享两层64维、4头关系感知GAT。关系类型独立于连续边属性，固定为self、task_robot、parent_next、parent_prev、route_next、route_prev、space_neighbor七种；边属性统一为4维：task_robot为`[eligible, owner, T_start, T_end]`，route为`[0, 0, T_empty, 0]`，space_neighbor为`[T00, T01, T10, T11]`，其他边填零。空间近邻保持K=4，反向转移分别计算。

坐标按固定20m×12m平台归一化至[-1,1]，长度除以固定24m，u区间和路线位置保留[0,1]，二值及One-hot不作连续归一化。全部时间特征采用仅由TRAIN计算的全局固定`T_ref`，当前3000个TRAIN的参考值为6193.805663528889秒；配置随新checkpoint保存，训练与推理共用，不使用随排样变化的workspace.scale。空走采用当前二维平面匀速模型；MOVE和WAIT统计来自已保存的原生调度操作，不追加调度调用。

`feedback_valid`是对应四个机器人的布尔向量。FEASIBLE保留完整调度反馈；DEADLOCK、INFEASIBLE、NUMERIC_FAILURE将empty_time、completion、waiting_time清零，processing_load保留。GAT将一个轻量的True/False嵌入加入机器人隐藏表示，使无效反馈与有效的零等待能够区分；节点输入仍为12维。

GNN和ActionMLP共用以下16维动作描述，顺序固定：`M1, M2, M3, M4, M5, M6, source_processing_delta, target_processing_delta, source_empty_delta, target_empty_delta, route_peak_delta, source_job_fraction, flip, portion_none, portion_start, portion_end`。时间变化量除以同一T_ref；route_peak_delta是最大路线代理代价的变化，不能当成实际协调Cmax变化。统一候选动作评分头结合源/目标作业、源/目标机器人嵌入及动作描述，输出完整联合动作的标量分数，不设置独立机器人分配预测头。

候选按邻域、目标机器人、锚点分配有限预算，取消随机截断和按局部代价淘汰半数候选。GNN对M1–M6统一评分，每个邻域最多3个候选进入原生评价，再从全部六邻域选择实际Cmax最小的可行改善。方向随动作处理，有限扰动始终保留全局可行最优。

M1也可沿已有原子边界迁移合并作业的端部原子，再按新归属强制合并，避免把长作业永久锁成不可调整的分配单位。若M5/M6没有可行扰动，有限扰动预算可用于GNN引导的M1/M2；不会增加切分点或接受不可行方案。

监督状态复用具有有效native_cmax_s、完整solution和正常终止状态的已保存TRAIN WAG/HGA教师路线、独立贪心初始化及少量合法扰动；恢复教师路线后仍由同一原生内核生成标签，不使用历史其他Cmax字段替代native_cmax_s，不重新运行基线。训练和推理共用候选与图构造逻辑，alpha=8、anchors=12、每邻域最多40候选、每状态最多256对比较。Pairwise Ranking只按真实原生Cmax比较FEASIBLE方案，失败状态之间不构造虚假Cmax优劣，NUMERIC_FAILURE不生成排序对；标签只在内存中生成。内部留出按完整TRAIN layout划分，不将同一排样的状态拆到两侧。

此前40个TRAIN排样训练的边特征模型models/gnn_vns_br_edgeaware.pt及3条Smoke保留；旧23维模型models/gnn_vns_br.pt及其60条VALIDATION结果也保留，三阶段结果分开记录。

历史scale1000模型`models/gnn_vns_br_edgeaware_scale1000.pt`及其旧输入定义和实验记录保留。当前12维节点、4维边、16维动作与历史模型不兼容，不转换或强行加载旧权重。新16维模型的默认路径为`models/gnn_vns_br_atomic_16d.pt`及`models/gnn_vns_br_atomic_16d_scale3000.pt`；Pilot使用独立的`models/gnn_vns_br_atomic_16d_pilot.pt`。已有checkpoint不会被覆盖。

## 数据、结果与执行

平台Excel读取、几何转换与原有哈希校验集中在 `src/mrta_data/platform_instances.py`；workbook/family角色与访问检查集中在 `src/mrta_data/dataset_split.py`。`data/manifests/`采用中性文件名；已保存的身份字段、历史几何路径和哈希保持原样。仓库外`../ppo`仅作为历史几何数据及生成工具的来源，现有算法不调用其中的训练代码。

- 原始排样：data/weld_layout/train.jsonl、validation.jsonl、test.jsonl。
- 固定任务及原始结果：[results/fixed_euclidean_split/](results/fixed_euclidean_split/)。
- 新方法统一入口：[scripts/run_gnn_vns_br.py](scripts/run_gnn_vns_br.py)。
- 完整实测、配对Cmax变化及失败记录：[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。
- 本轮删除清单、统一调用关系与针对性测试：[原生Cmax重构报告](docs/UNIFIED_NATIVE_CMAX_REPORT.md)。

准备检查和小规模监督学习使用同一入口；`--layouts`只接受不重复且实际存在的TRAIN排样ID。省略该参数才遍历全部TRAIN，Pilot必须显式指定8个排样，不执行全量训练或VALIDATION/TEST。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py prepare --dataset scale1000
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train --dataset scale1000 --variant full_gnn --epochs 24 --training-seed 20261126 --layouts train_12 train_04 train_27 train_35 train_36 train_29 train_28 train_14 --weights models/gnn_vns_br_atomic_16d_pilot.pt
```

入口默认使用--dataset scale1000。该路径现有TRAIN/VALIDATION/TEST分别为3000/500/500条，TRAIN包含原始40个排样，不重复计为3040个。逐条读取并通过原generate_taskset转换。Pilot从原始40个TRAIN按固定原子任务数量覆盖低、中、高规模，复用既有教师，不生成排样、不补跑WAG/HGA，也不保存单独Pairwise标签文件。训练结束只报告本轮训练记录并验证checkpoint重新加载，不自动调用validation、test或validation_admission。

历史VALIDATION矩阵为原始10排样×3种子×3模型，以及新100排样×1种子的Full GNN。旧B32终评不能与新原生结果配对。后续原生口径输出路径为results/native_priority/fixed_euclidean_split/与results/native_priority/new_dataset_final/，按数据、排样、种子和模型复用完整记录，包括完整失败记录；本轮未创建这些结果CSV。既有准入判断只读取新口径结果，不将历史B32基线混入比较。

既有WAG/HGA和旧23维VALIDATION不重跑。此前3条TRAIN Smoke仍保存在results/new_dataset_final/gnn_vns_br_edgeaware.csv；历史数据直接查阅旧CSV和既有实验报告。当前summary打印原生路径中的汇总，不重写历史报告。

动态拆分的激活、取消与切分点切换已从候选执行、生成及活动算子中移除；相关旧动态GNN搜索入口停用，必要的历史读取、几何校验和WAG/HGA共享函数保留。Scope仅移除恢复政策元数据，现有几何校验继续保留；它不能用于改变固定原子边界。

当前在线时间包含初始化、GNN推理、原生候选评价和结果封装，最终不再追加B32。离线样本生成和训练时间单列。新增文件均位于DRL分类目录，单文件小于100MB；不生成训练样本副本、额外调度器、hash、manifest或实验身份体系。

## 16维动作监督学习Pilot（本机实测）

已按上面的8排样命令完成full_gnn监督Pilot。所选原子任务数分别为train_12=120、train_04=141、train_27=152、train_35=160、train_36=165、train_29=170、train_28=180、train_14=199。复用WAG和HGA教师各8个（恢复后均为当前原生FEASIBLE），另有8次独立贪心初始化，没有补跑基线或生成排样。

生成48个状态，其中36个有有效Pairwise比较、12个无有效比较；候选标签11231个，FEASIBLE=3902、DEADLOCK=7329、INFEASIBLE=0、NUMERIC_FAILURE=0，有效比较对9216个。候选实际新增原生评价11193次，其余38个标签复用缓存；包含教师与初始化的原生调用合计11259次。标签生成43.37秒，24轮训练11.91秒，程序内离线总耗时63.78秒。

训练排样为train_12、train_14、train_27、train_28、train_35、train_36；内部留出为train_04、train_29，集合无交叉。保持两层64维4头、学习率3e-4、随机种子20261126。24轮均完成，训练损失由0.693014降至0.477017，内部留出损失由0.691408降至0.560897，末轮留出Pairwise准确率67.5781%，selected_epoch=24。所有评分、损失和梯度有限，最大裁剪前梯度范数1.107428。逐轮记录完整保存在checkpoint的learning.full_gnn.history中。

唯一新增文件为[models/gnn_vns_br_atomic_16d_pilot.pt](models/gnn_vns_br_atomic_16d_pilot.pt)，123940字节。已通过当前load_model重新加载，模型全部参数逐张量完全恢复，在已生成的train_12图上输出240个有限评分，节点12维、边4维、动作16维。16项针对性测试及prepare均通过；原数据、旧模型与历史结果保持不变。标签只存内存，未生成额外CSV、日志或报告文件，未运行VALIDATION/TEST、完整GNN-VNS-BR搜索或HPC全量训练。小规模结果用于验证流程，不作为优秀泛化能力的结论。

## scale1000实测结果（历史B32终评）

以下数字属于旧评价口径（原生搜索＋B32终评），原始记录与数值保持不变，不代表当前统一原生口径的性能。本轮没有新的完整算法实验结果。

实际输入1000个不同TRAIN排样，生成2160个状态，其中2074个有效状态、515946个候选标签、530723对排序样本。三个模型共用一次标签，均完成24轮；离线总耗时2412.64秒。

|原始VALIDATION模型|B32可行/运行数|平均Cmax(s)|中位Cmax(s)|中位在线(s)|
|---|---:|---:|---:|---:|
|full_gnn|25/30|5855.50|5907.12|6.64|
|no_edge_gnn|23/30|5879.97|5937.20|7.79|
|action_mlp|25/30|5664.43|5666.02|10.19|

新VALIDATION：Full GNN的B32可行数为88/100，平均/中位Cmax为5839.28/5792.97秒，中位在线耗时5.70秒。
Cmax仅汇总B32可行结果，运行时间包含失败运行；全部记录保存在新CSV。
原始VALIDATION准入：未通过；已保存TEST记录0条。配对差异、初始改善、失败详情及训练记录见[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。
