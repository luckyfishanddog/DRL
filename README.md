# 双滑轨四机器人焊缝分配与排序

当前研究路线：**固定拆分 → 四机器人任务分配 → 连续原子任务合并 → 焊缝排序及方向 → 调度评价**。

| 方法 | 定位 |
|---|---|
| Adapted-WAG+VNS | 独立文献对照组，保留既有210次旧口径正式结果 |
| Adapted-HGA | 独立文献对照组，保留既有210次旧口径正式结果 |
| GNN-VNS-BR | 本项目方法：监督学习的GNN候选排序与变邻域瓶颈重构 |

固定任务集GNN-PPO的210次失败实验保留在原CSV和报告中，不再作为主算法，也不继续训练。

## 固定任务域与调度

[src/mrta_reference/fixed_taskset.py](src/mrta_reference/fixed_taskset.py)是唯一切分来源。先按整条父焊缝的轨道可达性确定责任区域，无法由单侧承担时在共享带固定切分；区域段欧氏长度L>5m时使用n=ceil(L)-1等分，新生成子段严格大于1m。优化过程中不增删或移动切分点。

以固定原子任务确定机器人归属；同父焊缝中由同机器人承担的连续原子任务，在排序与评价前强制合并为实际作业。排序和方向作用于这些合并作业。原始TRAIN/VALIDATION/TEST分别有40/10/20个排样，共6446/1671/3201个固定原子任务。

当前正式评价统一使用原有Numba `_priority_arrays()` 内核。公共接口为 `native_schedule()` → `priority_schedule()` → `_priority_arrays()`；GNN搜索直接调用数组接口。WAG/HGA复用已选中的原生ScheduleResult，GNN最终封装已保存的数组，不再重复调度。可行方案的Cmax为四台机器人实际完工时间的最大值，等待时间计入完工时间。

B32正式入口、bounded/limited-discrepancy恢复及其预算和状态字段已删除。同轨双优先级、方向组合、跨轨干涉、准备/处理时间和连续原子合并规则保持不变。DEADLOCK只表示当前固定调度规则未完成派工，不证明物理上绝对不可行。通用Python参考调度器仅供独立几何工具、exact/oracle诊断和测试，不是当前实验的第二条正式评价路径。

## GNN-VNS-BR

[算法](src/mrta_search/gnn_vns_br.py)独立执行贪心初始化和搜索，不调用WAG/HGA求解器、种群进化或完整VND。由实际快速调度的四机器人完工时间确定瓶颈；尚未找到原生可行派工的初始状态仅使用路线代价辅助寻求可行状态。

瓶颈局部图包含合并作业、相关近邻和四机器人节点。共享两层64维、4头GAT使用16维节点输入：作业节点仅保留端点、长度、路线位置与方向，机器人节点保留身份和负载状态。3维边输入为可执行关系、当前归属和实际方向对应的空走时间；每条作业关联全部四个机器人，资格直接取自固定任务矩阵，正反向转移分别计算。每层用一个线性映射把连续边特征加入注意力。

有向空走边读取当前方向对应的w.travel，并除以workspace.scale，与机器人时间状态使用同一归一化尺度；空间近邻按有向转移时间选取。

候选按邻域、目标机器人、锚点分配有限预算，取消随机截断和按局部代价淘汰半数候选。GNN对M1–M6统一评分，每个邻域最多3个候选进入原生评价，再从全部六邻域选择实际Cmax最小的可行改善。方向随动作处理，有限扰动始终保留全局可行最优。

M1也可沿已有原子边界迁移合并作业的端部原子，再按新归属强制合并，避免把长作业永久锁成不可调整的分配单位。若M5/M6没有可行扰动，有限扰动预算可用于GNN引导的M1/M2；不会增加切分点或接受不可行方案。

监督状态可复用已保存且原生Cmax有限的TRAIN教师路线、独立贪心初始化及少量合法扰动；恢复教师路线后仍由同一原生内核生成标签，不依赖历史B32状态或终评值。训练和推理共用同一候选与图构造逻辑，alpha=8、anchors=12、每邻域最多40候选。标签使用实际原生Cmax及pairwise ranking loss；两个DEADLOCK之间不构造Cmax优劣，NUMERIC_FAILURE不生成排序对，不使用VALIDATION/TEST标签。本轮未对真实数据集生成训练标签或重新训练。

此前40个TRAIN排样训练的边特征模型models/gnn_vns_br_edgeaware.pt及3条Smoke保留；旧23维模型models/gnn_vns_br.pt及其60条VALIDATION结果也保留，三阶段结果分开记录。

历史scale1000实验在同一候选、图及Pairwise标签上分别训练Full GNN、屏蔽3维边特征的同结构GNN和仅输入18维动作特征的轻量MLP。三个模型及实际训练记录集中保存为models/gnn_vns_br_edgeaware_scale1000.pt。24轮、随机种子20261126和学习率0.0003的历史训练记录保留；本轮不修改网络、16/3/18特征或合并作业图。

## 数据、结果与执行

- 原始排样：data/weld_layout/train.jsonl、validation.jsonl、test.jsonl。
- 固定任务及原始结果：[results/fixed_euclidean_split/](results/fixed_euclidean_split/)。
- 新方法统一入口：[scripts/run_gnn_vns_br.py](scripts/run_gnn_vns_br.py)。
- 完整实测、配对Cmax变化及失败记录：[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。
- 本轮删除清单、统一调用关系与针对性测试：[原生Cmax重构报告](docs/UNIFIED_NATIVE_CMAX_REPORT.md)。

以下为保留的完整实验入口说明，使用指定环境；本轮没有执行这些训练或正式实验命令：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train --dataset scale1000
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py validation --dataset scale1000
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py test --dataset scale1000
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py summary --dataset scale1000
```

上述入口默认使用--dataset scale1000。现有scale1000 JSONL为1000/100/100条；TRAIN已完整包含原始40个排样，经用户确认采用1000个不同排样（原始40＋新增960），不重复计为1040个。逐条读取并通过原generate_taskset转换；保留原始40个教师方案，不重新生成数据，也不为新增960个运行WAG/HGA。

历史VALIDATION矩阵为原始10排样×3种子×3模型，以及新100排样×1种子的Full GNN。旧B32终评不能与新原生结果配对。后续原生口径输出路径为results/native_priority/fixed_euclidean_split/与results/native_priority/new_dataset_final/，按数据、排样、种子和模型复用完整记录，包括完整失败记录；本轮未创建这些结果CSV。既有准入判断只读取新口径结果，不将历史B32基线混入比较。

既有WAG/HGA、GNN-PPO和旧23维VALIDATION不重跑。此前3条TRAIN Smoke仍保存在results/new_dataset_final/gnn_vns_br_edgeaware.csv；历史数据直接查阅旧CSV和既有实验报告。当前summary打印原生路径中的汇总，不重写历史报告。

动态拆分的激活、取消与切分点切换已从候选执行、生成及活动算子中移除；相关旧动态GNN搜索入口停用，必要的历史读取、几何校验和WAG/HGA共享函数保留。Scope仅移除恢复政策元数据，现有几何校验继续保留；它不能用于改变固定原子边界。

当前在线时间包含初始化、GNN推理、原生候选评价和结果封装，最终不再追加B32。离线样本生成和训练时间单列。新增文件均位于DRL分类目录，单文件小于100MB；不生成训练样本副本、额外调度器、hash、manifest或实验身份体系。

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
