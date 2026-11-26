# 双滑轨四机器人焊缝分配与排序

当前研究路线：**固定拆分 → 四机器人任务分配 → 连续原子任务合并 → 焊缝排序及方向 → 调度评价**。

| 方法 | 定位 |
|---|---|
| Adapted-WAG+VNS | 独立文献对照组，保留既有210次正式结果 |
| Adapted-HGA | 独立文献对照组，保留既有210次正式结果 |
| GNN-VNS-BR | 本项目方法：监督学习的GNN候选排序与变邻域瓶颈重构 |

固定任务集GNN-PPO的210次失败实验保留在原CSV和报告中，不再作为主算法，也不继续训练。

## 固定任务域与调度

[src/mrta_reference/fixed_taskset.py](src/mrta_reference/fixed_taskset.py)是唯一切分来源。先按整条父焊缝的轨道可达性确定责任区域，无法由单侧承担时在共享带固定切分；区域段欧氏长度L>5m时使用n=ceil(L)-1等分，新生成子段严格大于1m。优化过程中不增删或移动切分点。

以固定原子任务确定机器人归属；同父焊缝中由同机器人承担的连续原子任务，在排序与评价前强制合并为实际作业。排序和方向作用于这些合并作业。原始TRAIN/VALIDATION/TEST分别有40/10/20个排样，共6446/1671/3201个固定原子任务。

搜索使用原有priority_schedule数值实现；每次最终方案仅调用一次原reference_schedule_formal。FORMAL_SCOPE_V2的B32 limited-discrepancy死锁恢复、同轨顺序及跨轨干涉政策保持原定义。历史scope字段不决定固定任务集边界。

## GNN-VNS-BR

[算法](src/mrta_search/gnn_vns_br.py)独立执行贪心初始化和搜索，不调用WAG/HGA求解器、种群进化或完整VND。由实际快速调度的四机器人完工时间确定瓶颈；不可行初始状态仅使用路线代价辅助寻求可行状态。

瓶颈局部图包含合并作业、相关近邻和四机器人节点。共享两层64维、4头GAT使用16维节点输入：作业节点仅保留端点、长度、路线位置与方向，机器人节点保留身份和负载状态。3维边输入为可执行关系、当前归属和实际方向对应的空走时间；每条作业关联全部四个机器人，资格直接取自固定任务矩阵，正反向转移分别计算。每层用一个线性映射把连续边特征加入注意力。

有向空走边读取当前方向对应的w.travel，并除以workspace.scale，与机器人时间状态使用同一归一化尺度；空间近邻按有向转移时间选取。

候选按邻域、目标机器人、锚点分配有限预算，取消随机截断和按局部代价淘汰半数候选。GNN对M1–M6统一评分，每个邻域最多3个候选进入原生评价，再从全部六邻域选择实际Cmax最小的可行改善。方向随动作处理，有限扰动始终保留全局可行最优。

M1也可沿已有原子边界迁移合并作业的端部原子，再按新归属强制合并，避免把长作业永久锁成不可调整的分配单位。若M5/M6没有可行扰动，有限扰动预算可用于GNN引导的M1/M2；不会增加切分点或接受不可行方案。

监督状态来自已完成且B32可行的TRAIN对照方案、独立贪心初始化及少量合法扰动。训练和推理共用同一候选与图构造逻辑，alpha=8、anchors=12、每邻域最多40候选。标签使用实际快速调度Cmax及pairwise ranking loss；按完整TRAIN排样留出，不使用VALIDATION/TEST标签。

新模型保存为models/gnn_vns_br_edgeaware.pt。旧23维模型models/gnn_vns_br.pt及其60条VALIDATION结果已保存并保留；新模型不加载旧输入结构的权重。

## 数据、结果与执行

- 原始排样：data/weld_layout/train.jsonl、validation.jsonl、test.jsonl。
- 固定任务及原始结果：[results/fixed_euclidean_split/](results/fixed_euclidean_split/)。
- 新方法统一入口：[scripts/run_gnn_vns_br.py](scripts/run_gnn_vns_br.py)。
- 完整实测、配对Cmax变化及失败记录：[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。

在DRL目录使用指定环境：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py smoke
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py summary
```

本轮仅做TRAIN训练，以及train_01、train_12、train_30各一个种子的Smoke，结果写入results/new_dataset_final/gnn_vns_br_edgeaware.csv；同键已有结果只读复用。已有WAG/HGA、GNN-PPO和旧VALIDATION不重跑，本轮不执行完整VALIDATION或TEST。

动态拆分的激活、取消与切分点切换已从候选执行、生成及活动算子中移除；相关旧动态GNN搜索入口停用，必要的摘要/审计、历史读取、几何校验和WAG/HGA共享函数保留。具体删除与修改文件见现有实验报告的本地清理清单。B32 Scope历史字段保持原定义，不能用于改变固定原子边界。

在线时间包含初始化、GNN推理、候选评价及最终B32；离线样本生成和训练时间单列。新增文件均位于DRL分类目录，单文件小于100MB；不生成训练样本副本、额外调度器、hash、manifest或实验身份体系。
