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

瓶颈局部图包含合并作业、相关近邻和四机器人节点。共享两层64维、4头GAT编码物理特征，一个标量动作头对M1–M6评分。路线动作同时处理方向，Top-K候选进入实际调度评价；严格改善时接受，有限M5/M6扰动后继续，始终保留全局可行最优解。

M1也可沿已有原子边界迁移合并作业的端部原子，再按新归属强制合并，避免把长作业永久锁成不可调整的分配单位。若M5/M6没有可行扰动，有限扰动预算可用于GNN引导的M1/M2；不会增加切分点或接受不可行方案。

监督样本仅来自已完成且B32可行的TRAIN对照方案及少量合法扰动。候选标签由实际快速调度生成，采用pairwise ranking loss，不可行候选排在可行候选之后。训练样本在内存中生成，唯一模型权重归类保存在models/gnn_vns_br.pt。VALIDATION/TEST均独立初始化，不使用对应对照最终解。

## 数据、结果与执行

- 原始排样：data/weld_layout/train.jsonl、validation.jsonl、test.jsonl。
- 固定任务及原始结果：[results/fixed_euclidean_split/](results/fixed_euclidean_split/)。
- 新方法统一入口：[scripts/run_gnn_vns_br.py](scripts/run_gnn_vns_br.py)。
- 完整实测、配对Cmax变化及失败记录：[现有实验报告](docs/FIXED_EUCLIDEAN_SPLIT_EXPERIMENT.md)。

在DRL目录使用指定环境：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py train
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py smoke --layout train_01
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py validation
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_vns_br.py test
```

已有WAG/HGA和GNN-PPO实验不重跑。VALIDATION共10排样×3种子；至少28/30可行、配对优于WAG、质量与HGA具有竞争力且中位耗时不超过WAG的1.5倍时，才使用选定模型执行一次20排样×3种子的TEST。当前实现将“与HGA具有竞争力”明确为配对Cmax中位退化不超过2%。

在线时间包含初始化、GNN推理、候选评价及最终B32；离线样本生成和训练时间单列。新增文件均位于DRL分类目录，单文件小于100MB；不生成训练样本副本、额外调度器、hash、manifest或实验身份体系。
