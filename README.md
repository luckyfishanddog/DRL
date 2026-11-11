# 双滑轨四机器人焊缝分配与排序

本项目联合决定父焊缝的合法拆分、四机器人归属、路线顺序和焊接方向，以 B32 调度返回的 Cmax 为优化目标。当前数据来自真实小组立排样，TRAIN、VALIDATION、TEST 分别包含 40、10、20 个排样和 3,749、936、2,060 条父焊缝。所有最终父焊缝的平面长度至少为 50 mm，TEST 的构件型号独立于训练侧。

## 正式模型

实验共用 `ScientificConfig()`、`FORMAL_SCOPE_V2` 和原 B32。每个 canonical 焊接块的加工时间为 `20 + L/0.0108 + 30` 秒，空走速度为 0.20 m/s，合法拆分的子焊缝最短 0.50 m。机器人的完成时间包括加工、相邻任务间空走和调度等待。

## 方法

- RULE/BEAM 用原单提案构造，作为快速基线。
- WAG+VNS 保留 `n=10`、`p=0.50`、`Imax=500`，用各轨的固定优先级顺序调度计分。HGA 用固定作业区的最长路线周期时间计分，执行 30 次子代迭代，范围为 TEST 前四个排样。两者的 5、30、60 秒观测值为各自搜索目标，选出的最终方案各评价一次 B32。
- 改进 GNN 在至多 8 个联合分配候选上先执行有向插入，再比较实际局部负载、空走和协调风险。TRAIN 使用 RULE/BEAM、WAG+VNS、HGA 教师及同前缀 B32 偏好。每组实验完成一次联合构造，重构版最多继续两轮瓶颈重构，记录完整流程的实际耗时。5、30、60 秒列保留时间观测值。

## 运行

在 `D:\pybullet_test\MRTA_GA\DRL` 中执行。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_weld_layout.py baselines --method HGA
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_weld_layout.py baselines --method 'WAG+VNS'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_weld_layout.py train
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_weld_layout.py test
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_gnn_weld_layout.py summary
```

## 新数据实验

上一轮新 TEST 上，RULE/BEAM、原始 GNN_VNEXT 和负载感知 GNN_VNEXT 的单提案 B32 可行数分别为 16/20、17/20、16/20。可行 Cmax 中位分别为 6,901.0、8,733.4、8,053.3 秒。详细结果与本轮实验进度见 [新排样实验报告](docs/GNN_WELD_LAYOUT_EXPERIMENT.md)。

完整流程设置下，两种联合 GNN 各完成 20 个 TEST 排样和三个种子的 60 次运行，均为 48/60 B32 可行。可行 Cmax 中位分别为 7,319.3 s、7,224.6 s，总耗时中位分别为 9.14 s、9.84 s。重构版相对 RULE/BEAM 的配对 Cmax 中位高 4.34%。当前汇总使用完整流程结果。

此前 60 秒多方案搜索中，两种联合 GNN 各完成 60 次 TEST，均为 58/60 B32 可行，可行 Cmax 中位分别为 7,172.3 s 和 7,143.1 s。带瓶颈重构版本在共同可行样本上相对负载感知 VNext 的配对 Cmax 中位下降 10.89%，相对 RULE/BEAM 高 3.49%。这些记录保留在 `gnn_results.csv` 中。HGA 论文评分版完成前四个排样的 12 次运行，最终 B32 可行 0/12。WAG+VNS 已暂停，保留两个排样各三个种子的 500 轮结果，B32 可行 5/6。[本轮已完成结果](results/new_dataset_final/final_comparison.csv)
