# 双滑轨四机器人焊缝分配与排序

本项目联合决定父焊缝的合法拆分、四机器人归属、路线顺序和焊接方向，以 B32 调度返回的 Cmax 为优化目标。当前数据来自真实小组立排样，TRAIN、VALIDATION、TEST 分别包含 40、10、20 个排样和 3,749、936、2,060 条父焊缝。所有最终父焊缝的平面长度至少为 50 mm，TEST 的构件型号独立于训练侧。

## 正式模型

实验共用 `ScientificConfig()`、`FORMAL_SCOPE_V2` 和原 B32。每个 canonical 焊接块的加工时间为 `20 + L/0.0108 + 30` 秒，空走速度为 0.20 m/s，合法拆分的子焊缝最短 0.50 m。机器人的完成时间包括加工、相邻任务间空走和调度等待。

## 方法

- RULE/BEAM 用原单提案构造，作为快速基线。
- 正式 WAG+VNS 使用 `n=10`、`p=0.50`、500 轮；正式 HGA 使用显式方向、完整路线 M1–M6 VND、`μ=20/λ=10/α=20`、200,000 个有效子代。两者使用原生协调评价选择完整方案，最终各调用一次 B32。正式结果与历史短测、120 秒预算结果分开统计。
- 4R-SAEAX 从双亲有向邻接组装路线片段，联合选择机器人、插入位置和方向，搜索使用既有原生协调内核，结束后最多 3 次 B32 选优。采用 12 个体、每代 4 子代、最多 400 代、至少 80 代、连续 120 代停滞停止，没有墙钟硬截止。
- EG-GNN-PVND 第一阶段只学习 HGA 已访问动作的优先级与真实原生评价成本。20 个 TRAIN 轨迹中，15 个排样拟合、5 个排样留出；评分器为两层 hidden32/2头，共 14,938 参数，未重新训练原焊缝分配 GNN。
- GNN 的联合构造、父级重构及子焊缝反馈实验保留为历史记录，本轮没有继续训练旧构造网络。

EG-GNN-PVND 已完成教师采集、固定 12 轮训练和离线检验。留出观察池总体排序准确率 59.49%，首次原生改善评价数均值 15.33→6.72，但两个排样退化；真正优化已有可行 Cmax 的子集在计入决策开销后净变慢。**当前不值得进入阶段二，未接入 HGA 搜索，未跑 VALIDATION/TEST。** 详见[第一阶段报告](docs/GNN_WELD_LAYOUT_EXPERIMENT.md)，数据与模型分别保存在 [TRAIN教师](data/weld_layout/hga_vnd_teacher.jsonl.gz)和[评分器 checkpoint](models/eg_gnn_pvnd.pt)。

2026-11-17 已完成 3 个 TRAIN 预实验与 30 次 SAEAX VALIDATION：B32 可行 30/30，完整耗时中位 22.59 s；对 WAG/HGA 的共同可行 Cmax 配对中位变化为 **+0.445% / +1.202%**（正值更差），分别 6 胜 16 负与 0 胜 30 负。质量未达主算法要求，未启动 TEST。完整逐例结果、停止计数与失败分析见[报告第 14 节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#14-2026-11-174r-saeax-实现一次-train-预实验与完整-validation)及[唯一实验 CSV](results/saeax_validation.csv)。

BR-SAEAX 随后完成同种子的三个 TRAIN：最终 Cmax 为 4432.852/5707.055/6100.694 s，相对 SAEAX 为 2 胜 1 负、配对中位变化 -0.401%；B32 可行 3/3，完整时间中位 26.90 s，实际停止于 364/313/400 代。但最终 Pmax 为 0 降、1 平、2 升，**未通过 TRAIN，不继续 VALIDATION/TEST**。1071 次合法联合重插及 32 次三机器人接力没有建立稳定负载改善，不能宣称超过 WAG/HGA。三条结果追加到原 CSV，详见[报告第 15 节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#15-2026-11-17br-saeax-多父焊缝瓶颈联合重插入train未通过)。

LA-4R-EAX-NSGA-II 已按用户提供的 Gao 论文第4.4节及图9实现独立增强邻接表交叉、四机器人联合归属/顺序/方向构造和三目标 NSGA-II。20个体、每代10子代、最多500代、120代停滞、最终一次B32的三个 TRAIN 已完成：B32可行2/3，两个共同可行配对均落后于SAEAX，Cmax变化中位 **+35.121%**；完整时间中位22.52 s，停止于427/133/330代。8900个有效子代仅73个原生可行，**TRAIN未通过，未进入VALIDATION/TEST**。实现见 [eax_nsga2.py](src/mrta_search/eax_nsga2.py)，入口见 [run_eax_nsga2.py](scripts/run_eax_nsga2.py)，详细负结果及命令见[报告第16节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#16-2026-11-17la-4r-eax-nsga-ii全文核对实现及train筛查负结果)。三条结果继续追加到原CSV，全部历史记录保留。

BC-ECS 已完成本次运行内实际生成 SAEAX 初解的3个 TRAIN及30次 VALIDATION，未读取历史最优解。TRAIN为3/3可行、对SAEAX 2胜1平，时间中位30.49 s，通过筛查；VALIDATION为30/30可行、对SAEAX 19胜11平，但对WAG共同可行22对为7胜15负、Cmax变化中位 **+0.0969%**，对HGA为0胜30负、**+1.0829%**。完整时间中位31.65 s，包含初始化、逐出链和全部B32。**质量未达标，本轮停止，未启动TEST或继续调参。** 实现见 [bc_ecs.py](src/mrta_search/bc_ecs.py)，独立入口见 [run_bc_ecs.py](scripts/run_bc_ecs.py)，33条结果追加到原CSV，调用计数、停止原因及全部逐例数据见[报告第17节](docs/WAG_HGA_BASELINE_REPORT_20261111.md#17-2026-11-18bc-ecs-实现train通过及完整validation负结果)。

## 运行

在 `D:\pybullet_test\MRTA_GA\DRL` 中执行。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -X utf8 scripts/run_eg_gnn_pvnd.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_eg_gnn_pvnd.py collect --workers 4
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_eg_gnn_pvnd.py fit
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py summary
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py pilot --workers 1
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -u -X utf8 scripts/run_saeax.py validation --workers 3
```

EG 入口在已有20个教师排样及模型时直接读取，不重新采集或训练。SAEAX 入口在已有3次预实验及30次验证时直接读取，不重复搜索或B32。旧GNN与对照入口保留在原脚本中。

## 新数据实验（早期记录）

以下为历史阶段结果；当前正式对照及 SAEAX 结论以上述报告为准。

上一轮新 TEST 上，RULE/BEAM、原始 GNN_VNEXT 和负载感知 GNN_VNEXT 的单提案 B32 可行数分别为 16/20、17/20、16/20。可行 Cmax 中位分别为 6,901.0、8,733.4、8,053.3 秒。详细结果与本轮实验进度见 [新排样实验报告](docs/GNN_WELD_LAYOUT_EXPERIMENT.md)。

完整流程设置下，两种联合 GNN 各完成 20 个 TEST 排样和三个种子的 60 次运行，均为 48/60 B32 可行。可行 Cmax 中位分别为 7,319.3 s、7,224.6 s，总耗时中位分别为 9.14 s、9.84 s。重构版相对 RULE/BEAM 的配对 Cmax 中位高 4.34%。当前汇总使用完整流程结果。

此前 60 秒多方案搜索中，两种联合 GNN 各完成 60 次 TEST，均为 58/60 B32 可行，可行 Cmax 中位分别为 7,172.3 s 和 7,143.1 s。带瓶颈重构版本在共同可行样本上相对负载感知 VNext 的配对 Cmax 中位下降 10.89%，相对 RULE/BEAM 高 3.49%。这些记录保留在 `gnn_results.csv` 中。HGA 论文评分版完成前四个排样的 12 次运行，最终 B32 可行 0/12。WAG+VNS 已暂停，保留两个排样各三个种子的 500 轮结果，B32 可行 5/6。[本轮已完成结果](results/new_dataset_final/final_comparison.csv)
