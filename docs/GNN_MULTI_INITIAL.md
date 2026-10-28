# GNN 四候选联合初解实验

实际结论：V2从Greedy 1/6到Multi4仍为1/6；Shared从1/6增至3/6，救活g21、g33，并降低g15的Cmax。少量采样可以挽救部分Shared初解，但尚未获得Beam的6/6可靠性。后续可靠求解主线采用 **Beam联合初解 + B32协调调度 + GNN V2瓶颈重构**；Shared-Multi4保留为有限研究分支，本轮不追加采样或训练。

冻结原V2与Shared checkpoint，model.eval() + torch.inference_mode()；无训练或归一化更新。seed=20261027，候选1为原greedy，候选2—4采样种子为20261028/20261029/20261030；父选择与联合插入均直接使用softmax(logits)，温度固定1，insertion_limit=8。
全部神经候选从纯几何、固定合法模式和四条空路线开始；保持原子子块插入及同一步邻块方向翻转。每个完成且不同的路线+方向只调用一次现有B32；重复候选复用结果，四次解码之后不再补采。Greedy行直接复用Multi4的第一个候选，并非另外再运行一条。

## TRAIN 正确性运行

从已有TRAIN构造数据按N选择15/53/90三个规模，仍通过原MLP_TRAIN白名单加载PPO原几何。所有完成候选检查父焊缝覆盖、子块原子性及方向长度，随后实际调用B32。不同路线数同时考虑显式方向。

|TRAIN几何|方法|可调度|最佳Cmax/s|B32调用|构造+B32秒|不同完整路线/方向|
|---|---|---|---:|---:|---:|---:|
|data/PPO_TRAIN/seed_0377964238::g15_w015|GNN-V2-Greedy|DEADLOCK|—|1|0.109|1|
|data/PPO_TRAIN/seed_0377964238::g15_w015|GNN-V2-Multi4|FEASIBLE|801.220|4|0.456|4|
|data/PPO_TRAIN/seed_0377964238::g15_w015|GNN-Shared-Greedy|DEADLOCK|—|1|0.148|1|
|data/PPO_TRAIN/seed_0377964238::g15_w015|GNN-Shared-Multi4|FEASIBLE|717.433|4|0.385|4|
|data/DEV_ONLY/seed_000404::g35_w053|GNN-V2-Greedy|DEADLOCK|—|1|0.829|1|
|data/DEV_ONLY/seed_000404::g35_w053|GNN-V2-Multi4|FAILED|—|4|4.359|4|
|data/DEV_ONLY/seed_000404::g35_w053|GNN-Shared-Greedy|DEADLOCK|—|1|1.463|1|
|data/DEV_ONLY/seed_000404::g35_w053|GNN-Shared-Multi4|FAILED|—|4|3.686|4|
|data/PPO_TRAIN/seed_1742672780::g42_w090|GNN-V2-Greedy|DEADLOCK|—|1|2.166|1|
|data/PPO_TRAIN/seed_1742672780::g42_w090|GNN-V2-Multi4|FAILED|—|4|7.871|4|
|data/PPO_TRAIN/seed_1742672780::g42_w090|GNN-Shared-Greedy|DEADLOCK|—|1|2.845|1|
|data/PPO_TRAIN/seed_1742672780::g42_w090|GNN-Shared-Multi4|FAILED|—|4|8.905|4|

## 六 DEV 结果

模型统一预加载，Excel读取不计时；计时包含几何模式/图、四次神经构造和实际B32。Beam保留现有实现，其4或8次B32预算如实列出。Beam在神经候选完成后独立运行，归属、路线与方向均未提供给GNN。不同预算不能称为严格等成本比较。

|方法|至少一条可调度/实例数|B32调用合计|构造+B32合计秒|
|---|---:|---:|---:|
|Beam-Init|6/6|32|17.314|
|GNN-V2-Greedy|1/6|6|6.743|
|GNN-V2-Multi4|1/6|24|24.847|
|GNN-Shared-Greedy|1/6|6|7.429|
|GNN-Shared-Multi4|3/6|24|28.492|

|实例|方法|结果|最佳真实Cmax/s|B32调用|构造+B32秒|不同完整路线/方向|
|---|---|---|---:|---:|---:|---:|
|g21_w035|GNN-V2-Greedy|FEASIBLE|2557.409|1|0.237|1|
|g21_w035|GNN-V2-Multi4|FEASIBLE|2557.409|4|1.857|4|
|g21_w035|GNN-Shared-Greedy|DEADLOCK|—|1|1.006|1|
|g21_w035|GNN-Shared-Multi4|FEASIBLE|2513.630|4|1.959|4|
|g21_w035|Beam-Init|FEASIBLE|2690.964|8|0.946|—|
|g15_w035|GNN-V2-Greedy|DEADLOCK|—|1|0.566|1|
|g15_w035|GNN-V2-Multi4|FAILED|—|4|1.752|4|
|g15_w035|GNN-Shared-Greedy|FEASIBLE|2543.703|1|0.361|1|
|g15_w035|GNN-Shared-Multi4|FEASIBLE|1865.463|4|1.191|4|
|g15_w035|Beam-Init|FEASIBLE|1653.687|4|1.240|—|
|g36_w086|GNN-V2-Greedy|DEADLOCK|—|1|1.745|1|
|g36_w086|GNN-V2-Multi4|FAILED|—|4|6.629|4|
|g36_w086|GNN-Shared-Greedy|DEADLOCK|—|1|1.720|1|
|g36_w086|GNN-Shared-Multi4|FAILED|—|4|7.584|4|
|g36_w086|Beam-Init|FEASIBLE|3869.494|4|5.925|—|
|g33_w055|GNN-V2-Greedy|DEADLOCK|—|1|0.992|1|
|g33_w055|GNN-V2-Multi4|FAILED|—|4|3.732|4|
|g33_w055|GNN-Shared-Greedy|DEADLOCK|—|1|0.785|1|
|g33_w055|GNN-Shared-Multi4|FEASIBLE|4475.523|4|3.641|4|
|g33_w055|Beam-Init|FEASIBLE|4023.515|8|2.226|—|
|g31_w055|GNN-V2-Greedy|DEADLOCK|—|1|0.969|1|
|g31_w055|GNN-V2-Multi4|FAILED|—|4|3.818|4|
|g31_w055|GNN-Shared-Greedy|DEADLOCK|—|1|1.988|1|
|g31_w055|GNN-Shared-Multi4|FAILED|—|4|4.971|4|
|g31_w055|Beam-Init|FEASIBLE|3126.807|4|5.027|—|
|g45_w082|GNN-V2-Greedy|DEADLOCK|—|1|2.234|1|
|g45_w082|GNN-V2-Multi4|FAILED|—|4|7.059|4|
|g45_w082|GNN-Shared-Greedy|DEADLOCK|—|1|1.569|1|
|g45_w082|GNN-Shared-Multi4|FAILED|—|4|9.146|4|
|g45_w082|Beam-Init|FEASIBLE|5447.518|4|1.950|—|

DEADLOCK只表示现有B32在该完整方案与方向下未找到调度，不是数学不可行证明。失败无虚构Cmax，无Beam或Regret修复/回退。

## 判断

- GNN-V2：Greedy 1/6 → Multi4 1/6；新增可调度实例 无；原Greedy已可调度实例中进一步降低Cmax 无。实际不同完整路线/方向总数 24/24。
- GNN-Shared：Greedy 1/6 → Multi4 3/6；新增可调度实例 ['g21_w035', 'g33_w055']；原Greedy已可调度实例中进一步降低Cmax ['g15_w035']。实际不同完整路线/方向总数 24/24。

本轮固定四候选，未继续增加采样、调整温度、改变模型或进行强化学习。研究路线按本次真实成功数、Cmax与成本作出判断。

- **是否更容易可调度：Shared有提高，V2没有。** Shared救活了原先Greedy失败的g21、g33，成功数1/6→3/6；g36、g31、g45仍四条全部DEADLOCK。V2没有新增成功。两个模型在六DEV均得到24/24条不同完整路线/方向；所有候选都完成父覆盖及显式方向检查，采样并非重复Greedy。
- **哪个模型适合纯初解：Shared-Multi4优于V2-Multi4的成功覆盖。** Shared在三个成功实例中，g21的2513.630秒优于Beam的2690.964秒；g15的1865.463秒、g33的4475.523秒均差于各自Beam的1653.687秒、4023.515秒。另外三例失败仍保留在分母，不能只比较共同成功例而宣称全面优于Beam。
- **是否改善真实Cmax：有局部实证。** Shared在原Greedy可调度的g15由2543.703降到1865.463秒，下降678.240秒（26.66%）；g21、g33从失败变为可调度。V2原成功g21仍为2557.409秒，没有改善。Multi4均保留自己的Greedy候选，没有发生可调度性或最佳Cmax退步。
- **额外成本是否值得：对挽救Shared有研究价值，对稳定替代Beam尚不值得。** V2由6次B32/6.743秒增加到24次/24.847秒，却无成功数或最佳Cmax收益。Shared由6次/7.429秒增加到24次/28.492秒，获得上述两例新增成功及一例Cmax改善；但Beam为32次/17.314秒、6/6成功。不同预算与方案导致的调度耗时不同，不能称作严格等成本优越性结论。
- **下一步路线：主线选Beam+B32+GNN V2。** 停止对V2纯初解继续盲目训练或调参。Shared-Multi4已显示多条神经构造路径能够救活两个失败实例，因此保留接口和结果作为受限研究分支；3/6成功尚不足以替代可靠初解器，不把继续扩大候选、温度搜索或更换模型作为下一步。若后续专门研究纯神经初解的终局反馈，应另行明确范围；当前优先在已有可调度解上研究神经联合重构与真实Cmax反馈。

运行核对：采用 `D:\pybullet_test\.venv\Scripts\python.exe`，工作目录 `DRL`，执行 `scripts/run_gnn_multi_initial.py --seed 20261027`。34项相关测试通过，包含两头softmax采样、局部种子复现、Greedy不变、子块原子性、结构检查及重复调度缓存。六DEV的Greedy状态和Cmax逐一复现上一轮CSV；运行前后两模型全部权重与归一化张量一致。三个TRAIN共24次神经B32，六DEV共48次神经B32，Beam另32次，总计104次；Greedy行是Multi4子集，不重复计入实际调用。原checkpoint、CSV、报告均保留，仅新增已确认的三个文件。
