# HIGH_WORKLOAD_04失败根因分析

日期：2026-11-04。本报告仅依据现有CSV及当前代码进行只读追踪，没有运行候选构造、B32、搜索、几何生成或模型训练，也未修改正式算法及原始结果。

## 1. 结论与证据边界

可以确认的流程原因是：每个种子的共同初始化菜单完成8次完整B32候选评价后，没有获得可行入口，MODE_NEW和FIXED_NEW随即退出，父模式调整与联合修复搜索没有启动。五个种子共执行五组共享初始化；两个方法各记8次属于等额计费，不能把10条失败行解释成80次相互独立的B32评价。按照任务分组代码，实际共同初始化调用为5×8=40次完整候选B32调用，B32内部恢复rollout不能另当成完整候选数。

具体几何冲突的根因仍无法由现有记录确定。失败行的details只有四个字段：status、cmax、actual_runtime、reference_calls。没有保留8个候选的分配、路线、方向、逐候选状态或阻塞操作。因此不能写成“已证明同轨死锁”“方向错误”或“负载失衡导致失败”。

已经确认一处状态分类缺陷：包装函数在“initial为空”与“耗时超预算”之间共用TIME_LIMIT_NO_FEASIBLE标签。这里实际耗时不足60秒，故原始标签不能解释为时间预算耗尽。有限菜单耗尽后提前退出是现有控制流，未发现预算时钟计算错误的证据。

## 2. 唯一几何与数据来源

|项目|值|
|---|---|
|唯一正式数据来源|[final_parent_mode_joint_reconstruction_05.csv](../results/final_parent_mode_joint_reconstruction_05.csv)|
|本报告实例名称|HIGH_WORKLOAD_04|
|几何种子|2026110512|
|父焊缝数量|92|
|总长度|135.22251660943434 m（显示为135.222517 m）|
|强制拆分父数量|1|
|装配组数量|45|
|配置|min_child_length=0.50 m，delta_x=delta_y=0.50 m，FORMAL_SCOPE_V2，现有B32|
|评价预算|60 s|
|求解种子|202611041、202611042、202611043、202611044、202611045|

诊断范围是此几何的10条失败行和15条成功对照行。几何元数据来自同一CSV的instance记录。历史[FINAL_COORDINATION_COMPARISON.md](FINAL_COORDINATION_COMPARISON.md)中同名HIGH_WORKLOAD_04是其他几何，不用于本报告。30秒开发集的N55_01等案例也不用于本次失败归因。

## 3. 十条失败运行

CSV行号含表头；所有失败Cmax均保持原始空值。每个种子两个方法的耗时完全相同，符合共享共同初始化的代码路径。

|方法|种子|CSV行|原始状态|B32计次|耗时/s|Cmax/s|
|---|---|---|---|---|---|---|
|FIXED_NEW|202611041|472|TIME_LIMIT_NO_FEASIBLE|8|27.755969|空值|
|MODE_NEW|202611041|471|TIME_LIMIT_NO_FEASIBLE|8|27.755969|空值|
|FIXED_NEW|202611042|477|TIME_LIMIT_NO_FEASIBLE|8|27.527932|空值|
|MODE_NEW|202611042|476|TIME_LIMIT_NO_FEASIBLE|8|27.527932|空值|
|FIXED_NEW|202611043|482|TIME_LIMIT_NO_FEASIBLE|8|27.607116|空值|
|MODE_NEW|202611043|481|TIME_LIMIT_NO_FEASIBLE|8|27.607116|空值|
|FIXED_NEW|202611044|487|TIME_LIMIT_NO_FEASIBLE|8|27.328573|空值|
|MODE_NEW|202611044|486|TIME_LIMIT_NO_FEASIBLE|8|27.328573|空值|
|FIXED_NEW|202611045|492|TIME_LIMIT_NO_FEASIBLE|8|31.943908|空值|
|MODE_NEW|202611045|491|TIME_LIMIT_NO_FEASIBLE|8|31.943908|空值|

每组尚有约28.056–32.671秒预算，但没有可供局部修复的current解。这里的“未用完预算”是正常提前退出，不应写成60秒到期。MODE_NEW尚未提出模式调整，FIXED_NEW也尚未进行联合修复；这10次失败不能用于判定哪个搜索算子更差。

## 4. 八次共同初解评价：能确定什么

入口是[common_initial_beam](../scripts/run_joint_oriented_repair_pilot.py#L152)。正式包装以interleave=True调用，未设置阶段1专用的max_reference_calls=4，因此最多遍历下列8个槽位。以下来源和次序是代码结合8次计数的推导，**不是补造的逐候选日志**。

|候选槽位（代码次序）|来源|Beam下标|能够确定的调度结论|未保存的具体内容|
|---|---|---|---|---|
|1|REGION_LOCAL|0|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|2|RULE_X_LOCAL|0|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|3|REGION_LOCAL|1|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|4|RULE_X_LOCAL|1|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|5|REGION_LOCAL|2|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|6|RULE_X_LOCAL|2|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|7|REGION_LOCAL|3|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|
|8|RULE_X_LOCAL|3|未返回可接纳的FEASIBLE结果|状态枚举、路线、方向、四机器人数量/长度/负载、阻塞轨迹|

该函数对每个槽位最多调用一次schedule_explicit，再记录返回的ScheduleStatus。正式包装的reference_calls是attempts中FEASIBLE/DEADLOCK/INFEASIBLE/NUMERIC_FAILURE枚举状态的条目数。八个槽位、计数为8且elapsed<60说明：两个参考构造均产生了可进入调度的四个Beam状态，8个结果均未成为可行入口。若全部只是几何/方向候选构造失败，此计数不可能达到8。

不能进一步断言8个结果全部为DEADLOCK：非FEASIBLE枚举还包括INFEASIBLE及NUMERIC_FAILURE，原始行没有保存其具体分布。也不能排除某个候选的B32数值失败；没有EXECUTION_ERROR记录只说明未进入包装函数的异常输出路径。

### 候选结构的代码事实

- REGION_LOCAL调用区域协同参考构造，规则上先选可行WHOLE，否则选一个合法强制Y拆分；任务在可达轨间按加工负载与区域边界代理分配，同轨边界同时考虑少量空走及X重叠代理。此处显式设置improve_routes=False，并传入局部方向插入成本，不启动独立方向优化。
- RULE_X_LOCAL实际调用construct_solution(..., model=None)，使用WAG式轨间平衡、沿X中点的连续边界分配。名称中的X不表示初始化就切换为X_SPLIT模式；初始规则仍是合法WHOLE优先、否则强制Y。
- build_oriented_insertion_beam保留各参考构造的机器人归属与父模式，只联合选择路线插入位置及方向。每个参考构造最多4个完整状态，所以8次调用最多来自两套归属结构，而非八次全新任务分配。
- Beam对路线与方向组合去重，并优先保留不同路线顺序；同一路线最多补入两个方向版本。两个参考构造之间没有完整B32候选去重。因此不能保证8次评价对应8个互不相同的完整方案。
- 由92个父焊缝和1个强制拆分父，以及这两种初始化的规则，可推导初始结构为91个WHOLE加1个Y_SPLIT、合计93个子段；这是规则推导。每个候选的具体切分点、四机器人子段数、长度、加工负载及执行次序未存入失败行，无法给出数值分配表。

可以确定候选多样性受固定归属、固定初始模式和Beam宽度限制。不能把这一结构性限制直接升级为“八个候选高度相似”或某种具体死锁的因果证明；这需要失败候选本身，现有数据没有。

## 5. 准确退出位置与错误分类

|位置|实际作用|与本次失败的关系|
|---|---|---|
|[正式任务分组](../scripts/run_gnn_multi_initial.py#L3472)，_final_formal_run|同一个几何/种子把MODE_NEW、FIXED_NEW放在一个job|五次真实共享初始化，结果分别记到两方法|
|[_final_experiment_worker](../scripts/run_gnn_multi_initial.py#L3173)|先调用共同初始化，记录elapsed|此时即发生失败，没有进入_mode_search|
|同函数约3187–3190行|若initial is None或elapsed>budget，统一输出TIME_LIMIT_NO_FEASIBLE|确认标签混合了两类退出原因|
|[common_initial_beam](../scripts/run_joint_oriented_repair_pilot.py#L152)|有限8槽位循环；成功则按轮次提前停止，否则完成菜单|8个已计次结果没有可行入口，返回None|
|[_mode_search](../scripts/run_gnn_multi_initial.py#L347)|需要一个可行current才能选择父、移除和修复|本次根本未调用该搜索；其自身初始化失败分支其实区分INITIALIZATION_FAILED与超时|

四类事件应分开解释：

1. **候选构造失败**：会记录CONSTRUCTION_FAILED等，不能按调度枚举计入上述8次。这里8次已计次评价并非全都停在构造之前。
2. **给定候选未被B32找到可行调度**：本次能够确认的8个非可行结果；具体DEADLOCK/INFEASIBLE/NUMERIC_FAILURE未知。
3. **60秒截止**：本次初始化耗时27.328573–31.943908秒，不能用此解释提前退出。
4. **程序异常**：包装捕获的异常应输出EXECUTION_ERROR和traceback，本次10行均无该状态；这不等于能够排除未保存的B32 NUMERIC_FAILURE状态。

共同初解返回None后没有路线和方向供select_bottleneck_parents等操作使用；包装函数没有“用剩余预算再换初始化结构”的分支。它也没有自动接入REGION的串行回退。以上属于现有算法的有界初始化设计，本次不将其改写为新的初始化或搜索方法。

## 6. 三个成功对照方法的实际记录

|方法|成功/次数|成功Cmax均值/s|成功Cmax中位/s|MOVE均值/s|WAIT均值/s|
|---|---|---|---|---|---|
|MODE_NEW|0/5|—|—|—|—|
|FIXED_NEW|0/5|—|—|—|—|
|WAG|5/5|4610.342|4652.329|477.404|6.582|
|HGA|5/5|4778.338|4778.338|759.435|0.000|
|REGION_FAST_ROUTE_REORDER|5/5|7696.989|7807.040|825.182|56.120|

以上均为此几何五个种子的记录，成功Cmax只对成功运行统计。initial_state的记录语义并非“第一次原生尝试”：WAG/HGA保存初始化阶段的initial_candidate，REGION保存首个被观察到的可行边界状态；它们未保存各自全部初始化候选。

### 6.1 可行初始状态的四机器人分配

三个方法各自的initial_state在五个种子间逐项相同。下表固定取最小求解种子202611041，不选择最快案例。加工P=SETUP+WELD+POST；子段数不是父焊缝数，长度由已存WELD操作端点计算。

|方法（seed=202611041）|机器人|子段数|焊接长度/m|加工P/s|MOVE/s|WAIT/s|完成/s|方向0/1数量|
|---|---|---|---|---|---|---|---|---|
|WAG|R0|26|32.547505|4313.658|94.703|0.000|4408.361|8/18|
|WAG|R1|24|31.616712|4127.473|104.290|0.000|4231.763|13/11|
|WAG|R2|15|36.853553|4162.366|98.588|0.000|4260.954|7/8|
|WAG|R3|28|34.204746|4567.106|92.315|0.000|4659.421|14/14|
|HGA|R0|26|32.547505|4313.658|171.082|0.000|4484.739|16/10|
|HGA|R1|24|31.616712|4127.473|207.032|0.000|4334.506|13/11|
|HGA|R2|15|36.853553|4162.366|170.089|0.000|4332.455|7/8|
|HGA|R3|28|34.204746|4567.106|211.232|0.000|4778.338|14/14|
|REGION_FAST_ROUTE_REORDER|R0|50|64.164217|8441.131|373.977|0.000|8815.108|26/24|
|REGION_FAST_ROUTE_REORDER|R1|0|0.000000|0.000|0.000|0.000|0.000|0/0|
|REGION_FAST_ROUTE_REORDER|R2|43|71.058299|8729.472|341.240|0.000|9070.713|20/23|
|REGION_FAST_ROUTE_REORDER|R3|0|0.000000|0.000|0.000|0.000|0.000|0/0|

WAG和HGA的四台机器人分配集合相同，子段数均为26/24/15/28，长度和加工负载也相同；但是四条路线顺序均不同，按相同block比较方向分别有14/14/10/18处差异。二者初始WAIT都为0，MOVE总时间分别为389.896秒和759.435秒，Cmax分别为4659.421秒和4778.338秒。因此成功不是只能依赖某一条唯一顺序或方向，且负载相同不保证完成时间相同。由于缺失MODE/FIXED的失败路线，不能据此把其中某种顺序或方向指认为失败的确定根因。

REGION的初始记录仅R0和R2有任务，R1和R3空闲，每轨一个活动机器人，WAIT为0；其加工负载明显集中但仍被B32返回FEASIBLE。当前build_initial_solution确有[RAIL_SERIAL_BOOTSTRAP路径](../src/mrta_search/initialization.py#L1876)，而MODE/FIXED的common_initial_beam没有此回退。**CSV未保存REGION的winning_strategy，故只确认实际的串行分配结构，不仅凭该结构断言本次一定由哪个策略槽生成。**

### 6.2 所有15条成功记录的最终结果

“运行总B32调用”含初始化及搜索，不能当成初始化调用数。

|方法|种子|Cmax/s|子段数R0/R1/R2/R3|焊接长度R0/R1/R2/R3（m）|MOVE总计/s|WAIT总计/s|运行总B32调用|
|---|---|---|---|---|---|---|---|
|HGA|202611041|4778.338|26/24/15/28|32.548/31.617/36.854/34.205|759.435|0.000|27|
|HGA|202611042|4778.338|26/24/15/28|32.548/31.617/36.854/34.205|759.435|0.000|34|
|HGA|202611043|4778.338|26/24/15/28|32.548/31.617/36.854/34.205|759.435|0.000|25|
|HGA|202611044|4778.338|26/24/15/28|32.548/31.617/36.854/34.205|759.435|0.000|27|
|HGA|202611045|4778.338|26/24/15/28|32.548/31.617/36.854/34.205|759.435|0.000|20|
|REGION_FAST_ROUTE_REORDER|202611041|8042.242|45/7/40/3|58.614/7.066/61.683/7.859|799.374|100.667|36|
|REGION_FAST_ROUTE_REORDER|202611042|7916.190|46/4/40/3|56.983/7.181/59.546/11.512|734.120|0.000|35|
|REGION_FAST_ROUTE_REORDER|202611043|6920.367|41/8/38/7|45.791/17.243/50.553/21.635|798.235|179.934|32|
|REGION_FAST_ROUTE_REORDER|202611044|7807.040|45/5/41/4|56.337/8.838/58.416/11.632|874.543|0.000|36|
|REGION_FAST_ROUTE_REORDER|202611045|7799.106|44/5/39/7|56.822/6.837/57.825/13.739|919.638|0.000|38|
|WAG|202611041|4659.421|26/24/15/28|32.548/31.617/36.854/34.205|389.896|0.000|15|
|WAG|202611042|4659.421|26/24/15/28|32.548/31.617/36.854/34.205|389.896|0.000|18|
|WAG|202611043|4652.329|26/24/16/28|32.548/31.617/38.826/32.233|801.145|25.860|19|
|WAG|202611044|4533.140|26/24/16/27|32.548/31.617/37.621/33.438|403.446|0.000|17|
|WAG|202611045|4547.400|26/24/16/27|32.548/31.617/37.562/33.497|402.636|7.048|22|

HGA五个最终状态与其已存初始可行状态相同，已有记录中的搜索迭代数分别为77/55/51/38/51，不能把“最终没有改善”写成“没有运行搜索”。WAG前两个种子最终保持初始状态，后三个种子有变化。REGION五个最终状态均改变，逐步给R1/R3分配任务，并可能改变父模式；这里仅描述已有对照方法的实际行为，不将其组合进MODE_NEW。

REGION seed=202611041的最终分解如下，显示可行性与负载均衡的区别：四台机器人均有任务，R3产生100.667秒WAIT，而Cmax由R0决定。

|机器人|子段数|长度/m|加工P/s|MOVE/s|WAIT/s|完成/s|
|---|---|---|---|---|---|---|
|R0|45|58.614434|7677.262|364.980|0.000|8042.242|
|R1|7|7.065695|1004.231|93.323|0.000|1097.554|
|R2|40|61.683019|7711.391|289.783|0.000|8001.173|
|R3|3|7.859367|877.719|51.288|100.667|1029.674|

### 6.3 可追溯的完整顺序与方向示例

仍固定seed=202611041。为便于阅读，省略每个原始ID中间的group字段，保留inst编号、源row及子段后缀；这些组合在本几何中唯一。方括号0表示保存端点start→end，1表示end→start。原始完整ID、坐标与全部操作仍在CSV的details.initial_state/final_state中。这些是成功对照的真实路线，不能替代失败候选的未知路线。

### WAG：存储的初始可行状态（CSV行478）

R0：

inst0006/row=0002::whole[1] → inst0012/row=0001::whole[0] → inst0012/row=0002::whole[1] → inst0012/row=0000::whole[1] → inst0017/row=0000::whole[0] → inst0044/row=0000::whole[1] → inst0043/row=0000::whole[1] → inst0015/row=0004::whole[0] → inst0015/row=0003::whole[1] → inst0015/row=0002::whole[0] → inst0027/row=0000::0[1] → inst0015/row=0006::whole[1] → inst0015/row=0007::whole[0] → inst0015/row=0008::whole[1] → inst0015/row=0009::whole[0] → inst0015/row=0010::whole[1] → inst0015/row=0001::whole[0] → inst0015/row=0005::whole[1] → inst0015/row=0000::whole[1] → inst0039/row=0000::whole[1] → inst0025/row=0000::whole[1] → inst0008/row=0002::whole[1] → inst0008/row=0000::whole[1] → inst0008/row=0001::whole[0] → inst0033/row=0000::whole[1] → inst0030/row=0000::whole[1]

R1：

inst0034/row=0000::whole[0] → inst0018/row=0001::whole[0] → inst0018/row=0000::whole[0] → inst0032/row=0000::whole[0] → inst0003/row=0000::whole[0] → inst0022/row=0000::whole[1] → inst0018/row=0002::whole[0] → inst0001/row=0000::whole[1] → inst0016/row=0000::whole[1] → inst0037/row=0000::whole[0] → inst0038/row=0000::whole[1] → inst0024/row=0004::whole[0] → inst0024/row=0006::whole[0] → inst0042/row=0000::whole[0] → inst0002/row=0000::whole[1] → inst0024/row=0005::whole[1] → inst0024/row=0003::whole[1] → inst0024/row=0000::whole[1] → inst0024/row=0001::whole[1] → inst0024/row=0002::whole[0] → inst0014/row=0000::whole[0] → inst0006/row=0001::whole[0] → inst0006/row=0000::whole[1] → inst0023/row=0000::whole[1]

R2：

inst0041/row=0000::whole[1] → inst0021/row=0000::whole[0] → inst0009/row=0000::whole[1] → inst0040/row=0000::whole[1] → inst0029/row=0000::whole[0] → inst0035/row=0000::whole[0] → inst0011/row=0000::whole[0] → inst0036/row=0000::whole[1] → inst0005/row=0000::whole[1] → inst0004/row=0000::whole[1] → inst0004/row=0002::whole[0] → inst0004/row=0003::whole[1] → inst0004/row=0007::whole[0] → inst0010/row=0000::whole[0] → inst0027/row=0000::1[1]

R3：

inst0004/row=0001::whole[0] → inst0004/row=0004::whole[0] → inst0004/row=0006::whole[0] → inst0028/row=0000::whole[0] → inst0004/row=0005::whole[1] → inst0031/row=0000::whole[1] → inst0022/row=0003::whole[0] → inst0022/row=0002::whole[1] → inst0022/row=0001::whole[0] → inst0022/row=0004::whole[1] → inst0013/row=0002::whole[0] → inst0013/row=0003::whole[1] → inst0013/row=0004::whole[0] → inst0013/row=0000::whole[1] → inst0013/row=0001::whole[0] → inst0013/row=0010::whole[1] → inst0013/row=0009::whole[0] → inst0013/row=0008::whole[1] → inst0013/row=0007::whole[0] → inst0013/row=0006::whole[1] → inst0013/row=0005::whole[0] → inst0007/row=0002::whole[1] → inst0007/row=0000::whole[1] → inst0007/row=0001::whole[1] → inst0026/row=0000::whole[1] → inst0019/row=0000::whole[0] → inst0000/row=0000::whole[0] → inst0020/row=0000::whole[1]

该种子的最终路线和方向与这里的初始可行状态逐项相同。

### HGA：存储的初始可行状态（CSV行479）

R0：

inst0030/row=0000::whole[0] → inst0033/row=0000::whole[0] → inst0008/row=0002::whole[1] → inst0008/row=0000::whole[0] → inst0008/row=0001::whole[1] → inst0027/row=0000::0[1] → inst0025/row=0000::whole[0] → inst0015/row=0002::whole[0] → inst0015/row=0003::whole[1] → inst0015/row=0004::whole[0] → inst0015/row=0005::whole[0] → inst0015/row=0006::whole[1] → inst0015/row=0007::whole[0] → inst0015/row=0008::whole[1] → inst0015/row=0009::whole[0] → inst0015/row=0010::whole[1] → inst0015/row=0000::whole[0] → inst0015/row=0001::whole[1] → inst0039/row=0000::whole[0] → inst0043/row=0000::whole[1] → inst0012/row=0002::whole[0] → inst0044/row=0000::whole[0] → inst0012/row=0000::whole[0] → inst0017/row=0000::whole[1] → inst0012/row=0001::whole[0] → inst0006/row=0002::whole[0]

R1：

inst0006/row=0000::whole[0] → inst0006/row=0001::whole[1] → inst0023/row=0000::whole[1] → inst0014/row=0000::whole[1] → inst0024/row=0001::whole[1] → inst0024/row=0002::whole[0] → inst0024/row=0003::whole[1] → inst0024/row=0004::whole[0] → inst0024/row=0006::whole[0] → inst0042/row=0000::whole[0] → inst0024/row=0000::whole[0] → inst0024/row=0005::whole[0] → inst0002/row=0000::whole[0] → inst0038/row=0000::whole[0] → inst0022/row=0000::whole[1] → inst0037/row=0000::whole[0] → inst0003/row=0000::whole[1] → inst0016/row=0000::whole[0] → inst0032/row=0000::whole[1] → inst0018/row=0002::whole[0] → inst0001/row=0000::whole[0] → inst0018/row=0000::whole[1] → inst0018/row=0001::whole[1] → inst0034/row=0000::whole[1]

R2：

inst0036/row=0000::whole[0] → inst0011/row=0000::whole[1] → inst0005/row=0000::whole[0] → inst0035/row=0000::whole[1] → inst0021/row=0000::whole[1] → inst0009/row=0000::whole[0] → inst0041/row=0000::whole[1] → inst0029/row=0000::whole[0] → inst0040/row=0000::whole[0] → inst0027/row=0000::1[1] → inst0010/row=0000::whole[1] → inst0004/row=0000::whole[1] → inst0004/row=0003::whole[0] → inst0004/row=0007::whole[1] → inst0004/row=0002::whole[0]

R3：

inst0004/row=0005::whole[0] → inst0028/row=0000::whole[1] → inst0004/row=0001::whole[0] → inst0004/row=0004::whole[0] → inst0004/row=0006::whole[0] → inst0022/row=0003::whole[0] → inst0022/row=0002::whole[1] → inst0031/row=0000::whole[0] → inst0022/row=0004::whole[0] → inst0022/row=0001::whole[1] → inst0007/row=0000::whole[0] → inst0013/row=0002::whole[1] → inst0013/row=0003::whole[0] → inst0007/row=0001::whole[1] → inst0013/row=0004::whole[1] → inst0007/row=0002::whole[1] → inst0026/row=0000::whole[1] → inst0020/row=0000::whole[1] → inst0013/row=0000::whole[0] → inst0000/row=0000::whole[1] → inst0013/row=0005::whole[1] → inst0013/row=0006::whole[0] → inst0013/row=0007::whole[0] → inst0019/row=0000::whole[1] → inst0013/row=0001::whole[1] → inst0013/row=0008::whole[0] → inst0013/row=0009::whole[1] → inst0013/row=0010::whole[0]

该种子的最终路线和方向与这里的初始可行状态逐项相同。

### REGION_FAST_ROUTE_REORDER：存储的初始可行状态（CSV行480）

R0：

inst0030/row=0000::whole[0] → inst0033/row=0000::whole[0] → inst0008/row=0000::whole[1] → inst0008/row=0001::whole[0] → inst0008/row=0002::whole[0] → inst0027/row=0000::0[0] → inst0015/row=0002::whole[1] → inst0015/row=0003::whole[0] → inst0015/row=0004::whole[1] → inst0015/row=0005::whole[1] → inst0015/row=0006::whole[0] → inst0015/row=0007::whole[1] → inst0015/row=0008::whole[0] → inst0015/row=0009::whole[1] → inst0015/row=0010::whole[0] → inst0025/row=0000::whole[0] → inst0015/row=0000::whole[0] → inst0015/row=0001::whole[1] → inst0039/row=0000::whole[0] → inst0043/row=0000::whole[0] → inst0044/row=0000::whole[0] → inst0012/row=0000::whole[0] → inst0012/row=0002::whole[0] → inst0017/row=0000::whole[1] → inst0012/row=0001::whole[0] → inst0006/row=0000::whole[0] → inst0006/row=0001::whole[1] → inst0006/row=0002::whole[1] → inst0023/row=0000::whole[1] → inst0014/row=0000::whole[1] → inst0024/row=0006::whole[1] → inst0024/row=0000::whole[0] → inst0042/row=0000::whole[0] → inst0024/row=0005::whole[1] → inst0024/row=0004::whole[0] → inst0024/row=0003::whole[1] → inst0024/row=0002::whole[0] → inst0024/row=0001::whole[1] → inst0002/row=0000::whole[0] → inst0038/row=0000::whole[0] → inst0022/row=0000::whole[1] → inst0037/row=0000::whole[0] → inst0003/row=0000::whole[1] → inst0016/row=0000::whole[0] → inst0032/row=0000::whole[1] → inst0001/row=0000::whole[1] → inst0018/row=0002::whole[1] → inst0018/row=0000::whole[1] → inst0018/row=0001::whole[1] → inst0034/row=0000::whole[1]

R1：

空路线

R2：

inst0036/row=0000::whole[0] → inst0035/row=0000::whole[1] → inst0011/row=0000::whole[0] → inst0021/row=0000::whole[1] → inst0005/row=0000::whole[0] → inst0009/row=0000::whole[1] → inst0041/row=0000::whole[1] → inst0029/row=0000::whole[0] → inst0040/row=0000::whole[0] → inst0027/row=0000::1[1] → inst0010/row=0000::whole[1] → inst0004/row=0001::whole[0] → inst0004/row=0004::whole[0] → inst0004/row=0006::whole[0] → inst0028/row=0000::whole[0] → inst0004/row=0000::whole[1] → inst0004/row=0003::whole[0] → inst0004/row=0007::whole[1] → inst0004/row=0002::whole[0] → inst0004/row=0005::whole[1] → inst0022/row=0003::whole[1] → inst0031/row=0000::whole[1] → inst0022/row=0004::whole[0] → inst0022/row=0002::whole[1] → inst0022/row=0001::whole[1] → inst0007/row=0000::whole[0] → inst0013/row=0002::whole[1] → inst0013/row=0003::whole[0] → inst0013/row=0004::whole[0] → inst0007/row=0001::whole[1] → inst0026/row=0000::whole[1] → inst0007/row=0002::whole[0] → inst0013/row=0000::whole[1] → inst0013/row=0005::whole[0] → inst0000/row=0000::whole[1] → inst0013/row=0006::whole[1] → inst0013/row=0007::whole[0] → inst0020/row=0000::whole[1] → inst0019/row=0000::whole[1] → inst0013/row=0001::whole[1] → inst0013/row=0008::whole[0] → inst0013/row=0009::whole[1] → inst0013/row=0010::whole[0]

R3：

空路线

### REGION：相同种子的最终可行状态

R0：

inst0030/row=0000::whole[0] → inst0033/row=0000::whole[0] → inst0008/row=0000::whole[1] → inst0008/row=0001::whole[0] → inst0008/row=0002::whole[0] → inst0027/row=0000::0[0] → inst0015/row=0002::whole[1] → inst0015/row=0003::whole[0] → inst0015/row=0004::whole[1] → inst0015/row=0005::whole[0] → inst0015/row=0006::whole[1] → inst0015/row=0008::whole[0] → inst0015/row=0009::whole[1] → inst0025/row=0000::whole[1] → inst0015/row=0001::whole[1] → inst0039/row=0000::whole[0] → inst0043/row=0000::whole[0] → inst0044/row=0000::whole[0] → inst0012/row=0000::whole[0] → inst0012/row=0002::whole[0] → inst0017/row=0000::whole[1] → inst0012/row=0001::whole[0] → inst0006/row=0002::0[0] → inst0006/row=0001::whole[1] → inst0023/row=0000::whole[1] → inst0014/row=0000::whole[1] → inst0024/row=0006::whole[1] → inst0024/row=0000::whole[0] → inst0042/row=0000::whole[0] → inst0024/row=0005::whole[1] → inst0024/row=0003::whole[1] → inst0024/row=0004::whole[0] → inst0024/row=0001::whole[1] → inst0002/row=0000::whole[0] → inst0038/row=0000::whole[0] → inst0022/row=0000::whole[1] → inst0037/row=0000::whole[0] → inst0003/row=0000::whole[1] → inst0016/row=0000::whole[0] → inst0032/row=0000::whole[1] → inst0001/row=0000::whole[1] → inst0018/row=0002::whole[1] → inst0018/row=0000::whole[1] → inst0018/row=0001::whole[1] → inst0034/row=0000::whole[1]

R1：

inst0022/row=0001::1[0] → inst0024/row=0002::whole[0] → inst0006/row=0000::whole[0] → inst0015/row=0000::whole[0] → inst0015/row=0007::whole[0] → inst0015/row=0010::whole[1] → inst0006/row=0002::1[0]

R2：

inst0036/row=0000::whole[0] → inst0035/row=0000::whole[1] → inst0011/row=0000::whole[0] → inst0021/row=0000::whole[1] → inst0005/row=0000::whole[0] → inst0009/row=0000::whole[1] → inst0041/row=0000::whole[0] → inst0040/row=0000::whole[1] → inst0027/row=0000::1[1] → inst0010/row=0000::whole[1] → inst0004/row=0001::whole[0] → inst0004/row=0004::whole[0] → inst0004/row=0006::whole[0] → inst0028/row=0000::whole[1] → inst0004/row=0000::whole[0] → inst0004/row=0003::whole[1] → inst0004/row=0007::whole[0] → inst0004/row=0005::whole[1] → inst0022/row=0003::whole[1] → inst0031/row=0000::whole[1] → inst0022/row=0004::whole[0] → inst0022/row=0002::whole[1] → inst0022/row=0001::0[1] → inst0007/row=0000::whole[0] → inst0013/row=0002::whole[1] → inst0013/row=0003::whole[0] → inst0013/row=0004::whole[0] → inst0007/row=0001::whole[1] → inst0026/row=0000::whole[1] → inst0007/row=0002::whole[0] → inst0013/row=0000::whole[1] → inst0013/row=0005::whole[0] → inst0000/row=0000::whole[1] → inst0013/row=0006::whole[1] → inst0013/row=0007::whole[0] → inst0019/row=0000::whole[1] → inst0013/row=0001::whole[1] → inst0013/row=0008::whole[0] → inst0013/row=0009::whole[1] → inst0013/row=0010::whole[0]

R3：

inst0020/row=0000::whole[0] → inst0004/row=0002::whole[0] → inst0029/row=0000::whole[0]


## 7. 根本原因、直接原因与可能原因

|层级|结论|证据强度|
|---|---|---|
|可确认的流程根因|有界共同初始化未产生可行搜索入口，算法又依赖该入口启动后续重构|CSV计数/耗时与代码分支共同支持|
|直接退出原因|initial=None；包装没有进入_mode_search并提前返回空Cmax|代码与两方法共享耗时支持|
|状态分类缺陷|初始化菜单耗尽被统一标成TIME_LIMIT_NO_FEASIBLE|明确可定位的实现问题|
|结构性局限|最多两套参考归属、每套四个有界顺序/方向变体，无串行初始化回退|代码事实；不等于已证明具体冲突原因|
|可能原因|同轨任务顺序、方向、局部P+MOVE代理与真实协调之间的偏差，或数值失败|仅是假设，缺失失败候选和B32轨迹，均未证实|
|不能接受的结论|任务数学不可行、8次必然全部DEADLOCK、某条焊缝/某台机器人必为阻塞根因、负载失衡已被证明|现有记录不支持|

WAG、HGA及REGION的成功证明：该几何在相同科学配置和B32下至少存在已找到的可行方案。它不能证明MODE_NEW的有限初始化一定能找到这些方案，更不能证明修改一个方向就能消除全部失败。

## 8. 是否有明确代码缺陷及最小修复建议

已确认的是**报告状态分类缺陷**，位置为_final_experiment_worker的共同初始化失败分支。影响是读者容易把提前失败误解为60秒耗尽，不会改变现有Cmax、FEASIBLE次数、19个完整配对几何或MODE_NEW/FIXED_NEW的实际搜索行为。没有足够证据确认正式联合修复或B32调度实现存在导致这些失败的其他代码错误。

若后续单独授权修正元数据，可在现有失败分支内最小区分：

```python
# 仅为修复建议，本次没有应用
status = "TIME_LIMIT_NO_FEASIBLE" if elapsed > budget else "INITIALIZATION_FAILED"
```

不要修改已完成CSV中的原始标签；可在论文中给出本报告的原因解释。现有attempts本来包含逐候选状态与diagnostics，但包装在本次失败分支没有保存它们，这是证据留存缺口，无法靠事后分析补回。本次不新增日志系统、不修改初始化策略、不重跑失败实例或正式矩阵。

## 9. 论文解释建议

可直接使用以下表述：

> 在独立实例HIGH_WORKLOAD_04（92条父焊缝，几何种子2026110512）上，MODE_NEW和FIXED_NEW在五个求解种子下均未取得可行共同初解。每个种子的有限初始化完成8次完整B32候选评价后即退出，耗时27.329–31.944秒，尚未达到60秒评价截止；后续父模式调整和联合重构未启动。原始TIME_LIMIT_NO_FEASIBLE标签在此表示未取得可行解，不应解释为时间耗尽。WAG、HGA和REGION在同一几何均成功5/5。由于失败候选和阻塞轨迹未保留，具体同轨顺序、焊接方向或资源阻塞原因尚不能确定。

因此应把该实例计入主方法95/100的可行率，并在Cmax配对统计中明确排除该不完整几何，不能填补失败Cmax、隐藏失败或解释为数学不可行。

## 10. 本次工作范围

只读核对25条相关运行、既有几何记录与实际代码，并整理报告。未改变MODE_NEW、FIXED_NEW、B32、科学配置、模型权重或实验原始数据；未新增任何调度或搜索模块。当前能够给出的根因结论止于有界共同初始化覆盖不足及提前退出，具体调度冲突根因保持未识别。

