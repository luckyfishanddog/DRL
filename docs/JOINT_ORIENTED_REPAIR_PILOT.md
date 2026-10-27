# 方向嵌入联合插入修复开发实验

seed=20261027；q=[4, 8]；attempts=8；insertion_limit=8；repair_budget=4096。
FORMAL_SCOPE_V2，0.0108/0.20 m/s，准备/后处理 20/30s，X/Y 窗口 0.50m，最短子段 0.20m；开放路线、免费首部署。
A0 固定每块初始方向；A1 枚举新块及前后邻块方向，最多保留两个局部配置。二者共享修复循环、不同结构 Regret-2、机器人均分预算和端点评分。A1_RANDOM 同时用于实验 A 和 B，只执行一遍。RANDOM 的 A0/A1 每次使用完全相同的移除集合、初解与修复 seed。
逐步插入后缓存更新受影响的最多四条弧。Regret 比较不同路线结构，方向不同而结构相同的方案仅保留较优者。有限预算按合法机器人/子块机器人组合分配，位置按 seed 打散；A1 每次局部配置评分计一次预算，故同预算下评估的结构数可能少于 A0。
共同初解复用 REGION 归属/边界逻辑，但通过新可选 cost hook 和关闭旧路线工具绕过内部 DP；从空状态逐块定向追加，锁定合法首方向。首个 B32 成功初解被所有组共享；失败才尝试其他首方向组合及现有 RULE/X 路线。
新实验无全路线方向 DP、方向 VNS、C2/C3/C4、独立 certifier、精确解或下界调用。每个组内唯一完整候选显式传入四个方向向量，仅调用一次原 B32。只把 FEASIBLE 的 Cmax 用于比较；基本检查覆盖、资格、固定模式和焊接操作方向。DEADLOCK 不代表数学不可行。
六例为历史已消耗开发几何。沿用 MLP_DEV/load_parents 白名单，未读取新封存测试几何。g15 的历史 ID_TEST 目录不表示当前封存角色。

## 共同初解与尝试

- g21_w035：Cmax=2456.922803492095，耗时 1.246s；选区生成 0.021s；模式 {'WHOLE': 34, 'Y_SPLIT': 1}；尝试状态 {'DEADLOCK': 16, 'FEASIBLE': 1}；成功构造 RULE_X_LOCAL。
- g15_w035：Cmax=1663.8411021760685，耗时 2.288s；选区生成 0.016s；模式 {'WHOLE': 35}；尝试状态 {'DEADLOCK': 16, 'FEASIBLE': 1}；成功构造 RULE_X_LOCAL。
- g36_w086：Cmax=3908.0208561480763，耗时 22.502s；选区生成 0.055s；模式 {'WHOLE': 86}；尝试状态 {'DEADLOCK': 16, 'FEASIBLE': 1}；成功构造 RULE_X_LOCAL。
- g33_w055：Cmax=4081.4384190260957，耗时 2.391s；选区生成 0.024s；模式 {'WHOLE': 55}；尝试状态 {'DEADLOCK': 16, 'FEASIBLE': 1}；成功构造 RULE_X_LOCAL。
- g31_w055：Cmax=3295.7162120184503，耗时 11.868s；选区生成 0.043s；模式 {'WHOLE': 50, 'Y_SPLIT': 5}；尝试状态 {'DEADLOCK': 8, 'FEASIBLE': 1}；成功构造 REGION_LOCAL。
- g45_w082：Cmax=6708.453657861711，耗时 0.560s；选区生成 0.061s；模式 {'WHOLE': 74, 'Y_SPLIT': 8}；尝试状态 {'FEASIBLE': 1}；成功构造 REGION_LOCAL。

## 全部分组结果

唯一候选按路线和方向共同去重。最佳值为成功候选中的最小值，可劣于初解；失败留空。仅严格改善候选作为专家保留。时间包含修复、检查和 B32，选区生成时间另列于初解日志；组间候选可能重合。

|实例|方法|q|初始 Cmax|尝试|唯一候选|B32 成功|改善|最佳 Cmax|改善%|秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|g21_w035|A0_RANDOM|4|2456.923|8|8|6|0|2456.923|0.000|0.51|
|g21_w035|A1_RANDOM|4|2456.923|8|8|7|0|2456.923|0.000|0.49|
|g21_w035|A1_LOAD|4|2456.923|8|8|5|2|2416.584|1.642|0.87|
|g21_w035|A1_TRAVEL|4|2456.923|8|8|7|0|2456.923|0.000|0.91|
|g21_w035|A1_WAIT_ANCHORED|4|2456.923|0|0|0|0|—|—|0.00|
|g21_w035|A0_RANDOM|8|2456.923|8|8|5|0|2456.923|0.000|1.24|
|g21_w035|A1_RANDOM|8|2456.923|8|8|3|0|2456.923|0.000|1.86|
|g21_w035|A1_LOAD|8|2456.923|8|8|5|1|2407.005|2.032|1.61|
|g21_w035|A1_TRAVEL|8|2456.923|8|8|7|3|2413.037|1.786|1.11|
|g21_w035|A1_WAIT_ANCHORED|8|2456.923|0|0|0|0|—|—|0.00|
|g15_w035|A0_RANDOM|4|1663.841|8|7|6|0|1663.841|0.000|1.03|
|g15_w035|A1_RANDOM|4|1663.841|8|8|7|0|1663.841|0.000|0.84|
|g15_w035|A1_LOAD|4|1663.841|8|8|7|0|1663.841|0.000|0.51|
|g15_w035|A1_TRAVEL|4|1663.841|8|8|8|2|1653.687|0.610|0.47|
|g15_w035|A1_WAIT_ANCHORED|4|1663.841|0|0|0|0|—|—|0.00|
|g15_w035|A0_RANDOM|8|1663.841|8|8|4|0|1663.841|0.000|1.55|
|g15_w035|A1_RANDOM|8|1663.841|8|8|4|0|1663.841|0.000|1.93|
|g15_w035|A1_LOAD|8|1663.841|8|8|5|0|1663.841|0.000|1.62|
|g15_w035|A1_TRAVEL|8|1663.841|8|8|4|1|1661.078|0.166|1.51|
|g15_w035|A1_WAIT_ANCHORED|8|1663.841|0|0|0|0|—|—|0.00|
|g36_w086|A0_RANDOM|4|3908.021|8|8|8|1|3900.928|0.181|5.45|
|g36_w086|A1_RANDOM|4|3908.021|8|8|8|2|3900.928|0.181|3.62|
|g36_w086|A1_LOAD|4|3908.021|8|8|4|0|3908.021|0.000|6.75|
|g36_w086|A1_TRAVEL|4|3908.021|8|8|6|2|3900.673|0.188|4.05|
|g36_w086|A1_WAIT_ANCHORED|4|3908.021|8|8|8|0|3908.021|0.000|8.94|
|g36_w086|A0_RANDOM|8|3908.021|8|8|7|0|3908.021|0.000|4.00|
|g36_w086|A1_RANDOM|8|3908.021|8|8|5|0|3908.021|0.000|7.31|
|g36_w086|A1_LOAD|8|3908.021|8|8|6|1|3889.118|0.484|5.91|
|g36_w086|A1_TRAVEL|8|3908.021|8|8|6|0|3908.021|0.000|3.70|
|g36_w086|A1_WAIT_ANCHORED|8|3908.021|8|8|7|2|3888.852|0.491|3.98|
|g33_w055|A0_RANDOM|4|4081.438|8|8|4|4|3606.916|11.626|1.38|
|g33_w055|A1_RANDOM|4|4081.438|8|8|5|5|3920.555|3.942|1.64|
|g33_w055|A1_LOAD|4|4081.438|8|8|7|3|3884.689|4.821|0.77|
|g33_w055|A1_TRAVEL|4|4081.438|8|8|7|2|3821.496|6.369|1.08|
|g33_w055|A1_WAIT_ANCHORED|4|4081.438|0|0|0|0|—|—|0.00|
|g33_w055|A0_RANDOM|8|4081.438|8|8|5|5|3665.740|10.185|1.31|
|g33_w055|A1_RANDOM|8|4081.438|8|8|7|7|3587.136|12.111|1.99|
|g33_w055|A1_LOAD|8|4081.438|8|8|5|1|3661.577|10.287|1.94|
|g33_w055|A1_TRAVEL|8|4081.438|8|8|4|0|4081.438|0.000|2.93|
|g33_w055|A1_WAIT_ANCHORED|8|4081.438|0|0|0|0|—|—|0.00|
|g31_w055|A0_RANDOM|4|3295.716|8|8|7|2|3263.210|0.986|1.01|
|g31_w055|A1_RANDOM|4|3295.716|8|8|7|2|3256.368|1.194|1.37|
|g31_w055|A1_LOAD|4|3295.716|8|8|7|2|3218.399|2.346|1.33|
|g31_w055|A1_TRAVEL|4|3295.716|8|8|5|2|3219.790|2.304|1.60|
|g31_w055|A1_WAIT_ANCHORED|4|3295.716|8|8|2|0|3419.762|-3.764|1.61|
|g31_w055|A0_RANDOM|8|3295.716|8|8|1|0|3313.923|-0.552|2.03|
|g31_w055|A1_RANDOM|8|3295.716|8|8|0|0|—|—|2.21|
|g31_w055|A1_LOAD|8|3295.716|8|8|3|2|3219.790|2.304|1.96|
|g31_w055|A1_TRAVEL|8|3295.716|8|8|3|1|3219.790|2.304|3.07|
|g31_w055|A1_WAIT_ANCHORED|8|3295.716|8|8|4|1|3234.317|1.863|2.08|
|g45_w082|A0_RANDOM|4|6708.454|8|8|4|2|6098.130|9.098|7.19|
|g45_w082|A1_RANDOM|4|6708.454|8|8|4|3|6098.130|9.098|5.67|
|g45_w082|A1_LOAD|4|6708.454|8|8|3|1|6698.443|0.149|6.41|
|g45_w082|A1_TRAVEL|4|6708.454|8|8|2|2|6356.676|5.244|3.47|
|g45_w082|A1_WAIT_ANCHORED|4|6708.454|8|8|1|1|6098.130|9.098|6.54|
|g45_w082|A0_RANDOM|8|6708.454|8|8|2|1|6580.799|1.903|8.63|
|g45_w082|A1_RANDOM|8|6708.454|8|8|3|2|6556.051|2.272|7.78|
|g45_w082|A1_LOAD|8|6708.454|8|8|0|0|—|—|7.14|
|g45_w082|A1_TRAVEL|8|6708.454|8|8|1|0|6708.454|0.000|6.64|
|g45_w082|A1_WAIT_ANCHORED|8|6708.454|8|8|0|0|—|—|5.67|

## A0/A1 同移除集配对

|实例|q|共同成功|A1 胜|平|A0 胜|仅 A1 成功|仅 A0 成功|
|---|---:|---:|---:|---:|---:|---:|---:|
|g21_w035|4|6|3|3|0|1|0|
|g21_w035|8|3|1|1|1|0|2|
|g15_w035|4|7|3|3|1|0|0|
|g15_w035|8|3|1|2|0|1|1|
|g36_w086|4|8|2|5|1|0|0|
|g36_w086|8|5|1|1|3|0|2|
|g33_w055|4|4|0|3|1|1|0|
|g33_w055|8|5|2|3|0|2|0|
|g31_w055|4|7|4|3|0|0|0|
|g31_w055|8|0|0|0|0|0|1|
|g45_w082|4|3|1|1|1|1|1|
|g45_w082|8|2|2|0|0|1|0|

## 构造与可调度性

- A0_RANDOM：{"attempts": 96, "unique_candidates": 95, "schedule_success": 59, "improved": 15, "restored_original": 3, "changed_assignment": 31, "changed_direction": 0, "reordered": 54, "joint_expert_candidates": 0, "direction_order_only_improved": 0, "movable_blocks": 597, "removed_from_critical_route": 140}；B32={'DEADLOCK': 36, 'FEASIBLE': 59}；修复拒绝={}
- A1_RANDOM：{"attempts": 96, "unique_candidates": 96, "schedule_success": 60, "improved": 21, "restored_original": 0, "changed_assignment": 30, "changed_direction": 60, "reordered": 58, "joint_expert_candidates": 12, "direction_order_only_improved": 9, "movable_blocks": 597, "removed_from_critical_route": 140}；B32={'DEADLOCK': 36, 'FEASIBLE': 60}；修复拒绝={}
- A1_LOAD：{"attempts": 96, "unique_candidates": 96, "schedule_success": 57, "improved": 13, "restored_original": 0, "changed_assignment": 31, "changed_direction": 55, "reordered": 55, "joint_expert_candidates": 6, "direction_order_only_improved": 7, "movable_blocks": 603, "removed_from_critical_route": 146}；B32={'DEADLOCK': 39, 'FEASIBLE': 57}；修复拒绝={}
- A1_TRAVEL：{"attempts": 96, "unique_candidates": 96, "schedule_success": 60, "improved": 15, "restored_original": 0, "changed_assignment": 34, "changed_direction": 60, "reordered": 59, "joint_expert_candidates": 7, "direction_order_only_improved": 8, "movable_blocks": 608, "removed_from_critical_route": 133}；B32={'DEADLOCK': 36, 'FEASIBLE': 60}；修复拒绝={}
- A1_WAIT_ANCHORED：{"attempts": 48, "unique_candidates": 48, "schedule_success": 22, "improved": 4, "restored_original": 0, "changed_assignment": 11, "changed_direction": 22, "reordered": 22, "joint_expert_candidates": 2, "direction_order_only_improved": 2, "movable_blocks": 324, "removed_from_critical_route": 84}；B32={'FEASIBLE': 22, 'DEADLOCK': 26}；修复拒绝={}

## 典型成功/失败与可重放动作

### 联合轨迹：g21_w035 A1_LOAD q4 attempt=1

状态 FEASIBLE；归属变化：inst0002:0004::whole R0→R1; inst0002:0002::whole R1→R0；方向变化：inst0002:0002::whole 0→1；重排路线 [0]。
加工负载 (2200.38, 2383.647, 1187.032, 1173.584)→(2240.581, 2343.445, 1187.032, 1173.584)；空走 357.612→353.480s；WAIT 0.000→0.000s；Cmax 2456.922803492095→2416.5843181653217。
移除父焊缝：inst0002|group=143-FR76A.dxf|row=0002; inst0002|group=143-FR76A.dxf|row=0003; inst0002|group=143-FR76A.dxf|row=0004; inst0016|group=804-FR315A.dxf|row=0000。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0002:0002::whole→R0@9/1; inst0002:0003::whole→R0@10/0; inst0016:0000::whole→R0@0/0; inst0002:0004::whole→R1@0/1。

- R0：`inst0016:0000::whole/0 → inst0016:0001::whole/1 → inst0016:0002::whole/0 → inst0001:0002::1/0 → inst0007:0000::whole/1 → inst0014:0000::whole/0 → inst0020:0000::whole/1 → inst0013:0000::whole/1 → inst0009:0000::whole/0 → inst0002:0006::whole/1 → inst0002:0000::whole/0 → inst0002:0005::whole/0 → inst0002:0004::whole/1 → inst0002:0003::whole/0` → `inst0016:0000::whole/0 → inst0016:0001::whole/1 → inst0016:0002::whole/0 → inst0001:0002::1/0 → inst0007:0000::whole/1 → inst0014:0000::whole/0 → inst0020:0000::whole/1 → inst0013:0000::whole/1 → inst0009:0000::whole/0 → inst0002:0006::whole/1 → inst0002:0002::whole/1 → inst0002:0003::whole/0 → inst0002:0000::whole/0 → inst0002:0005::whole/0`
- R1：`inst0002:0002::whole/0 → inst0002:0001::whole/0 → inst0008:0000::whole/1 → inst0004:0000::whole/1 → inst0019:0000::whole/0 → inst0018:0000::whole/1 → inst0010:0000::whole/1` → `inst0002:0004::whole/1 → inst0002:0001::whole/0 → inst0008:0000::whole/1 → inst0004:0000::whole/1 → inst0019:0000::whole/0 → inst0018:0000::whole/1 → inst0010:0000::whole/1`
- R2：`inst0017:0000::whole/0 → inst0011:0000::whole/1 → inst0011:0001::whole/0 → inst0001:0002::0/0 → inst0011:0002::whole/1 → inst0001:0000::whole/1 → inst0001:0001::whole/1 → inst0005:0000::whole/1` → `inst0017:0000::whole/0 → inst0011:0000::whole/1 → inst0011:0001::whole/0 → inst0001:0002::0/0 → inst0011:0002::whole/1 → inst0001:0000::whole/1 → inst0001:0001::whole/1 → inst0005:0000::whole/1`
- R3：`inst0006:0000::whole/0 → inst0012:0000::whole/0 → inst0012:0002::whole/0 → inst0003:0000::whole/0 → inst0012:0001::whole/0 → inst0015:0000::whole/0 → inst0000:0000::whole/1` → `inst0006:0000::whole/0 → inst0012:0000::whole/0 → inst0012:0002::whole/0 → inst0003:0000::whole/0 → inst0012:0001::whole/0 → inst0015:0000::whole/0 → inst0000:0000::whole/1`
### 仅方向和顺序改善：g15_w035 A1_TRAVEL q4 attempt=6

状态 FEASIBLE；归属变化：无；方向变化：inst0008:0000::whole 1→0; inst0008:0001::whole 0→1；重排路线 [3]。
加工负载 (1483.641, 1331.383, 1571.766, 1596.918)→(1483.641, 1331.383, 1571.766, 1596.918)；空走 238.128→227.973s；WAIT 0.000→0.000s；Cmax 1663.8411021760685→1653.686593316072。
移除父焊缝：inst0000|group=165-SR2A.dxf|row=0000; inst0000|group=165-SR2A.dxf|row=0001; inst0008|group=804-FR315A.dxf|row=0000; inst0008|group=804-FR315A.dxf|row=0001。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0008:0000::whole→R3@4/0; inst0000:0001::whole→R3@4/1; inst0008:0001::whole→R3@6/1; inst0000:0000::whole→R3@3/1。

### g33_w055:A1_LOAD：g33_w055 A1_LOAD q8 attempt=0

状态 FEASIBLE；归属变化：inst0025:0000::whole R1→R0; inst0017:0000::whole R3→R2; inst0000:0000::whole R3→R2; inst0023:0000::whole R3→R2；方向变化：inst0025:0000::whole 1→0; inst0006:0000::whole 0→1; inst0014:0000::whole 0→1; inst0010:0000::whole 1→0; inst0022:0000::whole 0→1; inst0019:0001::whole 1→0; inst0019:0002::whole 0→1; inst0032:0002::whole 0→1; inst0017:0000::whole 1→0; inst0000:0000::whole 1→0；重排路线 [1]。
加工负载 (3102.084, 3286.421, 2909.901, 3930.343)→(3426.078, 2962.426, 3305.112, 3535.133)；空走 546.778→513.743s；WAIT 0.000→0.000s；Cmax 4081.4384190260957→3661.5767548517365。
移除父焊缝：inst0000|group=183-CM2B.dxf|row=0000; inst0006|group=826BK312A|row=0000; inst0010|group=143-GR14A|row=0000; inst0017|group=143-FR76A.dxf|row=0000; inst0017|group=143-FR76A.dxf|row=0004; inst0019|group=145-LB17A.dxf|row=0002; inst0023|group=804-FR313A|row=0000; inst0025|group=312-GR2C|row=0000。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0023:0000::whole→R2@12/0; inst0000:0000::whole→R2@13/0; inst0019:0002::whole→R2@6/1(inst0019:0001::whole:1→0,inst0032:0002::whole:0→1); inst0017:0000::whole→R2@15/0; inst0025:0000::whole→R0@6/0; inst0006:0000::whole→R1@9/1(inst0014:0000::whole:0→1,inst0022:0000::whole:0→1); inst0010:0000::whole→R1@14/0; inst0017:0004::whole→R2@13/1。

### g33_w055:A1_RANDOM：g33_w055 A1_RANDOM q8 attempt=5

状态 FEASIBLE；归属变化：inst0030:0000::whole R1→R0; inst0017:0003::whole R3→R2; inst0023:0000::whole R3→R2; inst0002:0000::whole R3→R2；方向变化：inst0012:0000::whole 0→1; inst0030:0000::whole 1→0; inst0018:0000::whole 0→1; inst0005:0000::whole 1→0; inst0017:0004::whole 1→0; inst0016:0002::whole 1→0; inst0017:0001::whole 0→1; inst0016:0001::whole 1→0; inst0023:0000::whole 0→1；重排路线 [0, 2, 3]。
加工负载 (3102.084, 3286.421, 2909.901, 3930.343)→(3198.874, 3189.631, 3433.85, 3406.394)；空走 546.778→516.005s；WAIT 0.000→0.000s；Cmax 4081.4384190260957→3587.1357411961835。
移除父焊缝：inst0002|group=642-BL14A.dxf|row=0000; inst0005|group=642-BL14A.dxf|row=0000; inst0012|group=145-LB20A|row=0000; inst0016|group=143-FR65A.dxf|row=0003; inst0016|group=143-FR65A.dxf|row=0004; inst0017|group=143-FR76A.dxf|row=0003; inst0023|group=804-FR313A|row=0000; inst0030|group=165-SR2A|row=0000。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0002:0000::whole→R2@13/1; inst0023:0000::whole→R2@14/1; inst0030:0000::whole→R0@5/0; inst0005:0000::whole→R2@8/0(inst0018:0000::whole:0→1); inst0016:0003::whole→R3@3/1(inst0016:0002::whole:1→0); inst0012:0000::whole→R0@4/1; inst0017:0003::whole→R2@13/1(inst0017:0004::whole:1→0); inst0016:0004::whole→R3@5/1(inst0017:0001::whole:0→1,inst0016:0001::whole:1→0)。

### g36_w086:A1_WAIT_ANCHORED：g36_w086 A1_WAIT_ANCHORED q8 attempt=5

状态 FEASIBLE；归属变化：无；方向变化：inst0013:0003::whole 1→0; inst0013:0004::whole 0→1; inst0024:0001::whole 0→1; inst0024:0002::whole 1→0; inst0016:0002::whole 0→1; inst0000:0000::whole 1→0; inst0013:0002::whole 0→1；重排路线 [1, 2, 3]。
加工负载 (3179.116, 3438.85, 3725.559, 3762.89)→(3179.116, 3438.85, 3725.559, 3762.89)；空走 701.237→607.333s；WAIT 39.932→0.000s；Cmax 3908.0208561480763→3888.8515935142673。
移除父焊缝：inst0001|group=888-BK304A|row=0000; inst0011|group=183-CM2B.dxf|row=0000; inst0012|group=143-FR65A.dxf|row=0001; inst0013|group=165-SR1A.dxf|row=0002; inst0013|group=165-SR1A.dxf|row=0003; inst0013|group=165-SR1A.dxf|row=0004; inst0024|group=804-FR315A.dxf|row=0001; inst0024|group=804-FR315A.dxf|row=0002。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0024:0001::whole→R1@12/0; inst0024:0002::whole→R1@13/1(inst0024:0001::whole:0→1); inst0013:0004::whole→R1@5/1; inst0001:0000::whole→R1@15/0(inst0024:0002::whole:1→0); inst0013:0002::whole→R3@3/0(inst0000:0000::whole:1→0); inst0013:0003::whole→R1@6/0; inst0011:0000::whole→R3@3/1(inst0016:0002::whole:0→1,inst0013:0002::whole:0→1); inst0012:0001::whole→R2@12/1。

### g45_w082:A1_WAIT_ANCHORED：g45_w082 A1_WAIT_ANCHORED q4 attempt=1

状态 FEASIBLE；归属变化：无；方向变化：inst0003:0005::0 1→0; inst0039:0000::whole 1→0; inst0035:0000::whole 0→1；重排路线 [1, 2]。
加工负载 (4480.188, 4607.537, 5165.998, 5048.722)→(4480.188, 4607.537, 5165.998, 5048.722)；空走 909.263→867.912s；WAIT 1426.590→857.617s；Cmax 6708.453657861711→6098.1301991058335。
移除父焊缝：inst0001|group=826BK312A|row=0000; inst0006|group=804-FR315A.dxf|row=0000; inst0007|group=144-DK1B|row=0000; inst0039|group=143-FR69B|row=0000。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0039:0000::whole→R2@2/0; inst0007:0000::whole→R2@8/1(inst0035:0000::whole:0→1); inst0006:0000::whole→R2@5/0; inst0001:0000::whole→R1@4/1(inst0003:0005::0:1→0)。

### g36_w086:失败或未改善：g36_w086 A0_RANDOM q4 attempt=0

状态 FEASIBLE；归属变化：无；方向变化：无；重排路线 [1, 3]。
加工负载 (3179.116, 3438.85, 3725.559, 3762.89)→(3179.116, 3438.85, 3725.559, 3762.89)；空走 701.237→696.119s；WAIT 39.932→35.327s；Cmax 3908.0208561480763→3908.0208561480763。
移除父焊缝：inst0011|group=183-CM2B.dxf|row=0000; inst0013|group=165-SR1A.dxf|row=0001; inst0024|group=804-FR315A.dxf|row=0001; inst0031|group=804-FR315A.dxf|row=0001。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0031:0001::whole→R3@19/1; inst0011:0000::whole→R3@6/1; inst0013:0001::whole→R1@15/1; inst0024:0001::whole→R1@18/0。

### g33_w055:失败或未改善：g33_w055 A0_RANDOM q4 attempt=2

状态 DEADLOCK；归属变化：inst0000:0000::whole R3→R2; inst0002:0000::whole R3→R2；方向变化：无；重排路线 [1]。
加工负载 (3102.084, 3286.421, 2909.901, 3930.343)→(3102.084, 3286.421, 3295.134, 3545.111)；空走 546.778→564.971s；WAIT 0.000→无成功调度；Cmax 4081.4384190260957→None。
移除父焊缝：inst0000|group=183-CM2B.dxf|row=0000; inst0002|group=642-BL14A.dxf|row=0000; inst0004|group=804-FR315A|row=0000; inst0019|group=145-LB17A.dxf|row=0002。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0002:0000::whole→R2@13/1; inst0004:0000::whole→R1@12/0; inst0019:0002::whole→R2@6/0; inst0000:0000::whole→R2@15/1。

### g45_w082:失败或未改善：g45_w082 A0_RANDOM q4 attempt=0

状态 DEADLOCK；归属变化：无；方向变化：无；重排路线 [0, 1, 3]。
加工负载 (4480.188, 4607.537, 5165.998, 5048.722)→(4480.188, 4607.537, 5165.998, 5048.722)；空走 909.263→887.795s；WAIT 1426.590→无成功调度；Cmax 6708.453657861711→None。
移除父焊缝：inst0003|group=804-TB310A.dxf|row=0006; inst0023|group=144-DK1B|row=0000; inst0031|group=165-SR2A.dxf|row=0000; inst0042|group=801-FR322A|row=0000。
动作（block→R@位置/方向，括号内是邻块 old→new）：inst0023:0000::whole→R2@3/0; inst0003:0006::whole→R3@2/1; inst0031:0000::whole→R1@16/0; inst0042:0000::whole→R0@0/0。


## X_SPLIT 正确性运行

{"status": "FEASIBLE", "cmax": 235.18518518518516, "directions": [[0], [0], [], []], "actions": [{"block_id": "formal-x::0", "robot": 0, "position": 0, "direction": 0, "neighbour_changes": []}, {"block_id": "formal-x::1", "robot": 1, "position": 0, "direction": 0, "neighbour_changes": []}]}

## 判断

验证：相关回归测试 **139 项通过**。六个共同初解及九个典型成功/失败案例已在禁止方向 DP、旧 REGION 路线 DP 代理及独立 certifier 调用的条件下重放。每条真实插入动作及邻块翻转均逐步重放，最终路线、显式方向和 Cmax 一致；60 行 CSV 的计数、空值、固定预算和改善百分比已交叉核对。X_SPLIT 两子块固定 R0/R1、空路线方向为空且 B32 返回 FEASIBLE。

具体判断：A1_RANDOM 的逐次配对成功数与 A0_RANDOM 同为 60/96；去重后的 60 对 59 来自 A0 的一个重复候选，不能据此声称可调度性改善。共同成功的 53 对中，A1 胜/平/A0 胜为 20/25/8；按每例池内最佳 Cmax 比较为 2 胜、4 平。方向联动有小幅质量收益，尚非稳定全面优势。

TRAVEL 改善覆盖 6/6 例，LOAD 为 5/6；两者改善候选数仍少于 RANDOM。WAIT 在 3 个适用实例均找到改善，但总体仅 22/48 成功，尤其 g45 仅 1/16 成功。当前优先保留 RANDOM+TRAVEL，LOAD 保留为负载交换来源；WAIT 代理及局部空走评分对协调冲突的识别仍需改进。排序、归属和方向同时改变时，不能把全部收益单独归因于方向翻转。

1. 方向嵌入的质量机会：相同 RANDOM 移除输入，A0/A1 改善候选 15/21，组内唯一候选的 B32 成功数 59/60；运行时间 35.331/36.721s。逐次配对共同成功 53 次，A1 胜/平/A0 胜为 20/25/8；仅 A1/仅 A0 成功 7/7。配对按尝试计，重复候选复用该组已完成的调度结果，因此与去重成功数略有区别。
本轮方向嵌入比固定方向有更多改善机会，但不是逐例占优。逐次配对的调度成功变化应单独判断，不能把改善候选增加解释为可调度率同步提高。
2. 选区：改善覆盖实例数 {'A0_RANDOM': 4, 'A1_RANDOM': 4, 'A1_LOAD': 5, 'A1_TRAVEL': 6, 'A1_WAIT_ANCHORED': 3}。RANDOM 是必要对照；在本轮 LOAD/TRAVEL/WAIT 的比较中，应同时比较成功数、改善数和同实例最佳 Cmax。WAIT 不适用组不能计为失败，也不能用不同实例集合的原始成功率直接排名。
瓶颈选区的改善数：LOAD=13、TRAVEL=15、WAIT=4。LOAD/TRAVEL 与 RANDOM 使用同样尝试数，本轮仍应保留 RANDOM；WAIT 应只在有等待的共同实例上解释。
3. 联合专家：A1 中有 27 个组内改善候选同时改变归属、共同块相对顺序及方向，已经具备完整动作及邻块翻转记录。该数未跨选区去重，不能当作独特训练样本数；只说明联合补全动作可执行。
4. 下一步：已有方向嵌入和专家轨迹的开发证据，但当前可调度性、几何覆盖及初解质量仍不足以直接支持正式 GNN 训练结论。先用同一移除集合控制有效结构候选覆盖，改善局部评分对同轨交错/协调冲突的识别，并积累去重、多初解轨迹；避免把局部空走最优当作 Cmax 最优。本次不启动训练。
新初解经逐步定向构造，和上一轮初解不同；不可用本轮改善数或相对改善直接宣称优于上一轮独立 DP 流程。
- g33_w055：A1 组内完整候选 48，B32 成功 35，其中改变归属 27；移除块中有其他合法机器人可选的累计次数 288，来自当前最大完工路线的移除块累计 58；最佳 Cmax=3587.1357411961835。这些计数不能把无改善解释为最优，或直接归因于完全缺乏合法动作。
- g36_w086：A1 组内完整候选 64，B32 成功 50，其中改变归属 19；移除块中有其他合法机器人可选的累计次数 384，来自当前最大完工路线的移除块累计 71；最佳 Cmax=3888.8515935142673。这些计数不能把无改善解释为最优，或直接归因于完全缺乏合法动作。
- g45_w082：A1 组内完整候选 64，B32 成功 14，其中改变归属 3；移除块中有其他合法机器人可选的累计次数 435，来自当前最大完工路线的移除块累计 87；最佳 Cmax=6098.1301991058335。这些计数不能把无改善解释为最优，或直接归因于完全缺乏合法动作。

## 复现

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B DRL/scripts/run_joint_oriented_repair_pilot.py --instances g21_w035 g15_w035 g36_w086 g33_w055 g31_w055 g45_w082 --attempts 8 --q 4 8 --seed 20261027 --insertion-limit 8 --repair-budget 4096
```
