# 瓶颈驱动固定模式邻域开发实验（2026-10-27）

seed=20261027；q=[4, 8]；每组 8 次；REGRET_2_REPAIR；insertion_limit=8；max_insertion_evaluations=4096。
FORMAL_SCOPE_V2：焊速 0.0108、空走 0.20 m/s，准备/后处理 20/30 s，X/Y 窗口 0.50 m，最短子段 0.20 m。开放路线、首任务免费部署。
共同 REGION 初解经公共方向 DP、B32 和原 certifier；仅初解失败才使用已有 construct_solution 的 RULE/X。每组均从同一个不可变初解开始；候选逐个公共评价，无 C2/C4。模式规范化改变时统一拒绝。
WAIT_ANCHORED 是真实 WAIT 位置、时段和端点空间相关代理，未声称识别真实 blocker；调度器未改动。无 WAIT 的组不适用。DEADLOCK 只表示 B32 有限搜索失败；失败 Cmax 留空，不参与均值。
唯一候选按四条块路线去重（方向随后由公共 DP 决定）；组间独立评价。restored_original 按全部成功构造次数计数。方向可行数按实际进入 B32 的候选计；严格改善容差为 1e-9×初始 Cmax。运行秒为选区、修复和公共评价总墙钟，不含初解。

## 实例与共同初解

- g21_w035：`data/DEV_ONLY/seed_000303.xlsx`；35 父焊缝；FALLBACK:RULE_X；Cmax=2449.297911s；初解耗时 0.289s；模式 {'WHOLE': 34, 'Y_SPLIT': 1}；加工负载 [2200.38, 2383.647, 1187.032, 1173.584]；WAIT=0.000s。
- g15_w035：`data/ID_TEST/seed_1142967162.xlsx`；35 父焊缝；FALLBACK:RULE_X；Cmax=1663.724131s；初解耗时 0.755s；模式 {'WHOLE': 35}；加工负载 [1483.641, 1331.383, 1571.766, 1596.918]；WAIT=0.000s。
- g36_w086：`data/PPO_TRAIN/seed_0834257622.xlsx`；86 父焊缝；REGION；Cmax=3846.662442s；初解耗时 2.468s；模式 {'WHOLE': 86}；加工负载 [3179.116, 3438.85, 3725.559, 3762.89]；WAIT=0.000s。
- g33_w055：`data/PPO_TRAIN/seed_1745884954.xlsx`；55 父焊缝；REGION；Cmax=4107.881899s；初解耗时 0.900s；模式 {'WHOLE': 55}；加工负载 [3296.98, 3091.524, 3568.818, 3271.426]；WAIT=748.978s。
- g31_w055：`data/PPO_TRAIN/seed_1872211770.xlsx`；55 父焊缝；REGION；Cmax=3122.275010s；初解耗时 0.735s；模式 {'WHOLE': 50, 'Y_SPLIT': 5}；加工负载 [3023.985, 2922.997, 2763.495, 2709.909]；WAIT=0.000s。
- g45_w082：`data/VALIDATION/seed_0151520473.xlsx`；82 父焊缝；REGION；Cmax=5865.420310s；初解耗时 0.706s；模式 {'WHOLE': 74, 'Y_SPLIT': 8}；加工负载 [4480.188, 4607.537, 5165.998, 5048.722]；WAIT=637.133s。

数据只通过现有 representatives_v05 的 MLP_DEV 六实例和 load_parents 白名单读取。g15 的历史目录名为 ID_TEST，但它属于已消耗开发白名单；未读取当前封存测试角色的几何。

## 全部分组结果

|实例|选区|q|初始 Cmax|尝试/唯一部分解|唯一候选|方向可行|认证|改善|最佳 Cmax|最佳改善%|秒|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|g21_w035|RANDOM|4|2449.298|8/8|8|8|6|0|2449.298|0.000|0.72|
|g21_w035|LOAD|4|2449.298|8/8|8|7|6|0|2449.298|0.000|0.97|
|g21_w035|TRAVEL|4|2449.298|8/8|8|8|7|0|2449.298|0.000|1.29|
|g21_w035|WAIT_ANCHORED|4|2449.298|0/0|0|0|0|0|—|—|0.00|
|g21_w035|RANDOM|8|2449.298|8/8|8|8|6|1|2401.128|1.967|2.22|
|g21_w035|LOAD|8|2449.298|8/8|8|8|6|0|2471.248|-0.896|1.10|
|g21_w035|TRAVEL|8|2449.298|8/8|8|8|6|0|2449.298|0.000|1.80|
|g21_w035|WAIT_ANCHORED|8|2449.298|0/0|0|0|0|0|—|—|0.00|
|g15_w035|RANDOM|4|1663.724|8/8|7|6|6|0|1663.724|0.000|0.58|
|g15_w035|LOAD|4|1663.724|8/8|8|8|8|0|1663.724|0.000|0.85|
|g15_w035|TRAVEL|4|1663.724|8/8|7|6|6|1|1653.687|0.603|0.56|
|g15_w035|WAIT_ANCHORED|4|1663.724|0/0|0|0|0|0|—|—|0.00|
|g15_w035|RANDOM|8|1663.724|8/8|8|6|4|0|1663.724|0.000|1.94|
|g15_w035|LOAD|8|1663.724|8/8|8|6|5|0|1663.724|0.000|1.98|
|g15_w035|TRAVEL|8|1663.724|8/8|8|8|4|0|1679.680|-0.959|2.61|
|g15_w035|WAIT_ANCHORED|8|1663.724|0/0|0|0|0|0|—|—|0.00|
|g36_w086|RANDOM|4|3846.662|8/8|8|8|8|0|3846.662|0.000|3.14|
|g36_w086|LOAD|4|3846.662|8/8|8|8|8|0|3846.662|0.000|4.78|
|g36_w086|TRAVEL|4|3846.662|8/8|8|8|8|0|3846.662|0.000|2.89|
|g36_w086|WAIT_ANCHORED|4|3846.662|0/0|0|0|0|0|—|—|0.00|
|g36_w086|RANDOM|8|3846.662|8/8|8|8|8|0|3848.250|-0.041|4.63|
|g36_w086|LOAD|8|3846.662|8/8|8|8|7|0|3846.662|0.000|3.16|
|g36_w086|TRAVEL|8|3846.662|8/8|8|8|5|0|3854.945|-0.215|3.35|
|g36_w086|WAIT_ANCHORED|8|3846.662|0/0|0|0|0|0|—|—|0.00|
|g33_w055|RANDOM|4|4107.882|8/8|8|7|2|2|4104.482|0.083|4.73|
|g33_w055|LOAD|4|4107.882|8/8|8|7|2|0|4107.882|0.000|4.38|
|g33_w055|TRAVEL|4|4107.882|8/8|8|8|2|0|4107.882|0.000|5.28|
|g33_w055|WAIT_ANCHORED|4|4107.882|8/8|8|8|6|1|4104.564|0.081|6.30|
|g33_w055|RANDOM|8|4107.882|8/8|8|8|0|0|—|—|7.36|
|g33_w055|LOAD|8|4107.882|8/8|8|6|2|1|3623.632|11.788|5.96|
|g33_w055|TRAVEL|8|4107.882|8/8|8|8|4|2|3739.417|8.970|7.03|
|g33_w055|WAIT_ANCHORED|8|4107.882|8/8|8|8|3|2|3789.492|7.751|8.81|
|g31_w055|RANDOM|4|3122.275|8/8|8|8|7|0|3122.275|0.000|3.86|
|g31_w055|LOAD|4|3122.275|8/8|7|7|7|0|3122.275|0.000|2.07|
|g31_w055|TRAVEL|4|3122.275|8/8|8|8|7|0|3122.275|0.000|3.22|
|g31_w055|WAIT_ANCHORED|4|3122.275|0/0|0|0|0|0|—|—|0.00|
|g31_w055|RANDOM|8|3122.275|8/8|8|8|6|0|3170.343|-1.540|3.68|
|g31_w055|LOAD|8|3122.275|8/8|8|8|4|0|3147.774|-0.817|3.97|
|g31_w055|TRAVEL|8|3122.275|8/8|8|8|6|1|3121.741|0.017|4.30|
|g31_w055|WAIT_ANCHORED|8|3122.275|0/0|0|0|0|0|—|—|0.00|
|g45_w082|RANDOM|4|5865.420|8/8|8|8|8|1|5788.709|1.308|2.66|
|g45_w082|LOAD|4|5865.420|8/8|8|8|8|1|5722.790|2.432|2.49|
|g45_w082|TRAVEL|4|5865.420|8/8|8|8|7|1|5772.536|1.584|2.20|
|g45_w082|WAIT_ANCHORED|4|5865.420|8/8|8|8|7|2|5650.717|3.660|2.73|
|g45_w082|RANDOM|8|5865.420|8/8|8|7|7|0|5865.420|0.000|3.12|
|g45_w082|LOAD|8|5865.420|8/8|8|6|6|1|5820.264|0.770|2.62|
|g45_w082|TRAVEL|8|5865.420|8/8|8|8|8|0|5865.420|0.000|4.92|
|g45_w082|WAIT_ANCHORED|8|5865.420|8/8|8|8|8|2|5624.896|4.101|3.84|

## 各组选区的最佳相对改善候选

ID 简写保留 inst、row 编号和子块后缀。下面给出每类选区最佳相对改善，而非把不同实例的 Cmax 直接比较。

### RANDOM / g21_w035 / RANDOM:q8:0

归属：`inst0001|group=165-SR2A.dxf|row=0002::1` R0→R1; `inst0002|group=143-FR76A.dxf|row=0000::whole` R0→R1; `inst0004|group=312-GR2C|row=0000::whole` R1→R0。改变路线 [0, 1, 2, 3]；共同保留块相对顺序改变路线 [1, 2, 3]。
最大加工负载 2383.647→2303.307s；总空走 347.014→299.043s；总 WAIT 0.000→0.000s；Cmax 2449.298→2401.128s。

### LOAD / g33_w055 / LOAD:q8:2

归属：无归属变化。改变路线 [0]；共同保留块相对顺序改变路线 [0]。
最大加工负载 3568.818→3568.818s；总空走 290.833→301.674s；总 WAIT 748.978→0.000s；Cmax 4107.882→3623.632s。

### TRAVEL / g33_w055 / TRAVEL:q8:6

归属：`inst0013|group=888-BK304A|row=0000::whole` R1→R0; `inst0032|group=804-FR315A.dxf|row=0001::whole` R2→R3。改变路线 [0, 1, 2, 3]；共同保留块相对顺序改变路线 [0, 1, 3]。
最大加工负载 3568.818→3445.286s；总空走 290.833→356.022s；总 WAIT 748.978→237.549s；Cmax 4107.882→3739.417s。

### WAIT_ANCHORED / g33_w055 / WAIT_ANCHORED:q8:2

归属：`inst0013|group=888-BK304A|row=0000::whole` R1→R0。改变路线 [0, 1]；共同保留块相对顺序改变路线 [0, 1]。
最大加工负载 3568.818→3568.818s；总空走 290.833→290.585s；总 WAIT 748.978→274.071s；Cmax 4107.882→3789.492s。

## 联合改善案例路线和时间线：g33_w055 / TRAVEL:q8:6

任务标记为 inst:row::子块/方向。W=焊接，p=准备/后处理，m=空走，.=WAIT；时间线按各自 Cmax 缩放到 100 列，仅供解释。

- R0：`inst0003:0000::whole/1 → inst0015:0001::whole/0 → inst0015:0003::whole/0 → inst0012:0000::whole/1 → inst0011:0000::whole/1 → inst0009:0000::whole/0 → inst0015:0004::whole/1 → inst0008:0000::whole/1` → `inst0009:0000::whole/1 → inst0011:0000::whole/0 → inst0012:0000::whole/0 → inst0003:0000::whole/1 → inst0015:0001::whole/0 → inst0015:0003::whole/0 → inst0015:0004::whole/1 → inst0008:0000::whole/1 → inst0013:0000::whole/0`
- R1：`inst0013:0000::whole/1 → inst0030:0000::whole/1 → inst0015:0000::whole/0 → inst0021:0000::whole/0 → inst0015:0002::whole/0 → inst0029:0000::whole/0 → inst0029:0001::whole/0 → inst0029:0002::whole/1 → inst0025:0000::whole/1 → inst0010:0000::whole/1 → inst0024:0000::whole/0 → inst0004:0000::whole/0 → inst0020:0000::whole/0 → inst0022:0000::whole/0 → inst0014:0000::whole/1 → inst0006:0000::whole/1` → `inst0015:0002::whole/1 → inst0030:0000::whole/1 → inst0015:0000::whole/0 → inst0021:0000::whole/0 → inst0029:0000::whole/0 → inst0029:0001::whole/0 → inst0029:0002::whole/1 → inst0025:0000::whole/1 → inst0010:0000::whole/1 → inst0024:0000::whole/0 → inst0004:0000::whole/0 → inst0020:0000::whole/0 → inst0022:0000::whole/0 → inst0014:0000::whole/1 → inst0006:0000::whole/1`
- R2：`inst0028:0000::whole/1 → inst0032:0000::whole/0 → inst0032:0001::whole/1 → inst0018:0000::whole/0 → inst0005:0000::whole/1 → inst0032:0002::whole/0 → inst0019:0001::whole/0 → inst0019:0004::whole/0 → inst0019:0003::whole/1 → inst0019:0002::whole/1 → inst0019:0000::whole/1 → inst0007:0000::whole/1` → `inst0028:0000::whole/1 → inst0032:0000::whole/0 → inst0018:0000::whole/0 → inst0005:0000::whole/1 → inst0032:0002::whole/0 → inst0019:0001::whole/0 → inst0019:0004::whole/0 → inst0019:0003::whole/1 → inst0019:0002::whole/1 → inst0019:0000::whole/1 → inst0007:0000::whole/1`
- R3：`inst0017:0005::whole/0 → inst0017:0006::whole/1 → inst0017:0004::whole/1 → inst0017:0003::whole/0 → inst0016:0001::whole/1 → inst0016:0002::whole/0 → inst0016:0003::whole/1 → inst0016:0004::whole/0 → inst0017:0002::whole/1 → inst0017:0001::whole/0 → inst0017:0000::whole/1 → inst0002:0000::whole/1 → inst0027:0000::whole/0 → inst0016:0000::whole/0 → inst0000:0000::whole/1 → inst0001:0000::whole/0 → inst0023:0000::whole/0 → inst0031:0000::whole/1 → inst0026:0000::whole/0` → `inst0032:0001::whole/1 → inst0017:0005::whole/1 → inst0017:0006::whole/1 → inst0017:0004::whole/1 → inst0017:0003::whole/0 → inst0016:0001::whole/1 → inst0016:0002::whole/0 → inst0016:0003::whole/1 → inst0016:0004::whole/0 → inst0017:0002::whole/1 → inst0017:0001::whole/0 → inst0017:0000::whole/1 → inst0002:0000::whole/1 → inst0031:0000::whole/0 → inst0027:0000::whole/0 → inst0016:0000::whole/0 → inst0000:0000::whole/1 → inst0001:0000::whole/0 → inst0023:0000::whole/0 → inst0026:0000::whole/0`

```text
BEFORE
0                                                                                        4107.9s
R0 |..................WWWWWWWWWWWWWWWWWWWWWWWWWWWWppWWWWWWWpWWWWWWWppWWWWWWpWWWWWWWmWWppWWWWWWWmWWWWWWWp| 4107.9
R1 |WWppWpWWpWWpWWWWWWWppWWpWWWpWWWWWmWWWWWWWmWWWWWWWWWWmWppWpWWWpWWmWWWWWWWmWWWWp                      | 3199.6
R2 |WWmWWWpWWmWppWWWppWWWWpWWWWWWWpWWWWWWWpWWWWWWWppWWWWWWWpWppWWWWWWWWWWWWWWWWWWWWWWWWWWWWpp           | 3623.6
R3 |WWWpWWWpWWWWppWWWWpWWWpWWWWWppWppWppWWWWppWWWWWpWpWWWWpWWWWpWpWWWWpWppWpWWWWWppWpp                  | 3337.4
AFTER
0                                                                                        3739.4s
R0 |WWpmWWWWWWWpWWWWWWWpmWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWppWWWWWWWWpWWWWWWWWpWWWWWWWWpWWWWWWWWmWWWpp     | 3516.1
R1 |WWWWWWWWpWWmWWpWWppWWppWWWpWWWWWpWWWWWWWWpWWWWWWWWWWWpWWmWWpWWWmWWppWWWWWWWpWWWWpp                  | 3051.3
R2 |WWppWWppWppWWWpmWWWWppWWWWWWWWpWWWWWWWWpWWWWWWWWpWWWWWWWWpWWp.......WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWp| 3739.4
R3 |WWpmWWWppWWWppWWWWpWWWWWppWWppWWWWWWpWWpWWppWWWWWpWWWWWppWppWWWppWWWWWWmWWWWpWWpWWWWpWWpWWmWWpp     | 3515.5
```

## 构造与失败诊断

movable/positions 是每次部分解的静态合法机会累计（位置数不含两方向和双子块笛卡尔组合），不是已经枚举的动作数。CSV 另含真实插入评分次数、固定模式拒绝、原结构恢复、同轨/跨轨迁移和多路线重排计数。

- RANDOM：{"attempts": 96, "unique_partials": 96, "unique_candidates": 95, "certified": 68, "improved": 4, "restored_original": 3, "movable_blocks": 597, "same_rail_movable_blocks": 597, "cross_rail_movable_blocks": 27, "legal_positions": 9605, "changed_assignment": 32, "same_rail_candidates": 30, "cross_rail_candidates": 3, "reordered_multiple_routes": 42}
- LOAD：{"attempts": 96, "unique_partials": 96, "unique_candidates": 95, "certified": 69, "improved": 3, "restored_original": 3, "movable_blocks": 601, "same_rail_movable_blocks": 601, "cross_rail_movable_blocks": 24, "legal_positions": 9487, "changed_assignment": 34, "same_rail_candidates": 32, "cross_rail_candidates": 2, "reordered_multiple_routes": 36}
- TRAVEL：{"attempts": 96, "unique_partials": 96, "unique_candidates": 95, "certified": 70, "improved": 5, "restored_original": 3, "movable_blocks": 604, "same_rail_movable_blocks": 604, "cross_rail_movable_blocks": 40, "legal_positions": 9624, "changed_assignment": 27, "same_rail_candidates": 24, "cross_rail_candidates": 6, "reordered_multiple_routes": 47}
- WAIT_ANCHORED：{"attempts": 32, "unique_partials": 32, "unique_candidates": 32, "certified": 24, "improved": 7, "restored_original": 0, "movable_blocks": 204, "same_rail_movable_blocks": 204, "cross_rail_movable_blocks": 1, "legal_positions": 3161, "changed_assignment": 10, "same_rail_candidates": 10, "cross_rail_candidates": 0, "reordered_multiple_routes": 16}

少量实际未改善/失败案例（插入标记为 block→R@位置/方向）：

- RANDOM：COMPLETED；移除 ['inst0005:0000', 'inst0011:0001', 'inst0012:0001', 'inst0016:0001']；可迁移块 4，静态合法位置 58，合法机器人数量分布 {'2': 4}；实际插入 `inst0011:0001::whole→R2@2/0; inst0016:0001::whole→R0@1/1; inst0012:0001::whole→R3@2/0; inst0005:0000::whole→R2@5/0`。
  Cmax 2449.298→2449.298，改变归属父数 0，重排路线 [2, 3]。
- LOAD：COMPLETED；移除 ['inst0002:0005', 'inst0004:0000', 'inst0008:0000', 'inst0018:0000']；可迁移块 4，静态合法位置 52，合法机器人数量分布 {'2': 4}；实际插入 `inst0004:0000::whole→R1@2/1; inst0008:0000::whole→R1@5/0; inst0018:0000::whole→R0@13/1; inst0002:0005::whole→R0@10/1`。
  Cmax 2449.298→3676.271，改变归属父数 1，重排路线 [0, 1]。
- TRAVEL：COMPLETED；移除 ['inst0002:0006', 'inst0004:0000', 'inst0008:0000', 'inst0020:0000']；可迁移块 4，静态合法位置 56，合法机器人数量分布 {'2': 4}；实际插入 `inst0004:0000::whole→R1@2/1; inst0020:0000::whole→R1@0/0; inst0008:0000::whole→R1@7/0; inst0002:0006::whole→R0@10/1`。
  Cmax 2449.298→2728.254，改变归属父数 1，重排路线 [0, 1]。
- WAIT_ANCHORED：COMPLETED；移除 ['inst0003:0000', 'inst0004:0000', 'inst0015:0003', 'inst0029:0002']；可迁移块 4，静态合法位置 60，合法机器人数量分布 {'2': 4}；实际插入 `inst0015:0003::whole→R0@1/0; inst0003:0000::whole→R0@0/1; inst0004:0000::whole→R1@10/0; inst0029:0002::whole→R1@6/1`。
  Cmax 4107.882→4109.713，改变归属父数 0，重排路线 [1]。

## 结论

本轮完成 320 次适用选区尝试、317 个组内唯一完整候选；状态 {'COMPLETED': 231, 'DEADLOCK': 72, 'DIRECTION_INFEASIBLE': 14}；19 个组内改善候选。跨组可能重合，不能据此声称全局同样数量的独特专家。选区与候选评价总耗时 134.124s（不含初解与输入加载）。
原结构仅恢复 9/320 次；累计组内唯一部分解 320，且实际出现同轨左右迁移、跨轨迁移和多路线相对重排。因此当前没有证据把失败主要归因于选区重复或机器人完全不可迁移。
Regret-2 的 `_parent_alternatives` 依次比较最大加工负载和局部有向空走增量，没有 WAIT/同轨路线交错项；它的两优方案可能只是同一归属/位置的不同方向提示，方向随后又被公共 DP 重算。`remaining_budget` 按确定性机器人/位置/方向循环截断，双子块时可能只访问动作前缀；静态合法位置数不等于全部已搜索。这些是候选质量限制的代码机制，尚未做评分/枚举消融，不能把它们当成已证实因果。此次未改 Regret-2 核心评分。
方向可行而全部 B32 失败的分组：g33_w055/RANDOM/q8，8 个。
本轮无改善实例（实例、尝试、完整唯一候选）：[('g36_w086', 48, 48)]。请结合静态合法位置、迁移资格及真实修复 trace 判断，不能把无改善解释为数学不可改善。
各类选区改善覆盖实例数：{'RANDOM': 3, 'LOAD': 2, 'TRAVEL': 4, 'WAIT_ANCHORED': 2}。当前结果支持先研究共同修复器的联合补全和调度感知能力，并保持 RANDOM 对照。WAIT 的适用样本少，应按存在等待的共同实例比较，不能直接与全部六例比较成功率。
已有机会证据，尚无足够广度和独特专家数量直接支持启动神经训练。先在相同移除集合上改善 Regret-2 的有效归属/位置方案多样性和可调度率，再建立部分解补全专家。真正 blocker 证据值得在后续小范围旁路保留，用于解释 WAIT 相关重构；本轮的代理结果不构成 blocker 因果证明。
本轮实际初解只出现 WHOLE/Y_SPLIT，未实测 X_SPLIT 邻域收益；X 固定左右约束继续复用已有代码和相关测试。

## 复现

相关回归测试实际通过 **132 项**（LNS、初始化、方向 DP、公共评价、邻域、调度等价性及规范化）。四类最佳案例已从原种子、移除集合和初解重新修复、经公共评价重放，Cmax 与任务变化摘要一致。48 行 CSV 的共同初始 Cmax、状态计数、空值和改善百分比已交叉核对。

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B DRL/scripts/run_bottleneck_neighborhood_pilot.py --instances g21_w035 g15_w035 g36_w086 g33_w055 g31_w055 g45_w082 --attempts 8 --q 4 8 --seed 20261027 --insertion-limit 8 --repair-budget 4096
```
