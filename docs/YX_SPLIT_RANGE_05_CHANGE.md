# Y/X 拆分范围 0.5 m：正确性与 development sensitivity

当前默认 delta_y=delta_x=0.50 m；最短子焊缝仍为 0.20 m；干涉距离仍为 0.50 m。Y 候选为 5.5/5.8/6.0/6.2/6.5，X 为各轨固定中心 ±0.5/±0.2/0，保留合法 MIDPOINT。旧点身份保留，新增点使用 OUTER_LOWER/OUTER_UPPER；几何重复去重。按用户最新指示，本轮只保留 A/D 对比，不分析 B/C 单变量方案。

覆盖区域是任务层假设：上轨覆盖 y≥5.5，下轨覆盖 y≤6.5；未做 URDF/IK 物理可达性验证。Y 覆盖改变 WHOLE/mandatory Y 及 X 中心，A/D 不是同一可行域下的纯算法竞赛。

Phase3-Z、Phase4-0/0B、Phase4-1A/B/C 均保持历史 0.2 配置身份；旧标签、模型与结论未重写。本轮不训练 MLP/GAT、不采集标签，17 TRAIN / 6 DEV 划分未改变。

## 实验设置

A=(delta_y 0.2, delta_x 0.2)，D=(delta_y 0.5, delta_x 0.5)。六例为既有 Phase4-1A 的 I2/I3、I5/I6、I9/I12，每档两例，均为 V2_MODEL_DEVELOPMENT_CONSUMED；读取前检查既有数据角色，不访问 validation、ORACLE 或 sealed ID_TEST。seeds=20261081、20261082，实例与顺序在求解前固定。

SA_OI_ALNS_V2；M192=144 atomic+48 LNS；Kdp8/Kref2/Kref_total4；heuristic C2/C4；TWO_OPT_STAR OFF；B32、SA、direction 与候选生成器其余设置一致。初始化复用此前 construction_budget=5 / kinit_ref=5 / bootstrap=1。每 run 墙钟 60 s，四个 worker 各绑定一个物理核的一个逻辑线程，A/D 轮换顺序。特征/选择/观察记录耗时计入预算。Cmax@60 使用原生 anytime，绝不用 overshoot final_cmax 决胜。@60 解另做独立 certifier 复核；计数/负载/行走/等待均对应该解，final_patterns 另存原始记录。

完整测试：在 DRL 目录以 D:\pybullet_test\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider 执行；最终299 passed in 38.54s（实验前299 passed in 35.62s）。

| 实例 | 档位 | workbook / sheet |
| --- | --- | --- |
| I2 | SMALL | data/PPO_TRAIN/seed_1064648049.xlsx / g20_w026 |
| I3 | SMALL | data/DEV_ONLY/seed_000505.xlsx / g15_w026 |
| I5 | MEDIUM | data/PPO_TRAIN/seed_0900775608.xlsx / g30_w055 |
| I6 | MEDIUM | data/PPO_TRAIN/seed_1419626827.xlsx / g29_w055 |
| I9 | LARGE | data/PPO_TRAIN/seed_0194123089.xlsx / g43_w085 |
| I12 | LARGE | data/PPO_TRAIN/seed_0211781140.xlsx / g32_w085 |

## 静态目录（零 reference 调用）

| 实例 | 组 | WHOLE | mandatory Y | optional Y | Y patterns | X parents | X patterns | 无合法拆分 | 短 Y 删除 | 短 X 删除 | x_up | x_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | A | 24 | 2 | 6 | 11 | 18 | 32 | 5 | 5 | 1 | 11.7294 | 10.3730 |
| 2 | D | 25 | 1 | 10 | 24 | 18 | 38 | 5 | 5 | 8 | 12.4917 | 10.3730 |
| 3 | A | 20 | 6 | 4 | 24 | 16 | 21 | 4 | 1 | 2 | 8.1368 | 9.3670 |
| 3 | D | 20 | 6 | 4 | 41 | 16 | 26 | 4 | 1 | 2 | 8.1368 | 9.3670 |
| 5 | A | 54 | 1 | 1 | 5 | 46 | 61 | 8 | 1 | 1 | 12.7720 | 5.2779 |
| 5 | D | 54 | 1 | 2 | 8 | 46 | 72 | 7 | 3 | 1 | 12.7720 | 5.2779 |
| 6 | A | 54 | 1 | 0 | 2 | 43 | 53 | 11 | 2 | 1 | 9.2387 | 10.8322 |
| 6 | D | 55 | 0 | 2 | 5 | 44 | 62 | 11 | 3 | 1 | 9.2387 | 11.1876 |
| 9 | A | 84 | 1 | 3 | 6 | 59 | 65 | 24 | 3 | 4 | 11.1723 | 9.3367 |
| 9 | D | 85 | 0 | 5 | 12 | 59 | 67 | 24 | 3 | 7 | 11.1723 | 9.3367 |
| 12 | A | 78 | 7 | 3 | 25 | 45 | 51 | 33 | 2 | 7 | 10.1890 | 7.8020 |
| 12 | D | 80 | 5 | 11 | 43 | 46 | 71 | 32 | 6 | 8 | 10.1890 | 7.9211 |

WHOLE/mandatory/optional 统计父焊缝；X patterns 按轨计数。无合法拆分表示仅 WHOLE 可选，不等于无可行任务。短段删除为同一候选生成器在去除 0.2 m 下限前后、按几何去重后的差值。纵向/端点/区域外候选不计为短段删除。固定中心的旧 ±0.2 偏移保留；Y 资格改变时中心可移动，不能声称 A 的所有绝对 X 切点都嵌套于 D。

## 60 秒实验

已完成 24/24。

| 档位 | 组 | 认证 | 未认证 | Cmax均值 | 迭代均值 | ref均值 | 有效率 | 负载差 | 空行走s | 等待s | overshoot s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SMALL | A | 4 | 0 | 1754.0721 | 15.7500 | 53.5000 | 0.5565 | 195.3759 | 268.4646 | 48.8684 | 1.9387 |
| SMALL | D | 4 | 0 | 2039.1111 | 15.2500 | 49 | 0.5788 | 794.4566 | 280.6117 | 12.4753 | 2.1576 |
| MEDIUM | A | 4 | 0 | 5024.3957 | 11.2500 | 35.5000 | 0.5819 | 2844.1216 | 698.0307 | 791.0261 | 3.1974 |
| MEDIUM | D | 4 | 0 | 5065.9630 | 11.7500 | 42 | 0.5939 | 3132.3029 | 464.3888 | 469.2088 | 2.7544 |
| LARGE | A | 4 | 0 | 6534.5059 | 5.7500 | 23 | 0.6013 | 4319.0737 | 821.1971 | 81.6162 | 3.2357 |
| LARGE | D | 4 | 0 | 6418.9911 | 7.7500 | 28.5000 | 0.6176 | 4199.0373 | 781.2466 | 154.9501 | 3.9535 |

配对差以同实例同 seed 的 A 为参照，负数表示 Cmax 改善。均值只包含双方 @60 认证的完整配对。

| 档位 | 组-A | 配对数 | Cmax差均值s | 相对差中位% | 改善 | 相同 | 变差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SMALL | D | 4 | 285.0389 | 13.3969 | 2 | 0 | 2 |
| MEDIUM | D | 4 | 41.5673 | -6.1972 | 3 | 0 | 1 |
| LARGE | D | 4 | -115.5148 | -4.4568 | 3 | 0 | 1 |
| ALL | D | 12 | 70.3638 | -1.6688 | 8 | 0 | 4 |

逐次结果（WHOLE/Y/X 为 @60 解；空行走与等待单位秒）：

| I | seed | 组 | 认证 | Cmax@60 | 迭代 | ref | 有效率 | WHOLE/Y/X | 负载差 | 空行走 | 等待 | elapsed | overshoot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 20261081 | A | True | 1639.1546 | 21 | 58 | 0.5335 | 24/2/0 | 257.6845 | 286.0721 | 88.0128 | 61.6409 | 1.6409 |
| 2 | 20261081 | D | True | 1599.1751 | 17 | 53 | 0.5506 | 23/2/1 | 93.5300 | 333.1062 | 0 | 60.7151 | 0.7151 |
| 2 | 20261082 | A | True | 1624.5776 | 20 | 70 | 0.4977 | 23/2/1 | 425.8332 | 207.1845 | 0 | 60.9984 | 0.9984 |
| 2 | 20261082 | D | True | 1609.9801 | 18 | 55 | 0.5275 | 24/2/0 | 156.1409 | 270.8473 | 0 | 62.8559 | 2.8559 |
| 3 | 20261081 | A | True | 1962.3247 | 10 | 41 | 0.6016 | 19/7/0 | 21.2400 | 306.9330 | 107.4607 | 61.4855 | 1.4855 |
| 3 | 20261081 | D | True | 2505.7368 | 12 | 40 | 0.6276 | 20/6/0 | 1559.6477 | 255.5099 | 0 | 60.9531 | 0.9531 |
| 3 | 20261082 | A | True | 1790.2316 | 12 | 45 | 0.5933 | 20/6/0 | 76.7457 | 273.6686 | 0 | 63.6299 | 3.6299 |
| 3 | 20261082 | D | True | 2441.5524 | 14 | 48 | 0.6097 | 19/7/0 | 1368.5077 | 262.9834 | 49.9012 | 64.1064 | 4.1064 |
| 5 | 20261081 | A | True | 4305.5370 | 10 | 25 | 0.5859 | 54/1/0 | 396.1944 | 697.7502 | 1265.8216 | 64.0928 | 4.0928 |
| 5 | 20261081 | D | True | 4028.1646 | 8 | 30 | 0.6074 | 53/1/1 | 242.6587 | 510.4302 | 1102.4361 | 61.9779 | 1.9779 |
| 5 | 20261082 | A | True | 4305.5370 | 10 | 25 | 0.5901 | 54/1/0 | 396.1944 | 697.7502 | 1265.8216 | 63.9134 | 3.9134 |
| 5 | 20261082 | D | True | 3682.0091 | 10 | 40 | 0.5984 | 52/3/0 | 167.3800 | 466.9970 | 774.3992 | 65.7736 | 5.7736 |
| 6 | 20261081 | A | True | 4924.4110 | 12 | 45 | 0.5929 | 52/1/2 | 4579.1971 | 592.7014 | 216.8863 | 60.0413 | 0.0413 |
| 6 | 20261081 | D | True | 6382.1668 | 15 | 49 | 0.6035 | 55/0/0 | 6170.0892 | 436.1709 | 0 | 61.7183 | 1.7183 |
| 6 | 20261082 | A | True | 6562.0980 | 13 | 47 | 0.5585 | 51/1/3 | 6004.9005 | 803.9208 | 415.5748 | 64.7423 | 4.7423 |
| 6 | 20261082 | D | True | 6171.5114 | 14 | 49 | 0.5662 | 55/0/0 | 5949.0837 | 443.9570 | 0 | 61.5477 | 1.5477 |
| 9 | 20261081 | A | True | 7682.2815 | 8 | 34 | 0.5436 | 83/2/0 | 7097.8966 | 663.2129 | 57.1256 | 60.4095 | 0.4095 |
| 9 | 20261081 | D | True | 7666.8036 | 11 | 42 | 0.5720 | 85/0/0 | 7068.6157 | 627.2297 | 0 | 68.6404 | 8.6404 |
| 9 | 20261082 | A | True | 7600.4202 | 6 | 30 | 0.5503 | 83/2/0 | 6975.9992 | 646.3559 | 113.3579 | 62.4279 | 2.4279 |
| 9 | 20261082 | D | True | 8099.5720 | 10 | 42 | 0.5599 | 84/1/0 | 7626.4911 | 722.2023 | 29.3649 | 60.3013 | 0.3013 |
| 12 | 20261081 | A | True | 5427.6610 | 4 | 13 | 0.6549 | 78/7/0 | 1601.1996 | 987.6098 | 77.9906 | 60.1120 | 0.1120 |
| 12 | 20261081 | D | True | 4954.7944 | 5 | 15 | 0.6677 | 80/5/0 | 1050.5212 | 887.7772 | 295.2177 | 63.9957 | 3.9957 |
| 12 | 20261082 | A | True | 5427.6610 | 5 | 15 | 0.6562 | 78/7/0 | 1601.1996 | 987.6098 | 77.9906 | 69.9936 | 9.9936 |
| 12 | 20261082 | D | True | 4954.7944 | 5 | 15 | 0.6708 | 80/5/0 | 1050.5212 | 887.7772 | 295.2177 | 62.8764 | 2.8764 |

## 直接结论

1. **0.5 m 已真正生效。** 默认 delta_x/delta_y 均为0.50，实际枚举包括Y=5.5/6.5与各轨X中心±0.5。六例共332条父焊缝的目录统计如下；不是只改了文档或参数显示。

| 配置 | WHOLE父焊缝 | mandatory Y | optional Y | legal Y | X父焊缝 | legal X | 仅WHOLE无合法拆分 | 短Y删除 | 短X删除 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 314 | 18 | 17 | 73 | 227 | 283 | 85 | 14 | 16 |
| D | 319 | 13 | 34 | 133 | 229 | 336 | 83 | 21 | 27 |

2. **保留旧0.2切点，但要区分偏移与绝对位置。** Y的5.8/6.0/6.2及其旧point_id保留，X在同一固定中心下的±0.2与center保留，合法MIDPOINT继续保留。新增外点用OUTER_LOWER/UPPER，去重不改已有合法MIDPOINT身份。A→D时I2上轨中心11.7294→12.4917、I6下轨10.8322→11.1876、I12下轨7.8020→7.9211，其余中心不变；因此不能声称这些实例的所有旧绝对X坐标都原样保留。原因是Y覆盖改变了WHOLE资格和加权中位数输入，不是WAIT驱动重算。

3. **最短子焊缝仍为0.20 m，按实际欧氏长度检查。** 新测试覆盖0.199拒绝、0.200接受、斜焊缝轴向投影较短但实际长度合法、内部交点/端点/纵向/去重、固定X左右机器人、canonicalization、三方法共享目录、伪造FEASIBLE结果及连续干涉。独立certifier重构检查长度与时序。interference_dx/dy仍各0.50 m，未修改B32/scheduler、SA或邻域。

4. **mandatory Y减少18→13，optional Y父焊缝17→34，WHOLE资格314→319。** legal Y patterns从73→133。D的任务层共享覆盖为[5.5,6.5]，这会改变合法可行域；不是所有新切点都强制采用。没有进行URDF/IK验证，不能把任务层合法性当成真实机器人运动学可达性。

5. **legal X patterns增加283→336，即+53（+18.73%）；X-splittable父焊缝227→229。** 这同时包含窗口扩大、WHOLE资格和轨中心改变的影响。按用户最新要求仅保留A/D，不能从本实验定量分离X和Y各自的因果贡献。短段限制仍有效：D中删除21个Y和27个X几何候选。

6. **Cmax与效率呈混合结果，不能宣称D整体更优。** ALL mean Cmax@60为A=4437.658 s、D=4508.022 s（均值上升1.59%）；12个配对中8改善、4变差，配对相对变化中位数为−1.67%，相对变化均值为+4.37%。这些不同统计量不能互换。SMALL的配对中位变化+13.40%，MEDIUM为−6.20%；I3两个seed恶化27.69%/36.38%，I6的第一个seed恶化29.60%，保留全部结果。平均迭代10.92→11.58，平均ref37.33→39.83，逐run候选有效率均值57.99%→59.68%。reference和迭代统计包含最后一个跨过deadline的完整迭代；Cmax只用@60。process load imbalance是四机器人process time的max−min（秒），ALL均值2452.86→2708.60；empty travel为时间，595.90→508.75 s；waiting为307.17→212.21 s。等待减少并不保证makespan减少，负载失衡仍可恶化。

7. **LARGE有改善迹象，但并不一致。** mean Cmax为6534.506→6418.991 s（−1.77%），配对中位数−4.46%，4对中3改善、1变差。I12两个seed均改善8.71%；I9一个seed略改善0.20%，另一个变差6.57%。平均迭代5.75→7.75、ref23.0→28.5；等待81.62→154.95 s，空行走821.20→781.25 s，负载差4319.07→4199.04 s。两个实例、两个seed只支持development敏感性观察，不支持规模普适性或论文竞争结论。

8. **没有未认证运行或新增certifier/numeric异常，但出现了更多候选拒绝。** A和D各12/12的@60解及final解均独立认证。搜索阶段DEADLOCK为206→220，但调用数不同，比例54.21%→53.92%；搜索INFEASIBLE为0→8（SMALL 7、MEDIUM 1、LARGE 0）。初始化DEADLOCK另计56→58；最终均找到可行解。DEADLOCK不是物理无解证明，INFEASIBLE候选也不能外推为整个新任务无解。本报告保存状态计数与最终认证错误列表，没有保存每个被拒绝候选的完整诊断/几何，故不凭计数臆断这8次拒绝的具体约束原因。A/D平均overshoot为2.791/2.955 s，最大9.994/8.640 s；未用overshoot后的final_cmax替代@60。

| 配置 | 全部ref | init ref | 搜索DEADLOCK | 搜索DEADLOCK率% | 搜索INFEASIBLE | init DEADLOCK | NUMERIC | 认证异常 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 448 | 68 | 206 | 54.2105 | 0 | 56 | 0 | 0 |
| D | 478 | 70 | 220 | 53.9216 | 8 | 58 | 0 | 0 |

9. **可以作为后续任务层MLP研究的明确几何基础，性能优势尚未成立。** 代码、统一目录和独立认证已对0.5规则验证；是否采用覆盖区域假设，需要按真实设备能力确认。现有14,893条0.2 labels与旧MLP仍是历史证据，不能直接当作0.5域的正式训练集。本轮未重训、未重建标签、未改17 TRAIN/6 DEV划分，也未访问validation、ORACLE或最终sealed ID_TEST。production ranker继续HEURISTIC，TWO_OPT_STAR继续OFF。

10. **下一步：先确认覆盖假设，再决定新的训练工作。** 当前按指定0.5规则保留默认代码，记录其并非一致优于0.2。后续若接受这个任务模型，再另行授权基于新目录与既有workbook隔离采集训练标签、评估Top8/Top2排序；不复用旧域标签作正式训练，不因本轮混合结果马上调M/K、增加operator、重训MLP或上GAT。本轮到此停止。

## 执行边界与记录说明

用户在运行期间明确取消B/C；最终只保留24次完整A/D运行与12个配对，已完成的A/D均复用，未按结果筛选。切换时停止的未完成在途运行不构成60秒结果，不参与比较。报告中运行次序见原始记录，主表按实例/seed/配置排序。实验来自当前未提交工作区，仅为development sensitivity与正确性验证。唯一新增文件为本报告；复用现有run_ppo_smoke.py，未新增runner、JSON/CSV、数据库、protocol、manifest或hash管理体系，未清理历史实验文件。


## 可恢复的逐次原始记录

字段中 final_cmax/final_patterns 可能来自 overshoot，仅用于审计，不用于上述质量比较。

<!-- YX_RANGE_RESULTS -->
```json
{
  "entries": [
    {
      "selection_ordinal": 2,
      "tier": "SMALL",
      "N": 26,
      "relative_path": "data/PPO_TRAIN/seed_1064648049.xlsx",
      "sheet_name": "g20_w026",
      "instance_id": "data/PPO_TRAIN/seed_1064648049::g20_w026"
    },
    {
      "selection_ordinal": 3,
      "tier": "SMALL",
      "N": 26,
      "relative_path": "data/DEV_ONLY/seed_000505.xlsx",
      "sheet_name": "g15_w026",
      "instance_id": "data/DEV_ONLY/seed_000505::g15_w026"
    },
    {
      "selection_ordinal": 5,
      "tier": "MEDIUM",
      "N": 55,
      "relative_path": "data/PPO_TRAIN/seed_0900775608.xlsx",
      "sheet_name": "g30_w055",
      "instance_id": "data/PPO_TRAIN/seed_0900775608::g30_w055"
    },
    {
      "selection_ordinal": 6,
      "tier": "MEDIUM",
      "N": 55,
      "relative_path": "data/PPO_TRAIN/seed_1419626827.xlsx",
      "sheet_name": "g29_w055",
      "instance_id": "data/PPO_TRAIN/seed_1419626827::g29_w055"
    },
    {
      "selection_ordinal": 9,
      "tier": "LARGE",
      "N": 85,
      "relative_path": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet_name": "g43_w085",
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g43_w085"
    },
    {
      "selection_ordinal": 12,
      "tier": "LARGE",
      "N": 85,
      "relative_path": "data/PPO_TRAIN/seed_0211781140.xlsx",
      "sheet_name": "g32_w085",
      "instance_id": "data/PPO_TRAIN/seed_0211781140::g32_w085"
    }
  ],
  "catalog": [
    {
      "instance": 2,
      "tier": "SMALL",
      "group": "A",
      "N": 26,
      "whole": 24,
      "mandatory_y": 2,
      "optional_y": 6,
      "y_patterns": 11,
      "x_parents": 18,
      "x_patterns": 32,
      "no_split": 5,
      "removed_short_y": 5,
      "removed_short_x": 1,
      "x_up": 11.729369166069,
      "x_low": 10.373042978775
    },
    {
      "instance": 2,
      "tier": "SMALL",
      "group": "D",
      "N": 26,
      "whole": 25,
      "mandatory_y": 1,
      "optional_y": 10,
      "y_patterns": 24,
      "x_parents": 18,
      "x_patterns": 38,
      "no_split": 5,
      "removed_short_y": 5,
      "removed_short_x": 8,
      "x_up": 12.491658065642,
      "x_low": 10.373042978775
    },
    {
      "instance": 3,
      "tier": "SMALL",
      "group": "A",
      "N": 26,
      "whole": 20,
      "mandatory_y": 6,
      "optional_y": 4,
      "y_patterns": 24,
      "x_parents": 16,
      "x_patterns": 21,
      "no_split": 4,
      "removed_short_y": 1,
      "removed_short_x": 2,
      "x_up": 8.136845265161,
      "x_low": 9.366950945806
    },
    {
      "instance": 3,
      "tier": "SMALL",
      "group": "D",
      "N": 26,
      "whole": 20,
      "mandatory_y": 6,
      "optional_y": 4,
      "y_patterns": 41,
      "x_parents": 16,
      "x_patterns": 26,
      "no_split": 4,
      "removed_short_y": 1,
      "removed_short_x": 2,
      "x_up": 8.136845265161,
      "x_low": 9.366950945806
    },
    {
      "instance": 5,
      "tier": "MEDIUM",
      "group": "A",
      "N": 55,
      "whole": 54,
      "mandatory_y": 1,
      "optional_y": 1,
      "y_patterns": 5,
      "x_parents": 46,
      "x_patterns": 61,
      "no_split": 8,
      "removed_short_y": 1,
      "removed_short_x": 1,
      "x_up": 12.771984907791001,
      "x_low": 5.277945700075
    },
    {
      "instance": 5,
      "tier": "MEDIUM",
      "group": "D",
      "N": 55,
      "whole": 54,
      "mandatory_y": 1,
      "optional_y": 2,
      "y_patterns": 8,
      "x_parents": 46,
      "x_patterns": 72,
      "no_split": 7,
      "removed_short_y": 3,
      "removed_short_x": 1,
      "x_up": 12.771984907791001,
      "x_low": 5.277945700075
    },
    {
      "instance": 6,
      "tier": "MEDIUM",
      "group": "A",
      "N": 55,
      "whole": 54,
      "mandatory_y": 1,
      "optional_y": 0,
      "y_patterns": 2,
      "x_parents": 43,
      "x_patterns": 53,
      "no_split": 11,
      "removed_short_y": 2,
      "removed_short_x": 1,
      "x_up": 9.238730614972,
      "x_low": 10.832215807413
    },
    {
      "instance": 6,
      "tier": "MEDIUM",
      "group": "D",
      "N": 55,
      "whole": 55,
      "mandatory_y": 0,
      "optional_y": 2,
      "y_patterns": 5,
      "x_parents": 44,
      "x_patterns": 62,
      "no_split": 11,
      "removed_short_y": 3,
      "removed_short_x": 1,
      "x_up": 9.238730614972,
      "x_low": 11.187580565345
    },
    {
      "instance": 9,
      "tier": "LARGE",
      "group": "A",
      "N": 85,
      "whole": 84,
      "mandatory_y": 1,
      "optional_y": 3,
      "y_patterns": 6,
      "x_parents": 59,
      "x_patterns": 65,
      "no_split": 24,
      "removed_short_y": 3,
      "removed_short_x": 4,
      "x_up": 11.17226939128,
      "x_low": 9.3366835991735
    },
    {
      "instance": 9,
      "tier": "LARGE",
      "group": "D",
      "N": 85,
      "whole": 85,
      "mandatory_y": 0,
      "optional_y": 5,
      "y_patterns": 12,
      "x_parents": 59,
      "x_patterns": 67,
      "no_split": 24,
      "removed_short_y": 3,
      "removed_short_x": 7,
      "x_up": 11.17226939128,
      "x_low": 9.3366835991735
    },
    {
      "instance": 12,
      "tier": "LARGE",
      "group": "A",
      "N": 85,
      "whole": 78,
      "mandatory_y": 7,
      "optional_y": 3,
      "y_patterns": 25,
      "x_parents": 45,
      "x_patterns": 51,
      "no_split": 33,
      "removed_short_y": 2,
      "removed_short_x": 7,
      "x_up": 10.189030417996,
      "x_low": 7.801971877903
    },
    {
      "instance": 12,
      "tier": "LARGE",
      "group": "D",
      "N": 85,
      "whole": 80,
      "mandatory_y": 5,
      "optional_y": 11,
      "y_patterns": 43,
      "x_parents": 46,
      "x_patterns": 71,
      "no_split": 32,
      "removed_short_y": 6,
      "removed_short_x": 8,
      "x_up": 10.189030417996,
      "x_low": 7.921126479954
    }
  ],
  "runs": [
    {
      "instance": 2,
      "instance_id": "data/PPO_TRAIN/seed_1064648049::g20_w026",
      "tier": "SMALL",
      "seed": 20261081,
      "group": "D",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 1599.1750539165578,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 23,
        "Y_SPLIT": 2,
        "X_SPLIT": 1
      },
      "process_imbalance": 93.53000052433208,
      "empty_travel_s": 333.10620831265294,
      "waiting_s": 0,
      "iterations": 17,
      "reference_calls": 53,
      "raw_attempts": 3264,
      "valid_candidates": 1797,
      "candidate_valid_rate": 0.5505514705882353,
      "deadlock": 26,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "FEASIBLE": 1,
        "DEADLOCK": 2
      },
      "accepted_moves": 6,
      "global_best_updates": 1,
      "elapsed_s": 60.715101499999946,
      "overshoot_s": 0.7151014999999461,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1599.1750539165578,
      "final_patterns": {
        "WHOLE": 23,
        "Y_SPLIT": 2,
        "X_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/PPO_TRAIN/seed_1064648049::g20_w026",
      "tier": "SMALL",
      "seed": 20261081,
      "group": "A",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 1639.154648084687,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 24,
        "Y_SPLIT": 2
      },
      "process_imbalance": 257.6845369621685,
      "empty_travel_s": 286.0721109859862,
      "waiting_s": 88.01284859158795,
      "iterations": 21,
      "reference_calls": 58,
      "raw_attempts": 4032,
      "valid_candidates": 2151,
      "candidate_valid_rate": 0.5334821428571429,
      "deadlock": 38,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 3,
        "FEASIBLE": 1
      },
      "accepted_moves": 4,
      "global_best_updates": 0,
      "elapsed_s": 61.640878899999734,
      "overshoot_s": 1.6408788999997341,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1639.154648084687,
      "final_patterns": {
        "WHOLE": 24,
        "Y_SPLIT": 2
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/PPO_TRAIN/seed_1064648049::g20_w026",
      "tier": "SMALL",
      "seed": 20261082,
      "group": "D",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 1609.9800915874396,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 24,
        "Y_SPLIT": 2
      },
      "process_imbalance": 156.14090336152026,
      "empty_travel_s": 270.84730807522334,
      "waiting_s": 0,
      "iterations": 18,
      "reference_calls": 55,
      "raw_attempts": 3456,
      "valid_candidates": 1823,
      "candidate_valid_rate": 0.5274884259259259,
      "deadlock": 27,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "FEASIBLE": 1,
        "DEADLOCK": 2
      },
      "accepted_moves": 7,
      "global_best_updates": 1,
      "elapsed_s": 62.85586710000098,
      "overshoot_s": 2.855867100000978,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1609.9800915874396,
      "final_patterns": {
        "WHOLE": 24,
        "Y_SPLIT": 2
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/PPO_TRAIN/seed_1064648049::g20_w026",
      "tier": "SMALL",
      "seed": 20261082,
      "group": "A",
      "core_mask": 4,
      "certified_at60": true,
      "cmax_at60": 1624.577613471115,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 23,
        "Y_SPLIT": 2,
        "X_SPLIT": 1
      },
      "process_imbalance": 425.8332471563115,
      "empty_travel_s": 207.18446212914625,
      "waiting_s": 0,
      "iterations": 20,
      "reference_calls": 70,
      "raw_attempts": 3840,
      "valid_candidates": 1911,
      "candidate_valid_rate": 0.49765625,
      "deadlock": 24,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 3,
        "FEASIBLE": 1
      },
      "accepted_moves": 12,
      "global_best_updates": 2,
      "elapsed_s": 60.99836289999985,
      "overshoot_s": 0.9983628999998473,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1624.577613471115,
      "final_patterns": {
        "WHOLE": 24,
        "Y_SPLIT": 2
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/DEV_ONLY/seed_000505::g15_w026",
      "tier": "SMALL",
      "seed": 20261081,
      "group": "D",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 2505.7367628344514,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 20,
        "Y_SPLIT": 6
      },
      "process_imbalance": 1559.647697469803,
      "empty_travel_s": 255.50992417217276,
      "waiting_s": 0,
      "iterations": 12,
      "reference_calls": 40,
      "raw_attempts": 2304,
      "valid_candidates": 1446,
      "candidate_valid_rate": 0.6276041666666666,
      "deadlock": 19,
      "infeasible": 3,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 4,
      "global_best_updates": 4,
      "elapsed_s": 60.953088899999784,
      "overshoot_s": 0.9530888999997842,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 2505.7367628344514,
      "final_patterns": {
        "WHOLE": 20,
        "Y_SPLIT": 6
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/DEV_ONLY/seed_000505::g15_w026",
      "tier": "SMALL",
      "seed": 20261081,
      "group": "A",
      "core_mask": 4,
      "certified_at60": true,
      "cmax_at60": 1962.3246754416189,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 19,
        "Y_SPLIT": 7
      },
      "process_imbalance": 21.23995936524443,
      "empty_travel_s": 306.9330488243915,
      "waiting_s": 107.46065131321234,
      "iterations": 10,
      "reference_calls": 41,
      "raw_attempts": 1920,
      "valid_candidates": 1155,
      "candidate_valid_rate": 0.6015625,
      "deadlock": 17,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "FEASIBLE": 1,
        "DEADLOCK": 4
      },
      "accepted_moves": 6,
      "global_best_updates": 3,
      "elapsed_s": 61.48546990000068,
      "overshoot_s": 1.4854699000006804,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1962.324675441619,
      "final_patterns": {
        "WHOLE": 19,
        "Y_SPLIT": 7
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/DEV_ONLY/seed_000505::g15_w026",
      "tier": "SMALL",
      "seed": 20261082,
      "group": "D",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 2441.5523838826484,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 19,
        "Y_SPLIT": 7
      },
      "process_imbalance": 1368.507736167111,
      "empty_travel_s": 262.98341097388754,
      "waiting_s": 49.90116544766738,
      "iterations": 14,
      "reference_calls": 48,
      "raw_attempts": 2688,
      "valid_candidates": 1639,
      "candidate_valid_rate": 0.6097470238095238,
      "deadlock": 22,
      "infeasible": 4,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 6,
      "global_best_updates": 6,
      "elapsed_s": 64.1063635999999,
      "overshoot_s": 4.106363599999895,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 2441.5523838826484,
      "final_patterns": {
        "WHOLE": 19,
        "Y_SPLIT": 7
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/DEV_ONLY/seed_000505::g15_w026",
      "tier": "SMALL",
      "seed": 20261082,
      "group": "A",
      "core_mask": 16,
      "certified_at60": true,
      "cmax_at60": 1790.2315587982225,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 20,
        "Y_SPLIT": 6
      },
      "process_imbalance": 76.74572914316445,
      "empty_travel_s": 273.6685844469959,
      "waiting_s": 0,
      "iterations": 12,
      "reference_calls": 45,
      "raw_attempts": 2304,
      "valid_candidates": 1367,
      "candidate_valid_rate": 0.5933159722222222,
      "deadlock": 16,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "FEASIBLE": 1,
        "DEADLOCK": 4
      },
      "accepted_moves": 7,
      "global_best_updates": 3,
      "elapsed_s": 63.629934599999615,
      "overshoot_s": 3.629934599999615,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 1790.2315587982225,
      "final_patterns": {
        "WHOLE": 20,
        "Y_SPLIT": 6
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 5,
      "instance_id": "data/PPO_TRAIN/seed_0900775608::g30_w055",
      "tier": "MEDIUM",
      "seed": 20261081,
      "group": "D",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 4028.164647714563,
      "checkpoint_errors": [],
      "patterns": {
        "X_SPLIT": 1,
        "WHOLE": 53,
        "Y_SPLIT": 1
      },
      "process_imbalance": 242.65868021125198,
      "empty_travel_s": 510.43018243591894,
      "waiting_s": 1102.4360633189908,
      "iterations": 8,
      "reference_calls": 30,
      "raw_attempts": 1536,
      "valid_candidates": 933,
      "candidate_valid_rate": 0.607421875,
      "deadlock": 15,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 3,
        "FEASIBLE": 1
      },
      "accepted_moves": 5,
      "global_best_updates": 3,
      "elapsed_s": 61.97789299999931,
      "overshoot_s": 1.9778929999993125,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4028.164647714563,
      "final_patterns": {
        "X_SPLIT": 1,
        "WHOLE": 53,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 5,
      "instance_id": "data/PPO_TRAIN/seed_0900775608::g30_w055",
      "tier": "MEDIUM",
      "seed": 20261081,
      "group": "A",
      "core_mask": 16,
      "certified_at60": true,
      "cmax_at60": 4305.53700823084,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 54,
        "Y_SPLIT": 1
      },
      "process_imbalance": 396.1943860664119,
      "empty_travel_s": 697.7502282270408,
      "waiting_s": 1265.8215922577497,
      "iterations": 10,
      "reference_calls": 25,
      "raw_attempts": 1920,
      "valid_candidates": 1125,
      "candidate_valid_rate": 0.5859375,
      "deadlock": 20,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 64.09275609999895,
      "overshoot_s": 4.092756099998951,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4305.53700823084,
      "final_patterns": {
        "WHOLE": 54,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    },
    {
      "instance": 5,
      "instance_id": "data/PPO_TRAIN/seed_0900775608::g30_w055",
      "tier": "MEDIUM",
      "seed": 20261082,
      "group": "D",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 3682.009071584258,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 52,
        "Y_SPLIT": 3
      },
      "process_imbalance": 167.38000733926856,
      "empty_travel_s": 466.9969862499015,
      "waiting_s": 774.399166736348,
      "iterations": 10,
      "reference_calls": 40,
      "raw_attempts": 1920,
      "valid_candidates": 1149,
      "candidate_valid_rate": 0.5984375,
      "deadlock": 18,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 3,
        "FEASIBLE": 1
      },
      "accepted_moves": 7,
      "global_best_updates": 4,
      "elapsed_s": 65.77364119999947,
      "overshoot_s": 5.773641199999474,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 3682.009071584258,
      "final_patterns": {
        "WHOLE": 52,
        "Y_SPLIT": 3
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 5,
      "instance_id": "data/PPO_TRAIN/seed_0900775608::g30_w055",
      "tier": "MEDIUM",
      "seed": 20261082,
      "group": "A",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 4305.53700823084,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 54,
        "Y_SPLIT": 1
      },
      "process_imbalance": 396.1943860664119,
      "empty_travel_s": 697.7502282270408,
      "waiting_s": 1265.8215922577497,
      "iterations": 10,
      "reference_calls": 25,
      "raw_attempts": 1920,
      "valid_candidates": 1133,
      "candidate_valid_rate": 0.5901041666666667,
      "deadlock": 20,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 63.91335550000076,
      "overshoot_s": 3.913355500000762,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4305.53700823084,
      "final_patterns": {
        "WHOLE": 54,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    },
    {
      "instance": 6,
      "instance_id": "data/PPO_TRAIN/seed_1419626827::g29_w055",
      "tier": "MEDIUM",
      "seed": 20261081,
      "group": "D",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 6382.16683222046,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 55
      },
      "process_imbalance": 6170.08923349099,
      "empty_travel_s": 436.1709307273218,
      "waiting_s": 0,
      "iterations": 15,
      "reference_calls": 49,
      "raw_attempts": 2880,
      "valid_candidates": 1738,
      "candidate_valid_rate": 0.6034722222222222,
      "deadlock": 24,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "accepted_moves": 6,
      "global_best_updates": 6,
      "elapsed_s": 61.71827900000062,
      "overshoot_s": 1.7182790000006207,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 6382.16683222046,
      "final_patterns": {
        "WHOLE": 55
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 6,
      "instance_id": "data/PPO_TRAIN/seed_1419626827::g29_w055",
      "tier": "MEDIUM",
      "seed": 20261081,
      "group": "A",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 4924.410968982638,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 52,
        "X_SPLIT": 2,
        "Y_SPLIT": 1
      },
      "process_imbalance": 4579.197086176275,
      "empty_travel_s": 592.7013627547061,
      "waiting_s": 216.88629319472557,
      "iterations": 12,
      "reference_calls": 45,
      "raw_attempts": 2304,
      "valid_candidates": 1366,
      "candidate_valid_rate": 0.5928819444444444,
      "deadlock": 17,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "accepted_moves": 7,
      "global_best_updates": 5,
      "elapsed_s": 60.04127990000052,
      "overshoot_s": 0.041279900000517955,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4924.410968982638,
      "final_patterns": {
        "WHOLE": 52,
        "X_SPLIT": 2,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 6,
      "instance_id": "data/PPO_TRAIN/seed_1419626827::g29_w055",
      "tier": "MEDIUM",
      "seed": 20261082,
      "group": "A",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 6562.097967976422,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 51,
        "X_SPLIT": 3,
        "Y_SPLIT": 1
      },
      "process_imbalance": 6004.900491164934,
      "empty_travel_s": 803.9208285005539,
      "waiting_s": 415.5747656372025,
      "iterations": 13,
      "reference_calls": 47,
      "raw_attempts": 2496,
      "valid_candidates": 1394,
      "candidate_valid_rate": 0.5584935897435898,
      "deadlock": 19,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "accepted_moves": 6,
      "global_best_updates": 4,
      "elapsed_s": 64.74231990000044,
      "overshoot_s": 4.742319900000439,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 6562.097967976422,
      "final_patterns": {
        "WHOLE": 51,
        "X_SPLIT": 3,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 6,
      "instance_id": "data/PPO_TRAIN/seed_1419626827::g29_w055",
      "tier": "MEDIUM",
      "seed": 20261082,
      "group": "D",
      "core_mask": 16,
      "certified_at60": true,
      "cmax_at60": 6171.5114319877775,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 55
      },
      "process_imbalance": 5949.0836520504745,
      "empty_travel_s": 443.95701381365325,
      "waiting_s": 0,
      "iterations": 14,
      "reference_calls": 49,
      "raw_attempts": 2688,
      "valid_candidates": 1522,
      "candidate_valid_rate": 0.5662202380952381,
      "deadlock": 21,
      "infeasible": 1,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "accepted_moves": 7,
      "global_best_updates": 6,
      "elapsed_s": 61.547702600000775,
      "overshoot_s": 1.5477026000007754,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 5534.3688951848635,
      "final_patterns": {
        "WHOLE": 54,
        "X_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 9,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g43_w085",
      "tier": "LARGE",
      "seed": 20261082,
      "group": "D",
      "core_mask": 16,
      "certified_at60": true,
      "cmax_at60": 8099.571957835579,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 84,
        "Y_SPLIT": 1
      },
      "process_imbalance": 7626.4911158753,
      "empty_travel_s": 722.202314826863,
      "waiting_s": 29.364932920905403,
      "iterations": 10,
      "reference_calls": 42,
      "raw_attempts": 1920,
      "valid_candidates": 1075,
      "candidate_valid_rate": 0.5598958333333334,
      "deadlock": 13,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 7,
      "global_best_updates": 7,
      "elapsed_s": 60.301346000000194,
      "overshoot_s": 0.3013460000001942,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 8072.107350828733,
      "final_patterns": {
        "WHOLE": 83,
        "X_SPLIT": 1,
        "Y_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 9,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g43_w085",
      "tier": "LARGE",
      "seed": 20261081,
      "group": "A",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 7682.281481798071,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 83,
        "Y_SPLIT": 2
      },
      "process_imbalance": 7097.8966354674,
      "empty_travel_s": 663.2128971708752,
      "waiting_s": 57.12557607964004,
      "iterations": 8,
      "reference_calls": 34,
      "raw_attempts": 1536,
      "valid_candidates": 835,
      "candidate_valid_rate": 0.5436197916666666,
      "deadlock": 10,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 5,
      "global_best_updates": 4,
      "elapsed_s": 60.40945520000059,
      "overshoot_s": 0.4094552000005933,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 7682.281481798071,
      "final_patterns": {
        "WHOLE": 83,
        "Y_SPLIT": 2
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 9,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g43_w085",
      "tier": "LARGE",
      "seed": 20261082,
      "group": "A",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 7600.420154386237,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 83,
        "Y_SPLIT": 2
      },
      "process_imbalance": 6975.999176881748,
      "empty_travel_s": 646.3558663512371,
      "waiting_s": 113.35789604808136,
      "iterations": 6,
      "reference_calls": 30,
      "raw_attempts": 1152,
      "valid_candidates": 634,
      "candidate_valid_rate": 0.5503472222222222,
      "deadlock": 7,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 5,
      "global_best_updates": 5,
      "elapsed_s": 62.42789789999915,
      "overshoot_s": 2.4278978999991523,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 7525.604029874827,
      "final_patterns": {
        "WHOLE": 82,
        "Y_SPLIT": 2,
        "X_SPLIT": 1
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 9,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g43_w085",
      "tier": "LARGE",
      "seed": 20261081,
      "group": "D",
      "core_mask": 4,
      "certified_at60": true,
      "cmax_at60": 7666.803608246132,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 85
      },
      "process_imbalance": 7068.615682239676,
      "empty_travel_s": 627.2297258529758,
      "waiting_s": 0,
      "iterations": 11,
      "reference_calls": 42,
      "raw_attempts": 2112,
      "valid_candidates": 1208,
      "candidate_valid_rate": 0.571969696969697,
      "deadlock": 15,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 7,
        "FEASIBLE": 1
      },
      "accepted_moves": 6,
      "global_best_updates": 5,
      "elapsed_s": 68.64039480000065,
      "overshoot_s": 8.64039480000065,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 7666.803608246132,
      "final_patterns": {
        "WHOLE": 85
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 12,
      "instance_id": "data/PPO_TRAIN/seed_0211781140::g32_w085",
      "tier": "LARGE",
      "seed": 20261081,
      "group": "A",
      "core_mask": 1,
      "certified_at60": true,
      "cmax_at60": 5427.661043593272,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 78,
        "Y_SPLIT": 7
      },
      "process_imbalance": 1601.199588610491,
      "empty_travel_s": 987.6098147082422,
      "waiting_s": 77.99060074693125,
      "iterations": 4,
      "reference_calls": 13,
      "raw_attempts": 768,
      "valid_candidates": 503,
      "candidate_valid_rate": 0.6549479166666666,
      "deadlock": 8,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 60.112006099998325,
      "overshoot_s": 0.11200609999832523,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 5427.661043593272,
      "final_patterns": {
        "WHOLE": 78,
        "Y_SPLIT": 7
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    },
    {
      "instance": 12,
      "instance_id": "data/PPO_TRAIN/seed_0211781140::g32_w085",
      "tier": "LARGE",
      "seed": 20261081,
      "group": "D",
      "core_mask": 16,
      "certified_at60": true,
      "cmax_at60": 4954.79443722544,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 80,
        "Y_SPLIT": 5
      },
      "process_imbalance": 1050.5211588448906,
      "empty_travel_s": 887.7771609228547,
      "waiting_s": 295.2176776175115,
      "iterations": 5,
      "reference_calls": 15,
      "raw_attempts": 960,
      "valid_candidates": 641,
      "candidate_valid_rate": 0.6677083333333333,
      "deadlock": 10,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 63.995693499999106,
      "overshoot_s": 3.9956934999991063,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4954.79443722544,
      "final_patterns": {
        "WHOLE": 80,
        "Y_SPLIT": 5
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    },
    {
      "instance": 12,
      "instance_id": "data/PPO_TRAIN/seed_0211781140::g32_w085",
      "tier": "LARGE",
      "seed": 20261082,
      "group": "D",
      "core_mask": 4,
      "certified_at60": true,
      "cmax_at60": 4954.79443722544,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 80,
        "Y_SPLIT": 5
      },
      "process_imbalance": 1050.5211588448906,
      "empty_travel_s": 887.7771609228547,
      "waiting_s": 295.2176776175115,
      "iterations": 5,
      "reference_calls": 15,
      "raw_attempts": 960,
      "valid_candidates": 644,
      "candidate_valid_rate": 0.6708333333333333,
      "deadlock": 10,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 62.876427400000466,
      "overshoot_s": 2.876427400000466,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 4954.79443722544,
      "final_patterns": {
        "WHOLE": 80,
        "Y_SPLIT": 5
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    },
    {
      "instance": 12,
      "instance_id": "data/PPO_TRAIN/seed_0211781140::g32_w085",
      "tier": "LARGE",
      "seed": 20261082,
      "group": "A",
      "core_mask": 64,
      "certified_at60": true,
      "cmax_at60": 5427.661043593272,
      "checkpoint_errors": [],
      "patterns": {
        "WHOLE": 78,
        "Y_SPLIT": 7
      },
      "process_imbalance": 1601.199588610491,
      "empty_travel_s": 987.6098147082422,
      "waiting_s": 77.99060074693125,
      "iterations": 5,
      "reference_calls": 15,
      "raw_attempts": 960,
      "valid_candidates": 630,
      "candidate_valid_rate": 0.65625,
      "deadlock": 10,
      "infeasible": 0,
      "numeric_failure": 0,
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "accepted_moves": 0,
      "global_best_updates": 0,
      "elapsed_s": 69.99363830000038,
      "overshoot_s": 9.993638300000384,
      "status": "COMPLETED",
      "final_certified": true,
      "final_errors": [],
      "final_cmax": 5427.661043593272,
      "final_patterns": {
        "WHOLE": 78,
        "Y_SPLIT": 7
      },
      "max_kdp": 8,
      "max_reference_per_iteration": 2
    }
  ],
  "tests": "在 DRL 目录以 D:\\pybullet_test\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider 执行；最终299 passed in 38.54s（实验前299 passed in 35.62s）",
  "conclusions": "## 直接结论\n\n1. **0.5 m 已真正生效。** 默认 delta_x/delta_y 均为0.50，实际枚举包括Y=5.5/6.5与各轨X中心±0.5。六例共332条父焊缝的目录统计如下；不是只改了文档或参数显示。\n\n| 配置 | WHOLE父焊缝 | mandatory Y | optional Y | legal Y | X父焊缝 | legal X | 仅WHOLE无合法拆分 | 短Y删除 | 短X删除 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| A | 314 | 18 | 17 | 73 | 227 | 283 | 85 | 14 | 16 |\n| D | 319 | 13 | 34 | 133 | 229 | 336 | 83 | 21 | 27 |\n\n2. **保留旧0.2切点，但要区分偏移与绝对位置。** Y的5.8/6.0/6.2及其旧point_id保留，X在同一固定中心下的±0.2与center保留，合法MIDPOINT继续保留。新增外点用OUTER_LOWER/UPPER，去重不改已有合法MIDPOINT身份。A→D时I2上轨中心11.7294→12.4917、I6下轨10.8322→11.1876、I12下轨7.8020→7.9211，其余中心不变；因此不能声称这些实例的所有旧绝对X坐标都原样保留。原因是Y覆盖改变了WHOLE资格和加权中位数输入，不是WAIT驱动重算。\n\n3. **最短子焊缝仍为0.20 m，按实际欧氏长度检查。** 新测试覆盖0.199拒绝、0.200接受、斜焊缝轴向投影较短但实际长度合法、内部交点/端点/纵向/去重、固定X左右机器人、canonicalization、三方法共享目录、伪造FEASIBLE结果及连续干涉。独立certifier重构检查长度与时序。interference_dx/dy仍各0.50 m，未修改B32/scheduler、SA或邻域。\n\n4. **mandatory Y减少18→13，optional Y父焊缝17→34，WHOLE资格314→319。** legal Y patterns从73→133。D的任务层共享覆盖为[5.5,6.5]，这会改变合法可行域；不是所有新切点都强制采用。没有进行URDF/IK验证，不能把任务层合法性当成真实机器人运动学可达性。\n\n5. **legal X patterns增加283→336，即+53（+18.73%）；X-splittable父焊缝227→229。** 这同时包含窗口扩大、WHOLE资格和轨中心改变的影响。按用户最新要求仅保留A/D，不能从本实验定量分离X和Y各自的因果贡献。短段限制仍有效：D中删除21个Y和27个X几何候选。\n\n6. **Cmax与效率呈混合结果，不能宣称D整体更优。** ALL mean Cmax@60为A=4437.658 s、D=4508.022 s（均值上升1.59%）；12个配对中8改善、4变差，配对相对变化中位数为−1.67%，相对变化均值为+4.37%。这些不同统计量不能互换。SMALL的配对中位变化+13.40%，MEDIUM为−6.20%；I3两个seed恶化27.69%/36.38%，I6的第一个seed恶化29.60%，保留全部结果。平均迭代10.92→11.58，平均ref37.33→39.83，逐run候选有效率均值57.99%→59.68%。reference和迭代统计包含最后一个跨过deadline的完整迭代；Cmax只用@60。process load imbalance是四机器人process time的max−min（秒），ALL均值2452.86→2708.60；empty travel为时间，595.90→508.75 s；waiting为307.17→212.21 s。等待减少并不保证makespan减少，负载失衡仍可恶化。\n\n7. **LARGE有改善迹象，但并不一致。** mean Cmax为6534.506→6418.991 s（−1.77%），配对中位数−4.46%，4对中3改善、1变差。I12两个seed均改善8.71%；I9一个seed略改善0.20%，另一个变差6.57%。平均迭代5.75→7.75、ref23.0→28.5；等待81.62→154.95 s，空行走821.20→781.25 s，负载差4319.07→4199.04 s。两个实例、两个seed只支持development敏感性观察，不支持规模普适性或论文竞争结论。\n\n8. **没有未认证运行或新增certifier/numeric异常，但出现了更多候选拒绝。** A和D各12/12的@60解及final解均独立认证。搜索阶段DEADLOCK为206→220，但调用数不同，比例54.21%→53.92%；搜索INFEASIBLE为0→8（SMALL 7、MEDIUM 1、LARGE 0）。初始化DEADLOCK另计56→58；最终均找到可行解。DEADLOCK不是物理无解证明，INFEASIBLE候选也不能外推为整个新任务无解。本报告保存状态计数与最终认证错误列表，没有保存每个被拒绝候选的完整诊断/几何，故不凭计数臆断这8次拒绝的具体约束原因。A/D平均overshoot为2.791/2.955 s，最大9.994/8.640 s；未用overshoot后的final_cmax替代@60。\n\n| 配置 | 全部ref | init ref | 搜索DEADLOCK | 搜索DEADLOCK率% | 搜索INFEASIBLE | init DEADLOCK | NUMERIC | 认证异常 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| A | 448 | 68 | 206 | 54.2105 | 0 | 56 | 0 | 0 |\n| D | 478 | 70 | 220 | 53.9216 | 8 | 58 | 0 | 0 |\n\n9. **可以作为后续任务层MLP研究的明确几何基础，性能优势尚未成立。** 代码、统一目录和独立认证已对0.5规则验证；是否采用覆盖区域假设，需要按真实设备能力确认。现有14,893条0.2 labels与旧MLP仍是历史证据，不能直接当作0.5域的正式训练集。本轮未重训、未重建标签、未改17 TRAIN/6 DEV划分，也未访问validation、ORACLE或最终sealed ID_TEST。production ranker继续HEURISTIC，TWO_OPT_STAR继续OFF。\n\n10. **下一步：先确认覆盖假设，再决定新的训练工作。** 当前按指定0.5规则保留默认代码，记录其并非一致优于0.2。后续若接受这个任务模型，再另行授权基于新目录与既有workbook隔离采集训练标签、评估Top8/Top2排序；不复用旧域标签作正式训练，不因本轮混合结果马上调M/K、增加operator、重训MLP或上GAT。本轮到此停止。\n\n## 执行边界与记录说明\n\n用户在运行期间明确取消B/C；最终只保留24次完整A/D运行与12个配对，已完成的A/D均复用，未按结果筛选。切换时停止的未完成在途运行不构成60秒结果，不参与比较。报告中运行次序见原始记录，主表按实例/seed/配置排序。实验来自当前未提交工作区，仅为development sensitivity与正确性验证。唯一新增文件为本报告；复用现有run_ppo_smoke.py，未新增runner、JSON/CSV、数据库、protocol、manifest或hash管理体系，未清理历史实验文件。\n"
}
```


## SMALL_30_39复核

本章独立于上面的历史N26/55/85实验，不修改其SMALL标签或24条原始记录，不混合计算平均值。只选V2_MODEL_DEVELOPMENT_CONSUMED；目录名不定义角色，所选两个data/ID_TEST旧路径已明确属于消耗过的development，非ID_TEST_SEALED。未生成样本、未删除焊缝、未改角色或TRAIN/DEV划分。By=[5.5,6.5]仍是任务层假设，非URDF/IK验证。

已检查 38 个development workbook的META与实际行数；找到144个N30–39 sheet，来自31个workbook；距35最近的候选12个。

选择规则：META actual_weld_count equals actual rows; VALID unique consumed-development only; closest N to35; maximum legal X patterns, minimum X+Y patterns, median X patterns; three distinct workbooks; ties path/sheet; no solver/Cmax fields。清单在求解前固定，三例实际N均为35。

| 编号 | 角色 | workbook | sheet | N | 总长度m | X/Y patterns | 上/下/共享/mandatory | cross y6 | x_up/x_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | X_RICH | data/ID_TEST/seed_0533728435.xlsx | g24_w035 | 35 | 81.4011 | 52/11 | 14/18/2/1 | 2 | 9.8367/9.8367 |
| 2 | ORDINARY | data/ID_TEST/seed_0401115467.xlsx | g19_w035 | 35 | 47.9957 | 34/21 | 20/11/3/1 | 5 | 12.2937/12.2239 |
| 3 | SPLIT_SPARSE | data/PPO_TRAIN/seed_0194123089.xlsx | g17_w035 | 35 | 37.1424 | 26/1 | 20/15/0/0 | 0 | 6.5969/11.2972 |

A=0.2/0.2，D=0.5/0.5；seeds 20261081/20261082/20261083。其余沿用SA_OI_ALNS_V2、M192=144+48、Kdp8/Kref2/Kref_total4、heuristic C2/C4、TWO_OPT_STAR OFF、construction5/kinit5/bootstrap1、B32与既有certifier。18个真实60秒wall-clock run；worker各绑定独立物理核，A/D顺序交替。观察记录开销计入预算。@60独立认证，绝不以overshoot final Cmax补填。

first_certified_initial为初始化中首个通过独立认证的reference结果；chosen_initial为初始化完成后实际选中的解。iterations/accepted/valid/funnel取60秒内完成的最后一个iteration boundary；reference_status_counts_at60按实际认证完成时刻≤60计数，可能包含尚未完成的最后一轮ref。global-best另保留原生时间戳，次级目标改善也可触发update；同时报告Cmax是否真正下降。四机器人负载按R0/R1/R2/R3排序，含各child独立setup/weld/post。

执行完成 18/18。

| 例 | seed | 组 | 认证 | 首个init Cmax | 选中init Cmax | Cmax@60 | init s | iter≤60 | accept≤60 | best≤60 | ref≤60 | valid率 | WHOLE/Y/X | 四robot process load s | 负载差s | 空行走s | 等待s | runtime/overshoot s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20261081 | A | True | 5077.2943 | 5077.2943 | 3090.5796 | 1.9888 | 22 | 11 | 9 | 76 | 0.5859 | 30/2/3 | 2953.88/2008.16/2926.72/1648.38 | 1305.5051 | 393.0670 | 0 | 61.925/1.925 |
| 1 | 20261081 | D | True | 4956.8901 | 4956.8901 | 4795.6267 | 1.2250 | 22 | 2 | 1 | 58 | 0.5618 | 33/2/0 | 4178.18/0.00/4580.26/628.70 | 4580.2556 | 400.7927 | 0 | 60.165/0.165 |
| 1 | 20261082 | A | True | 5077.2943 | 5077.2943 | 3656.6443 | 2.1603 | 24 | 10 | 8 | 80 | 0.5458 | 31/2/2 | 3357.77/1604.28/2875.06/1650.04 | 1753.4856 | 565.4267 | 0 | 61.145/1.145 |
| 1 | 20261082 | D | True | 4956.8901 | 4956.8901 | 3904.1407 | 1.2802 | 21 | 3 | 3 | 61 | 0.5744 | 33/1/1 | 3627.00/2387.30/2782.38/590.46 | 3036.5317 | 400.3077 | 78.8513 | 62.148/2.148 |
| 1 | 20261083 | A | True | 5077.2943 | 5077.2943 | 3611.2002 | 2.3099 | 19 | 10 | 8 | 68 | 0.5387 | 32/2/1 | 3019.75/1892.30/3022.23/1502.87 | 1519.3557 | 460.1835 | 380.0927 | 61.598/1.598 |
| 1 | 20261083 | D | True | 4956.8901 | 4956.8901 | 3335.7544 | 1.4544 | 15 | 4 | 4 | 47 | 0.5944 | 32/1/2 | 2712.23/2094.66/3127.20/1503.05 | 1624.1477 | 492.2882 | 432.7724 | 64.792/4.792 |
| 2 | 20261081 | A | True | 2780.7783 | 2780.7783 | 2683.5496 | 3.2172 | 5 | 3 | 3 | 22 | 0.6167 | 29/6/0 | 2263.39/1691.94/2538.72/0.00 | 2538.7175 | 339.0801 | 549.4647 | 64.313/4.313 |
| 2 | 20261081 | D | True | 2127.4305 | 2127.4305 | 1871.8514 | 1.9562 | 16 | 11 | 7 | 63 | 0.6273 | 31/4/0 | 1735.01/1727.29/1272.26/1659.49 | 462.7484 | 392.7264 | 0 | 60.942/0.942 |
| 2 | 20261082 | A | True | 2780.7783 | 2780.7783 | 2702.0681 | 3.0227 | 7 | 2 | 2 | 24 | 0.5967 | 29/6/0 | 2244.87/1691.94/2557.24/0.00 | 2557.2360 | 338.6274 | 530.4935 | 65.168/5.168 |
| 2 | 20261082 | D | True | 2127.4305 | 2127.4305 | 1848.9965 | 2.2053 | 14 | 9 | 7 | 53 | 0.6183 | 30/5/0 | 1690.96/1701.96/1366.31/1684.82 | 335.6530 | 426.8836 | 28.4687 | 64.317/4.317 |
| 2 | 20261083 | A | True | 2780.7783 | 2780.7783 | 2721.0469 | 3.0027 | 6 | 1 | 1 | 23 | 0.5998 | 29/6/0 | 2244.87/1673.42/2575.75/0.00 | 2575.7545 | 339.1563 | 548.9434 | 60.384/0.384 |
| 2 | 20261083 | D | True | 2127.4305 | 2127.4305 | 2029.3808 | 1.8923 | 15 | 2 | 2 | 39 | 0.6087 | 34/1/0 | 1878.66/1863.93/1150.43/1351.02 | 728.2302 | 400.9733 | 13.6102 | 62.511/2.511 |
| 3 | 20261081 | A | True | 1465.5689 | 1465.5689 | 1465.5689 | 0.5313 | 20 | 7 | 0 | 71 | 0.6188 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 206.1164 | 75.2326 | 60.283/0.283 |
| 3 | 20261081 | D | True | 1465.5689 | 1465.5689 | 1465.5689 | 0.6743 | 19 | 11 | 1 | 70 | 0.5833 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 203.5168 | 58.6315 | 60.928/0.928 |
| 3 | 20261082 | A | True | 1465.5689 | 1465.5689 | 1465.5689 | 0.6230 | 20 | 9 | 0 | 66 | 0.6005 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 206.1164 | 75.2326 | 63.382/3.382 |
| 3 | 20261082 | D | True | 1465.5689 | 1465.5689 | 1465.5689 | 1.0687 | 19 | 11 | 0 | 66 | 0.5795 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 206.1164 | 75.2326 | 64.148/4.148 |
| 3 | 20261083 | A | True | 1465.5689 | 1465.5689 | 1465.5689 | 0.5015 | 26 | 9 | 1 | 80 | 0.6044 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 203.6279 | 62.3498 | 60.238/0.238 |
| 3 | 20261083 | D | True | 1465.5689 | 1465.5689 | 1465.5689 | 0.8507 | 28 | 13 | 0 | 94 | 0.5841 | 35/0/0 | 1438.56/1358.17/1108.55/1283.82 | 330.0147 | 206.1164 | 75.2326 | 61.523/1.523 |

配对差D−A；负值表示D更好。搜索改善为选中init−@60，正值表示ALNS降低Cmax。

| 例 | seed | @60绝对差s | @60相对差% | init绝对差s | init相对差% | A搜索改善s | D搜索改善s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20261081 | 1705.0471 | 55.1692 | -120.4043 | -2.3714 | 1986.7147 | 161.2633 |
| 1 | 20261082 | 247.4964 | 6.7684 | -120.4043 | -2.3714 | 1420.6500 | 1052.7493 |
| 1 | 20261083 | -275.4457 | -7.6275 | -120.4043 | -2.3714 | 1466.0942 | 1621.1356 |
| 2 | 20261081 | -811.6981 | -30.2472 | -653.3479 | -23.4951 | 97.2288 | 255.5790 |
| 2 | 20261082 | -853.0716 | -31.5711 | -653.3479 | -23.4951 | 78.7102 | 278.4340 |
| 2 | 20261083 | -691.6660 | -25.4191 | -653.3479 | -23.4951 | 59.7315 | 98.0496 |
| 3 | 20261081 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 20261082 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 20261083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| 范围 | 配对数 | 绝对差均值s | 绝对差中位s | 相对差均值% | 相对差中位% | 改善 | 持平 | 恶化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 9 | -75.4820 | 0.0000 | -3.6586 | 0.0000 | 4 | 3 | 2 |
| S1 | 3 | 559.0326 | 247.4964 | 18.1033 | 6.7684 | 1 | 0 | 2 |
| S2 | 3 | -785.4786 | -811.6981 | -29.0791 | -30.2472 | 3 | 0 | 0 |
| S3 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 3 | 0 |

### 直接回答：SMALL_30_39结果与定位

**结论属于情况C：没有在所有30多条实例上普遍恶化，但X丰富的S1仍有严重、依赖seed的退化。** 0.5仍保留为指定任务层配置，不作整体优越性结论，不因单个异常另设规模参数。以下只计算N35的9个配对，与历史N26完全分开。

1. **找到了真实N35实例，无需生成或删焊缝。** 38个允许读取的development workbook均检查了META和实际row count，144个有效N30–39 sheet来自31个workbook，N35候选12个。固定选择3个不同workbook的N35，分别代表X-rich、ordinary、split-sparse；没有访问V2_VALIDATION/ORACLE或ID_TEST_SEALED进行选样或模型选择。旧物理目录名不改变角色表中的development身份。

2. **按用户给出的应用规模，35比26更符合30–39定义。** 三例总焊缝长度为81.40/48.00/37.14 m，bbox面积覆盖比约0.898/0.790/0.815，网格占据率约0.938/0.813/0.813；几何和工作量不同。它们不是只改变N的控制实验，不能据此证明I3的异常由N=26导致。N35候选中还有Y非常密集的sheet（53个Y pattern），本轮固定三种选样未包含该几何极端；不能外推为所有35条实例均已覆盖。所有样本仍在20×12 m原平台规则内，未做设备IK验证。

3. **18个真实60秒run全部执行并取得独立认证解。** 9个配对的结果、三seed相对差如下；负值更好，所有异常seed保留。

| N35例 | A mean Cmax@60 s | D mean Cmax@60 s | seed81差% | seed82差% | seed83差% |
| --- | --- | --- | --- | --- | --- |
| 1 | 3452.8080 | 4011.8406 | 55.1692 | 6.7684 | -7.6275 |
| 2 | 2702.2215 | 1916.7429 | -30.2472 | -31.5711 | -25.4191 |
| 3 | 1465.5689 | 1465.5689 | 0.0000 | 0.0000 | 0.0000 |

ALL mean Cmax@60为A=2540.199 s、D=2464.717 s，平均绝对差−75.482 s（均值之比−2.97%）。配对绝对差中位数0 s；配对相对差均值−3.66%、中位数0%；4改善、3持平、2恶化。均值之比、配对相对差均值和中位数是不同统计量，不互换。

4. **整体相近并略有均值改善，局部严重恶化仍存在。** S1两次恶化+55.17%/+6.77%，第三次改善−7.63%；S2三次改善−30.25%/−31.57%/−25.42%；S3三次持平。S1 mean Cmax上升559.03 s，S2下降785.48 s，S3不变。不能用ALL均值或中位数掩盖S1，也不能把它概括成全部SMALL都变差。当前三个样本支持几何/轨迹相关问题，尚不足以识别一个适用于所有实例的几何因果判据。

5. **差异发生阶段已经分开。** 各run首个认证初始化解与最终选中的初始化解Cmax恰好相同，但两个字段均独立保留。下面恒等分解为：最终差 = 初始化差 + A搜索降低 − D搜索降低。

| 例 | A init Cmax | D init Cmax | init差D-A | A搜索降低s | D搜索降低s | 最终差D-A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5077.2943 | 4956.8901 | -120.4043 | 1624.4863 | 945.0494 | 559.0326 |
| 2 | 2780.7783 | 2127.4305 | -653.3479 | 78.5568 | 210.6875 | -785.4786 |
| 3 | 1465.5689 | 1465.5689 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

S1的A/D均退回RAIL_SERIAL_BOOTSTRAP，初始只有R0/R2承担任务，D初始反而低120.40 s；其最终变差来自后续ALNS改善不足，不能归咎为“D初始Cmax更差”。S1每次仍有15–24个完整迭代，不是零迭代；前两seed A/D完成轮数22/22、24/21，差距主要不在迭代数量。S2则主要由初始化选择从A的X_ORDER_AWARE变为D的RAIL_MONOTONE_BALANCED_BOOTSTRAP，初始下降653.35 s，再叠加约132.13 s的相对搜索收益；A/D平均完整迭代6/15，也有明显搜索效率差异。S3六次Cmax都等于初始化：A seeds81/82、D seeds82/83的global-best updates为0；另外两次虽有1次次级目标update，Cmax仍未下降。因此S3主目标表现主要依赖初始化，不能写成搜索能力提升。

6. **严重负载失衡仍出现，但与Cmax不是一一对应。** S1 D的前两seed负载差4580.26/3036.53 s，A为1305.51/1753.49 s，与Cmax恶化同向；最差D的R1负载仍为0。第三seed D负载差1624.15略高于A的1519.36 s，Cmax却改善7.63%，说明行走/等待/排程同样重要。S2负载差均值2557.24→508.88 s，与其改善一致；S3均约330.01 s不变。历史I3极端失衡仍保留，且本轮定向初始化重放已经证实其来源，详见下节。

7. **新外侧X/Y切点被实际采用，最终分配确实改变。** 九个D的@60解共采用10个Y外点（9个BY_OUTER_UPPER、1个BY_OUTER_LOWER）和2个X外点（BX_OUTER_UPPER/LOWER各1）；X外点均出现在S1 seed83的改善运行。S1三个D均采用外侧Y点；S2三个D均采用外侧Y点；S3始终全WHOLE。跨三seed统计，S1最终WHOLE/Y/X从A的93/6/6变为D的98/4/3，S2从87/18/0变为95/10/0，S3两组均105/0/0。这些是各配置三个独立解的次数合计，不是单个解的父焊缝数。不能据S1退化断言外点本身有害，改善seed也使用了它们。

新旧静态目录如下，零reference调用；S1还改变了上轨中心，S2改变了mandatory Y资格。因此A/D可行域不同，不是纯“切点数增加”的单变量算法竞赛。

| 例 | 配置 | WHOLE资格 | mandatory Y | optional Y | legal Y | legal X | x_up | x_low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 33 | 2 | 0 | 6 | 31 | 9.1178 | 9.8367 |
| 1 | D | 34 | 1 | 3 | 11 | 52 | 9.8367 | 9.8367 |
| 2 | A | 30 | 5 | 1 | 12 | 29 | 12.2937 | 12.2239 |
| 2 | D | 34 | 1 | 7 | 21 | 34 | 12.2937 | 12.2239 |
| 3 | A | 35 | 0 | 0 | 0 | 22 | 6.5969 | 11.2972 |
| 3 | D | 35 | 0 | 1 | 1 | 26 | 6.5969 | 11.2972 |

8. **更多合法候选没有保证有效搜索质量，S1问题在direction后/reference阶段已显现。** 下表是60秒内完成迭代的原生漏斗总计；C4 material只表示相对当时SA当前状态≥0.5%的认证改善，不是相对global-best，也不是对整个M192池的oracle capture。

| 例 | 配置 | iter≤60均值 | cheap有效率均值% | C2选中 | direction feasible | C4选中 | C4认证 | C4认证率% | C4 material |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 21.6667 | 55.6793 | 520 | 467 | 130 | 34 | 26.1538 | 21 |
| 1 | D | 19.3333 | 57.6880 | 464 | 435 | 116 | 13 | 11.2069 | 7 |
| 2 | A | 6 | 60.4406 | 144 | 102 | 36 | 6 | 16.6667 | 6 |
| 2 | D | 15 | 61.8088 | 360 | 356 | 90 | 36 | 40.0000 | 17 |
| 3 | A | 22 | 60.7879 | 528 | 524 | 132 | 41 | 31.0606 | 7 |
| 3 | D | 22 | 58.2302 | 528 | 525 | 132 | 49 | 37.1212 | 4 |

S1 D的cheap有效率55.68%→57.69%，direction-feasible比例467/520→435/464（约89.81%→93.75%），但C4认证率34/130→13/116（26.15%→11.21%），material候选21→7，搜索改善明显变弱。既有拒绝诊断包含机器人被相邻MOVE/WELD/POST/WAIT阻塞及B32 recovery后仍DEADLOCK；这不是物理不可行证明。S2 D的direction-feasible和C4认证比例均改善，S3认证候选更多仍没有降低global-best Cmax。**本轮没有额外评价未选候选，因此尚不能区分“池里没有好候选”与“C2/C4漏掉了好候选”，不能伪造Top8/Top2 capture结论。** S1少量迭代差不足以解释前两seed的差距，主要证据是实际认证候选和改善候选不足；更深原因需固定状态诊断，不能直接归咎SA或scheduler bug。

9. **认证与数值错误为0；候选失败分别记录，不当作run失败。** 所有18个@60解及final解独立认证，checkpoint/final认证错误列表为空。以下reference计数包含初始化及direction refinement，按实际完成时间≤60统计，与base C4候选计数不是同一分母。

| 例 | 配置 | ref≤60 | FEASIBLE | DEADLOCK | INFEASIBLE | NUMERIC |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 224 | 86 | 126 | 12 | 0 |
| 1 | D | 166 | 39 | 123 | 4 | 0 |
| 2 | A | 69 | 23 | 46 | 0 | 0 |
| 2 | D | 155 | 85 | 70 | 0 | 0 |
| 3 | A | 217 | 119 | 98 | 0 | 0 |
| 3 | D | 230 | 140 | 90 | 0 | 0 |

A/D平均runtime 62.048/62.386 s，平均overshoot 2.048/2.386 s，最大5.168/4.792 s。Cmax、分配与负载取原生≤60秒best事件对应解；global-best计数也对齐该时间戳，iteration/accepted/valid取完整iteration boundary。overshoot后的final_cmax单列原始记录，不参与比较。MLP不参与，本轮未改B32、certifier、干涉、M192/K或任何邻域。

10. **下一步先处理已定位的弱点，不直接启动大规模训练。** 这是情况C而非普遍的小规模退化：保留0.5任务配置与S1/I3异常，不设按规模切换参数。I3的串行fallback造成两机器人空闲，已有更好A初始解在D下合法认证，针对初始化鲁棒性做改进验证有明确必要；但仅改初始化不能解释S1，因为它的D初始略好、后续搜索反而弱。下一轮应从S1固定状态开始、保持M192做小范围C2/C4与reference筛选诊断，检查好候选是否生成、在哪层丢失，再决定排序或初始化改动。暂不进入全域新标签/MLP训练，不加operator、不调M/K、不改scheduler、不训练GAT。本轮没有执行这些下一步工作，完成复核后停止。


### I3历史边界案例的定向诊断

历史I3的26条结果不改写：A两seed Cmax@60为1962.32/1790.23 s，D为2505.74/2441.55 s（+27.69%/+36.38%）；D原始global-best updates为4/6，说明有搜索但未弥补初始损失。

本轮只用seed20261081各做一次A/D零迭代初始化重放，再做一次A初始解在D下的正常reference评价；共5+8+1=14次reference调用，没有重跑历史60秒组。A选择RAIL_MONOTONE_BALANCED_BOOTSTRAP，首个/选中初始化Cmax=2118.403 s，process loads=[1687.33,1610.59,1798.02,1675.70]，负载差187.44 s。D选择RAIL_SERIAL_BOOTSTRAP，首个/选中Cmax=3719.525 s，loads=[3440.52,0,3331.13,0]，负载差3440.52 s，R1/R3空闲。

对A初始解逐parent检查D目录中的kind/t/rail，将mandatory元信息按新catalog重建，再canonicalize；missing_patterns为空，原方向下重新reference评价并独立认证通过，Cmax仍为2118.403 s。这个解只有WHOLE/Y，未依赖旧X中心；没有未经检查假定所有旧X解在D下合法。本证据证明D可行域中存在更好的初始化分配，I3主要弱点是portfolio失败后退回负载极不均衡的串行fallback，而非新任务域必然无更好解。未发现合法性或数值bug；本轮不修改初始化、SA、ranking或scheduler。这项定向诊断只证明旧A初始解可重建，没有冒称重建了未保存完整schedule的旧A@60最优解。


完整pytest：D:\pybullet_test\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider，在DRL目录执行：303 passed in 37.34s；新增4项必要回归全部通过。

### SMALL_30_39逐次原始记录

<!-- SMALL30_RESULTS -->
```json
{
  "entries": [
    {
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "relative_path": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet_name": "g24_w035",
      "N": 35,
      "total_length_m": 81.40114003787251,
      "x_patterns": 52,
      "y_patterns": 11,
      "x_parents": 24,
      "cross_y6": 2,
      "cross_full_by": 1,
      "x_up": 9.836686700960001,
      "x_low": 9.836686700960001,
      "upper_only": 14,
      "lower_only": 18,
      "both": 2,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.8976498894895248,
      "grid_occupancy_ratio": 0.9375,
      "upper_lower_length_imbalance": 0.030943832904459935,
      "quadrant_length_cv": 0.2804116814589137,
      "selection_ordinal": 1,
      "tier": "SMALL_30_39",
      "geometry_role": "X_RICH"
    },
    {
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "relative_path": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet_name": "g19_w035",
      "N": 35,
      "total_length_m": 47.99570510627245,
      "x_patterns": 34,
      "y_patterns": 21,
      "x_parents": 24,
      "cross_y6": 5,
      "cross_full_by": 1,
      "x_up": 12.29372219962,
      "x_low": 12.223880778497,
      "upper_only": 20,
      "lower_only": 11,
      "both": 3,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.7896835556823183,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.355993984802175,
      "quadrant_length_cv": 0.6583822747544684,
      "selection_ordinal": 2,
      "tier": "SMALL_30_39",
      "geometry_role": "ORDINARY"
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "relative_path": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet_name": "g17_w035",
      "N": 35,
      "total_length_m": 37.14239830160228,
      "x_patterns": 26,
      "y_patterns": 1,
      "x_parents": 22,
      "cross_y6": 0,
      "cross_full_by": 0,
      "x_up": 6.5968820423225,
      "x_low": 11.297176045894,
      "upper_only": 20,
      "lower_only": 15,
      "both": 0,
      "mandatory_y": 0,
      "bbox_area_ratio": 0.8150082116584554,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.04488438892798867,
      "quadrant_length_cv": 0.23647125296854687,
      "selection_ordinal": 3,
      "tier": "SMALL_30_39",
      "geometry_role": "SPLIT_SPARSE"
    }
  ],
  "candidates": [
    {
      "instance_id": "data/DEV_ONLY/seed_000505::g20_w035",
      "relative_path": "data/DEV_ONLY/seed_000505.xlsx",
      "sheet_name": "g20_w035",
      "N": 35,
      "total_length_m": 73.07306533839702,
      "x_patterns": 29,
      "y_patterns": 53,
      "x_parents": 19,
      "cross_y6": 9,
      "cross_full_by": 8,
      "x_up": 7.447799622722,
      "x_low": 9.366950945806,
      "upper_only": 10,
      "lower_only": 13,
      "both": 4,
      "mandatory_y": 8,
      "bbox_area_ratio": 0.9589717748456108,
      "grid_occupancy_ratio": 0.875,
      "upper_lower_length_imbalance": 0.36559932344492757,
      "quadrant_length_cv": 0.4236355796504971
    },
    {
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "relative_path": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet_name": "g19_w035",
      "N": 35,
      "total_length_m": 47.99570510627245,
      "x_patterns": 34,
      "y_patterns": 21,
      "x_parents": 24,
      "cross_y6": 5,
      "cross_full_by": 1,
      "x_up": 12.29372219962,
      "x_low": 12.223880778497,
      "upper_only": 20,
      "lower_only": 11,
      "both": 3,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.7896835556823183,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.355993984802175,
      "quadrant_length_cv": 0.6583822747544684
    },
    {
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "relative_path": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet_name": "g24_w035",
      "N": 35,
      "total_length_m": 81.40114003787251,
      "x_patterns": 52,
      "y_patterns": 11,
      "x_parents": 24,
      "cross_y6": 2,
      "cross_full_by": 1,
      "x_up": 9.836686700960001,
      "x_low": 9.836686700960001,
      "upper_only": 14,
      "lower_only": 18,
      "both": 2,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.8976498894895248,
      "grid_occupancy_ratio": 0.9375,
      "upper_lower_length_imbalance": 0.030943832904459935,
      "quadrant_length_cv": 0.2804116814589137
    },
    {
      "instance_id": "data/ID_TEST/seed_2013829876::g21_w035",
      "relative_path": "data/ID_TEST/seed_2013829876.xlsx",
      "sheet_name": "g21_w035",
      "N": 35,
      "total_length_m": 72.35335366467412,
      "x_patterns": 47,
      "y_patterns": 11,
      "x_parents": 26,
      "cross_y6": 2,
      "cross_full_by": 1,
      "x_up": 11.90168146724,
      "x_low": 9.3101081543875,
      "upper_only": 9,
      "lower_only": 22,
      "both": 3,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.8255439023799446,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.21638976151538317,
      "quadrant_length_cv": 0.33267907433875094
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_0062056589::g18_w035",
      "relative_path": "data/PPO_TRAIN/seed_0062056589.xlsx",
      "sheet_name": "g18_w035",
      "N": 35,
      "total_length_m": 56.85802400549456,
      "x_patterns": 32,
      "y_patterns": 29,
      "x_parents": 19,
      "cross_y6": 5,
      "cross_full_by": 4,
      "x_up": 4.6310573872025,
      "x_low": 7.7847643085575005,
      "upper_only": 17,
      "lower_only": 12,
      "both": 2,
      "mandatory_y": 4,
      "bbox_area_ratio": 0.8076151548831428,
      "grid_occupancy_ratio": 0.875,
      "upper_lower_length_imbalance": 0.28873834846115665,
      "quadrant_length_cv": 0.46982474764153853
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "relative_path": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet_name": "g17_w035",
      "N": 35,
      "total_length_m": 37.14239830160228,
      "x_patterns": 26,
      "y_patterns": 1,
      "x_parents": 22,
      "cross_y6": 0,
      "cross_full_by": 0,
      "x_up": 6.5968820423225,
      "x_low": 11.297176045894,
      "upper_only": 20,
      "lower_only": 15,
      "both": 0,
      "mandatory_y": 0,
      "bbox_area_ratio": 0.8150082116584554,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.04488438892798867,
      "quadrant_length_cv": 0.23647125296854687
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_0442383381::g24_w035",
      "relative_path": "data/PPO_TRAIN/seed_0442383381.xlsx",
      "sheet_name": "g24_w035",
      "N": 35,
      "total_length_m": 60.73463581184139,
      "x_patterns": 49,
      "y_patterns": 12,
      "x_parents": 26,
      "cross_y6": 3,
      "cross_full_by": 2,
      "x_up": 5.57852700798,
      "x_low": 9.08399480618,
      "upper_only": 19,
      "lower_only": 13,
      "both": 1,
      "mandatory_y": 2,
      "bbox_area_ratio": 0.8277118994265132,
      "grid_occupancy_ratio": 0.9375,
      "upper_lower_length_imbalance": 0.09439244653673773,
      "quadrant_length_cv": 0.2644268414684331
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_0566744530::g16_w035",
      "relative_path": "data/PPO_TRAIN/seed_0566744530.xlsx",
      "sheet_name": "g16_w035",
      "N": 35,
      "total_length_m": 74.67267134299946,
      "x_patterns": 31,
      "y_patterns": 18,
      "x_parents": 24,
      "cross_y6": 5,
      "cross_full_by": 1,
      "x_up": 13.188647523439,
      "x_low": 10.792346591607998,
      "upper_only": 20,
      "lower_only": 13,
      "both": 1,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.8366247868825496,
      "grid_occupancy_ratio": 0.75,
      "upper_lower_length_imbalance": 0.23208228456592545,
      "quadrant_length_cv": 0.6703062412745205
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_1054426823::g20_w035",
      "relative_path": "data/PPO_TRAIN/seed_1054426823.xlsx",
      "sheet_name": "g20_w035",
      "N": 35,
      "total_length_m": 63.30822919049204,
      "x_patterns": 25,
      "y_patterns": 23,
      "x_parents": 19,
      "cross_y6": 5,
      "cross_full_by": 2,
      "x_up": 11.0828649425035,
      "x_low": 9.660533173496,
      "upper_only": 10,
      "lower_only": 21,
      "both": 2,
      "mandatory_y": 2,
      "bbox_area_ratio": 0.9728578850504209,
      "grid_occupancy_ratio": 0.9375,
      "upper_lower_length_imbalance": 0.4273925405898441,
      "quadrant_length_cv": 0.4590595422659161
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_1867256633::g19_w035",
      "relative_path": "data/PPO_TRAIN/seed_1867256633.xlsx",
      "sheet_name": "g19_w035",
      "N": 35,
      "total_length_m": 55.60014032050801,
      "x_patterns": 38,
      "y_patterns": 9,
      "x_parents": 21,
      "cross_y6": 2,
      "cross_full_by": 0,
      "x_up": 7.558168982675,
      "x_low": 9.742821864241,
      "upper_only": 25,
      "lower_only": 8,
      "both": 2,
      "mandatory_y": 0,
      "bbox_area_ratio": 0.8044859236569017,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.006786216780763627,
      "quadrant_length_cv": 0.2866615383839635
    },
    {
      "instance_id": "data/PPO_TRAIN/seed_1871665270::g17_w035",
      "relative_path": "data/PPO_TRAIN/seed_1871665270.xlsx",
      "sheet_name": "g17_w035",
      "N": 35,
      "total_length_m": 68.60835256179791,
      "x_patterns": 39,
      "y_patterns": 8,
      "x_parents": 24,
      "cross_y6": 2,
      "cross_full_by": 1,
      "x_up": 7.344368124008501,
      "x_low": 12.2381628395765,
      "upper_only": 15,
      "lower_only": 19,
      "both": 0,
      "mandatory_y": 1,
      "bbox_area_ratio": 0.931107281566514,
      "grid_occupancy_ratio": 0.8125,
      "upper_lower_length_imbalance": 0.28086902686133364,
      "quadrant_length_cv": 0.2914818081903489
    },
    {
      "instance_id": "data/VALIDATION/seed_0471570702::g23_w035",
      "relative_path": "data/VALIDATION/seed_0471570702.xlsx",
      "sheet_name": "g23_w035",
      "N": 35,
      "total_length_m": 71.22852352111282,
      "x_patterns": 32,
      "y_patterns": 14,
      "x_parents": 22,
      "cross_y6": 1,
      "cross_full_by": 0,
      "x_up": 8.686225968152,
      "x_low": 11.522825155396,
      "upper_only": 15,
      "lower_only": 20,
      "both": 0,
      "mandatory_y": 0,
      "bbox_area_ratio": 0.8689096197177215,
      "grid_occupancy_ratio": 0.9375,
      "upper_lower_length_imbalance": 0.5127514252471626,
      "quadrant_length_cv": 0.5693194658047855
    }
  ],
  "eligible_sheet_count": 144,
  "eligible_workbook_count": 31,
  "inspected_development_workbooks": 38,
  "runs": [
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "D",
      "core_mask": 4,
      "first_certified_initial": {
        "elapsed": 1.3765395999998873,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 4956.890053102886,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 4956.890053102886,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          4756.886948463563,
          0,
          4580.255647635742,
          0
        ],
        "process_imbalance": 4756.886948463563,
        "empty_travel_s": 415.3741916805848,
        "waiting_s": 0,
        "outer_cuts": [
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 1.2250360999987606,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 4795.626734677003,
      "at60": {
        "cmax": 4795.626734677003,
        "patterns": {
          "WHOLE": 33,
          "Y_SPLIT": 2
        },
        "process_loads": [
          4178.183244759859,
          0,
          4580.255647635742,
          628.7037037037038
        ],
        "process_imbalance": 4580.255647635742,
        "empty_travel_s": 400.7927282936136,
        "waiting_s": 0,
        "outer_cuts": [
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:Y_SPLIT:MIDPOINT",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::0",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            [
              "inst0020|group=144-DK1B|row=0000::1"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 22,
        "accepted_moves": 2,
        "raw_attempts": 4224,
        "valid_candidates": 2373,
        "global_best_updates": 1,
        "c4_feasible": 2,
        "c4_improving": 1,
        "c4_material": 1,
        "proposal_improving": 1,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2933,
            "constructed": 2169,
            "cheap_valid": 1652,
            "C2_selected": 115,
            "direction_evaluated": 115,
            "direction_feasible": 108,
            "C4_selected": 27,
            "reference_evaluated": 27,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 350,
            "constructed": 196,
            "cheap_valid": 196,
            "C2_selected": 17,
            "direction_evaluated": 17,
            "direction_feasible": 17,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 541,
            "constructed": 298,
            "cheap_valid": 298,
            "C2_selected": 22,
            "direction_evaluated": 22,
            "direction_feasible": 22,
            "C4_selected": 8,
            "reference_evaluated": 8,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_X": {
            "generated": 400,
            "constructed": 227,
            "cheap_valid": 227,
            "C2_selected": 22,
            "direction_evaluated": 22,
            "direction_feasible": 22,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 5.872762400029387,
          "cheap_screen_time": 15.039056899999196,
          "direction_dp_time": 0.11620310000216705,
          "reference_scheduler_time": 15.737332399992738,
          "certifier_time": 0.07020849999753409,
          "repair_time": 16.35402529998828
        }
      },
      "reference_calls_at60": 58,
      "reference_status_counts_at60": {
        "DEADLOCK": 49,
        "FEASIBLE": 8,
        "INFEASIBLE": 1
      },
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:POST', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=24', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=55', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R0:B2:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:SETUP', 'R0:B6:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=44', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=44', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=54', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=45', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=49', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=53', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD')\", \"R3 blocked by ('R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=16', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R2:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=66', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=81', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=16', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=53', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=49', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=13', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=93', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=50', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=49', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=49', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD', 'R3:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD', 'R3:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=70', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 4
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=52', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B18:MOVE', 'R2:B18:POST', 'R2:B18:SETUP', 'R2:B18:WELD', 'R2:B19:MOVE', 'R2:B19:POST', 'R2:B19:SETUP', 'R2:B19:WELD', 'R2:B1:POST', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=64', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B4:POST', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=45', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=41', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'R2 blocked by ()', \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=33', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=41', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B1:POST', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=54', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B18:MOVE', 'R2:B18:POST', 'R2:B18:SETUP', 'R2:B18:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=45', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B17:MOVE', 'R2:B17:POST', 'R2:B17:SETUP', 'R2:B17:WELD', 'R2:B18:MOVE', 'R2:B18:POST', 'R2:B18:SETUP', 'R2:B18:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=48', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=32', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=61', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=43', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "INFEASIBLE",
          "diagnostic": "['initial theoretical interference R0/R2']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5617897727272727,
      "best_events": [
        [
          1.3768171000010625,
          4956.890053102886
        ],
        [
          12.227100500000233,
          4795.626734677003
        ],
        [
          60.155232900000556,
          4363.604886012211
        ]
      ],
      "iterations_total": 23,
      "accepted_moves_total": 3,
      "global_best_updates_total": 2,
      "reference_calls_total": 59,
      "reference_status_counts_total": {
        "DEADLOCK": 49,
        "FEASIBLE": 9,
        "INFEASIBLE": 1
      },
      "elapsed_s": 60.165371299999606,
      "overshoot_s": 0.16537129999960598,
      "status": "COMPLETED",
      "final_cmax": 4363.604886012211,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "A",
      "core_mask": 64,
      "first_certified_initial": {
        "elapsed": 2.301197000000684,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 5077.294333210298,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 5077.294333210298,
        "patterns": {
          "WHOLE": 33,
          "Y_SPLIT": 2
        },
        "process_loads": [
          4880.849515120276,
          0,
          4506.293080979029,
          0
        ],
        "process_imbalance": 4880.849515120276,
        "empty_travel_s": 407.86286216582766,
        "waiting_s": 0,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_LOWER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 2.1602680000014516,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 3656.64429309039,
      "at60": {
        "cmax": 3656.64429309039,
        "patterns": {
          "WHOLE": 31,
          "X_SPLIT": 2,
          "Y_SPLIT": 2
        },
        "process_loads": [
          3357.7654440539804,
          1604.2798736347752,
          2875.0571308411045,
          1650.0401475694448
        ],
        "process_imbalance": 1753.4855704192053,
        "empty_travel_s": 565.426740587128,
        "waiting_s": 0,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:X_SPLIT:UPPER:BX_LOWER",
            "inst0005|group=144-DK1B|row=0000:X_SPLIT:UPPER:BX_CENTER",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_CENTER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0005|group=144-DK1B|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::0",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [
              "inst0005|group=144-DK1B|row=0000::1",
              "inst0004|group=144-DK1B.dxf|row=0000::1",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            [
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 24,
        "accepted_moves": 10,
        "raw_attempts": 4608,
        "valid_candidates": 2515,
        "global_best_updates": 8,
        "c4_feasible": 12,
        "c4_improving": 10,
        "c4_material": 7,
        "proposal_improving": 9,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 3154,
            "constructed": 2165,
            "cheap_valid": 1622,
            "C2_selected": 122,
            "direction_evaluated": 122,
            "direction_feasible": 109,
            "C4_selected": 30,
            "reference_evaluated": 30,
            "certified": 2,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_WHOLE": {
            "generated": 452,
            "constructed": 235,
            "cheap_valid": 235,
            "C2_selected": 22,
            "direction_evaluated": 22,
            "direction_feasible": 22,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_Y": {
            "generated": 282,
            "constructed": 83,
            "cheap_valid": 83,
            "C2_selected": 24,
            "direction_evaluated": 24,
            "direction_feasible": 24,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 6,
            "accepted": 5,
            "global_best_update": 3
          },
          "TARGET_X": {
            "generated": 720,
            "constructed": 575,
            "cheap_valid": 575,
            "C2_selected": 24,
            "direction_evaluated": 24,
            "direction_feasible": 23,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 3
          }
        },
        "timing_s": {
          "candidate_generation_time": 5.486638200032758,
          "cheap_screen_time": 15.001720800004477,
          "direction_dp_time": 0.13862800000424613,
          "reference_scheduler_time": 14.893089000002874,
          "certifier_time": 0.3660615000044345,
          "repair_time": 18.976681800013466
        }
      },
      "reference_calls_at60": 80,
      "reference_status_counts_at60": {
        "DEADLOCK": 46,
        "FEASIBLE": 32,
        "INFEASIBLE": 2
      },
      "init_status_counts": {
        "DEADLOCK": 9,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B9:POST', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=23', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=79', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R2 blocked by ('R0:B3:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=43', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=39', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=42', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "INFEASIBLE",
          "diagnostic": "['initial theoretical interference R0/R2']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=40', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B16:MOVE', 'R2:B16:POST', 'R2:B16:SETUP', 'R2:B16:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=35', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=43', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=50', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=109', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=66', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=45', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=45', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=76', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=49', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=92', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=42', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=55', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=64', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:SETUP', 'R0:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=33', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=24', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=25', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:SETUP', 'R1:B6:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=25', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5457899305555556,
      "best_events": [
        [
          2.301723000000493,
          5077.294333210298
        ],
        [
          4.158220900000742,
          5058.2035836444775
        ],
        [
          6.1478543000012,
          4737.522245453667
        ],
        [
          12.481616400000348,
          4700.48520841663
        ],
        [
          14.426174800000808,
          4554.952142774322
        ],
        [
          21.485327399999733,
          4517.800108122641
        ],
        [
          53.58010099999956,
          4254.204883588102
        ],
        [
          55.786678100001154,
          3668.065688664386
        ],
        [
          58.19764580000083,
          3656.64429309039
        ]
      ],
      "iterations_total": 25,
      "accepted_moves_total": 11,
      "global_best_updates_total": 8,
      "reference_calls_total": 84,
      "reference_status_counts_total": {
        "DEADLOCK": 47,
        "FEASIBLE": 34,
        "INFEASIBLE": 3
      },
      "elapsed_s": 61.14457560000119,
      "overshoot_s": 1.1445756000011897,
      "status": "COMPLETED",
      "final_cmax": 3656.64429309039,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "A",
      "core_mask": 1,
      "first_certified_initial": {
        "elapsed": 2.195637599999827,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 5077.294333210298,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 5077.294333210298,
        "patterns": {
          "WHOLE": 33,
          "Y_SPLIT": 2
        },
        "process_loads": [
          4880.849515120276,
          0,
          4506.293080979029,
          0
        ],
        "process_imbalance": 4880.849515120276,
        "empty_travel_s": 407.86286216582766,
        "waiting_s": 0,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_LOWER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 1.9887785000009899,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 3090.5796235693265,
      "at60": {
        "cmax": 3090.5796235693265,
        "patterns": {
          "WHOLE": 30,
          "X_SPLIT": 3,
          "Y_SPLIT": 2
        },
        "process_loads": [
          2953.882418848567,
          2008.1628988401894,
          2926.720003867641,
          1648.3772745429087
        ],
        "process_imbalance": 1305.5051443056582,
        "empty_travel_s": 393.0669863238701,
        "waiting_s": 0,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:X_SPLIT:UPPER:BX_LOWER",
            "inst0005|group=144-DK1B|row=0000:X_SPLIT:UPPER:BX_LOWER",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_CENTER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:X_SPLIT:LOWER:BX_LOWER",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=144-DK1B|row=0000::0",
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0004|group=144-DK1B.dxf|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::1",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::1",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0020|group=144-DK1B|row=0000::0",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole"
            ],
            [
              "inst0006|group=143-FR69B.dxf|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::1",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 22,
        "accepted_moves": 11,
        "raw_attempts": 4224,
        "valid_candidates": 2475,
        "global_best_updates": 9,
        "c4_feasible": 12,
        "c4_improving": 10,
        "c4_material": 7,
        "proposal_improving": 9,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2905,
            "constructed": 2086,
            "cheap_valid": 1603,
            "C2_selected": 113,
            "direction_evaluated": 113,
            "direction_feasible": 102,
            "C4_selected": 27,
            "reference_evaluated": 27,
            "certified": 2,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_WHOLE": {
            "generated": 393,
            "constructed": 258,
            "cheap_valid": 258,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 274,
            "constructed": 75,
            "cheap_valid": 75,
            "C2_selected": 22,
            "direction_evaluated": 22,
            "direction_feasible": 22,
            "C4_selected": 7,
            "reference_evaluated": 7,
            "certified": 7,
            "accepted": 6,
            "global_best_update": 4
          },
          "TARGET_X": {
            "generated": 652,
            "constructed": 539,
            "cheap_valid": 539,
            "C2_selected": 22,
            "direction_evaluated": 22,
            "direction_feasible": 22,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 3
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.625304399996821,
          "cheap_screen_time": 14.089658099996086,
          "direction_dp_time": 0.12872949999837147,
          "reference_scheduler_time": 12.76462380000521,
          "certifier_time": 0.34427990000040154,
          "repair_time": 23.406083699999726
        }
      },
      "reference_calls_at60": 76,
      "reference_status_counts_at60": {
        "DEADLOCK": 41,
        "FEASIBLE": 31,
        "INFEASIBLE": 4
      },
      "init_status_counts": {
        "DEADLOCK": 9,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B9:POST', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=23', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=79', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R2 blocked by ('R0:B3:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=43', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=39', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD', 'R2:B15:MOVE', 'R2:B15:POST', 'R2:B15:SETUP', 'R2:B15:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=92', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "INFEASIBLE",
          "diagnostic": "['initial theoretical interference R0/R2']",
          "count": 4
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:WELD',)\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=79', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=54', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=69', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=52', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=50', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=91', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B1:MOVE', 'R2:B1:SETUP', 'R2:B1:WELD')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=74', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=42', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=60', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:POST', 'R2:B0:WELD', 'R2:B1:MOVE')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=45', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:POST', 'R2:B0:WELD', 'R2:B1:MOVE')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=32', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R3:B2:MOVE', 'R3:B2:SETUP', 'R3:B2:WELD')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=36', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=10', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=40', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:POST', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=9', 'branch_points_considered=20', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=24', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=25', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:POST', 'R1:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=5', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B8:POST', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B7:POST', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=33', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=8', 'branch_points_considered=20', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=26', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B9:POST', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=33', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=87', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:SETUP', 'R1:B6:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B8:POST', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=13', 'branch_points_considered=21', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5859375,
      "best_events": [
        [
          2.196130499998617,
          5077.294333210298
        ],
        [
          4.190686599999026,
          5058.2035836444775
        ],
        [
          8.118206799999825,
          4719.003726935149
        ],
        [
          14.357273699999496,
          4699.192606536314
        ],
        [
          16.04040679999889,
          4468.16673131027
        ],
        [
          22.86298969999916,
          4449.268718601592
        ],
        [
          24.945126499998878,
          4122.584928903534
        ],
        [
          27.4856151999993,
          3624.999729874136
        ],
        [
          33.16857229999914,
          3109.4616042386974
        ],
        [
          58.552322899999126,
          3090.5796235693265
        ]
      ],
      "iterations_total": 23,
      "accepted_moves_total": 11,
      "global_best_updates_total": 9,
      "reference_calls_total": 78,
      "reference_status_counts_total": {
        "DEADLOCK": 43,
        "FEASIBLE": 31,
        "INFEASIBLE": 4
      },
      "elapsed_s": 61.92452679999951,
      "overshoot_s": 1.9245267999995121,
      "status": "COMPLETED",
      "final_cmax": 3090.5796235693265,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "D",
      "core_mask": 16,
      "first_certified_initial": {
        "elapsed": 1.439018799999758,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 4956.890053102886,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 4956.890053102886,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          4756.886948463563,
          0,
          4580.255647635742,
          0
        ],
        "process_imbalance": 4756.886948463563,
        "empty_travel_s": 415.3741916805848,
        "waiting_s": 0,
        "outer_cuts": [
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 1.2801712999989832,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 3904.1407258092167,
      "at60": {
        "cmax": 3904.1407258092167,
        "patterns": {
          "WHOLE": 33,
          "X_SPLIT": 1,
          "Y_SPLIT": 1
        },
        "process_loads": [
          3626.9950686688876,
          2387.2992872020836,
          2782.3848406729994,
          590.463399555335
        ],
        "process_imbalance": 3036.5316691135527,
        "empty_travel_s": 400.3077327898946,
        "waiting_s": 78.85133596456762,
        "outer_cuts": [
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:X_SPLIT:UPPER:BX_LOWER",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0005|group=144-DK1B|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::1",
              "inst0000|group=888-BK304A|row=0000::whole"
            ],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole"
            ],
            [
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 21,
        "accepted_moves": 3,
        "raw_attempts": 4032,
        "valid_candidates": 2316,
        "global_best_updates": 3,
        "c4_feasible": 6,
        "c4_improving": 2,
        "c4_material": 2,
        "proposal_improving": 3,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2763,
            "constructed": 1989,
            "cheap_valid": 1515,
            "C2_selected": 105,
            "direction_evaluated": 105,
            "direction_feasible": 97,
            "C4_selected": 26,
            "reference_evaluated": 26,
            "certified": 3,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_WHOLE": {
            "generated": 391,
            "constructed": 207,
            "cheap_valid": 207,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 474,
            "constructed": 294,
            "cheap_valid": 294,
            "C2_selected": 23,
            "direction_evaluated": 23,
            "direction_feasible": 22,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 404,
            "constructed": 300,
            "cheap_valid": 300,
            "C2_selected": 21,
            "direction_evaluated": 21,
            "direction_feasible": 21,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 3,
            "accepted": 1,
            "global_best_update": 1
          }
        },
        "timing_s": {
          "candidate_generation_time": 6.408464999998614,
          "cheap_screen_time": 17.43891729999814,
          "direction_dp_time": 0.12840069998856052,
          "reference_scheduler_time": 12.763426799998342,
          "certifier_time": 0.2255128999986482,
          "repair_time": 18.473412599994845
        }
      },
      "reference_calls_at60": 61,
      "reference_status_counts_at60": {
        "DEADLOCK": 43,
        "FEASIBLE": 18
      },
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:POST', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=24', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=55', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R0:B2:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:SETUP', 'R0:B6:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=44', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=37', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=54', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=51', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=69', 'branch_points_considered=14', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=38', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=73', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=36', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=6', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=10', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:POST', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=13', 'branch_points_considered=14', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B4:POST', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=6', 'branch_points_considered=22', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 4
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=63', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B4:POST', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=72', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=42', 'branch_points_considered=12', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:SETUP', 'R1:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=15', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B3:POST', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=12', 'branch_points_considered=17', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=27', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=10', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B8:POST', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:SETUP', 'R1:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=10', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=6', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:POST', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=49', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=21', 'branch_points_considered=16', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=25', 'branch_points_considered=10', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B4:POST', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=70', 'branch_points_considered=13', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=42', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5744047619047619,
      "best_events": [
        [
          1.4394344000011188,
          4956.890053102886
        ],
        [
          6.282750500000475,
          4795.626734677003
        ],
        [
          8.963027800000418,
          4203.8962425479485
        ],
        [
          11.881867100000818,
          3904.1407258092167
        ]
      ],
      "iterations_total": 22,
      "accepted_moves_total": 4,
      "global_best_updates_total": 3,
      "reference_calls_total": 65,
      "reference_status_counts_total": {
        "DEADLOCK": 44,
        "FEASIBLE": 21
      },
      "elapsed_s": 62.14755310000146,
      "overshoot_s": 2.14755310000146,
      "status": "COMPLETED",
      "final_cmax": 3904.1407258092167,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "A",
      "core_mask": 4,
      "first_certified_initial": {
        "elapsed": 2.4176446000001306,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 5077.294333210298,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 5077.294333210298,
        "patterns": {
          "WHOLE": 33,
          "Y_SPLIT": 2
        },
        "process_loads": [
          4880.849515120276,
          0,
          4506.293080979029,
          0
        ],
        "process_imbalance": 4880.849515120276,
        "empty_travel_s": 407.86286216582766,
        "waiting_s": 0,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_LOWER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 2.3098975000011706,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 3611.2001706026763,
      "at60": {
        "cmax": 3611.2001706026763,
        "patterns": {
          "WHOLE": 32,
          "X_SPLIT": 1,
          "Y_SPLIT": 2
        },
        "process_loads": [
          3019.7496259253153,
          1892.2956917634413,
          3022.2264714478074,
          1502.8708069627426
        ],
        "process_imbalance": 1519.3556644850648,
        "empty_travel_s": 460.1834623178984,
        "waiting_s": 380.0927340029831,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:X_SPLIT:UPPER:BX_UPPER",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:Y_SPLIT:BY_CENTER",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0009|group=826BK312A|row=0000::1",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::0",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0005|group=144-DK1B|row=0000::1"
            ],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::0",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            [
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 19,
        "accepted_moves": 10,
        "raw_attempts": 3648,
        "valid_candidates": 1965,
        "global_best_updates": 8,
        "c4_feasible": 10,
        "c4_improving": 8,
        "c4_material": 7,
        "proposal_improving": 8,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2490,
            "constructed": 1721,
            "cheap_valid": 1275,
            "C2_selected": 86,
            "direction_evaluated": 86,
            "direction_feasible": 58,
            "C4_selected": 21,
            "reference_evaluated": 21,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_WHOLE": {
            "generated": 370,
            "constructed": 182,
            "cheap_valid": 182,
            "C2_selected": 28,
            "direction_evaluated": 28,
            "direction_feasible": 28,
            "C4_selected": 8,
            "reference_evaluated": 8,
            "certified": 2,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_Y": {
            "generated": 214,
            "constructed": 70,
            "cheap_valid": 70,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 4,
            "accepted": 4,
            "global_best_update": 3
          },
          "TARGET_X": {
            "generated": 574,
            "constructed": 438,
            "cheap_valid": 438,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 2
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.6990846999506175,
          "cheap_screen_time": 12.138152000004993,
          "direction_dp_time": 0.10467120000066643,
          "reference_scheduler_time": 13.04729300001054,
          "certifier_time": 0.2642907000026753,
          "repair_time": 26.13955000000169
        }
      },
      "reference_calls_at60": 68,
      "reference_status_counts_at60": {
        "DEADLOCK": 39,
        "FEASIBLE": 23,
        "INFEASIBLE": 6
      },
      "init_status_counts": {
        "DEADLOCK": 9,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B9:POST', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=23', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=79', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R2 blocked by ('R0:B3:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:POST', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B3:MOVE', 'R0:B3:SETUP', 'R0:B3:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=43', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=39', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=67', 'branch_points_considered=10', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=60', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=92', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:POST', 'R2:B0:WELD', 'R2:B1:MOVE', 'R2:B1:SETUP', 'R2:B1:WELD')\", \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B3:POST', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=50', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=104', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:POST', 'R2:B0:WELD', 'R2:B1:MOVE', 'R2:B1:SETUP', 'R2:B1:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=59', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=44', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:WELD', 'R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=8', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "INFEASIBLE",
          "diagnostic": "['initial theoretical interference R0/R2']",
          "count": 6
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=46', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=40', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B0:POST', 'R2:B0:WELD', 'R2:B1:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=24', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:SETUP', 'R1:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=29', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:SETUP', 'R1:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=113', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=9', 'branch_points_considered=15', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=29', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=46', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=6', 'branch_points_considered=18', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:POST', 'R1:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=6', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:POST', 'R1:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=5', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=9', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=45', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5386513157894737,
      "best_events": [
        [
          2.417946600000505,
          5077.294333210298
        ],
        [
          5.073655000000144,
          4717.711125054833
        ],
        [
          11.643032600000879,
          4680.674088017796
        ],
        [
          18.818808199999694,
          4576.138499588117
        ],
        [
          20.84900370000105,
          4539.329349384146
        ],
        [
          24.178937900000165,
          4424.356466156256
        ],
        [
          27.352865400000155,
          3975.0369117499
        ],
        [
          33.602312600000005,
          3955.9461621840796
        ],
        [
          36.17098540000006,
          3611.2001706026763
        ]
      ],
      "iterations_total": 20,
      "accepted_moves_total": 11,
      "global_best_updates_total": 8,
      "reference_calls_total": 72,
      "reference_status_counts_total": {
        "DEADLOCK": 40,
        "FEASIBLE": 25,
        "INFEASIBLE": 7
      },
      "elapsed_s": 61.59776720000082,
      "overshoot_s": 1.597767200000817,
      "status": "COMPLETED",
      "final_cmax": 3611.2001706026763,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "D",
      "core_mask": 1,
      "first_certified_initial": {
        "elapsed": 0.9120855999990454,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2127.4304806137065,
        "diagnostics": [
          "wait-for cycles=()",
          "R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')",
          "R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')",
          "baseline DEADLOCK",
          "recovery_rollouts=16",
          "rollout_budget=32",
          "max_discrepancies_used=19",
          "branch_points_considered=4",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2127.4304806137065,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          1878.6608470408607,
          1863.9312263659199,
          1150.4306162107239,
          1351.024079481795
        ],
        "process_imbalance": 728.2302308301369,
        "empty_travel_s": 470.9702928226164,
        "waiting_s": 140.73069377088314,
        "outer_cuts": [
          {
            "parent": "inst0000|group=801-FR322A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_LOWER",
            "rail": null,
            "t": 0.12425678017329203,
            "coordinate": [
              15.791299288295,
              5.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_OUTER_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0002:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0003:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0004:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "init_time_s": 1.9562040000000707,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1871.8514437405865,
      "at60": {
        "cmax": 1871.8514437405865,
        "patterns": {
          "WHOLE": 31,
          "Y_SPLIT": 4
        },
        "process_loads": [
          1735.0063072373423,
          1727.2891416493458,
          1272.257886112483,
          1659.4934341001283
        ],
        "process_imbalance": 462.7484211248593,
        "empty_travel_s": 392.7264238501263,
        "waiting_s": 0,
        "outer_cuts": [
          {
            "parent": "inst0000|group=801-FR322A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.5102081503006561,
            "coordinate": [
              15.791299288295,
              6.5
            ]
          },
          {
            "parent": "inst0007|group=143-FR76A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.6312829826206405,
            "coordinate": [
              12.356223477627,
              6.5
            ]
          },
          {
            "parent": "inst0007|group=143-FR76A.dxf|row=0003",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.603759056976767,
            "coordinate": [
              11.656223477627,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_OUTER_UPPER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_OUTER_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_OUTER_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0004::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::0"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 16,
        "accepted_moves": 11,
        "raw_attempts": 3072,
        "valid_candidates": 1927,
        "global_best_updates": 7,
        "c4_feasible": 20,
        "c4_improving": 7,
        "c4_material": 7,
        "proposal_improving": 6,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2154,
            "constructed": 1650,
            "cheap_valid": 1309,
            "C2_selected": 68,
            "direction_evaluated": 68,
            "direction_feasible": 68,
            "C4_selected": 9,
            "reference_evaluated": 9,
            "certified": 6,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_WHOLE": {
            "generated": 226,
            "constructed": 173,
            "cheap_valid": 173,
            "C2_selected": 12,
            "direction_evaluated": 12,
            "direction_feasible": 12,
            "C4_selected": 2,
            "reference_evaluated": 2,
            "certified": 2,
            "accepted": 1,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 482,
            "constructed": 268,
            "cheap_valid": 268,
            "C2_selected": 32,
            "direction_evaluated": 32,
            "direction_feasible": 32,
            "C4_selected": 15,
            "reference_evaluated": 15,
            "certified": 10,
            "accepted": 7,
            "global_best_update": 5
          },
          "TARGET_X": {
            "generated": 210,
            "constructed": 177,
            "cheap_valid": 177,
            "C2_selected": 16,
            "direction_evaluated": 16,
            "direction_feasible": 16,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 2,
            "accepted": 1,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.4329301999914605,
          "cheap_screen_time": 12.647788299997046,
          "direction_dp_time": 0.08087040000827983,
          "reference_scheduler_time": 10.077853900009359,
          "certifier_time": 0.6506564000046637,
          "repair_time": 27.335526900016703
        }
      },
      "reference_calls_at60": 63,
      "reference_status_counts_at60": {
        "DEADLOCK": 16,
        "FEASIBLE": 47
      },
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=74', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B5:POST', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=72', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=73', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=55', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=75', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:WELD',)\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=72', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=52', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:POST', 'R1:B10:WELD')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R3:B3:POST', 'R3:B3:WELD', 'R3:B4:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=40', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=15', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6272786458333334,
      "best_events": [
        [
          2.091628700000001,
          2127.4304806137065
        ],
        [
          18.757656799998585,
          2015.7706104965791
        ],
        [
          22.034549399999378,
          1986.1775060051345
        ],
        [
          41.25804219999918,
          1918.1783755883685
        ],
        [
          47.12971759999891,
          1891.324832618528
        ],
        [
          50.7816716999987,
          1891.324832618528
        ],
        [
          54.12946609999926,
          1871.8514437405865
        ],
        [
          57.506581199999346,
          1871.8514437405865
        ]
      ],
      "iterations_total": 17,
      "accepted_moves_total": 12,
      "global_best_updates_total": 7,
      "reference_calls_total": 67,
      "reference_status_counts_total": {
        "DEADLOCK": 17,
        "FEASIBLE": 50
      },
      "elapsed_s": 60.942196500000136,
      "overshoot_s": 0.9421965000001364,
      "status": "COMPLETED",
      "final_cmax": 1871.8514437405865,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 1,
      "instance_id": "data/ID_TEST/seed_0533728435::g24_w035",
      "workbook": "data/ID_TEST/seed_0533728435.xlsx",
      "sheet": "g24_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "D",
      "core_mask": 64,
      "first_certified_initial": {
        "elapsed": 1.5676014999990002,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 4956.890053102886,
        "diagnostics": []
      },
      "chosen_initial": {
        "cmax": 4956.890053102886,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          4756.886948463563,
          0,
          4580.255647635742,
          0
        ],
        "process_imbalance": 4756.886948463563,
        "empty_travel_s": 415.3741916805848,
        "waiting_s": 0,
        "outer_cuts": [
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:WHOLE",
            "inst0005|group=144-DK1B|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::whole",
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0014|group=183-CM2B.dxf|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 1.4543731000012485,
      "init_strategy": "RAIL_SERIAL_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 3335.754439414221,
      "at60": {
        "cmax": 3335.754439414221,
        "patterns": {
          "WHOLE": 32,
          "X_SPLIT": 2,
          "Y_SPLIT": 1
        },
        "process_loads": [
          2712.2299724896957,
          2094.6569759738677,
          3127.201672674004,
          1503.0539749617378
        ],
        "process_imbalance": 1624.147697712266,
        "empty_travel_s": 492.2881696983285,
        "waiting_s": 432.7723729363011,
        "outer_cuts": [
          {
            "parent": "inst0004|group=144-DK1B.dxf|row=0000",
            "kind": "X_SPLIT",
            "point_id": "BX_OUTER_UPPER",
            "rail": "LOWER",
            "t": 0.40249306330599993,
            "coordinate": [
              10.336686700960001,
              6.293884350898
            ]
          },
          {
            "parent": "inst0005|group=144-DK1B|row=0000",
            "kind": "X_SPLIT",
            "point_id": "BX_OUTER_LOWER",
            "rail": "UPPER",
            "t": 0.26124037645736,
            "coordinate": [
              9.336686700960001,
              9.708758468913
            ]
          },
          {
            "parent": "inst0021|group=145-LB20A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.28313615601023767,
            "coordinate": [
              0.134072464869,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=888-BK304A|row=0000:WHOLE",
            "inst0001|group=801-FR322A|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0003|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0004|group=144-DK1B.dxf|row=0000:X_SPLIT:LOWER:BX_OUTER_UPPER",
            "inst0005|group=144-DK1B|row=0000:X_SPLIT:UPPER:BX_OUTER_LOWER",
            "inst0006|group=143-FR69B.dxf|row=0000:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0001:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0002:WHOLE",
            "inst0006|group=143-FR69B.dxf|row=0003:WHOLE",
            "inst0007|group=801-FR322A|row=0000:WHOLE",
            "inst0008|group=143-FR76A|row=0000:WHOLE",
            "inst0009|group=826BK312A|row=0000:WHOLE",
            "inst0010|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0011|group=145-LB17A|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0012|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0013|group=145-LB17A|row=0000:WHOLE",
            "inst0014|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=888-BK304A|row=0000:WHOLE",
            "inst0017|group=165-SR1A|row=0000:WHOLE",
            "inst0018|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0019|group=804-FR314A|row=0000:WHOLE",
            "inst0020|group=144-DK1B|row=0000:WHOLE",
            "inst0021|group=145-LB20A|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0022|group=804-FR314A|row=0000:WHOLE",
            "inst0023|group=143-FR65A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0021|group=145-LB20A|row=0000::0",
              "inst0013|group=145-LB17A|row=0000::whole",
              "inst0016|group=888-BK304A|row=0000::whole",
              "inst0000|group=888-BK304A|row=0000::whole",
              "inst0010|group=685-BK21A.dxf|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::0",
              "inst0003|group=801-FR322A.dxf|row=0003::whole",
              "inst0003|group=801-FR322A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0019|group=804-FR314A|row=0000::whole"
            ],
            [
              "inst0020|group=144-DK1B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0003|group=801-FR322A.dxf|row=0004::whole",
              "inst0003|group=801-FR322A.dxf|row=0000::whole",
              "inst0005|group=144-DK1B|row=0000::1"
            ],
            [
              "inst0021|group=145-LB20A|row=0000::1",
              "inst0017|group=165-SR1A|row=0000::whole",
              "inst0009|group=826BK312A|row=0000::whole",
              "inst0007|group=801-FR322A|row=0000::whole",
              "inst0023|group=143-FR65A|row=0000::whole",
              "inst0008|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0000::whole",
              "inst0012|group=165-SR2A.dxf|row=0002::whole",
              "inst0012|group=165-SR2A.dxf|row=0001::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0018|group=685-BK21A.dxf|row=0000::whole",
              "inst0006|group=143-FR69B.dxf|row=0002::whole",
              "inst0006|group=143-FR69B.dxf|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::0"
            ],
            [
              "inst0006|group=143-FR69B.dxf|row=0001::whole",
              "inst0006|group=143-FR69B.dxf|row=0003::whole",
              "inst0011|group=145-LB17A|row=0000::whole",
              "inst0022|group=804-FR314A|row=0000::whole",
              "inst0004|group=144-DK1B.dxf|row=0000::1",
              "inst0014|group=183-CM2B.dxf|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 15,
        "accepted_moves": 4,
        "raw_attempts": 2880,
        "valid_candidates": 1712,
        "global_best_updates": 4,
        "c4_feasible": 5,
        "c4_improving": 4,
        "c4_material": 4,
        "proposal_improving": 4,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 1969,
            "constructed": 1430,
            "cheap_valid": 1108,
            "C2_selected": 75,
            "direction_evaluated": 75,
            "direction_feasible": 62,
            "C4_selected": 19,
            "reference_evaluated": 19,
            "certified": 2,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_WHOLE": {
            "generated": 287,
            "constructed": 177,
            "cheap_valid": 177,
            "C2_selected": 15,
            "direction_evaluated": 15,
            "direction_feasible": 15,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 332,
            "constructed": 209,
            "cheap_valid": 209,
            "C2_selected": 15,
            "direction_evaluated": 15,
            "direction_feasible": 15,
            "C4_selected": 3,
            "reference_evaluated": 3,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 292,
            "constructed": 218,
            "cheap_valid": 218,
            "C2_selected": 15,
            "direction_evaluated": 15,
            "direction_feasible": 15,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 3,
            "accepted": 2,
            "global_best_update": 2
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.5574942999992345,
          "cheap_screen_time": 13.192355800001678,
          "direction_dp_time": 0.09636499999032822,
          "reference_scheduler_time": 9.117654900001071,
          "certifier_time": 0.15831200000320678,
          "repair_time": 29.78568620002443
        }
      },
      "reference_calls_at60": 47,
      "reference_status_counts_at60": {
        "DEADLOCK": 31,
        "FEASIBLE": 13,
        "INFEASIBLE": 3
      },
      "init_status_counts": {
        "DEADLOCK": 6,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:POST', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=24', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=55', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R0:B2:MOVE',)\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:WAIT:0')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:SETUP', 'R0:B6:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=44', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=53', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD')\", \"R2 blocked by ('R3:B2:POST', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=59', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=46', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=6', 'rollout_budget=32', 'max_discrepancies_used=49', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B12:MOVE', 'R0:B12:POST', 'R0:B12:SETUP', 'R0:B12:WELD', 'R0:B13:MOVE', 'R0:B13:POST', 'R0:B13:SETUP', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:POST', 'R0:B14:SETUP', 'R0:B14:WELD', 'R0:B15:MOVE', 'R0:B15:POST', 'R0:B15:SETUP', 'R0:B15:WELD', 'R0:B2:POST', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE', 'R0:B4:POST', 'R0:B4:SETUP', 'R0:B4:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=44', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "INFEASIBLE",
          "diagnostic": "['initial theoretical interference R0/R2']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", \"R2 blocked by ('R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=73', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:POST', 'R1:B3:WELD')\", \"R3 blocked by ('R2:B10:POST', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:POST', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD', 'R2:B14:MOVE', 'R2:B14:POST', 'R2:B14:SETUP', 'R2:B14:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=54', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", \"R3 blocked by ('R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B2:POST', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=10', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=21', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD', 'R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=5', 'branch_points_considered=23', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", \"R2 blocked by ('R3:B5:POST', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B5:POST', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD', 'R3:B9:MOVE', 'R3:B9:POST', 'R3:B9:SETUP', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=21', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B12:MOVE', 'R2:B12:POST', 'R2:B12:SETUP', 'R2:B12:WELD', 'R2:B13:MOVE', 'R2:B13:POST', 'R2:B13:SETUP', 'R2:B13:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B4:POST', 'R3:B4:WELD', 'R3:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:SETUP', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:POST', 'R0:B2:SETUP', 'R0:B2:WELD', 'R0:B3:MOVE', 'R0:B3:POST', 'R0:B3:SETUP', 'R0:B3:WELD', 'R0:B4:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=23', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD', 'R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R0:B6:POST', 'R0:B6:SETUP', 'R0:B6:WELD', 'R0:B7:MOVE', 'R0:B7:POST', 'R0:B7:SETUP', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=31', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:POST', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B11:MOVE', 'R2:B11:SETUP', 'R2:B11:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:POST', 'R1:B5:WELD')\", \"R2 blocked by ('R3:B4:POST', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5944444444444444,
      "best_events": [
        [
          1.5680556999996043,
          4956.890053102886
        ],
        [
          4.08805349999966,
          4795.626734677003
        ],
        [
          11.029267099998833,
          4296.906515570622
        ],
        [
          17.15062729999954,
          3789.0207309195393
        ],
        [
          28.175451899998734,
          3335.754439414221
        ]
      ],
      "iterations_total": 16,
      "accepted_moves_total": 4,
      "global_best_updates_total": 4,
      "reference_calls_total": 49,
      "reference_status_counts_total": {
        "DEADLOCK": 33,
        "FEASIBLE": 13,
        "INFEASIBLE": 3
      },
      "elapsed_s": 64.79195829999844,
      "overshoot_s": 4.791958299998441,
      "status": "COMPLETED",
      "final_cmax": 3335.754439414221,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "A",
      "core_mask": 16,
      "first_certified_initial": {
        "elapsed": 3.3188305999992735,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2780.7783420136006,
        "diagnostics": [
          "wait-for cycles=()",
          "R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=20",
          "rollout_budget=32",
          "max_discrepancies_used=106",
          "branch_points_considered=9",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2780.7783420136006,
        "patterns": {
          "WHOLE": 30,
          "Y_SPLIT": 5
        },
        "process_loads": [
          2244.87409103941,
          1562.445939797371,
          2636.726738262519,
          0
        ],
        "process_imbalance": 2636.726738262519,
        "empty_travel_s": 339.323886510636,
        "waiting_s": 657.254016946015,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 3.2171887000004062,
      "init_strategy": "X_ORDER_AWARE",
      "certified_at60": true,
      "cmax_at60": 2683.5495868958583,
      "at60": {
        "cmax": 2683.5495868958583,
        "patterns": {
          "WHOLE": 29,
          "Y_SPLIT": 6
        },
        "process_loads": [
          2263.392609557928,
          1691.9366805381117,
          2538.71747900326,
          0
        ],
        "process_imbalance": 2538.71747900326,
        "empty_travel_s": 339.08010922472425,
        "waiting_s": 549.4646978312145,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_CENTER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_CENTER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:Y_SPLIT:MIDPOINT",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0007|group=143-FR76A.dxf|row=0006::1",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::0",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 5,
        "accepted_moves": 3,
        "raw_attempts": 960,
        "valid_candidates": 592,
        "global_best_updates": 3,
        "c4_feasible": 3,
        "c4_improving": 3,
        "c4_material": 3,
        "proposal_improving": 3,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 665,
            "constructed": 512,
            "cheap_valid": 407,
            "C2_selected": 26,
            "direction_evaluated": 26,
            "direction_feasible": 16,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 83,
            "constructed": 56,
            "cheap_valid": 56,
            "C2_selected": 4,
            "direction_evaluated": 4,
            "direction_feasible": 1,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 78,
            "constructed": 35,
            "cheap_valid": 35,
            "C2_selected": 5,
            "direction_evaluated": 5,
            "direction_feasible": 5,
            "C4_selected": 3,
            "reference_evaluated": 3,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 3
          },
          "TARGET_X": {
            "generated": 134,
            "constructed": 94,
            "cheap_valid": 94,
            "C2_selected": 5,
            "direction_evaluated": 5,
            "direction_feasible": 5,
            "C4_selected": 1,
            "reference_evaluated": 1,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 1.3072115000122722,
          "cheap_screen_time": 3.9918917000013607,
          "direction_dp_time": 0.028182300005937577,
          "reference_scheduler_time": 14.83167709999725,
          "certifier_time": 0.16473369999948773,
          "repair_time": 31.392248800002562
        }
      },
      "reference_calls_at60": 22,
      "reference_status_counts_at60": {
        "DEADLOCK": 12,
        "FEASIBLE": 10
      },
      "init_status_counts": {
        "DEADLOCK": 5,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=80', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD', 'R3:B9:MOVE', 'R3:B9:POST', 'R3:B9:SETUP', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B9:MOVE', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R0:B5:MOVE', 'R0:B5:SETUP', 'R0:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:MOVE', 'R1:B11:SETUP', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=82', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R3 blocked by ('R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=26', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B13:MOVE', 'R1:B13:POST', 'R1:B13:SETUP', 'R1:B13:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD', 'R2:B3:MOVE', 'R2:B3:SETUP', 'R2:B3:WELD')\", \"R2 blocked by ('R1:B10:POST', 'R1:B10:WELD', 'R1:B11:MOVE')\", \"R3 blocked by ('R2:B10:MOVE', 'R2:B10:SETUP', 'R2:B10:WELD', 'R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE', 'R2:B9:POST', 'R2:B9:SETUP', 'R2:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=79', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=105', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B13:MOVE', 'R1:B13:POST', 'R1:B13:SETUP', 'R1:B13:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=45', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=32', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=46', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6166666666666667,
      "best_events": [
        [
          3.3202556999985973,
          2780.7783420136006
        ],
        [
          11.799637899999652,
          2721.0468588305844
        ],
        [
          23.67350019999867,
          2702.0681054143765
        ],
        [
          52.6026583999992,
          2683.5495868958583
        ],
        [
          64.29927919999864,
          2665.1668316329983
        ]
      ],
      "iterations_total": 6,
      "accepted_moves_total": 4,
      "global_best_updates_total": 4,
      "reference_calls_total": 26,
      "reference_status_counts_total": {
        "DEADLOCK": 13,
        "FEASIBLE": 13
      },
      "elapsed_s": 64.31268439999985,
      "overshoot_s": 4.312684399999853,
      "status": "COMPLETED",
      "final_cmax": 2665.1668316329983,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "A",
      "core_mask": 4,
      "first_certified_initial": {
        "elapsed": 3.122621299999082,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2780.7783420136006,
        "diagnostics": [
          "wait-for cycles=()",
          "R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=20",
          "rollout_budget=32",
          "max_discrepancies_used=106",
          "branch_points_considered=9",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2780.7783420136006,
        "patterns": {
          "WHOLE": 30,
          "Y_SPLIT": 5
        },
        "process_loads": [
          2244.87409103941,
          1562.445939797371,
          2636.726738262519,
          0
        ],
        "process_imbalance": 2636.726738262519,
        "empty_travel_s": 339.323886510636,
        "waiting_s": 657.254016946015,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 3.0226846000005025,
      "init_strategy": "X_ORDER_AWARE",
      "certified_at60": true,
      "cmax_at60": 2702.0681054143765,
      "at60": {
        "cmax": 2702.0681054143765,
        "patterns": {
          "WHOLE": 29,
          "Y_SPLIT": 6
        },
        "process_loads": [
          2244.87409103941,
          1691.9366805381117,
          2557.2359975217787,
          0
        ],
        "process_imbalance": 2557.2359975217787,
        "empty_travel_s": 338.62743715065227,
        "waiting_s": 530.4935072367269,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_CENTER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:Y_SPLIT:MIDPOINT",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0007|group=143-FR76A.dxf|row=0006::1",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::0",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 7,
        "accepted_moves": 2,
        "raw_attempts": 1344,
        "valid_candidates": 802,
        "global_best_updates": 2,
        "c4_feasible": 2,
        "c4_improving": 2,
        "c4_material": 2,
        "proposal_improving": 2,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 968,
            "constructed": 741,
            "cheap_valid": 584,
            "C2_selected": 39,
            "direction_evaluated": 39,
            "direction_feasible": 30,
            "C4_selected": 12,
            "reference_evaluated": 12,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 61,
            "constructed": 35,
            "cheap_valid": 35,
            "C2_selected": 3,
            "direction_evaluated": 3,
            "direction_feasible": 1,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 175,
            "constructed": 73,
            "cheap_valid": 73,
            "C2_selected": 7,
            "direction_evaluated": 7,
            "direction_feasible": 6,
            "C4_selected": 2,
            "reference_evaluated": 2,
            "certified": 2,
            "accepted": 2,
            "global_best_update": 2
          },
          "TARGET_X": {
            "generated": 140,
            "constructed": 110,
            "cheap_valid": 110,
            "C2_selected": 7,
            "direction_evaluated": 7,
            "direction_feasible": 4,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 1.75740169999699,
          "cheap_screen_time": 4.472333200001231,
          "direction_dp_time": 0.03691310000613157,
          "reference_scheduler_time": 14.492815399999017,
          "certifier_time": 0.1150813999975071,
          "repair_time": 32.804857599998286
        }
      },
      "reference_calls_at60": 24,
      "reference_status_counts_at60": {
        "DEADLOCK": 17,
        "FEASIBLE": 7
      },
      "init_status_counts": {
        "DEADLOCK": 5,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=80', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD', 'R3:B9:MOVE', 'R3:B9:POST', 'R3:B9:SETUP', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B9:MOVE', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R0:B5:MOVE', 'R0:B5:SETUP', 'R0:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:MOVE', 'R1:B11:SETUP', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=82', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=17', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=64', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=34', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R3 blocked by ('R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=26', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R3 blocked by ('R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=82', 'branch_points_considered=12', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=37', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=46', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B0:POST', 'R0:B0:WELD', 'R0:B1:MOVE', 'R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=58', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=19', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5967261904761905,
      "best_events": [
        [
          3.1232655000003433,
          2780.7783420136006
        ],
        [
          27.99183339999945,
          2721.0468588305844
        ],
        [
          38.19643579999865,
          2702.0681054143765
        ],
        [
          65.15302510000038,
          2683.5495868958583
        ]
      ],
      "iterations_total": 8,
      "accepted_moves_total": 3,
      "global_best_updates_total": 3,
      "reference_calls_total": 28,
      "reference_status_counts_total": {
        "DEADLOCK": 18,
        "FEASIBLE": 10
      },
      "elapsed_s": 65.16838610000013,
      "overshoot_s": 5.168386100000134,
      "status": "COMPLETED",
      "final_cmax": 2683.5495868958583,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "A",
      "core_mask": 16,
      "first_certified_initial": {
        "elapsed": 3.101649999998699,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2780.7783420136006,
        "diagnostics": [
          "wait-for cycles=()",
          "R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=20",
          "rollout_budget=32",
          "max_discrepancies_used=106",
          "branch_points_considered=9",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2780.7783420136006,
        "patterns": {
          "WHOLE": 30,
          "Y_SPLIT": 5
        },
        "process_loads": [
          2244.87409103941,
          1562.445939797371,
          2636.726738262519,
          0
        ],
        "process_imbalance": 2636.726738262519,
        "empty_travel_s": 339.323886510636,
        "waiting_s": 657.254016946015,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 3.002729100000579,
      "init_strategy": "X_ORDER_AWARE",
      "certified_at60": true,
      "cmax_at60": 2721.0468588305844,
      "at60": {
        "cmax": 2721.0468588305844,
        "patterns": {
          "WHOLE": 29,
          "Y_SPLIT": 6
        },
        "process_loads": [
          2244.87409103941,
          1673.4181620195932,
          2575.754516040297,
          0
        ],
        "process_imbalance": 2575.754516040297,
        "empty_travel_s": 339.1563355016321,
        "waiting_s": 548.9433623019551,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:Y_SPLIT:MIDPOINT",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0005|group=143-FR69B|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0007|group=143-FR76A.dxf|row=0006::1",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0008|group=804-FR313A|row=0000::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0007|group=143-FR76A.dxf|row=0006::0",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0009|group=642-BL14A.dxf|row=0000::whole"
            ],
            []
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 6,
        "accepted_moves": 1,
        "raw_attempts": 1152,
        "valid_candidates": 691,
        "global_best_updates": 1,
        "c4_feasible": 1,
        "c4_improving": 1,
        "c4_material": 1,
        "proposal_improving": 1,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 836,
            "constructed": 648,
            "cheap_valid": 499,
            "C2_selected": 34,
            "direction_evaluated": 34,
            "direction_feasible": 27,
            "C4_selected": 9,
            "reference_evaluated": 9,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 42,
            "constructed": 31,
            "cheap_valid": 31,
            "C2_selected": 2,
            "direction_evaluated": 2,
            "direction_feasible": 0,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 165,
            "constructed": 72,
            "cheap_valid": 72,
            "C2_selected": 6,
            "direction_evaluated": 6,
            "direction_feasible": 4,
            "C4_selected": 1,
            "reference_evaluated": 1,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_X": {
            "generated": 109,
            "constructed": 89,
            "cheap_valid": 89,
            "C2_selected": 6,
            "direction_evaluated": 6,
            "direction_feasible": 3,
            "C4_selected": 2,
            "reference_evaluated": 2,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 1.5559085000113555,
          "cheap_screen_time": 4.737346899999466,
          "direction_dp_time": 0.03562600000623206,
          "reference_scheduler_time": 11.791001099998539,
          "certifier_time": 0.07143779999933031,
          "repair_time": 30.714555700000346
        }
      },
      "reference_calls_at60": 23,
      "reference_status_counts_at60": {
        "DEADLOCK": 17,
        "FEASIBLE": 6
      },
      "init_status_counts": {
        "DEADLOCK": 5,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=80', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:POST', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE', 'R3:B10:MOVE', 'R3:B10:POST', 'R3:B10:SETUP', 'R3:B10:WELD', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD', 'R3:B9:MOVE', 'R3:B9:POST', 'R3:B9:SETUP', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B9:MOVE', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R0:B5:MOVE', 'R0:B5:SETUP', 'R0:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B11:MOVE', 'R1:B11:SETUP', 'R1:B11:WELD')\", \"R2 blocked by ('R0:B5:POST', 'R0:B5:WELD', 'R0:B6:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=82', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=48', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=106', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=72', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R3 blocked by ('R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=26', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R1:B8:POST', 'R1:B8:WELD', 'R1:B9:MOVE')\", \"R3 blocked by ('R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=37', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD', 'R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=34', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B1:POST', 'R3:B1:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=84', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R2 blocked by ('R3:B0:POST', 'R3:B0:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=98', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B1:POST', 'R1:B1:WELD', 'R1:B2:MOVE')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B11:MOVE', 'R1:B11:POST', 'R1:B11:SETUP', 'R1:B11:WELD', 'R1:B12:MOVE', 'R1:B12:POST', 'R1:B12:SETUP', 'R1:B12:WELD', 'R1:B13:MOVE', 'R1:B13:POST', 'R1:B13:SETUP', 'R1:B13:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R1:B5:MOVE', 'R1:B5:SETUP', 'R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD', 'R2:B9:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=86', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE', 'R2:B2:POST', 'R2:B2:SETUP', 'R2:B2:WELD', 'R2:B3:MOVE', 'R2:B3:POST', 'R2:B3:SETUP', 'R2:B3:WELD', 'R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=38', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5998263888888888,
      "best_events": [
        [
          3.102276300000085,
          2780.7783420136006
        ],
        [
          31.136067299999922,
          2721.0468588305844
        ],
        [
          60.36862429999928,
          2702.0681054143765
        ]
      ],
      "iterations_total": 7,
      "accepted_moves_total": 2,
      "global_best_updates_total": 2,
      "reference_calls_total": 24,
      "reference_status_counts_total": {
        "DEADLOCK": 17,
        "FEASIBLE": 7
      },
      "elapsed_s": 60.383718099999896,
      "overshoot_s": 0.38371809999989637,
      "status": "COMPLETED",
      "final_cmax": 2702.0681054143765,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "D",
      "core_mask": 1,
      "first_certified_initial": {
        "elapsed": 1.022270200001003,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2127.4304806137065,
        "diagnostics": [
          "wait-for cycles=()",
          "R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')",
          "R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')",
          "baseline DEADLOCK",
          "recovery_rollouts=16",
          "rollout_budget=32",
          "max_discrepancies_used=19",
          "branch_points_considered=4",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2127.4304806137065,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          1878.6608470408607,
          1863.9312263659199,
          1150.4306162107239,
          1351.024079481795
        ],
        "process_imbalance": 728.2302308301369,
        "empty_travel_s": 470.9702928226164,
        "waiting_s": 140.73069377088314,
        "outer_cuts": [
          {
            "parent": "inst0000|group=801-FR322A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_LOWER",
            "rail": null,
            "t": 0.12425678017329203,
            "coordinate": [
              15.791299288295,
              5.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_OUTER_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0002:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0003:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0004:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "init_time_s": 2.2053280000000086,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1848.9965105011386,
      "at60": {
        "cmax": 1848.9965105011386,
        "patterns": {
          "WHOLE": 30,
          "Y_SPLIT": 5
        },
        "process_loads": [
          1690.9568151133608,
          1701.9603747446236,
          1366.3073782364645,
          1684.8222010048505
        ],
        "process_imbalance": 335.65299650815905,
        "empty_travel_s": 426.88357223750916,
        "waiting_s": 28.468667181310593,
        "outer_cuts": [
          {
            "parent": "inst0007|group=143-FR76A.dxf|row=0001",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.6356343283518083,
            "coordinate": [
              13.079223477627,
              6.5
            ]
          },
          {
            "parent": "inst0007|group=143-FR76A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.6312829826206405,
            "coordinate": [
              12.356223477627,
              6.5
            ]
          },
          {
            "parent": "inst0007|group=143-FR76A.dxf|row=0003",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.603759056976767,
            "coordinate": [
              11.656223477627,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:MIDPOINT",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:Y_SPLIT:BY_OUTER_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0002:Y_SPLIT:BY_OUTER_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0003:Y_SPLIT:BY_OUTER_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0004:Y_SPLIT:BY_UPPER",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::0",
              "inst0007|group=143-FR76A.dxf|row=0003::0",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::0"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::0",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0007|group=143-FR76A.dxf|row=0004::1",
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::1",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::1",
              "inst0007|group=143-FR76A.dxf|row=0002::1",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 14,
        "accepted_moves": 9,
        "raw_attempts": 2688,
        "valid_candidates": 1662,
        "global_best_updates": 7,
        "c4_feasible": 14,
        "c4_improving": 9,
        "c4_material": 9,
        "proposal_improving": 7,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 1880,
            "constructed": 1430,
            "cheap_valid": 1108,
            "C2_selected": 60,
            "direction_evaluated": 60,
            "direction_feasible": 57,
            "C4_selected": 7,
            "reference_evaluated": 7,
            "certified": 1,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 205,
            "constructed": 163,
            "cheap_valid": 163,
            "C2_selected": 10,
            "direction_evaluated": 10,
            "direction_feasible": 10,
            "C4_selected": 2,
            "reference_evaluated": 2,
            "certified": 2,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_Y": {
            "generated": 455,
            "constructed": 243,
            "cheap_valid": 243,
            "C2_selected": 28,
            "direction_evaluated": 28,
            "direction_feasible": 28,
            "C4_selected": 14,
            "reference_evaluated": 14,
            "certified": 10,
            "accepted": 7,
            "global_best_update": 6
          },
          "TARGET_X": {
            "generated": 148,
            "constructed": 148,
            "cheap_valid": 148,
            "C2_selected": 14,
            "direction_evaluated": 14,
            "direction_feasible": 14,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 3.9438176000021485,
          "cheap_screen_time": 10.66959659999884,
          "direction_dp_time": 0.07413250000718108,
          "reference_scheduler_time": 11.568057400008911,
          "certifier_time": 0.5164360000017041,
          "repair_time": 29.872813800020594
        }
      },
      "reference_calls_at60": 53,
      "reference_status_counts_at60": {
        "DEADLOCK": 21,
        "FEASIBLE": 32
      },
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=74', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B5:POST', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=72', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=64', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=56', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:WELD',)\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=71', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=3', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:POST', 'R1:B10:WELD')\", \"R3 blocked by ('R2:B2:POST', 'R2:B2:WELD', 'R2:B3:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=81', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R3:B1:MOVE', 'R3:B1:SETUP', 'R3:B1:WELD')\", \"R2 blocked by ('R0:B5:WELD',)\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=85', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B9:POST', 'R1:B9:WELD')\", \"R2 blocked by ('R0:B5:WELD',)\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B7:POST', 'R0:B7:WELD', 'R0:B8:MOVE', 'R0:B8:POST', 'R0:B8:SETUP', 'R0:B8:WELD', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=13', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=37', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:MOVE', 'R0:B8:SETUP', 'R0:B8:WELD')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B7:POST', 'R0:B7:WELD', 'R0:B8:MOVE')\", \"R2 blocked by ('R0:B6:MOVE', 'R0:B6:SETUP', 'R0:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", \"R2 blocked by ('R3:B10:POST', 'R3:B10:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B5:MOVE', 'R0:B5:POST', 'R0:B5:SETUP', 'R0:B5:WELD', 'R0:B6:MOVE')\", \"R3 blocked by ('R2:B7:POST', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=51', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6183035714285714,
      "best_events": [
        [
          2.3540290000000823,
          2127.4304806137065
        ],
        [
          15.699448500001381,
          2029.3808433641825
        ],
        [
          18.878739000001588,
          1939.8812097042082
        ],
        [
          22.334025699999984,
          1918.1783755883685
        ],
        [
          25.978997200001686,
          1896.5225039046404
        ],
        [
          30.18268950000129,
          1869.6989931725245
        ],
        [
          47.6904426000001,
          1869.6989931725245
        ],
        [
          52.65453550000166,
          1848.9965105011386
        ]
      ],
      "iterations_total": 15,
      "accepted_moves_total": 9,
      "global_best_updates_total": 7,
      "reference_calls_total": 55,
      "reference_status_counts_total": {
        "DEADLOCK": 23,
        "FEASIBLE": 32
      },
      "elapsed_s": 64.31709670000055,
      "overshoot_s": 4.317096700000548,
      "status": "COMPLETED",
      "final_cmax": 1848.9965105011386,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 2,
      "instance_id": "data/ID_TEST/seed_0401115467::g19_w035",
      "workbook": "data/ID_TEST/seed_0401115467.xlsx",
      "sheet": "g19_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "D",
      "core_mask": 64,
      "first_certified_initial": {
        "elapsed": 0.817591099999845,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2127.4304806137065,
        "diagnostics": [
          "wait-for cycles=()",
          "R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')",
          "R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')",
          "baseline DEADLOCK",
          "recovery_rollouts=16",
          "rollout_budget=32",
          "max_discrepancies_used=19",
          "branch_points_considered=4",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2127.4304806137065,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          1878.6608470408607,
          1863.9312263659199,
          1150.4306162107239,
          1351.024079481795
        ],
        "process_imbalance": 728.2302308301369,
        "empty_travel_s": 470.9702928226164,
        "waiting_s": 140.73069377088314,
        "outer_cuts": [
          {
            "parent": "inst0000|group=801-FR322A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_LOWER",
            "rail": null,
            "t": 0.12425678017329203,
            "coordinate": [
              15.791299288295,
              5.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_OUTER_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0002:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0003:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0004:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "init_time_s": 1.8922521000004053,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 2029.3808433641825,
      "at60": {
        "cmax": 2029.3808433641825,
        "patterns": {
          "WHOLE": 34,
          "Y_SPLIT": 1
        },
        "process_loads": [
          1878.660847040861,
          1863.9312263659199,
          1150.4306162107239,
          1351.024079481795
        ],
        "process_imbalance": 728.2302308301371,
        "empty_travel_s": 400.9732830437725,
        "waiting_s": 13.610232867603372,
        "outer_cuts": [
          {
            "parent": "inst0000|group=801-FR322A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_LOWER",
            "rail": null,
            "t": 0.12425678017329203,
            "coordinate": [
              15.791299288295,
              5.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0002:Y_SPLIT:BY_OUTER_LOWER",
            "inst0000|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0000|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0001|group=312-GR2C|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0002|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0003|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0000:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0001:WHOLE",
            "inst0004|group=165-SR2A.dxf|row=0002:WHOLE",
            "inst0005|group=143-FR69B|row=0000:WHOLE",
            "inst0006|group=143-FR76A|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0000:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0001:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0002:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0003:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0004:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0005:WHOLE",
            "inst0007|group=143-FR76A.dxf|row=0006:WHOLE",
            "inst0008|group=804-FR313A|row=0000:WHOLE",
            "inst0009|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0013|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0014|group=143-FR76A|row=0000:WHOLE",
            "inst0015|group=888-BK304A|row=0000:WHOLE",
            "inst0016|group=143-GR14A|row=0000:WHOLE",
            "inst0017|group=143-FR76A|row=0000:WHOLE",
            "inst0018|group=165-SR2A|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0005|group=143-FR69B|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR314A.dxf|row=0001::whole",
              "inst0002|group=804-FR314A.dxf|row=0002::whole",
              "inst0008|group=804-FR313A|row=0000::whole",
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0004::whole",
              "inst0007|group=143-FR76A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0003::whole",
              "inst0013|group=804-FR315A.dxf|row=0000::whole"
            ],
            [
              "inst0000|group=801-FR322A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0004::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::1",
              "inst0004|group=165-SR2A.dxf|row=0001::whole",
              "inst0000|group=801-FR322A.dxf|row=0000::whole",
              "inst0004|group=165-SR2A.dxf|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0003::whole",
              "inst0004|group=165-SR2A.dxf|row=0002::whole",
              "inst0007|group=143-FR76A.dxf|row=0001::whole",
              "inst0013|group=804-FR315A.dxf|row=0002::whole",
              "inst0013|group=804-FR315A.dxf|row=0001::whole"
            ],
            [
              "inst0009|group=642-BL14A.dxf|row=0000::whole",
              "inst0001|group=312-GR2C|row=0000::whole",
              "inst0012|group=685-BK21A.dxf|row=0000::whole",
              "inst0014|group=143-FR76A|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0005::whole",
              "inst0007|group=143-FR76A.dxf|row=0006::whole"
            ],
            [
              "inst0015|group=888-BK304A|row=0000::whole",
              "inst0006|group=143-FR76A|row=0000::whole",
              "inst0000|group=801-FR322A.dxf|row=0002::0",
              "inst0003|group=685-BK21A.dxf|row=0000::whole",
              "inst0007|group=143-FR76A.dxf|row=0000::whole",
              "inst0017|group=143-FR76A|row=0000::whole",
              "inst0010|group=804-FR313A|row=0000::whole",
              "inst0018|group=165-SR2A|row=0000::whole",
              "inst0016|group=143-GR14A|row=0000::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 15,
        "accepted_moves": 2,
        "raw_attempts": 2880,
        "valid_candidates": 1753,
        "global_best_updates": 2,
        "c4_feasible": 2,
        "c4_improving": 1,
        "c4_material": 1,
        "proposal_improving": 1,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 1995,
            "constructed": 1546,
            "cheap_valid": 1205,
            "C2_selected": 36,
            "direction_evaluated": 36,
            "direction_feasible": 35,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 247,
            "constructed": 152,
            "cheap_valid": 152,
            "C2_selected": 12,
            "direction_evaluated": 12,
            "direction_feasible": 12,
            "C4_selected": 3,
            "reference_evaluated": 3,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_Y": {
            "generated": 482,
            "constructed": 240,
            "cheap_valid": 240,
            "C2_selected": 57,
            "direction_evaluated": 57,
            "direction_feasible": 57,
            "C4_selected": 19,
            "reference_evaluated": 19,
            "certified": 1,
            "accepted": 1,
            "global_best_update": 1
          },
          "TARGET_X": {
            "generated": 156,
            "constructed": 156,
            "cheap_valid": 156,
            "C2_selected": 15,
            "direction_evaluated": 15,
            "direction_feasible": 15,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.234510400014187,
          "cheap_screen_time": 11.703627099997902,
          "direction_dp_time": 0.07544250001046748,
          "reference_scheduler_time": 14.640283000006093,
          "certifier_time": 0.08649660000082804,
          "repair_time": 25.458359900001597
        }
      },
      "reference_calls_at60": 39,
      "reference_status_counts_at60": {
        "DEADLOCK": 33,
        "FEASIBLE": 6
      },
      "init_status_counts": {
        "DEADLOCK": 4,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=74', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 4
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B5:POST', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=2', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B1:POST', 'R0:B1:WELD', 'R0:B2:MOVE', 'R0:B2:SETUP', 'R0:B2:WELD')\", \"R2 blocked by ('R3:B1:MOVE', 'R3:B1:POST', 'R3:B1:SETUP', 'R3:B1:WELD', 'R3:B2:MOVE', 'R3:B2:POST', 'R3:B2:SETUP', 'R3:B2:WELD', 'R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE', 'R3:B4:POST', 'R3:B4:SETUP', 'R3:B4:WELD', 'R3:B5:MOVE', 'R3:B5:POST', 'R3:B5:SETUP', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:POST', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=29', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=72', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:WELD',)\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=65', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=73', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=78', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B9:POST', 'R1:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=4', 'rollout_budget=32', 'max_discrepancies_used=4', 'branch_points_considered=1', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B8:POST', 'R0:B8:WELD', 'R0:B9:MOVE')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=76', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R3:B2:POST', 'R3:B2:WELD', 'R3:B3:MOVE')\", \"R2 blocked by ('R0:B6:POST', 'R0:B6:WELD', 'R0:B7:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=91', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:POST', 'R1:B10:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=74', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", \"R1 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE')\", \"R3 blocked by ('R2:B4:MOVE', 'R2:B4:POST', 'R2:B4:SETUP', 'R2:B4:WELD', 'R2:B5:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=45', 'branch_points_considered=11', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:POST', 'R1:B10:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 10
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B9:POST', 'R1:B9:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=52', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R3:B3:MOVE', 'R3:B3:POST', 'R3:B3:SETUP', 'R3:B3:WELD', 'R3:B4:MOVE')\", \"R2 blocked by ('R3:B8:POST', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=79', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=2', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=1', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B6:MOVE',)\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=69', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", \"R3 blocked by ('R1:B2:MOVE', 'R1:B2:SETUP', 'R1:B2:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=66', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6086805555555556,
      "best_events": [
        [
          1.9949796000000788,
          2127.4304806137065
        ],
        [
          13.086213699998552,
          2029.3808433641825
        ],
        [
          58.23300459999882,
          2029.3808433641825
        ]
      ],
      "iterations_total": 16,
      "accepted_moves_total": 3,
      "global_best_updates_total": 2,
      "reference_calls_total": 43,
      "reference_status_counts_total": {
        "DEADLOCK": 33,
        "FEASIBLE": 10
      },
      "elapsed_s": 62.51072889999887,
      "overshoot_s": 2.510728899998867,
      "status": "COMPLETED",
      "final_cmax": 2029.3808433641825,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "A",
      "core_mask": 4,
      "first_certified_initial": {
        "elapsed": 0.6346049000003404,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 0.5312692999996216,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 20,
        "accepted_moves": 7,
        "raw_attempts": 3840,
        "valid_candidates": 2376,
        "global_best_updates": 0,
        "c4_feasible": 15,
        "c4_improving": 1,
        "c4_material": 1,
        "proposal_improving": 1,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2899,
            "constructed": 2130,
            "cheap_valid": 1603,
            "C2_selected": 117,
            "direction_evaluated": 117,
            "direction_feasible": 116,
            "C4_selected": 25,
            "reference_evaluated": 25,
            "certified": 1,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 456,
            "constructed": 291,
            "cheap_valid": 291,
            "C2_selected": 23,
            "direction_evaluated": 23,
            "direction_feasible": 23,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 4,
            "accepted": 2,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 0,
            "constructed": 0,
            "cheap_valid": 0,
            "C2_selected": 0,
            "direction_evaluated": 0,
            "direction_feasible": 0,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 485,
            "constructed": 482,
            "cheap_valid": 482,
            "C2_selected": 20,
            "direction_evaluated": 20,
            "direction_feasible": 20,
            "C4_selected": 11,
            "reference_evaluated": 11,
            "certified": 10,
            "accepted": 5,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.145474999981161,
          "cheap_screen_time": 13.266969999995126,
          "direction_dp_time": 0.1014171000024362,
          "reference_scheduler_time": 12.96645929999977,
          "certifier_time": 0.6413691000016115,
          "repair_time": 22.872713199978534
        }
      },
      "reference_calls_at60": 71,
      "reference_status_counts_at60": {
        "DEADLOCK": 27,
        "FEASIBLE": 44
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=10', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=37', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=15', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=16', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=47', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=36', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B5:POST', 'R3:B5:WELD', 'R3:B6:MOVE', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B8:POST', 'R3:B8:WELD', 'R3:B9:MOVE', 'R3:B9:POST', 'R3:B9:SETUP', 'R3:B9:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=12', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=10', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=25', 'branch_points_considered=12', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.61875,
      "best_events": [
        [
          0.635007200000473,
          1465.5689310673554
        ]
      ],
      "iterations_total": 21,
      "accepted_moves_total": 8,
      "global_best_updates_total": 0,
      "reference_calls_total": 74,
      "reference_status_counts_total": {
        "DEADLOCK": 27,
        "FEASIBLE": 47
      },
      "elapsed_s": 60.28284480000002,
      "overshoot_s": 0.28284480000002077,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261081,
      "group": "D",
      "core_mask": 16,
      "first_certified_initial": {
        "elapsed": 0.7744413999989774,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 0.6743426000011823,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471897,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685391,
        "empty_travel_s": 203.51676179259135,
        "waiting_s": 58.631527514830054,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 19,
        "accepted_moves": 11,
        "raw_attempts": 3648,
        "valid_candidates": 2128,
        "global_best_updates": 1,
        "c4_feasible": 18,
        "c4_improving": 2,
        "c4_material": 2,
        "proposal_improving": 3,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2679,
            "constructed": 1973,
            "cheap_valid": 1483,
            "C2_selected": 112,
            "direction_evaluated": 112,
            "direction_feasible": 112,
            "C4_selected": 23,
            "reference_evaluated": 23,
            "certified": 4,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 384,
            "constructed": 217,
            "cheap_valid": 217,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 4,
            "accepted": 4,
            "global_best_update": 1
          },
          "TARGET_Y": {
            "generated": 30,
            "constructed": 12,
            "cheap_valid": 12,
            "C2_selected": 2,
            "direction_evaluated": 2,
            "direction_feasible": 2,
            "C4_selected": 2,
            "reference_evaluated": 2,
            "certified": 2,
            "accepted": 2,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 555,
            "constructed": 416,
            "cheap_valid": 416,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 9,
            "reference_evaluated": 9,
            "certified": 8,
            "accepted": 5,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.348875700015924,
          "cheap_screen_time": 12.607805700004974,
          "direction_dp_time": 0.08434040000065579,
          "reference_scheduler_time": 10.284795899995515,
          "certifier_time": 0.687622100003864,
          "repair_time": 27.645844699991358
        }
      },
      "reference_calls_at60": 70,
      "reference_status_counts_at60": {
        "DEADLOCK": 23,
        "FEASIBLE": 47
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=10', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=21', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=16', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=15', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=10', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=10', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=3', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5833333333333334,
      "best_events": [
        [
          0.775177800000165,
          1465.5689310673554
        ],
        [
          53.49143109999932,
          1465.5689310673554
        ]
      ],
      "iterations_total": 20,
      "accepted_moves_total": 11,
      "global_best_updates_total": 1,
      "reference_calls_total": 72,
      "reference_status_counts_total": {
        "DEADLOCK": 25,
        "FEASIBLE": 47
      },
      "elapsed_s": 60.927880499999446,
      "overshoot_s": 0.9278804999994463,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "D",
      "core_mask": 1,
      "first_certified_initial": {
        "elapsed": 1.1616174999999203,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 1.0686654000001,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 19,
        "accepted_moves": 11,
        "raw_attempts": 3648,
        "valid_candidates": 2114,
        "global_best_updates": 0,
        "c4_feasible": 13,
        "c4_improving": 3,
        "c4_material": 1,
        "proposal_improving": 3,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2670,
            "constructed": 1978,
            "cheap_valid": 1496,
            "C2_selected": 105,
            "direction_evaluated": 105,
            "direction_feasible": 103,
            "C4_selected": 23,
            "reference_evaluated": 23,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 353,
            "constructed": 211,
            "cheap_valid": 211,
            "C2_selected": 23,
            "direction_evaluated": 23,
            "direction_feasible": 23,
            "C4_selected": 5,
            "reference_evaluated": 5,
            "certified": 3,
            "accepted": 2,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 69,
            "constructed": 29,
            "cheap_valid": 29,
            "C2_selected": 5,
            "direction_evaluated": 5,
            "direction_feasible": 5,
            "C4_selected": 3,
            "reference_evaluated": 3,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 556,
            "constructed": 378,
            "cheap_valid": 378,
            "C2_selected": 19,
            "direction_evaluated": 19,
            "direction_feasible": 19,
            "C4_selected": 7,
            "reference_evaluated": 7,
            "certified": 7,
            "accepted": 6,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.257013700018433,
          "cheap_screen_time": 11.862406600002942,
          "direction_dp_time": 0.08424690001083945,
          "reference_scheduler_time": 10.576968000003035,
          "certifier_time": 0.47989310000048135,
          "repair_time": 29.917408999968757
        }
      },
      "reference_calls_at60": 66,
      "reference_status_counts_at60": {
        "DEADLOCK": 28,
        "FEASIBLE": 38
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=11', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=13', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=11', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=25', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=2', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=1', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=15', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=61', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=50', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=7', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=41', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5794956140350878,
      "best_events": [
        [
          1.1623400999997102,
          1465.5689310673554
        ]
      ],
      "iterations_total": 20,
      "accepted_moves_total": 12,
      "global_best_updates_total": 0,
      "reference_calls_total": 70,
      "reference_status_counts_total": {
        "DEADLOCK": 29,
        "FEASIBLE": 41
      },
      "elapsed_s": 64.14781829999993,
      "overshoot_s": 4.147818299999926,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261082,
      "group": "A",
      "core_mask": 64,
      "first_certified_initial": {
        "elapsed": 0.7395532999998977,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 0.6229886000000988,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 20,
        "accepted_moves": 9,
        "raw_attempts": 3840,
        "valid_candidates": 2306,
        "global_best_updates": 0,
        "c4_feasible": 13,
        "c4_improving": 5,
        "c4_material": 4,
        "proposal_improving": 5,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 2956,
            "constructed": 2147,
            "cheap_valid": 1591,
            "C2_selected": 114,
            "direction_evaluated": 114,
            "direction_feasible": 113,
            "C4_selected": 25,
            "reference_evaluated": 25,
            "certified": 3,
            "accepted": 2,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 384,
            "constructed": 218,
            "cheap_valid": 218,
            "C2_selected": 26,
            "direction_evaluated": 26,
            "direction_feasible": 26,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 4,
            "accepted": 3,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 0,
            "constructed": 0,
            "cheap_valid": 0,
            "C2_selected": 0,
            "direction_evaluated": 0,
            "direction_feasible": 0,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 500,
            "constructed": 497,
            "cheap_valid": 497,
            "C2_selected": 20,
            "direction_evaluated": 20,
            "direction_feasible": 20,
            "C4_selected": 11,
            "reference_evaluated": 11,
            "certified": 6,
            "accepted": 4,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.101987500027462,
          "cheap_screen_time": 12.947165899997344,
          "direction_dp_time": 0.09209779998309386,
          "reference_scheduler_time": 14.344435999993948,
          "certifier_time": 0.4874290000043402,
          "repair_time": 25.16272709997611
        }
      },
      "reference_calls_at60": 66,
      "reference_status_counts_at60": {
        "DEADLOCK": 31,
        "FEASIBLE": 35
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B5:MOVE', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=30', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=34', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R3 blocked by ('R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=8', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=5', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=96', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=37', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=12', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B4:POST', 'R2:B4:WELD', 'R2:B5:MOVE', 'R2:B5:POST', 'R2:B5:SETUP', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=14', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=5', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B5:MOVE', 'R3:B5:SETUP', 'R3:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=48', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD', 'R2:B8:MOVE', 'R2:B8:POST', 'R2:B8:SETUP', 'R2:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=10', 'branch_points_considered=13', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6005208333333333,
      "best_events": [
        [
          0.7402856000007887,
          1465.5689310673554
        ]
      ],
      "iterations_total": 21,
      "accepted_moves_total": 9,
      "global_best_updates_total": 0,
      "reference_calls_total": 68,
      "reference_status_counts_total": {
        "DEADLOCK": 33,
        "FEASIBLE": 35
      },
      "elapsed_s": 63.382208200000605,
      "overshoot_s": 3.382208200000605,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "A",
      "core_mask": 4,
      "first_certified_initial": {
        "elapsed": 0.6046315999992657,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 0.501505599999291,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471897,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685391,
        "empty_travel_s": 203.6278738929466,
        "waiting_s": 62.34977188298376,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 26,
        "accepted_moves": 9,
        "raw_attempts": 4992,
        "valid_candidates": 3017,
        "global_best_updates": 1,
        "c4_feasible": 13,
        "c4_improving": 2,
        "c4_material": 2,
        "proposal_improving": 2,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 3974,
            "constructed": 2877,
            "cheap_valid": 2151,
            "C2_selected": 157,
            "direction_evaluated": 157,
            "direction_feasible": 155,
            "C4_selected": 33,
            "reference_evaluated": 33,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 336,
            "constructed": 188,
            "cheap_valid": 188,
            "C2_selected": 25,
            "direction_evaluated": 25,
            "direction_feasible": 25,
            "C4_selected": 4,
            "reference_evaluated": 4,
            "certified": 3,
            "accepted": 3,
            "global_best_update": 1
          },
          "TARGET_Y": {
            "generated": 0,
            "constructed": 0,
            "cheap_valid": 0,
            "C2_selected": 0,
            "direction_evaluated": 0,
            "direction_feasible": 0,
            "C4_selected": 0,
            "reference_evaluated": 0,
            "certified": 0,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 682,
            "constructed": 678,
            "cheap_valid": 678,
            "C2_selected": 26,
            "direction_evaluated": 26,
            "direction_feasible": 26,
            "C4_selected": 15,
            "reference_evaluated": 15,
            "certified": 10,
            "accepted": 6,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 4.667561300006128,
          "cheap_screen_time": 14.43714920000275,
          "direction_dp_time": 0.10988950000137265,
          "reference_scheduler_time": 12.587074999986726,
          "certifier_time": 0.47078090000650263,
          "repair_time": 22.01930419999553
        }
      },
      "reference_calls_at60": 80,
      "reference_status_counts_at60": {
        "DEADLOCK": 40,
        "FEASIBLE": 40
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B6:MOVE', 'R0:B6:SETUP', 'R0:B6:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=8', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=19', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=12', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 5
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:SETUP', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=21', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 3
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B6:MOVE', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=32', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:MOVE', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:MOVE', 'R3:B6:SETUP', 'R3:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=32', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=10', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=38', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=33', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R3 blocked by ('R2:B5:MOVE', 'R2:B5:SETUP', 'R2:B5:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=26', 'rollout_budget=32', 'max_discrepancies_used=48', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.6043669871794872,
      "best_events": [
        [
          0.6050312999996095,
          1465.5689310673554
        ],
        [
          15.883956300000136,
          1465.5689310673554
        ]
      ],
      "iterations_total": 27,
      "accepted_moves_total": 9,
      "global_best_updates_total": 1,
      "reference_calls_total": 82,
      "reference_status_counts_total": {
        "DEADLOCK": 42,
        "FEASIBLE": 40
      },
      "elapsed_s": 60.23845479999909,
      "overshoot_s": 0.23845479999909003,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    },
    {
      "instance": 3,
      "instance_id": "data/PPO_TRAIN/seed_0194123089::g17_w035",
      "workbook": "data/PPO_TRAIN/seed_0194123089.xlsx",
      "sheet": "g17_w035",
      "N": 35,
      "tier": "SMALL_30_39",
      "seed": 20261083,
      "group": "D",
      "core_mask": 16,
      "first_certified_initial": {
        "elapsed": 0.9625332999985403,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 1465.5689310673554,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=26",
          "rollout_budget=32",
          "max_discrepancies_used=31",
          "branch_points_considered=6",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "init_time_s": 0.8507048999999824,
      "init_strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "certified_at60": true,
      "cmax_at60": 1465.5689310673554,
      "at60": {
        "cmax": 1465.5689310673554,
        "patterns": {
          "WHOLE": 35
        },
        "process_loads": [
          1438.5649477157287,
          1358.1717260199034,
          1108.5502286471894,
          1283.8240514692409
        ],
        "process_imbalance": 330.0147190685393,
        "empty_travel_s": 206.11644467671402,
        "waiting_s": 75.23256055021886,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=801-FR322A|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0000:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0001:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0002:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0003:WHOLE",
            "inst0001|group=801-FR322A.dxf|row=0004:WHOLE",
            "inst0002|group=804-FR313A|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0000:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0001:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0002:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0003:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0004:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0005:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0006:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0007:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0008:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0009:WHOLE",
            "inst0003|group=165-SR1A.dxf|row=0010:WHOLE",
            "inst0004|group=801-FR322A|row=0000:WHOLE",
            "inst0005|group=642-BL14A.dxf|row=0000:WHOLE",
            "inst0006|group=804-FR315A|row=0000:WHOLE",
            "inst0007|group=145-LB17A|row=0000:WHOLE",
            "inst0008|group=888-BK304A|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0009|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0010|group=183-CM2B.dxf|row=0000:WHOLE",
            "inst0011|group=801-FR322A|row=0000:WHOLE",
            "inst0012|group=888-BK304A|row=0000:WHOLE",
            "inst0013|group=143-FR76A|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0000:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0001:WHOLE",
            "inst0014|group=804-FR314A.dxf|row=0002:WHOLE",
            "inst0015|group=143-FR76A|row=0000:WHOLE",
            "inst0016|group=143-FR69B|row=0000:WHOLE"
          ],
          "routes": [
            [
              "inst0004|group=801-FR322A|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0002::whole",
              "inst0003|group=165-SR1A.dxf|row=0003::whole",
              "inst0003|group=165-SR1A.dxf|row=0004::whole",
              "inst0003|group=165-SR1A.dxf|row=0000::whole",
              "inst0003|group=165-SR1A.dxf|row=0005::whole",
              "inst0003|group=165-SR1A.dxf|row=0006::whole",
              "inst0003|group=165-SR1A.dxf|row=0007::whole",
              "inst0003|group=165-SR1A.dxf|row=0001::whole",
              "inst0003|group=165-SR1A.dxf|row=0008::whole",
              "inst0003|group=165-SR1A.dxf|row=0009::whole",
              "inst0003|group=165-SR1A.dxf|row=0010::whole"
            ],
            [
              "inst0011|group=801-FR322A|row=0000::whole",
              "inst0014|group=804-FR314A.dxf|row=0001::whole",
              "inst0014|group=804-FR314A.dxf|row=0002::whole",
              "inst0014|group=804-FR314A.dxf|row=0000::whole",
              "inst0002|group=804-FR313A|row=0000::whole",
              "inst0012|group=888-BK304A|row=0000::whole",
              "inst0016|group=143-FR69B|row=0000::whole",
              "inst0000|group=801-FR322A|row=0000::whole"
            ],
            [
              "inst0009|group=804-FR314A.dxf|row=0000::whole",
              "inst0009|group=804-FR314A.dxf|row=0001::whole",
              "inst0009|group=804-FR314A.dxf|row=0002::whole",
              "inst0007|group=145-LB17A|row=0000::whole",
              "inst0008|group=888-BK304A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0003::whole",
              "inst0001|group=801-FR322A.dxf|row=0000::whole"
            ],
            [
              "inst0013|group=143-FR76A|row=0000::whole",
              "inst0010|group=183-CM2B.dxf|row=0000::whole",
              "inst0006|group=804-FR315A|row=0000::whole",
              "inst0005|group=642-BL14A.dxf|row=0000::whole",
              "inst0015|group=143-FR76A|row=0000::whole",
              "inst0001|group=801-FR322A.dxf|row=0001::whole",
              "inst0001|group=801-FR322A.dxf|row=0004::whole",
              "inst0001|group=801-FR322A.dxf|row=0002::whole"
            ]
          ]
        }
      },
      "checkpoint_errors": [],
      "counters_at60": {
        "iterations": 28,
        "accepted_moves": 13,
        "raw_attempts": 5376,
        "valid_candidates": 3140,
        "global_best_updates": 0,
        "c4_feasible": 18,
        "c4_improving": 1,
        "c4_material": 1,
        "proposal_improving": 1,
        "decision_family_funnel": {
          "STRUCTURAL": {
            "generated": 4204,
            "constructed": 3063,
            "cheap_valid": 2291,
            "C2_selected": 163,
            "direction_evaluated": 163,
            "direction_feasible": 162,
            "C4_selected": 35,
            "reference_evaluated": 35,
            "certified": 1,
            "accepted": 0,
            "global_best_update": 0
          },
          "TARGET_WHOLE": {
            "generated": 456,
            "constructed": 225,
            "cheap_valid": 225,
            "C2_selected": 24,
            "direction_evaluated": 24,
            "direction_feasible": 24,
            "C4_selected": 6,
            "reference_evaluated": 6,
            "certified": 6,
            "accepted": 6,
            "global_best_update": 0
          },
          "TARGET_Y": {
            "generated": 134,
            "constructed": 45,
            "cheap_valid": 45,
            "C2_selected": 9,
            "direction_evaluated": 9,
            "direction_feasible": 9,
            "C4_selected": 7,
            "reference_evaluated": 7,
            "certified": 7,
            "accepted": 7,
            "global_best_update": 0
          },
          "TARGET_X": {
            "generated": 582,
            "constructed": 579,
            "cheap_valid": 579,
            "C2_selected": 28,
            "direction_evaluated": 28,
            "direction_feasible": 28,
            "C4_selected": 8,
            "reference_evaluated": 8,
            "certified": 4,
            "accepted": 0,
            "global_best_update": 0
          }
        },
        "timing_s": {
          "candidate_generation_time": 5.302232099997127,
          "cheap_screen_time": 15.491493499997887,
          "direction_dp_time": 0.10357179998391075,
          "reference_scheduler_time": 14.79070979998869,
          "certifier_time": 0.6001179000049888,
          "repair_time": 20.852889600000708
        }
      },
      "reference_calls_at60": 94,
      "reference_status_counts_at60": {
        "DEADLOCK": 39,
        "FEASIBLE": 55
      },
      "init_status_counts": {
        "DEADLOCK": 1,
        "FEASIBLE": 1
      },
      "rejection_diagnostics_at60": [
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B0:POST', 'R1:B0:WELD', 'R1:B10:MOVE', 'R1:B10:POST', 'R1:B10:SETUP', 'R1:B10:WELD', 'R1:B1:MOVE', 'R1:B1:POST', 'R1:B1:SETUP', 'R1:B1:WELD', 'R1:B2:MOVE', 'R1:B2:POST', 'R1:B2:SETUP', 'R1:B2:WELD', 'R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD', 'R1:B9:MOVE', 'R1:B9:POST', 'R1:B9:SETUP', 'R1:B9:WELD')\", \"R3 blocked by ('R2:B1:MOVE', 'R2:B1:POST', 'R2:B1:SETUP', 'R2:B1:WELD', 'R2:B2:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=1', 'branch_points_considered=47', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:POST', 'R2:B7:SETUP', 'R2:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=9', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B9:MOVE', 'R0:B9:POST', 'R0:B9:SETUP', 'R0:B9:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=18', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=20', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B3:MOVE', 'R1:B3:POST', 'R1:B3:SETUP', 'R1:B3:WELD', 'R1:B4:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=21', 'branch_points_considered=11', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R2 blocked by ('R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD', 'R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=2', 'branch_points_considered=7', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=28', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=35', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R3 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=14', 'rollout_budget=32', 'max_discrepancies_used=17', 'branch_points_considered=3', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=36', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B5:POST', 'R3:B5:WELD', 'R3:B6:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=22', 'rollout_budget=32', 'max_discrepancies_used=40', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=20', 'rollout_budget=32', 'max_discrepancies_used=11', 'branch_points_considered=5', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=28', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B11:MOVE', 'R0:B11:POST', 'R0:B11:SETUP', 'R0:B11:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE')\", 'baseline DEADLOCK', 'recovery_rollouts=16', 'rollout_budget=32', 'max_discrepancies_used=31', 'branch_points_considered=4', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=24', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 2
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B8:MOVE', 'R1:B8:POST', 'R1:B8:SETUP', 'R1:B8:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=24', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=26', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=22', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=24', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=6', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R1 blocked by ('R0:B10:MOVE', 'R0:B10:POST', 'R0:B10:SETUP', 'R0:B10:WELD')\", \"R2 blocked by ('R3:B6:POST', 'R3:B6:WELD', 'R3:B7:MOVE', 'R3:B7:POST', 'R3:B7:SETUP', 'R3:B7:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=23', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=18', 'branch_points_considered=11', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=29', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B4:MOVE', 'R1:B4:POST', 'R1:B4:SETUP', 'R1:B4:WELD', 'R1:B5:MOVE', 'R1:B5:POST', 'R1:B5:SETUP', 'R1:B5:WELD', 'R1:B6:MOVE')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=32', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=15', 'recovery_exhausted=True', 'frontier_exhausted=False']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B7:MOVE', 'R1:B7:POST', 'R1:B7:SETUP', 'R1:B7:WELD')\", \"R2 blocked by ('R3:B7:POST', 'R3:B7:WELD', 'R3:B8:MOVE', 'R3:B8:POST', 'R3:B8:SETUP', 'R3:B8:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=27', 'branch_points_considered=8', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        },
        {
          "status": "DEADLOCK",
          "diagnostic": "['wait-for cycles=()', \"R0 blocked by ('R1:B6:MOVE', 'R1:B6:POST', 'R1:B6:SETUP', 'R1:B6:WELD', 'R1:B7:MOVE')\", \"R3 blocked by ('R2:B5:POST', 'R2:B5:WELD', 'R2:B6:MOVE', 'R2:B6:SETUP', 'R2:B6:WELD')\", 'baseline DEADLOCK', 'recovery_rollouts=30', 'rollout_budget=32', 'max_discrepancies_used=14', 'branch_points_considered=9', 'recovery_exhausted=False', 'frontier_exhausted=True']",
          "count": 1
        }
      ],
      "candidate_valid_rate_at60": 0.5840773809523809,
      "best_events": [
        [
          0.9629398999986734,
          1465.5689310673554
        ]
      ],
      "iterations_total": 29,
      "accepted_moves_total": 13,
      "global_best_updates_total": 0,
      "reference_calls_total": 98,
      "reference_status_counts_total": {
        "DEADLOCK": 40,
        "FEASIBLE": 58
      },
      "elapsed_s": 61.522742399998606,
      "overshoot_s": 1.5227423999986058,
      "status": "COMPLETED",
      "final_cmax": 1465.5689310673554,
      "final_certified": true,
      "final_errors": [],
      "max_kdp": 8,
      "max_reference_per_iteration": 4
    }
  ],
  "i3_diagnostics": [
    {
      "kind": "INITIALIZATION_ONLY",
      "group": "A",
      "seed": 20261081,
      "N": 26,
      "first_certified_initial": {
        "elapsed": 0.3234695000010106,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 2118.4033350744635,
        "diagnostics": [
          "wait-for cycles=()",
          "R0 blocked by ('R2:B6:POST', 'R2:B6:WELD', 'R2:B7:MOVE', 'R2:B7:SETUP', 'R2:B7:WELD')",
          "R2 blocked by ('R1:B6:POST', 'R1:B6:WELD')",
          "R3 blocked by ('R2:B7:POST', 'R2:B7:WELD', 'R2:B8:MOVE')",
          "baseline DEADLOCK",
          "recovery_rollouts=30",
          "rollout_budget=32",
          "max_discrepancies_used=35",
          "branch_points_considered=8",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 2118.4033350744635,
        "patterns": {
          "WHOLE": 20,
          "Y_SPLIT": 6
        },
        "process_loads": [
          1687.3334091422164,
          1610.587679999052,
          1798.0239975251727,
          1675.6992975349515
        ],
        "process_imbalance": 187.43631752612077,
        "empty_travel_s": 257.62774599684826,
        "waiting_s": 270.1385406523143,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=145-LB17A|row=0000:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0000:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0001:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0004:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0005:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0006:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0007:Y_SPLIT:BY_UPPER",
            "inst0002|group=801-FR322A|row=0000:WHOLE",
            "inst0003|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0005|group=145-LB17A|row=0000:WHOLE",
            "inst0006|group=804-FR313A|row=0000:WHOLE",
            "inst0007|group=642-BL14A|row=0000:WHOLE",
            "inst0008|group=143-FR65A|row=0000:WHOLE",
            "inst0009|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0011|group=804-FR313A|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0001:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0002:WHOLE",
            "inst0013|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0014|group=143-GR14A|row=0000:Y_SPLIT:BY_UPPER"
          ],
          "routes": [
            [
              "inst0012|group=888-BK304A.dxf|row=0002::whole",
              "inst0014|group=143-GR14A|row=0000::0",
              "inst0012|group=888-BK304A.dxf|row=0000::whole",
              "inst0012|group=888-BK304A.dxf|row=0001::whole",
              "inst0008|group=143-FR65A|row=0000::whole",
              "inst0007|group=642-BL14A|row=0000::whole",
              "inst0002|group=801-FR322A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::0"
            ],
            [
              "inst0006|group=804-FR313A|row=0000::whole",
              "inst0003|group=804-FR313A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::0",
              "inst0001|group=804-TB310A.dxf|row=0005::0",
              "inst0001|group=804-TB310A.dxf|row=0003::0",
              "inst0010|group=804-FR313A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0002::0"
            ],
            [
              "inst0014|group=143-GR14A|row=0000::1",
              "inst0004|group=804-FR315A.dxf|row=0000::whole",
              "inst0005|group=145-LB17A|row=0000::whole",
              "inst0004|group=804-FR315A.dxf|row=0001::whole",
              "inst0004|group=804-FR315A.dxf|row=0002::whole",
              "inst0011|group=804-FR313A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::1",
              "inst0001|group=804-TB310A.dxf|row=0001::whole",
              "inst0001|group=804-TB310A.dxf|row=0002::1"
            ],
            [
              "inst0000|group=145-LB17A|row=0000::whole",
              "inst0013|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::1",
              "inst0009|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0006::whole",
              "inst0001|group=804-TB310A.dxf|row=0005::1",
              "inst0001|group=804-TB310A.dxf|row=0004::whole",
              "inst0001|group=804-TB310A.dxf|row=0003::1"
            ]
          ]
        }
      },
      "init_time_s": 1.2783127999991848,
      "strategy": "RAIL_MONOTONE_BALANCED_BOOTSTRAP",
      "reference_calls": 5,
      "certified": true
    },
    {
      "kind": "INITIALIZATION_ONLY",
      "group": "D",
      "seed": 20261081,
      "N": 26,
      "first_certified_initial": {
        "elapsed": 1.3841563000005408,
        "initialization": true,
        "status": "FEASIBLE",
        "cmax": 3719.524738948981,
        "diagnostics": [
          "wait-for cycles=()",
          "R2 blocked by ('R0:B13:POST', 'R0:B13:WELD', 'R0:B14:MOVE', 'R0:B14:SETUP', 'R0:B14:WELD')",
          "baseline DEADLOCK",
          "recovery_rollouts=1",
          "rollout_budget=32",
          "max_discrepancies_used=1",
          "branch_points_considered=1",
          "recovery_exhausted=False",
          "frontier_exhausted=True"
        ]
      },
      "chosen_initial": {
        "cmax": 3719.524738948981,
        "patterns": {
          "WHOLE": 20,
          "Y_SPLIT": 6
        },
        "process_loads": [
          3440.515205250109,
          0,
          3331.1291789512848,
          0
        ],
        "process_imbalance": 3440.515205250109,
        "empty_travel_s": 265.6534684333753,
        "waiting_s": 139.67713332195217,
        "outer_cuts": [
          {
            "parent": "inst0001|group=804-TB310A.dxf|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.4478702720529121,
            "coordinate": [
              8.223950945806,
              6.5
            ]
          },
          {
            "parent": "inst0001|group=804-TB310A.dxf|row=0002",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.46193765397840597,
            "coordinate": [
              9.793950945806,
              6.5
            ]
          },
          {
            "parent": "inst0001|group=804-TB310A.dxf|row=0003",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.448724334356598,
            "coordinate": [
              10.519950945806,
              6.5
            ]
          },
          {
            "parent": "inst0001|group=804-TB310A.dxf|row=0005",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.46193765397840597,
            "coordinate": [
              11.245950945806,
              6.5
            ]
          },
          {
            "parent": "inst0001|group=804-TB310A.dxf|row=0007",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.448724334356598,
            "coordinate": [
              12.815950945806,
              6.5
            ]
          },
          {
            "parent": "inst0014|group=143-GR14A|row=0000",
            "kind": "Y_SPLIT",
            "point_id": "BY_OUTER_UPPER",
            "rail": null,
            "t": 0.3554675438474192,
            "coordinate": [
              1.428552415587,
              6.5
            ]
          }
        ],
        "solution": {
          "pattern_ids": [
            "inst0000|group=145-LB17A|row=0000:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0000:Y_SPLIT:BY_OUTER_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0001:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0002:Y_SPLIT:BY_OUTER_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0003:Y_SPLIT:BY_OUTER_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0004:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0005:Y_SPLIT:BY_OUTER_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0006:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0007:Y_SPLIT:BY_OUTER_UPPER",
            "inst0002|group=801-FR322A|row=0000:WHOLE",
            "inst0003|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0005|group=145-LB17A|row=0000:WHOLE",
            "inst0006|group=804-FR313A|row=0000:WHOLE",
            "inst0007|group=642-BL14A|row=0000:WHOLE",
            "inst0008|group=143-FR65A|row=0000:WHOLE",
            "inst0009|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0011|group=804-FR313A|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0001:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0002:WHOLE",
            "inst0013|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0014|group=143-GR14A|row=0000:Y_SPLIT:BY_OUTER_UPPER"
          ],
          "routes": [
            [
              "inst0014|group=143-GR14A|row=0000::0",
              "inst0012|group=888-BK304A.dxf|row=0000::whole",
              "inst0012|group=888-BK304A.dxf|row=0002::whole",
              "inst0012|group=888-BK304A.dxf|row=0001::whole",
              "inst0008|group=143-FR65A|row=0000::whole",
              "inst0007|group=642-BL14A|row=0000::whole",
              "inst0002|group=801-FR322A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::0",
              "inst0001|group=804-TB310A.dxf|row=0002::0",
              "inst0010|group=804-FR313A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0003::0",
              "inst0001|group=804-TB310A.dxf|row=0004::whole",
              "inst0001|group=804-TB310A.dxf|row=0005::0",
              "inst0001|group=804-TB310A.dxf|row=0006::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::0",
              "inst0003|group=804-FR313A.dxf|row=0000::whole",
              "inst0006|group=804-FR313A|row=0000::whole"
            ],
            [],
            [
              "inst0014|group=143-GR14A|row=0000::1",
              "inst0004|group=804-FR315A.dxf|row=0000::whole",
              "inst0005|group=145-LB17A|row=0000::whole",
              "inst0004|group=804-FR315A.dxf|row=0001::whole",
              "inst0004|group=804-FR315A.dxf|row=0002::whole",
              "inst0011|group=804-FR313A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::1",
              "inst0001|group=804-TB310A.dxf|row=0001::whole",
              "inst0001|group=804-TB310A.dxf|row=0002::1",
              "inst0001|group=804-TB310A.dxf|row=0003::1",
              "inst0001|group=804-TB310A.dxf|row=0005::1",
              "inst0009|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::1",
              "inst0013|group=685-BK21A.dxf|row=0000::whole",
              "inst0000|group=145-LB17A|row=0000::whole"
            ],
            []
          ]
        }
      },
      "init_time_s": 1.2923656000002666,
      "strategy": "RAIL_SERIAL_BOOTSTRAP",
      "reference_calls": 8,
      "certified": true
    },
    {
      "kind": "A_INITIAL_REBUILT_IN_D",
      "missing_patterns": [],
      "certified": true,
      "status": "FEASIBLE",
      "certification_errors": [],
      "reference_calls": 1,
      "details": {
        "cmax": 2118.4033350744635,
        "patterns": {
          "WHOLE": 20,
          "Y_SPLIT": 6
        },
        "process_loads": [
          1687.3334091422164,
          1610.587679999052,
          1798.0239975251727,
          1675.6992975349515
        ],
        "process_imbalance": 187.43631752612077,
        "empty_travel_s": 257.62774599684826,
        "waiting_s": 270.1385406523143,
        "outer_cuts": [],
        "solution": {
          "pattern_ids": [
            "inst0000|group=145-LB17A|row=0000:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0000:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0001:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0002:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0003:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0004:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0005:Y_SPLIT:BY_UPPER",
            "inst0001|group=804-TB310A.dxf|row=0006:WHOLE",
            "inst0001|group=804-TB310A.dxf|row=0007:Y_SPLIT:BY_UPPER",
            "inst0002|group=801-FR322A|row=0000:WHOLE",
            "inst0003|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0000:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0001:WHOLE",
            "inst0004|group=804-FR315A.dxf|row=0002:WHOLE",
            "inst0005|group=145-LB17A|row=0000:WHOLE",
            "inst0006|group=804-FR313A|row=0000:WHOLE",
            "inst0007|group=642-BL14A|row=0000:WHOLE",
            "inst0008|group=143-FR65A|row=0000:WHOLE",
            "inst0009|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0010|group=804-FR313A.dxf|row=0000:WHOLE",
            "inst0011|group=804-FR313A|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0000:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0001:WHOLE",
            "inst0012|group=888-BK304A.dxf|row=0002:WHOLE",
            "inst0013|group=685-BK21A.dxf|row=0000:WHOLE",
            "inst0014|group=143-GR14A|row=0000:Y_SPLIT:BY_UPPER"
          ],
          "routes": [
            [
              "inst0012|group=888-BK304A.dxf|row=0002::whole",
              "inst0014|group=143-GR14A|row=0000::0",
              "inst0012|group=888-BK304A.dxf|row=0000::whole",
              "inst0012|group=888-BK304A.dxf|row=0001::whole",
              "inst0008|group=143-FR65A|row=0000::whole",
              "inst0007|group=642-BL14A|row=0000::whole",
              "inst0002|group=801-FR322A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::0"
            ],
            [
              "inst0006|group=804-FR313A|row=0000::whole",
              "inst0003|group=804-FR313A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::0",
              "inst0001|group=804-TB310A.dxf|row=0005::0",
              "inst0001|group=804-TB310A.dxf|row=0003::0",
              "inst0010|group=804-FR313A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0002::0"
            ],
            [
              "inst0014|group=143-GR14A|row=0000::1",
              "inst0004|group=804-FR315A.dxf|row=0000::whole",
              "inst0005|group=145-LB17A|row=0000::whole",
              "inst0004|group=804-FR315A.dxf|row=0001::whole",
              "inst0004|group=804-FR315A.dxf|row=0002::whole",
              "inst0011|group=804-FR313A|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0000::1",
              "inst0001|group=804-TB310A.dxf|row=0001::whole",
              "inst0001|group=804-TB310A.dxf|row=0002::1"
            ],
            [
              "inst0000|group=145-LB17A|row=0000::whole",
              "inst0013|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0007::1",
              "inst0009|group=685-BK21A.dxf|row=0000::whole",
              "inst0001|group=804-TB310A.dxf|row=0006::whole",
              "inst0001|group=804-TB310A.dxf|row=0005::1",
              "inst0001|group=804-TB310A.dxf|row=0004::whole",
              "inst0001|group=804-TB310A.dxf|row=0003::1"
            ]
          ]
        }
      }
    }
  ],
  "selection_rule": "META actual_weld_count equals actual rows; VALID unique consumed-development only; closest N to35; maximum legal X patterns, minimum X+Y patterns, median X patterns; three distinct workbooks; ties path/sheet; no solver/Cmax fields",
  "tests": "D:\\pybullet_test\\.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider，在DRL目录执行：303 passed in 37.34s；新增4项必要回归全部通过",
  "static_catalog": [
    {
      "instance": 1,
      "tier": "SMALL_30_39",
      "group": "A",
      "N": 35,
      "whole": 33,
      "mandatory_y": 2,
      "optional_y": 0,
      "y_patterns": 6,
      "x_parents": 23,
      "x_patterns": 31,
      "no_split": 10,
      "removed_short_y": 1,
      "removed_short_x": 0,
      "x_up": 9.117849992285,
      "x_low": 9.836686700960001
    },
    {
      "instance": 1,
      "tier": "SMALL_30_39",
      "group": "D",
      "N": 35,
      "whole": 34,
      "mandatory_y": 1,
      "optional_y": 3,
      "y_patterns": 11,
      "x_parents": 24,
      "x_patterns": 52,
      "no_split": 10,
      "removed_short_y": 1,
      "removed_short_x": 1,
      "x_up": 9.836686700960001,
      "x_low": 9.836686700960001
    },
    {
      "instance": 2,
      "tier": "SMALL_30_39",
      "group": "A",
      "N": 35,
      "whole": 30,
      "mandatory_y": 5,
      "optional_y": 1,
      "y_patterns": 12,
      "x_parents": 24,
      "x_patterns": 29,
      "no_split": 6,
      "removed_short_y": 4,
      "removed_short_x": 0,
      "x_up": 12.29372219962,
      "x_low": 12.223880778497
    },
    {
      "instance": 2,
      "tier": "SMALL_30_39",
      "group": "D",
      "N": 35,
      "whole": 34,
      "mandatory_y": 1,
      "optional_y": 7,
      "y_patterns": 21,
      "x_parents": 24,
      "x_patterns": 34,
      "no_split": 6,
      "removed_short_y": 6,
      "removed_short_x": 0,
      "x_up": 12.29372219962,
      "x_low": 12.223880778497
    },
    {
      "instance": 3,
      "tier": "SMALL_30_39",
      "group": "A",
      "N": 35,
      "whole": 35,
      "mandatory_y": 0,
      "optional_y": 0,
      "y_patterns": 0,
      "x_parents": 22,
      "x_patterns": 22,
      "no_split": 13,
      "removed_short_y": 1,
      "removed_short_x": 4,
      "x_up": 6.5968820423225,
      "x_low": 11.297176045894
    },
    {
      "instance": 3,
      "tier": "SMALL_30_39",
      "group": "D",
      "N": 35,
      "whole": 35,
      "mandatory_y": 0,
      "optional_y": 1,
      "y_patterns": 1,
      "x_parents": 22,
      "x_patterns": 26,
      "no_split": 13,
      "removed_short_y": 1,
      "removed_short_x": 4,
      "x_up": 6.5968820423225,
      "x_low": 11.297176045894
    }
  ],
  "conclusions": "### 直接回答：SMALL_30_39结果与定位\n\n**结论属于情况C：没有在所有30多条实例上普遍恶化，但X丰富的S1仍有严重、依赖seed的退化。** 0.5仍保留为指定任务层配置，不作整体优越性结论，不因单个异常另设规模参数。以下只计算N35的9个配对，与历史N26完全分开。\n\n1. **找到了真实N35实例，无需生成或删焊缝。** 38个允许读取的development workbook均检查了META和实际row count，144个有效N30–39 sheet来自31个workbook，N35候选12个。固定选择3个不同workbook的N35，分别代表X-rich、ordinary、split-sparse；没有访问V2_VALIDATION/ORACLE或ID_TEST_SEALED进行选样或模型选择。旧物理目录名不改变角色表中的development身份。\n\n2. **按用户给出的应用规模，35比26更符合30–39定义。** 三例总焊缝长度为81.40/48.00/37.14 m，bbox面积覆盖比约0.898/0.790/0.815，网格占据率约0.938/0.813/0.813；几何和工作量不同。它们不是只改变N的控制实验，不能据此证明I3的异常由N=26导致。N35候选中还有Y非常密集的sheet（53个Y pattern），本轮固定三种选样未包含该几何极端；不能外推为所有35条实例均已覆盖。所有样本仍在20×12 m原平台规则内，未做设备IK验证。\n\n3. **18个真实60秒run全部执行并取得独立认证解。** 9个配对的结果、三seed相对差如下；负值更好，所有异常seed保留。\n\n| N35例 | A mean Cmax@60 s | D mean Cmax@60 s | seed81差% | seed82差% | seed83差% |\n| --- | --- | --- | --- | --- | --- |\n| 1 | 3452.8080 | 4011.8406 | 55.1692 | 6.7684 | -7.6275 |\n| 2 | 2702.2215 | 1916.7429 | -30.2472 | -31.5711 | -25.4191 |\n| 3 | 1465.5689 | 1465.5689 | 0.0000 | 0.0000 | 0.0000 |\n\nALL mean Cmax@60为A=2540.199 s、D=2464.717 s，平均绝对差−75.482 s（均值之比−2.97%）。配对绝对差中位数0 s；配对相对差均值−3.66%、中位数0%；4改善、3持平、2恶化。均值之比、配对相对差均值和中位数是不同统计量，不互换。\n\n4. **整体相近并略有均值改善，局部严重恶化仍存在。** S1两次恶化+55.17%/+6.77%，第三次改善−7.63%；S2三次改善−30.25%/−31.57%/−25.42%；S3三次持平。S1 mean Cmax上升559.03 s，S2下降785.48 s，S3不变。不能用ALL均值或中位数掩盖S1，也不能把它概括成全部SMALL都变差。当前三个样本支持几何/轨迹相关问题，尚不足以识别一个适用于所有实例的几何因果判据。\n\n5. **差异发生阶段已经分开。** 各run首个认证初始化解与最终选中的初始化解Cmax恰好相同，但两个字段均独立保留。下面恒等分解为：最终差 = 初始化差 + A搜索降低 − D搜索降低。\n\n| 例 | A init Cmax | D init Cmax | init差D-A | A搜索降低s | D搜索降低s | 最终差D-A |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | 5077.2943 | 4956.8901 | -120.4043 | 1624.4863 | 945.0494 | 559.0326 |\n| 2 | 2780.7783 | 2127.4305 | -653.3479 | 78.5568 | 210.6875 | -785.4786 |\n| 3 | 1465.5689 | 1465.5689 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |\n\nS1的A/D均退回RAIL_SERIAL_BOOTSTRAP，初始只有R0/R2承担任务，D初始反而低120.40 s；其最终变差来自后续ALNS改善不足，不能归咎为“D初始Cmax更差”。S1每次仍有15–24个完整迭代，不是零迭代；前两seed A/D完成轮数22/22、24/21，差距主要不在迭代数量。S2则主要由初始化选择从A的X_ORDER_AWARE变为D的RAIL_MONOTONE_BALANCED_BOOTSTRAP，初始下降653.35 s，再叠加约132.13 s的相对搜索收益；A/D平均完整迭代6/15，也有明显搜索效率差异。S3六次Cmax都等于初始化：A seeds81/82、D seeds82/83的global-best updates为0；另外两次虽有1次次级目标update，Cmax仍未下降。因此S3主目标表现主要依赖初始化，不能写成搜索能力提升。\n\n6. **严重负载失衡仍出现，但与Cmax不是一一对应。** S1 D的前两seed负载差4580.26/3036.53 s，A为1305.51/1753.49 s，与Cmax恶化同向；最差D的R1负载仍为0。第三seed D负载差1624.15略高于A的1519.36 s，Cmax却改善7.63%，说明行走/等待/排程同样重要。S2负载差均值2557.24→508.88 s，与其改善一致；S3均约330.01 s不变。历史I3极端失衡仍保留，且本轮定向初始化重放已经证实其来源，详见下节。\n\n7. **新外侧X/Y切点被实际采用，最终分配确实改变。** 九个D的@60解共采用10个Y外点（9个BY_OUTER_UPPER、1个BY_OUTER_LOWER）和2个X外点（BX_OUTER_UPPER/LOWER各1）；X外点均出现在S1 seed83的改善运行。S1三个D均采用外侧Y点；S2三个D均采用外侧Y点；S3始终全WHOLE。跨三seed统计，S1最终WHOLE/Y/X从A的93/6/6变为D的98/4/3，S2从87/18/0变为95/10/0，S3两组均105/0/0。这些是各配置三个独立解的次数合计，不是单个解的父焊缝数。不能据S1退化断言外点本身有害，改善seed也使用了它们。\n\n新旧静态目录如下，零reference调用；S1还改变了上轨中心，S2改变了mandatory Y资格。因此A/D可行域不同，不是纯“切点数增加”的单变量算法竞赛。\n\n| 例 | 配置 | WHOLE资格 | mandatory Y | optional Y | legal Y | legal X | x_up | x_low |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| 1 | A | 33 | 2 | 0 | 6 | 31 | 9.1178 | 9.8367 |\n| 1 | D | 34 | 1 | 3 | 11 | 52 | 9.8367 | 9.8367 |\n| 2 | A | 30 | 5 | 1 | 12 | 29 | 12.2937 | 12.2239 |\n| 2 | D | 34 | 1 | 7 | 21 | 34 | 12.2937 | 12.2239 |\n| 3 | A | 35 | 0 | 0 | 0 | 22 | 6.5969 | 11.2972 |\n| 3 | D | 35 | 0 | 1 | 1 | 26 | 6.5969 | 11.2972 |\n\n8. **更多合法候选没有保证有效搜索质量，S1问题在direction后/reference阶段已显现。** 下表是60秒内完成迭代的原生漏斗总计；C4 material只表示相对当时SA当前状态≥0.5%的认证改善，不是相对global-best，也不是对整个M192池的oracle capture。\n\n| 例 | 配置 | iter≤60均值 | cheap有效率均值% | C2选中 | direction feasible | C4选中 | C4认证 | C4认证率% | C4 material |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n| 1 | A | 21.6667 | 55.6793 | 520 | 467 | 130 | 34 | 26.1538 | 21 |\n| 1 | D | 19.3333 | 57.6880 | 464 | 435 | 116 | 13 | 11.2069 | 7 |\n| 2 | A | 6 | 60.4406 | 144 | 102 | 36 | 6 | 16.6667 | 6 |\n| 2 | D | 15 | 61.8088 | 360 | 356 | 90 | 36 | 40.0000 | 17 |\n| 3 | A | 22 | 60.7879 | 528 | 524 | 132 | 41 | 31.0606 | 7 |\n| 3 | D | 22 | 58.2302 | 528 | 525 | 132 | 49 | 37.1212 | 4 |\n\nS1 D的cheap有效率55.68%→57.69%，direction-feasible比例467/520→435/464（约89.81%→93.75%），但C4认证率34/130→13/116（26.15%→11.21%），material候选21→7，搜索改善明显变弱。既有拒绝诊断包含机器人被相邻MOVE/WELD/POST/WAIT阻塞及B32 recovery后仍DEADLOCK；这不是物理不可行证明。S2 D的direction-feasible和C4认证比例均改善，S3认证候选更多仍没有降低global-best Cmax。**本轮没有额外评价未选候选，因此尚不能区分“池里没有好候选”与“C2/C4漏掉了好候选”，不能伪造Top8/Top2 capture结论。** S1少量迭代差不足以解释前两seed的差距，主要证据是实际认证候选和改善候选不足；更深原因需固定状态诊断，不能直接归咎SA或scheduler bug。\n\n9. **认证与数值错误为0；候选失败分别记录，不当作run失败。** 所有18个@60解及final解独立认证，checkpoint/final认证错误列表为空。以下reference计数包含初始化及direction refinement，按实际完成时间≤60统计，与base C4候选计数不是同一分母。\n\n| 例 | 配置 | ref≤60 | FEASIBLE | DEADLOCK | INFEASIBLE | NUMERIC |\n| --- | --- | --- | --- | --- | --- | --- |\n| 1 | A | 224 | 86 | 126 | 12 | 0 |\n| 1 | D | 166 | 39 | 123 | 4 | 0 |\n| 2 | A | 69 | 23 | 46 | 0 | 0 |\n| 2 | D | 155 | 85 | 70 | 0 | 0 |\n| 3 | A | 217 | 119 | 98 | 0 | 0 |\n| 3 | D | 230 | 140 | 90 | 0 | 0 |\n\nA/D平均runtime 62.048/62.386 s，平均overshoot 2.048/2.386 s，最大5.168/4.792 s。Cmax、分配与负载取原生≤60秒best事件对应解；global-best计数也对齐该时间戳，iteration/accepted/valid取完整iteration boundary。overshoot后的final_cmax单列原始记录，不参与比较。MLP不参与，本轮未改B32、certifier、干涉、M192/K或任何邻域。\n\n10. **下一步先处理已定位的弱点，不直接启动大规模训练。** 这是情况C而非普遍的小规模退化：保留0.5任务配置与S1/I3异常，不设按规模切换参数。I3的串行fallback造成两机器人空闲，已有更好A初始解在D下合法认证，针对初始化鲁棒性做改进验证有明确必要；但仅改初始化不能解释S1，因为它的D初始略好、后续搜索反而弱。下一轮应从S1固定状态开始、保持M192做小范围C2/C4与reference筛选诊断，检查好候选是否生成、在哪层丢失，再决定排序或初始化改动。暂不进入全域新标签/MLP训练，不加operator、不调M/K、不改scheduler、不训练GAT。本轮没有执行这些下一步工作，完成复核后停止。\n",
  "i3_analysis": "历史I3的26条结果不改写：A两seed Cmax@60为1962.32/1790.23 s，D为2505.74/2441.55 s（+27.69%/+36.38%）；D原始global-best updates为4/6，说明有搜索但未弥补初始损失。\n\n本轮只用seed20261081各做一次A/D零迭代初始化重放，再做一次A初始解在D下的正常reference评价；共5+8+1=14次reference调用，没有重跑历史60秒组。A选择RAIL_MONOTONE_BALANCED_BOOTSTRAP，首个/选中初始化Cmax=2118.403 s，process loads=[1687.33,1610.59,1798.02,1675.70]，负载差187.44 s。D选择RAIL_SERIAL_BOOTSTRAP，首个/选中Cmax=3719.525 s，loads=[3440.52,0,3331.13,0]，负载差3440.52 s，R1/R3空闲。\n\n对A初始解逐parent检查D目录中的kind/t/rail，将mandatory元信息按新catalog重建，再canonicalize；missing_patterns为空，原方向下重新reference评价并独立认证通过，Cmax仍为2118.403 s。这个解只有WHOLE/Y，未依赖旧X中心；没有未经检查假定所有旧X解在D下合法。本证据证明D可行域中存在更好的初始化分配，I3主要弱点是portfolio失败后退回负载极不均衡的串行fallback，而非新任务域必然无更好解。未发现合法性或数值bug；本轮不修改初始化、SA、ranking或scheduler。这项定向诊断只证明旧A初始解可重建，没有冒称重建了未保存完整schedule的旧A@60最优解。\n"
}
```
