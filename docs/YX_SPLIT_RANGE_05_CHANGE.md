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
