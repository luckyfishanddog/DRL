# Phase4-1A：真实60秒生产候选池与MLP数据准备

Phase4-1A: PASS；M_prod=192；Feature pipeline READY；下一步 Phase4-1B MLP Candidate Ranker。

## 测量口径

原样复用 Phase3-Y/YR 的 I2/I3、I5/I6、I9/I12；全部属于 V2_MODEL_DEVELOPMENT_CONSUMED，未使用8个ORACLE_DEV或23个MLP候选workbook做搜索。固定三个新seed 20261111/20261112/20261113。硬件为Intel i5-1135G7（4物理核/8逻辑CPU）。每seed一个独立物理核、每核只使用一个逻辑CPU；四个M按实例/seed轮换顺序，三个seed进程同时运行。测量对应本机这种固定并行负载，不声称换硬件仍有相同迭代数。

72次均调用原生 SA_OI_ALNS_V2、60秒、初始化计时。只改变 M/M_lns；M_atomic由M−M_lns自动得到。Kdp8/Kref2/Kref_total4、C2/C4、SA、direction refinement、B32、TWO_OPT_STAR=OFF均保持。统计计时包装只测量函数耗时，不改变返回值、随机数或搜索状态。

Cmax@60 仅来自原生认证 best_events 中 elapsed≤60 的记录，overshoot 后 final_cmax 不参与选择。generation_seconds 覆盖整个 generate_candidate_pool（含cheap screening、complete构造、LNS repair）；lns_seconds是其 destroy+repair 子集，不能再加到generation_seconds。DP/reference时间含初始化。吞吐以actual_runtime为分母，额外记录60秒检查点与overshoot。unique是每个iteration内去重的有效候选数之和，不是跨搜索状态去重。iterations/attempts/calls采用原生完整run计数，包含最后一轮可能超时的工作；只有主指标Cmax严格截到60秒。

超时后final_cmax继续改善的run为7/72；最大overshoot=8.441秒。这些超时后改善全部排除出M选择。

## 配置与实测

| M | atomic/LNS | certified@60 | overall ratio | SMALL | MEDIUM | LARGE | median iterations | retention | gen fraction | reference calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 48/16 | 18/18 | 1.000000 | 1.000000 | 1.097991 | 1.029938 | 25.0 | 100.00% | 66.97% | 84.0 |
| 128 | 96/32 | 18/18 | 1.018015 | 1.000640 | 1.048323 | 1.018015 | 16.5 | 66.00% | 78.47% | 56.0 |
| 160 | 120/40 | 18/18 | 1.018015 | 1.000000 | 1.039553 | 1.018015 | 13.5 | 54.00% | 83.43% | 48.0 |
| 192 | 144/48 | 18/18 | 1.000000 | 1.000000 | 1.000000 | 1.011709 | 11.5 | 46.00% | 85.58% | 45.0 |

## 14个问题的回答

1. 完成迭代中位数：M64: 25.0；M128: 16.5；M160: 13.5；M192: 11.5。
2. 候选生成耗时中位占比：M64: 66.97%；M128: 78.47%；M160: 83.43%；M192: 85.58%。LNS是generation子集。
3. 每run reference calls中位数：M64: 84.0；M128: 56.0；M160: 48.0；M192: 45.0。其余吞吐见成本表，Kref预算没有增加。
4. overall配对ratio中位数最小的M为[64, 192]；这是18组instance×seed的Cmax@60比较，final_cmax未参与。
5. 分tier配对ratio中位数最小的M：SMALL: M=[64, 160, 192]；MEDIUM: M=[192]；LARGE: M=[192]。SMALL有多个M并列，MEDIUM/LARGE均以M192最小，分层最优集合不完全一致；生产设置仍统一。
6. 离线M192覆盖结论仍保留。在线M192：四项条件均满足，但仍按最小合格M选择。其迭代保留率为46.00%；本轮需要采用M192。
7. M_prod=192。
8. 更小的M未全部满足预先指定条件，选择首个合格M。 更小候选的失败项：M64未满足：all_tiers_within2pct；M128未满足：overall_within1pct、all_tiers_within2pct；M160未满足：overall_within1pct、all_tiers_within2pct。M192的46%迭代保留率满足40%下限。详细ratio见上表；未比较本轮范围外的M。
9. Kdp=8、Kref=2、Kref_total=4保持。SearchConfig()仍为历史M64。SA_OI_ALNS_V2省略search_config时使用生产M192/48；显式传入的历史配置原样保留，旧实验runner不改。未来60秒调用沿用原有60秒配置，仅将m/m_lns替换为公开的SA_OI_ALNS_V2_PRODUCTION_POOL_SIZE及其1/4。
10. 剩余23个未被Phase4用于机制决策的workbook已按静态几何farthest-first划为17 MLP_TRAIN + 6 MLP_DEV，同workbook不跨集合。
11. 原8个workbook在新简单split文件中标为ORACLE_DEV_CONSUMED，不再用于早停、超参数选择或MLP/GAT比较；历史manifest不改写。
12. 纯函数特征已完成，C2=87维，C4=130维。
13. C2接口只接收当前解/当前方向/当前Cmax、完整candidate和科学配置，不能接收candidate方向DP/调度标签；C4额外接收方向DP和cheap rank。输入特征与训练targets在不同函数中。既有10,701候选全部通过determinism/finite/information-boundary检查，未发现label leakage；没有用这些label选择特征。
14. 下一轮可以围绕选定M_prod生成MLP_TRAIN/DEV数据并进入Phase4-1B训练；本轮没有生成17个TRAIN的oracle labels，没有网络、scaler或optimizer拟合。

## 分层吞吐

| M | tier | iterations | gen fraction | reference calls | median Cmax@60 |
|---|---|---:|---:|---:|---:|
| 64 | SMALL | 53.5 | 69.15% | 206.5 | 1686.203540 |
| 64 | MEDIUM | 25.0 | 57.63% | 84.0 | 4990.629136 |
| 64 | LARGE | 13.5 | 54.60% | 44.5 | 6326.262359 |
| 128 | SMALL | 26.0 | 83.11% | 98.5 | 1699.195141 |
| 128 | MEDIUM | 18.5 | 72.34% | 65.0 | 4642.110473 |
| 128 | LARGE | 9.0 | 71.73% | 29.5 | 6029.233324 |
| 160 | SMALL | 21.0 | 86.28% | 61.5 | 1714.693103 |
| 160 | MEDIUM | 14.5 | 75.56% | 53.0 | 4730.160247 |
| 160 | LARGE | 7.5 | 74.41% | 25.5 | 6029.233324 |
| 192 | SMALL | 17.5 | 87.94% | 61.5 | 1693.813247 |
| 192 | MEDIUM | 12.5 | 77.66% | 47.0 | 4493.101844 |
| 192 | LARGE | 6.5 | 76.61% | 23.5 | 5686.461199 |

## 额外成本

| M | gen s | LNS s (子集) | DP s | reference s | calls/s | unique/s | runtime s | overshoot s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 41.1789 | 27.7118 | 0.1361 | 18.8331 | 1.3436 | 16.8707 | 61.3485 | 1.3485 |
| 128 | 49.3140 | 35.8372 | 0.0972 | 12.6062 | 0.8946 | 20.9585 | 62.2165 | 2.2165 |
| 160 | 52.0386 | 38.3242 | 0.0761 | 10.2471 | 0.7723 | 21.0913 | 61.8379 | 1.8379 |
| 192 | 52.3723 | 38.4166 | 0.0673 | 8.6664 | 0.7188 | 20.8393 | 61.7320 | 1.7320 |

## 干净的数据集合

MLP split仅有角色列表、已有workbook identity和静态选择说明；没有新hash/protocol。静态descriptor的标准化只用于几何多样性选择，不是MLP scaler拟合。workbook路径延续既有dataset manifest，均相对于DRL相邻的../ppo目录。

```json
{
  "MLP_TRAIN": [
    "data/DEV_ONLY/seed_000101.xlsx",
    "data/DEV_ONLY/seed_000404.xlsx",
    "data/ID_TEST/seed_0357172400.xlsx",
    "data/ID_TEST/seed_0394443053.xlsx",
    "data/ID_TEST/seed_0539338415.xlsx",
    "data/ID_TEST/seed_1050940519.xlsx",
    "data/ID_TEST/seed_1846918101.xlsx",
    "data/PPO_TRAIN/seed_0155539492.xlsx",
    "data/PPO_TRAIN/seed_0377964238.xlsx",
    "data/PPO_TRAIN/seed_0514719396.xlsx",
    "data/PPO_TRAIN/seed_0765477955.xlsx",
    "data/PPO_TRAIN/seed_1098469608.xlsx",
    "data/PPO_TRAIN/seed_1631571109.xlsx",
    "data/PPO_TRAIN/seed_1742672780.xlsx",
    "data/PPO_TRAIN/seed_1783157644.xlsx",
    "data/PPO_TRAIN/seed_1861847868.xlsx",
    "data/VALIDATION/seed_1910136750.xlsx"
  ],
  "MLP_DEV": [
    "data/DEV_ONLY/seed_000303.xlsx",
    "data/ID_TEST/seed_1142967162.xlsx",
    "data/PPO_TRAIN/seed_0834257622.xlsx",
    "data/PPO_TRAIN/seed_1745884954.xlsx",
    "data/PPO_TRAIN/seed_1872211770.xlsx",
    "data/VALIDATION/seed_0151520473.xlsx"
  ],
  "ORACLE_DEV_CONSUMED": [
    "data/PPO_TRAIN/seed_0180141104.xlsx",
    "data/PPO_TRAIN/seed_0703048389.xlsx",
    "data/PPO_TRAIN/seed_1062621171.xlsx",
    "data/PPO_TRAIN/seed_1636194744.xlsx",
    "data/PPO_TRAIN/seed_1914517550.xlsx",
    "data/VALIDATION/seed_1653613181.xlsx",
    "data/VALIDATION/seed_1960958179.xlsx",
    "outputs/v9_frozen_dataset_round_20260808/dev_only_dgseed_900001.xlsx"
  ]
}
```

## 特征与标签

C2包含move/family/source与LNS operator、当前认证Cmax、处理负载和cheap travel proxy前后变化、各robot route长度/块数、pattern counts、受影响机器人/parent/位置及焊缝几何。C4额外包含direction feasibility/成本/变化、各robot方向count/alternations/首尾方向和flips、定向后route端点（变长方向向量不作为固定维输入直接展开）以及C3 rerank components。candidate reference/WAIT/DEADLOCK/certifier/timing不在输入中。缺失direction用显式可用性mask，正式C4仍只筛direction-feasible候选。

Target：仅FEASIBLE_CERTIFIED有y_feasible=1与y_improvement=(Cs−Ccandidate)/Cs；其他有效candidate状态y_feasible=0、improvement=NULL；NUMERIC_FAILURE整条排除。没有人工penalty、绝对Cmax回归或训练拟合。

完整特征列与逐候选回归计数在结果JSON；函数位于src/mrta_ranker/features.py。

验证：{
  "passed": true,
  "states": 72,
  "counts": {
    "candidates": 10701,
    "FEASIBLE_CERTIFIED": 5540,
    "DEADLOCK": 4871,
    "DIRECTION_INFEASIBLE": 290
  },
  "C2_feature_count": 87,
  "C4_feature_count": 130,
  "C2_feature_names": [
    "source_atomic",
    "source_lns_repaired",
    "move_intra_relocate",
    "move_inter_relocate",
    "move_swap",
    "move_two_opt",
    "move_split_activate",
    "move_split_deactivate",
    "move_split_point_switch",
    "family_structural",
    "family_target_whole",
    "family_target_y",
    "family_target_x",
    "destroy_random_removal",
    "destroy_critical_load_removal",
    "repair_greedy_repair",
    "repair_regret_2_repair",
    "current_cmax",
    "projected_process_before",
    "projected_process_after",
    "projected_process_delta",
    "travel_proxy_before",
    "travel_proxy_after",
    "travel_delta",
    "split_count_delta",
    "load_spread_delta",
    "before_r0_process_load",
    "before_r0_block_count",
    "before_r1_process_load",
    "before_r1_block_count",
    "before_r2_process_load",
    "before_r2_block_count",
    "before_r3_process_load",
    "before_r3_block_count",
    "before_max_robot_load",
    "before_load_range",
    "before_load_std",
    "before_whole_count",
    "before_x_split_count",
    "before_y_split_count",
    "after_r0_process_load",
    "after_r0_block_count",
    "after_r1_process_load",
    "after_r1_block_count",
    "after_r2_process_load",
    "after_r2_block_count",
    "after_r3_process_load",
    "after_r3_block_count",
    "after_max_robot_load",
    "after_load_range",
    "after_load_std",
    "after_whole_count",
    "after_x_split_count",
    "after_y_split_count",
    "affected_r0",
    "affected_r1",
    "affected_r2",
    "affected_r3",
    "affected_robot_count",
    "affected_upper_rail",
    "affected_lower_rail",
    "affected_same_rail",
    "affected_parent_count",
    "remove_position_count",
    "remove_position_min",
    "remove_position_max",
    "remove_position_mean",
    "remove_position_std",
    "insert_position_count",
    "insert_position_min",
    "insert_position_max",
    "insert_position_mean",
    "insert_position_std",
    "parent_length_min",
    "parent_length_max",
    "parent_length_mean",
    "parent_length_std",
    "affected_length_min",
    "affected_length_max",
    "affected_length_mean",
    "affected_length_std",
    "affected_x_span",
    "affected_y_span",
    "affected_mid_x",
    "affected_mid_y",
    "affected_upper_fraction",
    "affected_left_fraction"
  ],
  "C4_feature_names": [
    "source_atomic",
    "source_lns_repaired",
    "move_intra_relocate",
    "move_inter_relocate",
    "move_swap",
    "move_two_opt",
    "move_split_activate",
    "move_split_deactivate",
    "move_split_point_switch",
    "family_structural",
    "family_target_whole",
    "family_target_y",
    "family_target_x",
    "destroy_random_removal",
    "destroy_critical_load_removal",
    "repair_greedy_repair",
    "repair_regret_2_repair",
    "current_cmax",
    "projected_process_before",
    "projected_process_after",
    "projected_process_delta",
    "travel_proxy_before",
    "travel_proxy_after",
    "travel_delta",
    "split_count_delta",
    "load_spread_delta",
    "before_r0_process_load",
    "before_r0_block_count",
    "before_r1_process_load",
    "before_r1_block_count",
    "before_r2_process_load",
    "before_r2_block_count",
    "before_r3_process_load",
    "before_r3_block_count",
    "before_max_robot_load",
    "before_load_range",
    "before_load_std",
    "before_whole_count",
    "before_x_split_count",
    "before_y_split_count",
    "after_r0_process_load",
    "after_r0_block_count",
    "after_r1_process_load",
    "after_r1_block_count",
    "after_r2_process_load",
    "after_r2_block_count",
    "after_r3_process_load",
    "after_r3_block_count",
    "after_max_robot_load",
    "after_load_range",
    "after_load_std",
    "after_whole_count",
    "after_x_split_count",
    "after_y_split_count",
    "affected_r0",
    "affected_r1",
    "affected_r2",
    "affected_r3",
    "affected_robot_count",
    "affected_upper_rail",
    "affected_lower_rail",
    "affected_same_rail",
    "affected_parent_count",
    "remove_position_count",
    "remove_position_min",
    "remove_position_max",
    "remove_position_mean",
    "remove_position_std",
    "insert_position_count",
    "insert_position_min",
    "insert_position_max",
    "insert_position_mean",
    "insert_position_std",
    "parent_length_min",
    "parent_length_max",
    "parent_length_mean",
    "parent_length_std",
    "affected_length_min",
    "affected_length_max",
    "affected_length_mean",
    "affected_length_std",
    "affected_x_span",
    "affected_y_span",
    "affected_mid_x",
    "affected_mid_y",
    "affected_upper_fraction",
    "affected_left_fraction",
    "direction_feasible",
    "direction_cost_available",
    "direction_dp_travel",
    "direction_dp_travel_delta",
    "direction_r0_route_available",
    "direction_r0_reverse_count",
    "direction_r0_alternations",
    "direction_r0_first_orientation",
    "direction_r0_last_orientation",
    "direction_r0_first_x",
    "direction_r0_first_y",
    "direction_r0_last_x",
    "direction_r0_last_y",
    "direction_r1_route_available",
    "direction_r1_reverse_count",
    "direction_r1_alternations",
    "direction_r1_first_orientation",
    "direction_r1_last_orientation",
    "direction_r1_first_x",
    "direction_r1_first_y",
    "direction_r1_last_x",
    "direction_r1_last_y",
    "direction_r2_route_available",
    "direction_r2_reverse_count",
    "direction_r2_alternations",
    "direction_r2_first_orientation",
    "direction_r2_last_orientation",
    "direction_r2_first_x",
    "direction_r2_first_y",
    "direction_r2_last_x",
    "direction_r2_last_y",
    "direction_r3_route_available",
    "direction_r3_reverse_count",
    "direction_r3_alternations",
    "direction_r3_first_orientation",
    "direction_r3_last_orientation",
    "direction_r3_first_x",
    "direction_r3_first_y",
    "direction_r3_last_x",
    "direction_r3_last_y",
    "direction_shared_blocks",
    "direction_flips",
    "c3_cheap_rank"
  ],
  "historical_inputs_unchanged": true,
  "new_reference_calls": 0,
  "new_direction_optimizations": 0,
  "purpose": "consumed ORACLE_DEV pipeline regression only; true frozen M256 cheap ranks; no feature selection or fitting"
}

回归：{"passed": 365, "interpreter": "D:\\pybullet_test\\.venv\\Scripts\\python.exe", "command": "pytest -q -p no:cacheprovider"}
