# Phase4-0 V2 Candidate-Pool Oracle Recall Audit

本轮使用用户指定的本地 DRL 与 Python 环境。独立 Git 要求由用户明确撤销；source_commit 为本地标签，commit_verified=False、dirty=True，未声称正式 clean Git provenance。未上传 GitHub。

固定生产 M64 / atomic48 + LNS16 / Kdp8 / Kref2 / Kref_total4 / B32。影子池按 48+16、48+16、96+32 的真实尝试顺序继续生成。坏候选、重复和 identity 均占预算，不补抽。Oracle 仅使用基础方向 DP；方向细化收益独立记录。

## 状态

```
PHASE4_0_EXECUTION_STATUS = PASS
AUTHORITATIVE_PLAN_V2_SYNC = PASS
RANKER_SPLIT_STATUS = FROZEN
AUDIT_STATE_CONTEXTS = 72 / 72
CANDIDATE_POOL_REPLAY_STATUS = PASS
OPPORTUNITY_EVIDENCE_SUFFICIENT = YES
GENERATION_COVERAGE_STATUS = FAIL
RANKING_GAP_STATUS = NOT_EVALUABLE
PRIMARY_RANKING_STAGE = NOT_EVALUABLE
PRIMARY_BOTTLENECK = CANDIDATE_GENERATION
POOL_SIZE_SENSITIVITY = SUPPORTED
PHASE4_1_MLP_AUTHORIZED = NO
TRADITIONAL_HEURISTIC_TUNING_STATUS = CLOSED
V2_VALIDATION_STATUS = CONSUMED
ID_TEST_STATUS = SEALED
NEXT_PHASE = Phase4-0B Bounded Candidate Generation Scaling Audit
```

## 16 个问题的测量回答

1. M64、M128、M256 严格嵌套，72 个实际轨迹池逐尝试、构造解、身份及 C2/C4 重放一致。
2. M64 有效 unique candidate：平均 42.417，中位 43.000。
3. M256 有 68 个改善机会状态；M64 有 58 个，相差 10 个。
4. 在 68 个改善机会状态中，M64 捕获改善的中位比例为 99.98%；达到 75% 捕获率的状态占 60.29%，完全未捕获的占 14.71%。
5. M128 捕获比例中位数为 100.00%，均值为 83.76%；典型状态接近 M256，但尾部仍有损失，不能说所有状态已足够。按冻结规则，池大小敏感性 SUPPORTED。
6. 在 58 个 M64 有改善的状态中，C2 中位保留 9.27%，中位损失 90.73%；完全未捕获占 37.93%。
7. C4 最终中位保留 0.00%，完全未捕获占 67.24%；分母始终为 M64 可利用改善 I64。
8. 机械判定主要瓶颈 CANDIDATE_GENERATION，ranking stage NOT_EVALUABLE。全局 generation median 虽高，但达到 75% 捕获率的状态比例为 60.29%，未满足至少 2/3 的覆盖条件。冻结规则要求先通过 generation 才判 ranking；C2/C4 低保留率仍是测量事实，不能解释成不存在筛选损失。
9. SMALL/MEDIUM/LARGE 的等状态权重分层结果见下表。
10. EARLY/MID/LATE 结果见下表；相同解但不同阶段/上下文仍保留。
11. M256 改善机会状态中的 oracle-best family ties：{"STRUCTURAL":92,"TARGET_WHOLE":21,"TARGET_X":2,"TARGET_Y":2}；并列最优分别计数，不推成独占贡献。
12. X 在 M64 每状态平均尝试 5.194、有效 4.764、生产 reference 评价 0.333；X 达到 M64 oracle-best 但未被生产评价的候选平均 0.069。原始拒绝不能确定 pattern 的另记 UNRESOLVED_PATTERN；未评价的并列最优不自动等于改善损失。
13. LARGE：更符合筛选排名损失。24 个状态中有 24 个 M256 改善机会；M64 捕获中位 100.00%、均值 78.55%，达到 75% 捕获率的占 70.83%，完全未捕获占 12.50%。M128 捕获中位 100.00%、均值 83.40%。在 21 个 M64 有改善的状态中，C2 中位保留 44.56%，C4 中位保留 0.00%，两层都存在损失；C4 完全未捕获占 57.14%。所以不能将全局机械标签直接当作 LARGE 退化的成因。这只是开发状态上的机会损失证据。
14. MLP 授权 NO；必须同时通过执行、机会数、生成覆盖和排名差距条件。
15. 本轮未验证图表示相对固定候选特征的收益；即使 MLP 获准，也不能据此推进 GAT。
16. 下一阶段：Phase4-0B Bounded Candidate Generation Scaling Audit。本轮不执行该阶段。

## 分层结果

| 分组 | 状态 | 改善机会 | median gen64 | median gen128 | median C2 | median C4 |
|---|---:|---:|---:|---:|---:|---:|
| SMALL | 24 | 21 | 0.931686 | 1.000000 | 0.022790 | 0.000000 |
| MEDIUM | 24 | 23 | 0.866865 | 1.000000 | 0.085855 | 0.000000 |
| LARGE | 24 | 24 | 1.000000 | 1.000000 | 0.445627 | 0.000000 |
| EARLY | 24 | 22 | 1.000000 | 1.000000 | 0.424190 | 0.000000 |
| MID | 24 | 24 | 0.999836 | 1.000000 | 0.000000 | 0.000000 |
| LATE | 24 | 22 | 0.959455 | 1.000000 | 0.008503 | 0.000000 |

## Family（每状态平均；M64）

| family | 尝试 | 有效 | C2选择 | 生产reference | reference可行 | 改善 | oracle-best ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| STRUCTURAL | 46.222 | 28.847 | 5.292 | 0.931 | 15.000 | 2.278 | 2.097 |
| TARGET_WHOLE | 2.653 | 2.236 | 0.528 | 0.167 | 1.236 | 0.306 | 0.139 |
| TARGET_Y | 9.931 | 6.569 | 1.167 | 0.569 | 4.028 | 1.125 | 0.181 |
| TARGET_X | 5.194 | 4.764 | 1.014 | 0.333 | 2.361 | 0.500 | 0.111 |

## LARGE 扩池后的 oracle-best family

在同一批 M256 改善机会状态上，以下统计各 prefix 的最优候选并列数量；扩池会替换原有最优，数量并不单调，也不是独立状态数。M256 中以 STRUCTURAL 和 WHOLE 为主，Y/X 没有达到扩池后的最优；不能据此修改 family quota。

| prefix | STRUCTURAL | WHOLE | Y | X |
|---|---:|---:|---:|---:|
| M64 | 56 | 7 | 3 | 1 |
| M128 | 67 | 13 | 3 | 1 |
| M256 | 31 | 17 | 0 | 0 |

## 候选与状态计数

24 条原生轨迹全部完成，72 个状态 M64 exact replay 全部通过；10,701 个 valid unique M256 候选全部完成标签。独立候选标签：{"DEADLOCK":4871,"DIRECTION_INFEASIBLE":290,"FEASIBLE_CERTIFIED":5540}；NUMERIC_FAILURE=0、certifier mismatch=0。其他已声明状态计数为 0。

| prefix | 有效 unique 总数 | 每状态平均 | 每状态中位 |
|---|---:|---:|---:|
| M64 | 3054 | 42.417 | 43.000 |
| M128 | 5680 | 78.889 | 80.000 |
| M256 | 10701 | 148.625 | 150.500 |

尝试级状态计数（每状态固定 64/128/256；重复/identity/拒绝也占预算）：

| status | M64 | M128 | M256 |
|---|---:|---:|---:|
| RAW_REJECTED | 0 | 0 | 0 |
| CONSTRUCTION_REJECTED | 687 | 1372 | 2722 |
| IDENTITY | 175 | 365 | 732 |
| DUPLICATE | 692 | 1799 | 4277 |
| DIRECTION_INFEASIBLE | 85 | 152 | 290 |
| FEASIBLE_CERTIFIED | 1629 | 2975 | 5540 |
| DEADLOCK | 1340 | 2553 | 4871 |
| INFEASIBLE | 0 | 0 | 0 |
| NUMERIC_FAILURE | 0 | 0 | 0 |

## 成本与复现

```json
{
  "candidates": 10701,
  "reference_calls": 10411,
  "reference_seconds": 5832.6174065997875,
  "direction_seconds": 12.32623450011306,
  "candidate_evaluator_cpu_seconds": 5967.46875,
  "candidate_evaluator_wall_seconds": 6102.830628099848,
  "offline_complete_process_cpu": {
    "method": "Windows GetProcessTimes after exit; parent/venv launcher and all three worker processes",
    "processes": [
      {
        "pid": 1800,
        "kernel_cpu_seconds": 0.0,
        "user_cpu_seconds": 0.0,
        "cpu_seconds": 0.0
      },
      {
        "pid": 22884,
        "kernel_cpu_seconds": 29.875,
        "user_cpu_seconds": 16.875,
        "cpu_seconds": 46.75
      },
      {
        "pid": 12744,
        "kernel_cpu_seconds": 2.46875,
        "user_cpu_seconds": 1997.765625,
        "cpu_seconds": 2000.234375
      },
      {
        "pid": 13348,
        "kernel_cpu_seconds": 2.875,
        "user_cpu_seconds": 1999.21875,
        "cpu_seconds": 2002.09375
      },
      {
        "pid": 7116,
        "kernel_cpu_seconds": 2.21875,
        "user_cpu_seconds": 1993.375,
        "cpu_seconds": 1995.59375
      }
    ],
    "total_cpu_seconds": 6044.671875
  },
  "offline_total_cpu_seconds": 6044.671875,
  "offline_execution_mode": "3 isolated processes; single SQLite writer",
  "offline_elapsed_wall_seconds": 2070.496243400001,
  "replay_wall_seconds": 443.6395640000028,
  "replay_cpu_seconds": 439.1875,
  "trajectory_cpu_seconds": 1474.46875,
  "trajectory_wall_seconds": 1480.3360127000024,
  "trajectory_reference_calls": 2868
}
```

无损存储优化：{"all_logical_rows_equal_after_expansion":true,"bytes_after":70144000,"bytes_before":147804160,"cpu_seconds":166.609375,"foreign_keys":"ok","format":"compressed complete identity + shared state packets for candidates/attempts/diagnostics + SQLite VACUUM","reduction_percent":52.54260773174449,"sqlite_integrity":"ok","wall_seconds":167.56453399999737}

压缩块按状态保存，标量标签继续逐候选索引。可用审计脚本的 read_candidate_payload、read_candidate_diagnostic、read_attempt_detail 读取完整记录；传入对应实例 parents 即可恢复共享几何。完整身份通过 expanded_identity 展开，未替换为摘要或截断标识。

轨迹的原生 60 秒预算与离线 replay/oracle 计时分离。离线 oracle 使用 3 个独立进程，主进程按确定顺序逐候选写入；candidate wall/scheduler time 为调用耗时之和，offline_elapsed_wall_seconds 为并行阶段实际经过时间。候选标签按协议、上下文、完整候选身份、scope 和源码身份联合主键逐条提交事务，可断点续跑；只读取已完成且身份匹配的标签。状态 JSON 共享几何并使用无损压缩；SQLite 候选与诊断也使用无损压缩。

FEASIBLE_CERTIFIED 才有有限 Cmax；其他状态 Cmax/delta 为 NULL。无可行候选时 C* 为 NULL、I 为零并保留 no_feasible 标记。无改善机会时 recall 为 NA。排名 recall 仅在 I64>epsilon 状态上聚合；generation recall 仅在 I256>epsilon 状态上聚合。

辅助的 C4/C2 条件 recall：{"count":36,"maximum":1.0,"mean":0.31542535994815574,"median":0.023737101990840278,"minimum":0.0}；不替换以 I64 为分母的主指标。POST_C4_DIRECTION_GAIN：{"count":43,"maximum":92.73846349673477,"mean":2.3819115241833146,"median":0.0,"minimum":0.0}，仅单独描述方向细化。

候选标签文件中的 production_reference_evaluated 与 audit_oracle_reference_evaluated 分列。方向细化收益在每状态 POST_C4_DIRECTION_GAIN 列中，不进入基础 oracle 排名。

历史 Phase3-Z/Phase3-ZR 文件不改写。V2 validation 已消耗，ID_TEST 继续封存。本轮仅运行 RANKER_DEV；路径含 VALIDATION 的文件按既有 manifest 角色判定，文件夹名字不决定角色。

INITIAL 回退状态数：7 / 72。

回归验证：{"duration_seconds":58.33,"import_root":"D:\\pybullet_test\\MRTA_GA\\DRL\\src","interpreter":"D:\\pybullet_test\\.venv\\Scripts\\python.exe","passed":352,"phase":"after_complete_labels_compaction_final_reporting_and_document_sync"}
源身份：{"commit_verified":false,"repository_id":"luckyfishanddog/DRL","source_commit":"LOCAL_DRL_PHASE4_0_USER_AUTHORIZED","source_tree_hash":"ffdcb5533336f5443439361af392a0d86535d29212253d4a5d400b7aeb0af051","worktree_dirty":true}

协议与科学身份：{"protocol_hash":"b10b603f85e46b7343d9d3173bae867d75f679151d8fd8aa25fbd47c6b8a4365","ranker_split_hash":"a5e3a9d4fc1a3bfe108f0492305144a08fe6539ca54b9c6078be3a4a52889dc8","reference_scheduler_policy_id":"FORMAL_LIMITED_DISCREPANCY_DISPATCH_POLICY_V1","scientific_config_hash":"791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e","scope_hash":"16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599"}
