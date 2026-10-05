# Phase 3-X2 — Frozen X-Candidate Deterministic Potential Audit

日期：2026-10-05。证据类型：DEVELOPMENT mechanism diagnosis，非公平算法 benchmark。

## 最终结论

```text
PHASE3X2_EXECUTION_STATUS = PASS
X_ONE_STEP_POTENTIAL = PRESENT
X_SEARCH_RETENTION = ADEQUATE
X_DOMAIN_VALUE_STATUS = SUPPORTED
FORMAL_SCOPE_V2_AUTHORIZED = YES
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1_1
ID_TEST_STATUS = SEALED
NEXT_PHASE = Phase 3-Y — FORMAL_SCOPE_V2 Core Closure + common-domain HGA/WAG/ALNS adaptation
```

历史 Phase 3-X 的 `PASS / NOT_SUPPORTED / WEAK / V2 NO` 完整保留。本轮独立证据为 `PHASE3X2_X_DOMAIN_POTENTIAL = SUPPORTED`，不把历史负 Gate 重解释为 PASS。Phase 3-X2 预注册 CASE 3 成立：冻结有限 X domain 中有 replicated certified one-step material benefit；已知好 X seed 的 retention 达到冻结支持规则。不是每个实例均受益，也不是全局 X-domain exact optimization。

ACTIVE 仍为 FORMAL_SCOPE_V1_1，直到 Phase 3-Y core closure；本轮没有实现或激活 V2，没有修改 HGA/WAG、生产 ALNS、scheduler、certifier。

## 1. 文件、执行环境与保护范围

仅新增已获逐项批准的四个文件：

- `scripts/run_phase3x2_x_oracle_audit.py`
- `data/manifests/PHASE3X2_X_ORACLE_PROTOCOL_V1.json`
- `data/development/phase3x2_x_oracle_audit_v1.json`
- 本报告。

在现有 `tests/test_phase3_validation.py` 增加 14 个测试用例，保留原有 237 个测试。README 仅追加本轮最终状态。没有新增 src 模块、临时结果文件、数据分区或 plan 文件。

运行目录：`D:\pybullet_test\MRTA_GA\DRL`。解释器：`D:\pybullet_test\.venv\Scripts\python.exe`。关闭 bytecode/cache 文件生成。PPO Excel、references、历史 Phase 3 和 Phase 3-X artifacts 保持原样；未 commit、push、上传 GitHub。

外层 MRTA_GA HEAD 为 `0935ad8a723057084be190e31c6be4733ce3cb62`，DRL 是外层未跟踪目录，不能当作已验证的独立 GitHub checkout。用户提供 main `f75ddc4044d9d066a6a5a89d91c03ef50c61386d` 只作为 source label，`source_commit_verified=false`；本轮不宣称已验证该远端提交。科学源树与历史 Phase 3-X 一致，见下列 source hash。

## 2. 冻结身份与数据治理

| 身份 | 值 |
|---|---|
| 原 gate set | 4bfe998a9caef567a66db24a77e9a598b24bda4be7967e51d0484c1548558c93 |
| Experimental X scope | 28236ee1caa75ce669af2a62ddbd28e2c557ed4b00eb5d4a1d8a29335aa209c8 |
| ACTIVE V1.1 scope | 5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc |
| Scientific config | 791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e |
| Unchanged scientific source | e6fd58b6c56e53866c19760403a9094cdd99b6dc4b794b2413e78003d1681eaf |
| Historical Phase 3-X protocol | 73fb5ff0c17d53439eacd097b79a19a0a98e6ebad762bd7ba21d9d7cc66b7c18 |
| Historical Phase 3-X artifact file SHA-256 | 854183d399c11240a42577e2c86c7a5ad838e249b788eae79cc57280dc6c3b25 |
| X_PATTERN_CENSUS_V1 | 5607e7538ad6e17bab05738d976b619d79d2dc43114891f845a6a8dd57651052 |
| Phase 3-X2 protocol | 49f11419df30cb079357b3199db0a1c312c0852c168d03ee784e614531e4035e |
| Independent audit runner | e7d977a96960fcf91160b39557f460ec350e0699ed3df6bd0d764b36b7cd3a14 |
| Stage A scientific payload | 6907f896c1ba461b4bad8c01fc677b161d7a79598fc9038516ec1a9cbfeef267 |

仍是原 12 instances、12 different workbooks，没有增减或替换。所有 workbook 在 `PPO_PHASE3_DATA_SPLIT_V1` 的当前角色均核对为 DEVELOPMENT_CONSUMED。目录名中的 PPO_TRAIN/VALIDATION/ID_TEST 不是当前科学角色；真正 TRAIN_POOL、VALIDATION、ID_TEST 拒绝访问，原 15 ID_TEST 继续 SEALED。

12 个 common seed 从历史定义重建，canonical hash 和 Cmax 均与原 artifact 完全一致，且 V1.1 certified。未换 seed、未重新优化 common seed。所有可审计 X parent 在 common seed 中为 WHOLE。

原 Phase 3-X manifest 没有逐 pattern identity 列表或 census hash，不能声称对不存在的历史 hash 做了比较。身份延续由相同 generator/scientific source、相同 workbook/geometry metadata 和原 pattern 聚合数量建立；本轮首次冻结完整 539 行 census，并在独立复跑中逐行核对。

## 3. 方法与边界

本轮名称为 BOUNDED_DETERMINISTIC_X_PATTERN_POTENTIAL_AUDIT，**不是全局最优、exact X oracle 或 best possible X_SPLIT solution**。

X scientific definition 未变：Bx lower/center/upper/midpoint；delta_x=Lmin=0.20 m；cut_time=split_time=0；每 child 都保留 SETUP+WELD+POST，真正 split 总加工量固定增加 50 s。每 parent 最多 split 一次，mandatory Y 不可被 X 替换。UPPER left/right→R0/R1，LOWER→R2/R3；方向仍可优化；shared split point 无 collision exception。

对每 pattern 从同一个 common seed 激活单个 X，先移除 WHOLE，再完整枚举目标两条 route 的所有插入位置乘积。每对经 canonical construction、exactly-once/coverage 和 eligibility validation，再 cheap scoring；没有随机 parent/pattern/insertion。

cheap 排序为 max robot process load、process spread、已有 common-direction empty proxy、已有 zero-direction empty proxy、canonical solution hash。新 child proxy orientation 使用 0；不用 WAIT/reference Cmax 排序。

每 pattern top min(16,n) 调用已有 initial-feasibility direction DP；按 DP total empty travel、cheap tuple、canonical hash 取 top min(4,n)。每个 direction-feasible pattern 至少获得一次正式 reference B32 评价。FEASIBLE 一律 independent certifier PASS 后才作为证据；DEADLOCK 保持政策调度结果，不写 penalty Cmax。

若 top16 全方向不可行但其他插入存在合法初始方向，runner 会立即 FAIL，不增加 cap，也不静默遗漏。本轮没有触发此冲突。没有 per-instance wallclock cutoff，没有按 N85 缩小预算。

blocking_proxy=p_i*x_span_i（另存归一化值）、cross_x_g、midpoint distance 只作事后解释；2 m 支持条件和 10 m 描述组均未用于 legality、候选过滤或预算分配。

## 4. Stage A 全量结果

| 项目 | 数量 |
|---|---:|
| 冻结 patterns / 完成系统访问 / 有 direction-feasible insertion | 539 / 539 / 539 |
| 全量 insertion pairs / canonical-valid / cheap-valid / cheap-scored | 145646 / 145646 / 145646 / 145646 |
| DP evaluations | 8623 |
| Reference evaluations | 2153（上限 2156） |
| Reference FEASIBLE + certified | 1074 |
| Reference DEADLOCK | 1079 |
| INFEASIBLE / NUMERIC_FAILURE / certifier mismatch | 0 / 0 / 0 |
| 至少有一个 certified FEASIBLE 的 patterns | 301 |
| one-step 优于 common seed 的 patterns | 116 |
| one-step 改善 >=1% 的 patterns | 63 |
| 有正收益 / 有 >=1% 收益的不同 workbooks | 11 / 8 |
| 每 pattern reference evaluations | 538 个为 4，1 个为 1 |
| 实际 cache hit | 0 |

Stage A pattern audit wallclock 合计 968.522 s（约 16.14 min），不含 common-seed portfolio 重建、结果刷新/持久化和独立复跑。这里统计的是第一遍 bounded audit 的 reference 调用；复跑、common-seed 重建、Stage B seed replay 和 search reference 调用另计，不能混入 2156 上限。

8 个支持 workbooks 满足 >=1%，其中 6 个 N>=50；best parent x_span>=2 的支持实例包含 I3/I5/I6/I7/I9。因此 Stage A 机械判定 PRESENT，没有根据结果调阈值。

### 4.1 instance-level best one-step X

I1–I12 以下述表格顺序固定。时间单位 s，长度/x_span 单位 m；blocking_proxy 单位 s·m。所有 C_X1 均为 reference FEASIBLE + independent certifier PASS。

| ID | instance_id | N | patterns | common Cmax | C_X1 | 改善 | best source | parent length | x_span | blocking_proxy |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| I1 | data/PPO_TRAIN/seed_0442383381::g15_w024 | 24 | 28 | 1552.220 | 1543.775 | 0.5440% | BX_UPPER | 1.693 | 0.846 | 174.903 |
| I2 | data/PPO_TRAIN/seed_1064648049::g20_w026 | 26 | 32 | 1639.155 | 1639.155 | 0.0000% | BX_UPPER | 1.249 | 1.082 | 179.270 |
| I3 | data/DEV_ONLY/seed_000505::g15_w026 | 26 | 21 | 2118.403 | 1905.782 | 10.0369% | MIDPOINT | 2.278 | 2.278 | 594.390 |
| I4 | data/PPO_TRAIN/seed_0760656368::g16_w025 | 25 | 23 | 2028.733 | 1924.108 | 5.1571% | MIDPOINT | 2.651 | 0.005 | 1.477 |
| I5 | data/PPO_TRAIN/seed_0900775608::g30_w055 | 55 | 61 | 4305.537 | 3499.313 | 18.7253% | BX_CENTER | 12.500 | 12.500 | 15092.593 |
| I6 | data/PPO_TRAIN/seed_1419626827::g29_w055 | 55 | 53 | 7174.129 | 6541.654 | 8.8161% | BX_LOWER | 12.500 | 12.500 | 15092.593 |
| I7 | data/PPO_TRAIN/seed_1137713392::g23_w055 | 55 | 39 | 3791.216 | 3608.448 | 4.8208% | MIDPOINT | 2.959 | 2.959 | 958.745 |
| I8 | data/VALIDATION/seed_1964628001::g24_w055 | 55 | 45 | 3659.596 | 3635.947 | 0.6462% | MIDPOINT | 0.540 | 0.468 | 46.764 |
| I9 | data/PPO_TRAIN/seed_0194123089::g43_w085 | 85 | 65 | 9810.208 | 9186.201 | 6.3608% | BX_LOWER | 12.500 | 12.500 | 15092.593 |
| I10 | data/ID_TEST/seed_0401115467::g45_w085 | 85 | 71 | 4652.070 | 4532.263 | 2.5754% | MIDPOINT | 1.509 | 0.742 | 140.788 |
| I11 | data/ID_TEST/seed_1411387107::g28_w085 | 85 | 50 | 6547.100 | 6516.829 | 0.4624% | MIDPOINT | 2.129 | 2.129 | 526.139 |
| I12 | data/PPO_TRAIN/seed_0211781140::g32_w085 | 85 | 51 | 5427.661 | 5331.614 | 1.7696% | MIDPOINT | 1.090 | 1.042 | 157.256 |

I2 的最佳 X 仅与 common seed 持平，标记 NOT_APPLICABLE_NO_POSITIVE_X_SEED；其余全部 11 个实例进入 Stage B，而非只挑选 >=1% 的成功实例。完整 best parent/rail/t、插入位置、方向、canonical solution、hash 和 scope/policy identity 均在 artifact 的 instance_summaries 和 pattern_records 中。

### 4.2 来源分布与 MIDPOINT dominance

| source | 全部 patterns | 正收益 patterns | >=1% patterns | 12 个实例的 best source |
|---|---:|---:|---:|---:|
| MIDPOINT | 424 | 90 | 39 | 7 |
| BX_LOWER | 37 | 8 | 7 | 2 |
| BX_CENTER | 42 | 8 | 8 | 1 |
| BX_UPPER | 36 | 10 | 9 | 2 |

8 个 >=1% 的 instance best 中 MIDPOINT=5，BX_LOWER=2，BX_CENTER=1。MIDPOINT 按数量占优，但 Bx 合计只有 115 patterns，却提供 24 个 >=1% patterns（20.87%；MIDPOINT 为 39/424=9.20%），并包含长焊缝的主要收益；不能得出 Bx 贡献弱的结论，也没有据此删除任何来源。

source 是 generator 去重后保留的标签，不是互斥的几何机制：例如 I5 的 BX_CENTER t=0.5 与 midpoint 重合。不能仅凭标签证明 Bx 相对 midpoint 的独立因果优势。

116 个正收益 pattern 的 parent 描述统计（pattern-weighted，同一 parent 可重复出现）：

| 字段 | 最小 | 中位 | 最大 |
|---|---:|---:|---:|
| parent_length | 0.504 | 1.249 | 12.500 |
| x_span | 0.001 | 1.042 | 12.500 |
| whole_process_time | 96.655 | 165.684 | 1207.407 |
| blocking_proxy | 0.094 | 158.497 | 15092.593 |

## 5. 收益分解：不能只看 Cmax

以下 delta=best one-step X−common seed，负值表示该指标下降；spread/WAIT 变化不是因果消融。每个 schedule 的四机器人 process、empty、WAIT、completion、makespan robot、makespan_robot_wait 以及 B32 recovery usage 已完整记录于 artifact。

| ID | delta process spread | delta empty travel | delta total WAIT | delta makespan_robot_wait | common scheduler / rollouts | best-X scheduler / rollouts |
|---|---:|---:|---:|---:|---|---|
| I1 | -22.877 | 12.675 | 0.000 | 0.000 | BASELINE/0 | BASELINE/0 |
| I2 | -17.256 | 0.367 | -44.589 | 0.000 | LIMITED_DISCREPANCY_RECOVERY/14 | BASELINE/0 |
| I3 | 118.262 | 6.608 | -270.139 | -270.139 | LIMITED_DISCREPANCY_RECOVERY/30 | BASELINE/0 |
| I4 | 36.937 | 45.043 | -325.178 | -325.178 | LIMITED_DISCREPANCY_RECOVERY/2 | BASELINE/0 |
| I5 | 675.597 | -228.623 | -1265.822 | -1265.822 | LIMITED_DISCREPANCY_RECOVERY/14 | BASELINE/0 |
| I6 | -597.222 | -35.253 | 0.000 | 0.000 | BASELINE/0 | BASELINE/0 |
| I7 | -91.759 | -132.763 | -324.571 | -324.571 | LIMITED_DISCREPANCY_RECOVERY/2 | LIMITED_DISCREPANCY_RECOVERY/2 |
| I8 | -25.000 | 20.404 | 0.000 | 0.000 | BASELINE/0 | BASELINE/0 |
| I9 | -597.222 | -26.784 | 0.000 | 0.000 | BASELINE/0 | BASELINE/0 |
| I10 | 69.865 | -135.255 | -90.147 | -90.147 | LIMITED_DISCREPANCY_RECOVERY/2 | LIMITED_DISCREPANCY_RECOVERY/1 |
| I11 | 148.565 | -294.392 | -80.332 | -100.095 | LIMITED_DISCREPANCY_RECOVERY/6 | LIMITED_DISCREPANCY_RECOVERY/6 |
| I12 | -100.452 | -229.814 | 323.470 | 0.000 | LIMITED_DISCREPANCY_RECOVERY/32 | LIMITED_DISCREPANCY_RECOVERY/24 |

I5 的长焊缝 best-X：spread 反而增加 675.597 s，但 empty 减少 228.623 s、WAIT 减少 1265.822 s，Cmax 改善 18.7253%；主要可观测变化是 WAIT/route geometry 的组合，而非单纯全局 process spread 改善。I6/I9 的长焊缝 best-X 无 WAIT 改变，spread 均减少 597.222 s，empty 分别减少 35.253/26.784 s，符合 load sharing+route geometry 的解释。I3/I4 主要表现为 WAIT 降低。I12 的 WAIT 增加 323.470 s，但 empty 降低 229.814 s、spread 降低 100.452 s，makespan robot WAIT 未变；不能用总 WAIT 单指标判定收益。

50 s 额外 pre/post 已计入全部候选；301 个 certified-feasible patterns 中 116（38.54%）正收益，185 不优于 common seed。238 个 pattern 没有在有限 shortlist 内得到 certified schedule。不能把这些负结果全归因于额外 50 s，因没有工艺开销消融；只能说额外开销不会普遍阻断收益，但许多合法候选仍无收益。

1079/2153 次 reference 为 DEADLOCK（50.12%）。这显示协调/政策调度困难，但没有 shared-point 专项因果消融，不能断言 shared split point 是主要负面因素。1074 certified FEASIBLE 调度和长焊缝收益均遵守原 collision/non-passing 规则，无 shared-point exception。

## 6. Top 20 parents by blocking_proxy

排序范围为原 12 实例的全部 parents，并非只排序可 X split parents；此表不参与候选选择。delta Cmax<0 为改善；“无 certified X”仅指本 bounded audit 未得到证据，不是数学不可行。

| 排名 | 实例 | parent_id | length / x_span | whole p_i | blocking_proxy | X patterns | best-X certified FEASIBLE | delta Cmax | 历史随机访问 | 历史最终保留 |
|---:|---|---|---:|---:|---:|---:|---|---:|---|---|
| 1 | I9 | inst0042\|group=144-DK1B\|row=0000 | 12.500 / 12.500 | 1207.407 | 15092.593 | 3 | PASS | -624.006 | UNKNOWN | 无 |
| 2 | I5 | inst0001\|group=144-DK1B\|row=0000 | 12.500 / 12.500 | 1207.407 | 15092.593 | 3 | PASS | -806.224 | UNKNOWN | 无 |
| 3 | I2 | inst0009\|group=144-DK1B.dxf\|row=0000 | 12.500 / 12.500 | 1207.407 | 15092.593 | 3 | PASS | 620.190 | UNKNOWN | 无 |
| 4 | I6 | inst0006\|group=144-DK1B\|row=0000 | 12.500 / 12.500 | 1207.407 | 15092.593 | 3 | PASS | -632.476 | UNKNOWN | 无 |
| 5 | I5 | inst0018\|group=804-TB310A.dxf\|row=0000 | 5.254 / 5.254 | 536.481 | 2818.674 | 4 | 无 certified X | — | UNKNOWN | 无 |
| 6 | I5 | inst0018\|group=804-TB310A.dxf\|row=0003 | 5.244 / 5.244 | 535.556 | 2808.453 | 4 | 无 certified X | — | UNKNOWN | 无 |
| 7 | I5 | inst0018\|group=804-TB310A.dxf\|row=0007 | 5.244 / 5.244 | 535.556 | 2808.453 | 1 | 无 certified X | — | UNKNOWN | 无 |
| 8 | I5 | inst0018\|group=804-TB310A.dxf\|row=0002 | 5.094 / 5.094 | 521.667 | 2657.370 | 3 | 无 certified X | — | UNKNOWN | 无 |
| 9 | I5 | inst0018\|group=804-TB310A.dxf\|row=0005 | 5.094 / 5.094 | 521.667 | 2657.370 | 3 | 无 certified X | — | UNKNOWN | 无 |
| 10 | I3 | inst0012\|group=888-BK304A.dxf\|row=0002 | 5.375 / 4.704 | 547.688 | 2576.466 | 1 | PASS | -96.083 | UNKNOWN | 无 |
| 11 | I9 | inst0029\|group=888-BK304A.dxf\|row=0002 | 5.375 / 4.704 | 547.688 | 2576.466 | 1 | PASS | 0.000 | UNKNOWN | 无 |
| 12 | I8 | inst0004\|group=888-BK304A.dxf\|row=0002 | 5.375 / 4.704 | 547.688 | 2576.466 | 1 | PASS | 0.000 | UNKNOWN | 无 |
| 13 | I10 | inst0016\|group=143-GR14A\|row=0000 | 4.213 / 4.213 | 440.136 | 1854.499 | 1 | 无 certified X | — | UNKNOWN | 无 |
| 14 | I11 | inst0010\|group=143-GR14A\|row=0000 | 4.213 / 4.213 | 440.136 | 1854.499 | 4 | 无 certified X | — | UNKNOWN | 无 |
| 15 | I6 | inst0002\|group=143-GR14A\|row=0000 | 4.213 / 4.213 | 440.136 | 1854.499 | 1 | PASS | 0.000 | UNKNOWN | 无 |
| 16 | I4 | inst0000\|group=143-GR14A\|row=0000 | 4.213 / 4.213 | 440.136 | 1854.499 | 7 | PASS | 0.000 | UNKNOWN | 无 |
| 17 | I12 | inst0001\|group=888-BK304A.dxf\|row=0002 | 5.375 / 2.600 | 547.688 | 1424.082 | 0 | 无 certified X | — | UNKNOWN | 无 |
| 18 | I7 | inst0020\|group=145-LB17A.dxf\|row=0003 | 3.130 / 3.130 | 339.769 | 1063.306 | 4 | PASS | 129.537 | UNKNOWN | 无 |
| 19 | I7 | inst0020\|group=145-LB17A.dxf\|row=0004 | 3.130 / 3.130 | 339.769 | 1063.306 | 4 | PASS | 129.537 | UNKNOWN | 无 |
| 20 | I12 | inst0014\|group=145-LB17A.dxf\|row=0003 | 3.129 / 3.129 | 339.769 | 1063.306 | 1 | 无 certified X | — | UNKNOWN | 无 |

以上每行的 UNKNOWN 是 UNKNOWN_NOT_RECORDED，不是 NO：旧 artifact 没有逐 parent/pattern generation、DP 或 reference-access 日志，不能根据最终没保留而倒推从没访问。“无”只表示历史 72 个 run 的 chosen_x_patterns 没有该 parent。具体 best-X pattern identity 保存于 artifact 的 top20_parents_by_blocking_proxy。

12 个 instance-level best pattern 均未出现在历史 final-retained patterns 中；其历史实际访问仍未知。只能支持“原随机访问未建立稳定收益、系统访问建立了收益”的证据链，不能证明每个好 pattern 从未被随机生成或 reference 评价。

### 6.1 x_span>=10 m 描述组

只有 4 个 instance-parent 组合，均为 144-DK1B、length=x_span=12.5 m、whole processing=1207.407 s、blocking_proxy=15092.593 s·m，每个 3 个冻结 patterns。全部完成系统访问且均有 certified FEASIBLE best-X：

- I5: BX_CENTER，delta Cmax=−806.224 s，改善 18.7253%。
- I6: BX_LOWER，delta Cmax=−632.476 s，改善 8.8161%。
- I9: BX_LOWER，delta Cmax=−624.006 s，改善 6.3608%。
- I2: BX_UPPER，delta Cmax=+620.190 s，恶化 37.8360%。

因此，确有一个长 X-span parent 系统审计后仍无收益，但其余三个明显获益；不能概括为“真正长焊缝 X 仍没有价值”。10 m 仅描述，没有新增 legality threshold。

## 7. Stage B：已知好 X state 的 retention

Stage A PRESENT 后，对全部 11 个 positive best-X seed 的实例，每个 seeds=20261004/20261005/20261006、两个 60 s arms，共 66 runs、33 pairs。I2 不适用，仍保留在原 12 实例审计范围。

NO_X_FROM_COMMON 用原 common seed、V1.1、X disabled；FINITE_X_FROM_BEST_X 用冻结 certified X seed、experimental finite-X scope、X enabled。M=64、Kdp=8、Kref=2、Kref_total=4、direction_refinement_budget=4、SA、LNS 和全部 SearchConfig 相同；TWO_OPT_STAR OFF。未调整任何生产搜索参数或行为。

66/66 final schedules certified，numeric failures=0，termination=TIME_LIMIT。实际 search runtime 合计 4066.651 s（目标 3960 s）；overshoot 中位 1.393 s、最大 6.669 s。B32 正常 overshoot 允许完成，**判据只使用 deadline-safe Cmax@60，不用 overshoot 后 final Cmax 替代**。

6 个 run 的 final Cmax 在超时完成阶段继续改善，与 Cmax@60 不同；最大差值 588.242 s（I9 的 NO_X、seed 20261006）。artifact 同时保留二者，本表配对只用 @60。最终 X 保留指标使用实际 final certified best solution，按预注册规则独立记录。

### 7.1 冻结 retention 支持判定

| ID | X-seeded wins | paired median improvement @60 | final X 存在 | workbook 支持条件 |
|---|---:|---:|---|---|
| I3 | 2/3 | 0.6791% | 无 | 不支持 |
| I10 | 1/3 | -0.1563% | 有 | 不支持 |
| I11 | 1/3 | -5.2273% | 有 | 不支持 |
| I9 | 1/3 | -0.0844% | 有 | 不支持 |
| I12 | 3/3 | 1.7696% | 有 | 支持 |
| I1 | 0/3 | -0.9922% | 有 | 不支持 |
| I4 | 1/3 | -1.0499% | 有 | 不支持 |
| I5 | 3/3 | 19.7729% | 有 | 支持 |
| I7 | 3/3 | 0.9142% | 有 | 不支持 |
| I6 | 1/3 | -4.7674% | 有 | 不支持 |
| I8 | 2/3 | 0.1711% | 有 | 不支持 |

仅 I5、I12 同时满足 >=2/3 wins 和 paired median>=1%；两个都是不同 workbooks，且均真实保留 X。因此 X_SEARCH_RETENTION=ADEQUATE，CASE 3。

I5 的 median=19.7729%，3/3 wins，seed 后 global-best updates 为 3/2/4；存在进一步利用收益的证据。I12 的 median=1.7696%，3/3 wins，但三次 X arm 的 global-best updates 均为 0、Cmax 保持 5331.614 s：这是好初始 X state 的保存证据，不应说成进一步优化证据。I7 的 0.9142% 不足 1%，未算支持。

X arm final 保留原 best-X pattern 为 26/33，removed=7/33，switched=0。27/33 X arm 有起始后 global-best updates；没有把无 X 的最终优势当作 X retained。

### 7.2 每个 instance×seed 的配对 @60

| ID | seed | NO_X Cmax@60 | X-seeded Cmax@60 | improvement | final 含 X |
|---|---:|---:|---:|---:|---|
| I3 | 20261004 | 1827.027 | 1791.675 | 1.9349% | 无 |
| I3 | 20261005 | 1787.766 | 1775.624 | 0.6791% | 无 |
| I3 | 20261006 | 1769.724 | 1772.819 | -0.1749% | 无 |
| I10 | 20261004 | 4465.553 | 4398.624 | 1.4988% | 无 |
| I10 | 20261005 | 4403.129 | 4410.013 | -0.1563% | 无 |
| I10 | 20261006 | 4377.645 | 4465.553 | -2.0081% | 有 |
| I11 | 20261004 | 5295.208 | 5761.003 | -8.7965% | 有 |
| I11 | 20261005 | 6193.095 | 6516.829 | -5.2273% | 有 |
| I11 | 20261006 | 5744.959 | 5454.617 | 5.0539% | 有 |
| I9 | 20261004 | 8437.186 | 8425.298 | 0.1409% | 有 |
| I9 | 20261005 | 7938.267 | 7944.967 | -0.0844% | 有 |
| I9 | 20261006 | 5721.171 | 7300.264 | -27.6009% | 有 |
| I12 | 20261004 | 5427.661 | 5331.614 | 1.7696% | 有 |
| I12 | 20261005 | 5427.661 | 5331.614 | 1.7696% | 有 |
| I12 | 20261006 | 5357.187 | 5331.614 | 0.4773% | 有 |
| I1 | 20261004 | 1478.299 | 1501.756 | -1.5868% | 有 |
| I1 | 20261005 | 1487.608 | 1501.756 | -0.9511% | 有 |
| I1 | 20261006 | 1489.446 | 1504.225 | -0.9922% | 有 |
| I4 | 20261004 | 1777.403 | 1885.536 | -6.0838% | 有 |
| I4 | 20261005 | 1772.307 | 1748.124 | 1.3645% | 无 |
| I4 | 20261006 | 1760.821 | 1779.309 | -1.0499% | 无 |
| I5 | 20261004 | 3679.581 | 3420.647 | 7.0370% | 有 |
| I5 | 20261005 | 4074.626 | 3268.955 | 19.7729% | 有 |
| I5 | 20261006 | 4305.537 | 2963.450 | 31.1712% | 有 |
| I7 | 20261004 | 3712.330 | 3454.606 | 6.9424% | 有 |
| I7 | 20261005 | 3594.897 | 3570.660 | 0.6742% | 有 |
| I7 | 20261006 | 3605.805 | 3572.839 | 0.9142% | 有 |
| I6 | 20261004 | 5966.560 | 6541.654 | -9.6386% | 有 |
| I6 | 20261005 | 6243.978 | 6541.654 | -4.7674% | 有 |
| I6 | 20261006 | 5801.233 | 5119.532 | 11.7510% | 有 |
| I8 | 20261004 | 3534.331 | 3559.976 | -0.7256% | 有 |
| I8 | 20261005 | 3566.077 | 3559.976 | 0.1711% | 有 |
| I8 | 20261006 | 3566.077 | 3559.976 | 0.1711% | 有 |

### 7.3 X arm final parent/t、retention、updates 与两种时间口径

search_only 字段是完整该 arm 的实际 runtime，目标为 60 s，含正常 overshoot；科学比较仍用 Cmax@60。audit_cost_inclusive=该实例完整 Stage A pattern 审计成本+该 arm runtime，是一次实例级诊断程序的成本口径，不代表每 seed 重新构造全部候选。一次构造成本可供三个 seeds 复用，不能把表中 33 行的 inclusive 值相加当成实际墙钟成本。

artifact 对两个 arms 都保存“该实例共同诊断窗口”的 audit_cost_inclusive；**不能用这两个同时加相同 audit 成本的字段宣称 X 的 seed 是免费的或作公平算法速度比较**。若按独立算法部署口径收费，NO_X 只付原 common 初始化+自身 search，X arm 必须额外付下面单列的 Stage A audit cost；本轮不做该公平 benchmark。

| ID | Stage A X-seed audit cost s | X arm 目标 search 60 s + audit cost s |
|---|---:|---:|
| I1 | 7.264 | 67.264 |
| I2 | 16.053 | NOT_APPLICABLE |
| I3 | 13.264 | 73.264 |
| I4 | 5.654 | 65.654 |
| I5 | 75.026 | 135.026 |
| I6 | 20.524 | 80.524 |
| I7 | 41.354 | 101.354 |
| I8 | 41.088 | 101.088 |
| I9 | 50.434 | 110.434 |
| I10 | 340.617 | 400.617 |
| I11 | 165.698 | 225.698 |
| I12 | 191.545 | 251.545 |

全量首遍 Stage A=968.522 s，一次性成本；Stage B search=4066.651 s，两者合计 5035.173 s（不含 common-seed 重建、seed 回放、artifact persistence 和独立确定性复跑）。这远非只运行 60 s，不能将所得 X seed 当作公平 benchmark 的零成本 warm start。common_seed_construction_seconds 在每个 instance summary 另记，独立复跑是验证成本，不是 Stage A 首遍预算。

下表 X-seeded arm 的 inclusive 使用实例完整 audit cost；final parent/t 对应实际 certified final solution，best-X seed 的方向完整保存于 Stage A evidence；Stage B 另存 final solution hash 与 parent/rail/t。

| ID | seed | retention | final X parent / rail / t | global-best updates after start | search_only actual s | audit_cost_inclusive s |
|---|---:|---|---|---:|---:|---:|
| I1 | 20261004 | RETAINED | inst0004\|group=804-FR315A.dxf\|row=0002 / UPPER / t=0.736451 | 7 | 60.644 | 67.908 |
| I1 | 20261005 | RETAINED | inst0004\|group=804-FR315A.dxf\|row=0002 / UPPER / t=0.736451 | 11 | 60.443 | 67.707 |
| I1 | 20261006 | RETAINED | inst0004\|group=804-FR315A.dxf\|row=0002 / UPPER / t=0.736451 | 14 | 60.882 | 68.146 |
| I3 | 20261004 | REMOVED | 无 | 11 | 61.075 | 74.339 |
| I3 | 20261005 | REMOVED | 无 | 11 | 60.093 | 73.357 |
| I3 | 20261006 | REMOVED | 无 | 11 | 60.848 | 74.112 |
| I4 | 20261004 | RETAINED | inst0005\|group=143-FR69B.dxf\|row=0002 / UPPER / t=0.500000 | 3 | 60.209 | 65.863 |
| I4 | 20261005 | REMOVED | 无 | 10 | 60.052 | 65.706 |
| I4 | 20261006 | REMOVED | 无 | 7 | 60.164 | 65.818 |
| I5 | 20261004 | RETAINED | inst0001\|group=144-DK1B\|row=0000 / UPPER / t=0.500000 | 3 | 61.269 | 136.295 |
| I5 | 20261005 | RETAINED | inst0001\|group=144-DK1B\|row=0000 / UPPER / t=0.500000 | 2 | 63.261 | 138.287 |
| I5 | 20261006 | RETAINED | inst0001\|group=144-DK1B\|row=0000 / UPPER / t=0.500000 | 4 | 60.431 | 135.457 |
| I6 | 20261004 | RETAINED | inst0006\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 0 | 60.905 | 81.428 |
| I6 | 20261005 | RETAINED | inst0006\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 0 | 61.823 | 82.347 |
| I6 | 20261006 | RETAINED | inst0006\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 1 | 61.734 | 82.257 |
| I7 | 20261004 | RETAINED | inst0009\|group=312-GR2C\|row=0000 / UPPER / t=0.500000 | 3 | 61.596 | 102.950 |
| I7 | 20261005 | RETAINED | inst0009\|group=312-GR2C\|row=0000 / UPPER / t=0.500000 | 1 | 62.607 | 103.961 |
| I7 | 20261006 | RETAINED | inst0009\|group=312-GR2C\|row=0000 / UPPER / t=0.500000 | 6 | 61.588 | 102.942 |
| I8 | 20261004 | RETAINED | inst0014\|group=804-FR315A\|row=0000 / LOWER / t=0.500000 | 7 | 66.669 | 107.758 |
| I8 | 20261005 | RETAINED | inst0014\|group=804-FR315A\|row=0000 / LOWER / t=0.500000 | 6 | 61.527 | 102.615 |
| I8 | 20261006 | RETAINED | inst0014\|group=804-FR315A\|row=0000 / LOWER / t=0.500000 | 7 | 61.504 | 102.592 |
| I9 | 20261004 | RETAINED | inst0042\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 1 | 64.010 | 114.444 |
| I9 | 20261005 | RETAINED | inst0042\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 1 | 63.796 | 114.230 |
| I9 | 20261006 | RETAINED | inst0042\|group=144-DK1B\|row=0000 / UPPER / t=0.516000 | 2 | 63.524 | 113.958 |
| I10 | 20261004 | REMOVED | 无 | 6 | 60.102 | 400.719 |
| I10 | 20261005 | REMOVED | 无 | 11 | 61.047 | 401.664 |
| I10 | 20261006 | RETAINED | inst0009\|group=642-BL14A.dxf\|row=0000 / LOWER / t=0.500000 | 2 | 61.623 | 402.240 |
| I11 | 20261004 | RETAINED | inst0008\|group=143-FR76A.dxf\|row=0001 / UPPER / t=0.500000 | 2 | 60.660 | 226.358 |
| I11 | 20261005 | RETAINED | inst0008\|group=143-FR76A.dxf\|row=0001 / UPPER / t=0.500000 | 0 | 60.506 | 226.204 |
| I11 | 20261006 | RETAINED | inst0008\|group=143-FR76A.dxf\|row=0001 / UPPER / t=0.500000 | 9 | 62.126 | 227.824 |
| I12 | 20261004 | RETAINED | inst0010\|group=801-FR322A.dxf\|row=0001 / LOWER / t=0.500000 | 0 | 63.359 | 254.904 |
| I12 | 20261005 | RETAINED | inst0010\|group=801-FR322A.dxf\|row=0001 / LOWER / t=0.500000 | 0 | 64.483 | 256.028 |
| I12 | 20261006 | RETAINED | inst0010\|group=801-FR322A.dxf\|row=0001 / LOWER / t=0.500000 | 0 | 62.258 | 253.803 |

## 8. 确定性、测试与历史保护核对

独立第二遍完整重算 539 patterns，cache disabled，另执行 2153 次 reference evaluation；全部 pattern scientific payload 逐行一致。`determinism_verification.match=true`，两遍科学 hash 均为 `6907f896c1ba461b4bad8c01fc677b161d7a79598fc9038516ec1a9cbfeef267`。首遍 2153 次 + 独立验证 2153 次 = 4306 次 candidate reference calls，分别属于实验和复现验证，不把两遍混作首遍预算。

科学 payload 明确剔除所有 *_seconds、cache、scientific_hash 字段；未剔除 pattern identity、cheap rank、shortlist、方向、reference status、certification、Cmax、canonical solution/hash、recovery usage 或 official metrics。Stage A 没有 solver seeds；Stage B 是固定 seeds 的 wallclock-limited stochastic search，不能宣称整个包含 Stage B 的 raw artifact 字节级 deterministic。

基线完整回归：237 passed in 32.52 s。新增 14 个用例后：251 passed in 69.47 s。最终命令 `pytest -q -p no:cacheprovider`：251 passed in 90.20 s；未删除旧 tests。最终测试与确定性复跑并行执行，但均在全部 Stage B 搜索结束后，不改变配对搜索运行预算；复跑墙钟不用于科学判据。

新增测试覆盖：冻结 539 census/hash 与 common seed、移除 WHOLE 后完整插入乘积、无 random insertion、左右 robot 固定、全部有效 pair cheap-scored、top16/top4 与 canonical tie-break、正式访问和 <=4 cap、common seed 不变、serialization roundtrip、V1.1 reject X/experimental accept、cache 输出不变、角色 TRAIN_POOL/VALIDATION/ID_TEST 拒绝、Stage A 精确阈值和 N/span/two-workbook 条件、Stage B trigger 和 2/3/median 条件、四类 decision、certifier/numeric/scope mismatch FAIL、resume identity/cap 检查。

最终再次核对：历史 Phase 3-X artifact 文件 SHA-256 仍为 `854183d399c11240a42577e2c86c7a5ad838e249b788eae79cc57280dc6c3b25`；scientific source-tree hash 仍为 `e6fd58b6c56e53866c19760403a9094cdd99b6dc4b794b2413e78003d1681eaf`；12 common seed hashes/Cmax 未变。历史 Gate 的 NOT_SUPPORTED / WEAK / V2 NO 仍原样，ACTIVE 仍 V1.1。PPO workbook 内容在重新加载阶段按冻结 manifest 校验；本轮未写 PPO 或 references。

上述新 hashes 是本轮明确要求的 census/protocol/evidence reproducibility identity，并非新增泛化 gate 或用前置检查替代模拟。本轮实际完成全部插入枚举、方向评估、reference 调度和独立 certification。

## 9. 16 个核心 handoff 问题

1. **539 个冻结 X patterns 是否全部系统访问？** 是，539/539，全量 cheap insertion enumeration，每个都获得正式 reference opportunity。
2. **多少 patterns 有 direction-feasible insertion？** 539。8623 是 top16 内实际 DP 调用总数，不代表完整插入域的方向可行解总数。
3. **多少 patterns 产生 certified FEASIBLE？** 301；候选调度层为 1074 FEASIBLE/PASS。
4. **多少 patterns one-step 优于 common seed？** 116。
5. **多少 patterns 改善 >=1%？** 63，阈值未改。
6. **来自多少不同 workbooks？** 正收益来自 11；>=1% 来自 8；Stage B 完整支持来自 2。
7. **MIDPOINT 还是 Bx？** 正收益 MIDPOINT=90/Bx=26；>=1% MIDPOINT=39/Bx=24。8 个 >=1% instance best 为 MIDPOINT=5/Bx=3；两类都有贡献，长焊缝 best 来源为 Bx。
8. **收益 pattern parent 的 length/x_span/p_i/blocking_proxy 如何？** 正收益 pattern-weighted 中位分别 1.249 m/1.042 m/165.684 s/158.497 s·m；范围见 4.2，instance best 和 top20 另列，未据此筛选。
9. **真正长 X-span 是否被审计且仍无收益？** 四个 12.5 m parent 全审计；一个变差、三个有 6.36%–18.73% 收益，不能概括为长跨度全无收益。
10. **历史随机搜索曾访问这些 best patterns 吗？** UNKNOWN_NOT_RECORDED；12 个 instance-best 均没有历史 final retention，不能据此判未访问。
11. **若没有，是否是 access 问题？** 不能前提化“没有访问”。本轮系统访问建立收益、原 stochastic Gate 未建立，支持把 domain potential 与 search-access/retention 分开；Stage B ADEQUATE 也不证明原搜索已稳定访问所有好 pattern。
12. **best-X 主要来自 spread/empty/WAIT 哪项？** 实例不同：I5 显著 WAIT+empty 改善；I6/I9 load sharing+empty；I3/I4 WAIT；I12 即使总 WAIT 上升仍因 route/spread 改善降低 makespan。是观测性分解，不是因果消融。
13. **50 s 额外开销是否通常吞掉收益？** 38.54% 的 feasible patterns 仍正收益，185 个不改善；开销、route 和协调变化未做独立消融，不能将负结果全归因于 50 s，也不能说该成本可忽略。
14. **shared split point 是主要负面因素吗？** 无法因果确认。50.12% reference DEADLOCK，但没有按 shared-point 原因做独立归因；合法 certified 收益无需 exception。
15. **现有 ALNS 能保留好 X seed 吗？** 26/33 final 保留；I5/I12 满足预注册 workbook-level 支持。I5 进一步改善，I12 保存种子但无新 global-best updates；总体 ADEQUATE，不是全实例稳定优势。
16. **最终 V1.1 还是 V2？** CASE 3，V2 authorized 进入 Phase 3-Y；实际 ACTIVE 仍 V1.1，不能提前宣称 V2 FINAL。

## 10. X 方向收口与下一阶段

冻结当前 finite-X scientific definition 的 domain potential 和 seeded retention 已建立，本轮不再添加切点、X-specific operators、Gate instances 或调整 1%/N/span/budget。不是继续试到成功，而是按冻结 Stage A/Stage B 规则结束诊断；禁止把本轮当全局数学证明或正式算法 superiority evidence。

NEXT：Phase 3-Y — FORMAL_SCOPE_V2 Core Closure + common-domain HGA/WAG/ALNS adaptation。下一阶段才使三个算法面对相同 WHOLE/Y/X domain；candidate-pool audit 必须等 V2 domain 闭合，不在本轮混入。未来 candidate ranking 必须显式涵盖合法 X candidates；不继续手工追加 X-specific operator。

数据治理：旧 Phase 3 的 15 VALIDATION workbooks 对未来 V2 的科学角色记录为 **V2_MODEL_DEVELOPMENT_CONSUMED**，不能声称 untouched V2 validation；本轮仅记录该使用历史，**不修改现有分区 manifest、不重新划数据、不访问 TRAIN_POOL**。Phase 3-Y 在任何 V2 solver result 之前，从尚未用于 V2 性能开发的 TRAIN_POOL solver-independent 冻结新的 V2_VALIDATION。原 15 ID_TEST 继续 SEALED。

历史 V1.1、Phase 3 VALIDATION 和 Phase 3-X negative Gate 原样保留；ACTIVE/production scope 不变，直到 Phase 3-Y core closure 获得实际闭合证据。

## 11. 本地复现

以下均在 DRL 目录执行，不会上传 GitHub；已有 protocol 不允许因结果变化重新冻结：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B scripts\run_phase3x2_x_oracle_audit.py --freeze-protocol --run
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B scripts\run_phase3x2_x_oracle_audit.py --verify-determinism
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider
```

已完成的 audit artifact 由 --run 按冻结身份继续读取，不替换历史 Phase 3-X 结果。--verify-determinism 独立重算完整 Stage A 并更新既有 artifact 的 verification 字段；不要与仍在写同一个 artifact 的 Stage B 同时运行。

本轮文件索引：[protocol](../data/manifests/PHASE3X2_X_ORACLE_PROTOCOL_V1.json)、[完整 evidence artifact](../data/development/phase3x2_x_oracle_audit_v1.json)、[独立 runner](../scripts/run_phase3x2_x_oracle_audit.py)、[现有测试文件](../tests/test_phase3_validation.py)。

