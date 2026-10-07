# Phase 3-ZR — Reference Runtime 与 Certified-Incumbent 审计

执行日期：2026-10-06 至 2026-10-07（Asia/Shanghai）。文件名沿用本轮开始日期及用户批准的路径。

本轮预注册 DEVELOPMENT runtime gate 通过，按任务第 63 节关闭为 `CLOSED_WITHOUT_CODE_CHANGE`。HGA 为 37/40 certified，分类 `INTERMITTENT_NO_CERTIFIED`。Phase 4-0 获得执行授权，本轮未执行 candidate-pool/oracle audit、标签生成或训练。历史 Phase 3-Z 保持 FAIL / 179 certified，未改写。

这里的关闭仅适用于预注册开发样本及本机测量。历史 Z11 的 746.78 秒长尾没有复现，历史内部耗时归因仍无法精确恢复；本轮不能声称已修复该历史长尾或获得任意输入的硬实时保证。

## 1. 来源、冻结范围与执行边界

- 当前项目目录：`D:\pybullet_test\MRTA_GA\DRL`。科学测量时曾使用 `DRL\workspaces\phase3z-validation` 独立副本；现按用户要求将 7 个新增文件、README、两个测试文件归并到 DRL 主目录，清理重复副本和本轮额外 Git 记录。用户已有的独立 Git 副本不动；原始运行记录保留当时来源，不回填路径。
- 测量时 remote：`https://github.com/luckyfishanddog/DRL.git`；分支：`codex/phase3zr-runtime-audit`。无 push、PR、上传操作。
- 初始本地 HEAD：`43a004c2b40382110b642fa009b7a30ff124470b`。用户提供的远程 main `94ea6c1ab081348891b232b65e5758501cad95e0` 因网络不可用未验证，不宣称本地已对齐该 main。
- 观察层本地提交：`6ce0ce8`（首版 observer）、`fa55a0f`（保留 certifier identity）、`011ae220199b8dbe4c8f4d793a9d66e059955da4`（最终机械判定）。每条运行记录保留实际 commit、dirty 状态，不将整个批次伪装成单一 clean HEAD。
- 迁移前交接 HEAD：`fa0705e3937bd3636d03323847c549f533407d52`（仅 README 更新），当时 verified=true、worktree_dirty=false；科学源树 identity 与测量时一致。
- 全轮科学源树 identity：`533372e63cef39a1ee967b8cae8afa9dac3b1a259da08f6199d6fc0943f332a7`。未修改 `src/` 科学实现，未优化 scheduler，未修改算法配置、X/Y、物理、B32、branch/frontier、ESS、tie-break 或 tolerance。
- `FORMAL_SCOPE_V2.scope_hash = 16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599`。
- `reference_implementation_id = PRE_ZR_B32_43a004c_OBSERVER_ONLY`，其字符串标识观察路径，未创建新的科学 policy。
- 解释器：`D:\pybullet_test\.venv\Scripts\python.exe`。该 venv 原有 editable install 指向外层 DRL；runner 在导入科学模块前明确选择本独立仓库的 `src`，pytest 显式设置该仓库的 `PYTHONPATH`。
- Excel 只从既有 `D:\pybullet_test\MRTA_GA\ppo` 读取，未复制或生成新 Excel。

只新增用户批准的 7 个永久文件：runner、两个 manifests、三个 development JSON、本报告。测试复用 `tests/test_baseline_common.py` 和 `tests/test_scheduler_equivalence.py`；README 最小更新。写入器仅允许本轮 5 个 JSON 路径，原子写入的临时文件随完成清理。历史 JSON 作为只读证据使用，最终与既有未触碰的 handoff 副本逐字节相等。未新增额外 gate、baseline、contract、hash 机制；protocol/runtime/候选 identity 使用本任务明确要求的既有摘要方式。

## 2. 预注册、静态选择与数据隔离

协议于 `2026-10-06T23:55:36.236471+08:00` 写入，Stage A 于 `23:58:38` 开始，`2026-10-07T01:27:55` 完成。协议与 runtime set 不允许由 runner 重新 prepare 覆盖。

- Protocol hash：`00379447601705e48de54775f72979dc0f187bbb9c522595e217931c18b5a262`。
- Runtime set hash：`e6dd978e799520ffab2ca110c74915878d336bda4467d63a5ea552d75e3f1a2e`。
- Stage A：12 不同 workbooks × 3 方法 × seeds `20261021/20261022`，每次 native 60 秒，初始化计入时钟。
- HGA：8 LARGE × seeds `20261021` 至 `20261025`，复用同一科学 identity 的 16 条 Stage A 记录，另执行 24 条。
- 每 workbook 按首选 N 区间、离目标 N 的距离、geometry hash 选代表；LARGE 取 24 个最近候选，MEDIUM 在剩余 workbooks 取 12 个最近候选。随后对 12 个静态几何描述维度归一化，deterministic farthest-first 选择，距离并列按 geometry hash。未读取 solver 表现、Phase 3-Z Cmax、WAIT、deadlock rate 或 Z11 相似度。
- role metadata 校验在 Excel 加载之前。所有普通运行均为 `V2_MODEL_DEVELOPMENT_CONSUMED`；历史物理文件夹名 `ID_TEST/VALIDATION/PPO_TRAIN` 不覆盖 V2 workbook role。

| Tier | N | Instance ID |
|---|---:|---|
| LARGE | 85 | data/PPO_TRAIN/seed_0211781140::g32_w085 |
| LARGE | 75 | data/PPO_TRAIN/seed_0767803876::g45_w075 |
| LARGE | 79 | data/PPO_TRAIN/seed_1916546866::g45_w079 |
| LARGE | 85 | data/ID_TEST/seed_1411387107::g28_w085 |
| LARGE | 77 | data/PPO_TRAIN/seed_2091884649::g45_w077 |
| LARGE | 85 | data/PPO_TRAIN/seed_0194123089::g43_w085 |
| LARGE | 79 | data/VALIDATION/seed_2145450783::g45_w079 |
| LARGE | 76 | data/PPO_TRAIN/seed_0088690758::g45_w076 |
| MEDIUM | 55 | data/VALIDATION/seed_0967455456::g27_w055 |
| MEDIUM | 55 | data/ID_TEST/seed_2013829876::g37_w055 |
| MEDIUM | 55 | data/PPO_TRAIN/seed_0463639632::g25_w055 |
| MEDIUM | 55 | data/PPO_TRAIN/seed_0817833068::g22_w055 |

唯一 validation 访问是完成开发 A/B 和 HGA 审计之后，专用入口读取冻结 Z11/HGA/20261015。该结果只写入 ZR JSON 的 `external_frozen_stress`，未写入 Phase 3-Z、未用于调参或正式 Cmax@60。

## 3. Observer 与执行中修正

LEVEL 1 包装现有调用并返回同一个 scheduler result；记录候选、方向、状态、native 时间、四段耗时及现有 recovery accounting。完整候选序列化和 JSON 落盘在 native solver 时钟外。准备、baseline、recovery 直接计时；packaging 为总时长扣除前三段的剩余值，含包装和 observer 的微小残余开销。LEVEL 2 仅在冻结 slow subset 中启用既有 SchedulerProfile 和每 rollout 计时。前三条另开 cProfile，数据嵌入既有 JSON；全部 LEVEL 2 时间排除在正式阈值之外。

首版 observer 包装了 certifier 函数，触发已有 ALNS 初始化的函数 identity 安全检查。24 次 ALNS 启动在方向/reference 科学评估前失败，reference calls 全为 0。这是本轮 observer 实现错误。修正为读取返回后的 certification accounting，保留原 certifier identity 及安全检查，并增加真实 ALNS 初始化回归测试。

全部 24 条失败尝试保存在 `observer_startup_attempts`，未删除。48 条已经正常完成的 HGA/WAG 科学运行继续保留；修正后只重新执行 24 条尚未进入科学评估的 ALNS 启动。最终是 **72 条科学运行 + 24 条明确隔离的 observer 启动失败尝试**，共 96 次 invocation attempts。没有重试任何合法 no-certified outcome 或 numeric failure。HGA/WAG 的首版 certifier wrapper 仅透传原结果；其科学源树与算法配置和修正后的 ALNS 相同，但实际 observer 版本不同，均逐条记录。

因此 Stage A `execution_failure=0` 指完成的 72 条科学 corpus，而非声称 observer 开发全过程从未失败。报告和 artifact 均保留这一限制。

## 4. Stage A 实测

72 条科学运行全部完成：70 `CERTIFIED_INCUMBENT`、2 `NO_CERTIFIED_INCUMBENT`，numeric failure、certifier mismatch、runner exception 均为 0。两条 no-certified 为 `g45_w079 / seed_1916546866` 的 HGA seeds 21/22，未补解。共 4,742 条 reference trace，每条含可重建 parents/patterns/routes/directions/config/scope 的 replay payload。

以下均为 LEVEL 1，单位秒；分位数使用有序样本的线性插值。按调用聚合，调用更多的运行权重更大，非 instance 等权结论。

| 分层 | count | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 4742 | 0.158464 | 1.108277 | 1.372721 | 2.100753 | 4.706792 |
| ALNS | 1560 | 0.247537 | 1.046964 | 1.272183 | 1.730303 | 2.665705 |
| HGA | 1492 | 0.391765 | 1.202497 | 1.564730 | 2.275160 | 4.333224 |
| WAG | 1690 | 0.084206 | 1.075748 | 1.324556 | 2.399574 | 4.706792 |
| LARGE | 2315 | 0.401359 | 1.359020 | 1.707757 | 2.657827 | 4.706792 |
| MEDIUM | 2427 | 0.060878 | 0.784889 | 0.978363 | 1.220195 | 2.319006 |
| FEASIBLE | 2513 | 0.058469 | 0.830538 | 1.111685 | 1.713885 | 2.691178 |
| baseline FEASIBLE | 1556 | 0.044846 | 0.069945 | 0.073256 | 0.080486 | 0.217077 |
| recovered FEASIBLE | 957 | 0.262029 | 1.182550 | 1.416858 | 1.962290 | 2.691178 |
| remaining DEADLOCK | 2229 | 0.516489 | 1.277102 | 1.637600 | 2.649948 | 4.706792 |
| X-present | 784 | 0.282334 | 1.095959 | 1.324382 | 1.733951 | 2.319006 |
| Y-present | 4742 | 0.158464 | 1.108277 | 1.372721 | 2.100753 | 4.706792 |
| SMALL / WHOLE-only（各自） | 0 | NULL | NULL | NULL | NULL | NULL |

method × tier 的完整分层同样保存在 audit JSON，空分层不能解释为零耗时。这里没有 SMALL 或 WHOLE-only 的实测覆盖。

| Run 级指标 | count | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| actual runtime | 72 | 60.576118 | 62.652940 | 63.486869 | 65.098681 | 65.583580 |
| overshoot | 72 | 0.576118 | 2.652940 | 3.486869 | 5.098681 | 5.583580 |
| first certified（仅已取得者） | 70 | 0.859629 | 5.097770 | 6.323283 | 17.805011 | 29.364616 |

两条无解运行 first-certified 为 NULL，并非从总体中抹去；上表第三行是有条件时间分布。baseline DEADLOCK 为 3186，recovered 为 957，remaining DEADLOCK 为 2229。

四段累计：preparation 23.216525、baseline dispatch 329.642508、recovery 1620.182929、packaging/residual 6.301072 秒。recovery 占四段总量约 81.9%。开发慢调用主要关联 remaining DEADLOCK 和 recovery，而 baseline FEASIBLE 很快。

## 5. 冻结 slow-call 与深度诊断

top 20 unique calls，加至少 5 FEASIBLE、5 recovered FEASIBLE、5 remaining DEADLOCK（允许重叠），最终冻结 24 个可重放候选。可用 unique 分层分别为 1860 FEASIBLE、705 recovered FEASIBLE、1808 remaining DEADLOCK，数量充足。24/24 完成深度重放，所比较的状态、Cmax、baseline flag、source、rollout/branch/frontier accounting 全部一致；canonical schedule 和选中 rollout/plan 保存在 JSON。

全部 24 条进入 recovery，包含 19 remaining DEADLOCK、5 recovered FEASIBLE。586 个完整 rollout，单调用实际 14–32 次，frontier 提前穷尽时少于 32，未改变 B32 上限。深度 baseline 总量 6.772222、recovery 总量 99.853110 秒；dominant component 为 `B32_RECOVERY`。

前三条 cProfile reference duration 约 14.39、14.61、13.74 秒，而原 LEVEL 1 约 4.71、4.60、4.33 秒，显示 instrumentation 会显著放大计时。这些值不能用于 performance gate。其余 21 条中，单个最慢 rollout 占全部 rollout 耗时比例为 11.6%–37.0%，中位 20.2%。观测是多个完整 rollout 累积并伴随成本差异，不能描述成固定恰好 32 次，也未发现一个足以解释数百秒的独立异常 rollout。

代表性非 cProfile 调用：376 operations、20536 ESS calls、ESS cache 15631 hits / 4905 misses；456016 relevant fixed operation scans；forbidden interval cache 433189 hits / 22827 misses；conflict cache 714533 hits / 37150 misses。ESS 累计 3.2148 秒，forbidden interval 1.2774、conflict 1.0138 秒（嵌套计时，不能相加作为互斥阶段）；state identity 0.0119、frontier 0.0131 秒。26 次完整 rollout 总计 3.6932 秒。

cProfile 可见 ESS、polygon projection、既有 cache key 的 hash/equality 和 dictionary 查询成本。已有精确缓存大量命中，未找到本轮需要修复的实现缺陷；这些热函数本身不证明存在可无条件省略的重复计算。开发 gate 已自然通过，故没有新增 cache、timeout、剪枝或 scheduler 优化。此处不使用 `POLICY_COST_DOMINATED` 声称已证明历史几百秒均为正常 B32 成本。

调用耗时与 N、block count、X count、Y count 的描述性 Pearson r 分别为 0.3344、0.3442、0.0099、0.2719。LARGE 尾部比 MEDIUM 更高，但这些调用在 12 个固定实例及方法内相关，且 recovery 是混杂因素；不能作显著性或因果结论。X count 在本 corpus 没有明显线性信号；所有调用 Y-present，不能作 Y 有无的比较。

无 scheduler 修改，因此 old/new optimized differential、独立 differential artifact 和 POST-OPT 72-run **NOT_APPLICABLE**。不把 observer 的 24/24 重放一致性包装成优化后的 100% 全语义证明；fixture 测试另覆盖完整返回对象、操作/时间/WAIT 等相等性。

## 6. HGA 40-run 审计

40/40 完成，37 certified（92.5%）、3 no-certified，numeric/certifier/exception 为 0。复用 16 条，新增 24 条，不重复计算为 56 次审计。35/40 的初始化候选集合取得 certified incumbent；初始化 reference 次数范围 4–16，逐条保留。

| LARGE instance（简写） | certified / 5 | no-certified seeds |
|---|---:|---|
| seed_0211781140::g32_w085 | 5 | 无 |
| seed_0767803876::g45_w075 | 5 | 无 |
| seed_1916546866::g45_w079 | 2 | 20261021 / 20261022 / 20261023 |
| seed_1411387107::g28_w085 | 5 | 无 |
| seed_2091884649::g45_w077 | 5 | 无 |
| seed_0194123089::g43_w085 | 5 | 无 |
| seed_2145450783::g45_w079 | 5 | 无 |
| seed_0088690758::g45_w076 | 5 | 无 |

reference 共 1728 次，baseline DEADLOCK 1329、recovered 203、remaining DEADLOCK 1126。reference p50/p90/p95/p99/max = 0.685089 / 1.597479 / 2.023517 / 2.974587 / 5.686627 秒；run max 63.713171 秒。37 条已取得 certified 的 first-time p50 0.453471、p95 8.309391、max 59.076334 秒，另外 3 条 NULL。

预注册 32–37/40 对应 `INTERMITTENT_NO_CERTIFIED`。这是有限开发样本中的描述性分类，无统计显著性/泛化可靠率承诺，未据此调整 HGA，也不自动阻止 Phase 4。

## 7. Z11 独立 external forensic

在 DEVELOPMENT A/B 诊断和 HGA 审计完成后执行。完成于 `2026-10-07T02:06:26.260829+08:00`，科学源树与本轮其他运行一致；native HGA、seed `20261015`、requested 60 秒，没有新增 initializer、延长预算或补 seed。

| 指标 | 冻结历史 Phase 3-Z | 本次 external replay |
|---|---:|---:|
| actual runtime（秒） | 746.775917 | 60.222375 |
| overshoot（秒） | 686.775917 | 0.222375 |
| scheduler total（秒） | 727.6306（约） | 37.780165（trace） |
| reference calls | 15 | 18 |
| baseline / remaining DEADLOCK | 15 / 15 | 18 / 18 |
| recovered / numeric failure | 0 / 0 | 0 / 0 |
| final certified | false | false |
| solver outcome（独立解释） | NO_CERTIFIED_INCUMBENT | NO_CERTIFIED_INCUMBENT |

本次保留原 `termination_reason = INITIALIZATION_FAILED`，不根据这个字符串误判 crash。本次 first-certified 为 NULL；未生成正式 Cmax@5/30/60，也没有历史回填。

18 次 reference p50/p90/p95/p99/max = 1.496948 / 4.350476 / 4.749220 / 6.512538 / 6.953368 秒。四段累计 preparation 0.123755、baseline 3.807859、recovery 33.796261、packaging 0.052291 秒，recovery 约 89.5%。实际 rollouts 为 10–32。全部调用耗时依序（秒）：

```text
3.470331, 0.901243, 1.071340, 0.767022, 0.771702, 0.767504,
2.664645, 0.914779, 1.213788, 4.360253, 1.509359, 1.484536,
0.897513, 2.386216, 1.544586, 4.346286, 1.755696, 6.953368
```

历史未保存逐调用 candidate replay/计时，无法精确匹配历史 15 条与本次 18 条，也无法判定原 727 秒来自一个还是多个 outlier。时间预算算法的轨迹可随执行速度、环境和 observer 开销变化，即使科学源代码和 seed 不变也不能用新 trajectory 代替旧结果。这里观察到长尾未复现，而非证明已有代码优化使它缩短。未发现本次 numeric、certifier 或异常恢复问题。

## 8. Future Solver Outcome Policy V1

此策略写在本报告，不另新增 policy 文件；用于后续协议设计，不能追溯修改已 consumed 的 Phase 3-Z。

分别报告 `EXECUTION_INTEGRITY` 和 `SOLVER_OUTCOME`：完整执行、正确 protocol/data role、无 exception、numeric=0、certifier mismatch=0 才有完整执行可信度。合法无 certified incumbent 是预设预算内的 solver outcome；它仍需如实进入总分母。

派生分类按以下优先顺序：runner exception/crash → `EXECUTION_FAILURE`；否则 numeric>0 → `NUMERIC_FAILURE`；否则 final certified → `CERTIFIED_INCUMBENT`；否则 → `NO_CERTIFIED_INCUMBENT`。certifier mismatch 独立作为 integrity failure，不能藏在合法 outcome 里。保留旧 termination_reason，不从字符串推断完整科学意义。

无 certified 的 Cmax 为 NULL，first-certified 为 NULL/预算内未观察到，不填 0、不借其他 seed、不删失败 seed。后续协议需预先规定质量聚合遇到缺失值的处理；同时公布成功率和所有 outcome 分母，不能用成功样本的 Cmax median 假冒完整竞争结果。numeric failure 不能当作正常负训练样本。

明确 requested budget、actual native runtime、overshoot、reference start/end、候选完成时间及严格 deadline 内的 anytime 值。reference 非抢占，60 秒 requested 不等于硬终止；在截止前启动但截止后完成的 candidate 不得倒填 Cmax@60。将尾部和 integrity 按预注册阈值单独检查；不能通过 rollout timeout、少跑 B32 或更换 policy 消除 overshoot。

本輪 HGA 37/40 的 no-certified 分类不阻止 Phase 4，因为 runtime/integrity 检查通过。后续 Phase 4-0 使用 31 `V2_TRAIN_POOL` workbooks 和必要的 development debug subset；不得用 12 个 consumed validation 实例训练，ID_TEST 保持 SEALED。执行新阶段前沿用科学 identity 和新的阶段协议；本轮仅授权，不执行。

## 9. 回归与机械判定

开始时原测试 314 passed；明确修正独立仓库 import 路径后，原测试 314 passed in 59.81s。新增 observer/outcome/isolation 测试后多次完整回归通过（326、328），最终 **329 passed in 61.89s**，returncode=0，完整 stdout/stderr、执行时间在 audit JSON 的 `final_regression` 中。

测试覆盖 LEVEL 1/2 与旧返回对象完全一致、FEASIBLE/recovered/DEADLOCK fixtures、timer 分解、observer 异常 fail-closed 且还原 wrapper、真实 ALNS certifier identity、可重建候选、outcome taxonomy、role 先于数据加载拒绝、forensic 限定 Z11、禁止历史写入、runtime/integrity 失败不能授权。判定器另检查样本 key 与数量完整性。没有删除旧测试。

预注册 reference 阈值 p95≤8 / p99≤30 / max≤60 秒，development run max≤120 秒；实际 1.372721 / 2.100753 / 4.706792，run max 65.583580，均通过。HGA max reference 5.686627、run max 63.713171；numeric/certifier/exception 均为 0。

机械决策的 11 项 checks 全 True：Stage A 完整及 runtime、Stage B 完整及 observer 对照、HGA 完整及 execution/runtime integrity、历史字节未变、最终回归、专用 forensic 完成、development role 隔离。优化条件不适用。本轮执行状态 PASS 不抹去第 3 节公开的 observer 启动失败，也不表示历史 Phase 3-Z release gate 已通过。

```text
PHASE3ZR_EXECUTION_STATUS = PASS
REFERENCE_RUNTIME_STATUS = CLOSED_WITHOUT_CODE_CHANGE
REFERENCE_SEMANTIC_EQUIVALENCE = NOT_APPLICABLE
HGA_CERTIFIED_INCUMBENT_STATUS = INTERMITTENT_NO_CERTIFIED
PHASE3Z_HISTORICAL_STATUS = FAIL_179_OF_180_CERTIFIED
DETERMINISTIC_V2_BACKBONE_STATUS = NOT_EVALUABLE_IN_PHASE3Z
TRADITIONAL_HEURISTIC_TUNING_STATUS = CLOSED
V2_VALIDATION_STATUS = CONSUMED
ID_TEST_STATUS = SEALED
PHASE4_0_AUTHORIZED = YES
NEXT_PHASE = Phase 4-0 — V2 Candidate-Pool Oracle Recall Audit
```

`TRADITIONAL_HEURISTIC_TUNING_STATUS = CLOSED` 表示不再依据 validation 修改 ALNS/HGA/WAG，非 Phase 3-Z 竞争通过。保留 Phase 3-Z 的 `FORMAL_SCOPE_V2_STATUS = BLOCKED` 历史 release decision，与活动 scope 定义、ZR 开发 execution 授权分别解释。没有重算不完整的 12-instance competition median 或 LARGE median。

## 10. Handoff 必答 15 问

1. **历史 727 秒主要 baseline 还是 recovery？** 原记录没有阶段计时，无法直接证明。开发集 recovery 占约 81.9%，本次 Z11 replay 占约 89.5%，定位当前主要成本在 recovery；不能反推历史的精确分配。
2. **历史 15 条是单个极端还是多个累积？** 缺逐调用历史 trace，无法判定。本次 18 条 0.767–6.953 秒，多个 recovery 累积，没有百秒 outlier，未复现历史。
3. **DEVELOPMENT p50/p90/p95/p99/max？** 0.158464 / 1.108277 / 1.372721 / 2.100753 / 4.706792 秒，4742 calls。
4. **哪个状态最相关？** remaining DEADLOCK 尾部最高；recovered 同样有 recovery 成本；baseline FEASIBLE 明显较快。完整分层见第 4 节。
5. **N/block/X/Y 的关系？** N/block/Y count 有描述性正相关，X count 线性相关接近零；有限且相关样本，不作显著性/因果推断。见第 5 节。
6. **优化去除了什么重复计算？** 未实施优化，没有声称去除重复计算。
7. **为何 semantics-preserving？** 科学代码完全未修改；observer 透传结果、关闭后恢复原函数。此处不宣称新优化实现的等价性。
8. **old/new differential 100%？** 优化 differential NOT_APPLICABLE；观察 replay 24/24 所列字段一致，fixture 完整对象相等测试通过。
9. **POST-OPT gate？** 无优化所以 NOT_APPLICABLE；原实现的预注册 Stage A gate 自然通过。
10. **Z11 当前 runtime？** actual 60.222375、overshoot 0.222375 秒，18 calls，max reference 6.953368；非优化收益证明。
11. **Z11 仍 no-certified？** 是，18 baseline/remaining DEADLOCK，numeric=0，无 exception，保留 INITIALIZATION_FAILED。
12. **HGA 40-run certified 率？** 37/40，92.5%，另 3 条 no-certified 全保留。
13. **HGA 分类？** INTERMITTENT_NO_CERTIFIED，未调参。
14. **为什么 Phase 3-Z 仍 FAIL？** 冻结协议要求的 certified 完整性未达到，历史179/180，validation已 consumed；新 taxonomy 或 replay 无权改变原 release decision。
15. **Phase 4-0 可开始？** YES，依据任务第 63/66 节，开发 runtime/integrity gate、HGA 审计和回归完成。适用范围见首段；历史长尾未复现及其精确原因未恢复，不宣称全输入硬实时保证。本轮未执行 Phase 4。

## 11. 文件与后续操作

原始证据分别位于：

- `data/manifests/PHASE3ZR_RUNTIME_PROTOCOL_V1.json`
- `data/manifests/PPO_PHASE3ZR_RUNTIME_SET_V1.json`
- `data/development/phase3zr_reference_runtime_audit_v1.json`（含 24 startup attempts、72 science runs、external replay、final regression/decision）
- `data/development/phase3zr_slow_reference_calls_v1.json`（frozen 24 calls、LEVEL 2 与诊断）
- `data/development/phase3zr_hga_incumbent_robustness_v1.json`（40 runs 与复用标识）

runner 入口：`prepare / stage-a / repair-startup / freeze-slow / stage-b / hga / forensic / summarize / finalize / compact-storage`。已完成 corpus 不重复执行；forensic 已记录后拒绝再次运行；repair-startup 仅允许特定 observer identity 启动错误且 reference count 为 0 的已保留尝试。

可读汇总命令（在 DRL 主目录）：

```powershell
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -m scripts.run_phase3zr_runtime_audit summarize
```

用户手动上传前请以 `D:\pybullet_test\MRTA_GA\DRL` 主目录的文件为准。额外的独立 Git 副本与本轮 `.git` 记录按用户要求删除，原有项目文件和本地 Excel 缓存保留。未上传到 GitHub。

目录迁移不改变已完成的测量与授权。`finalize` 对已完成 artifact 仅读取并输出既有回归/决策，不重跑或覆盖，也不再依赖已删除的重复副本进行历史字节比较；完成时的 `historical_evidence_unchanged=true` 保留为原始证据。

迁移后在 `D:\pybullet_test\MRTA_GA\DRL` 主目录使用指定解释器和显式 `PYTHONPATH` 再次完整回归：**330 passed in 64.82s**。新增的迁移测试复用既有测试文件，验证已完成 `finalize` 不重跑实验、不改写 artifact。测量原始 JSON 保持不变，仍保留完成时的 329 项审计回归记录。


## 12. 无损存储去重（2026-10-07）

按用户要求，原 `phase3zr_reference_runtime_audit_v1.json` 在原路径内优化存储，没有新增永久文件、删除任何运行或重跑实验。大小由 **109154315 bytes（104.10 MiB）** 降为 **11009546 bytes（10.50 MiB）**，减少 **89.9138%**。HGA 和 slow-call JSON 的格式及内容未修改。

磁盘格式标记为 `storage_format = phase3zr_replay_tables_v1`。原每条调用的内嵌 `replay` 改为整数 `replay_ref`，引用同一文件内 `replay_tables.replays`。这只是存储格式版本；科学 scope、协议、方法配置、运行来源、测量值和授权决策保持原值，没有新增科学 hash、冻结 contract 或 baseline。

共享表使用普通数组下标，数量如下：

| 表 | 条目数 | 含义 |
|---|---:|---|
| parents | 13 | 12 个开发实例及 Z11 的焊缝几何 |
| patterns | 1108 | 去重后的单个切分定义 |
| pattern_sets | 400 | 有序切分组合，引用 patterns 下标 |
| block_ids | 1257 | 路线中的 block 标识字符串 |
| routes | 7696 | 保留 robot_id 和有序 block_ids 的路线 |
| directions | 3380 | 有序方向向量 |
| metadata | 28 | 科学配置、scope、revision 等原重放字段 |
| replays | 3718 | 完整候选重放定义，引用上述表 |

调用记录仍逐条保留各自的 duration/status/accounting。3718 条去重重放定义供 4760 条调用引用，并不代表删掉调用：Stage A 仍为 72 runs / 4742 calls，external 仍为 18 calls，24 个 observer startup attempts 全部保留。坐标、数值精度、父焊缝、pattern/route 顺序、方向和所有原字段均无损保留。

读取方式：runner 的 `read(AUDIT)` 兼容旧内嵌格式和新格式，返回与原 JSON 相同的展开结构；因此 `reconstruct(call["replay"])`、汇总和重放入口继续使用原 API。直接 `json.load` 则读取磁盘的去重表和引用，需要通过 reader 展开。

```python
from scripts import run_phase3zr_runtime_audit as zr

audit = zr.read(zr.AUDIT)
call = audit["records"][0]["reference_trace"][0]
solution, config, directions = zr.reconstruct(call["replay"])
```

写入器今后仅对本轮 AUDIT 自动去重，其他 artifacts 沿用既有格式。去重后先全量展开比较整个原始审计对象，再原子替换；落盘后再次读取比较。验证为全字段一致，重新计算的 summary 与原 summary 一致，原 decision/final_regression 和历史证据均保留。

新增测试复用既有 `tests/test_baseline_common.py`，覆盖重复项共用、不同方向不合并、额外 metadata 保留、展开对象相互独立、旧格式兼容、非法引用拒绝、原文件写入/读取和候选重建。相关测试 **58 passed**，最终完整回归 **333 passed in 63.26s**。这些是存储层验证，不是新的 solver runtime 测量；原始 artifact 的 329 项审计回归记录不覆盖为 333。
