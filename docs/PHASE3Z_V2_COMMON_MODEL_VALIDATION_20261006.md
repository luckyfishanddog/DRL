# Phase 3-Z — V2 Common-Model Validation（2026-10-06）

## 结论与执行边界

本轮正式执行 **FAIL**，不是竞争性结论。180/180 次 native solver 调用已完成，179 次返回可独立认证的最终解。第 165 次 HGA 没有产生任何 certified solution，runner 首次以 exit code 1 停止。用户随后明确要求继续余下运行；使用同一个已冻结 runner 从第 166 次恢复，第 165 次保留原记录，不重跑失败 seed、不补 seed、不调算法。续跑 15/15 次 final certified。验证数据已经 CONSUMED，不再是 untouched validation。

```text
PHASE3Z_EXECUTION_STATUS = FAIL
VALIDATION_RUNS = 180/180
CERTIFIED_RUNS = 179/180
FORMAL_SCOPE_V2_STATUS = BLOCKED
DETERMINISTIC_V2_BACKBONE_STATUS = NOT_EVALUABLE
TRADITIONAL_HEURISTIC_TUNING_STATUS = NOT_CLOSED
V2_VALIDATION_STATUS = CONSUMED
ID_TEST_STATUS = SEALED
NEXT_PHASE = Resolve Phase3-Z execution blocker only
```

这里 BLOCKED 表示 Phase3-Z 最终冻结未获通过；已完成的 Phase3-YR access closure 不撤销，`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V2` 保持不变。Phase3-Y 历史 FAIL、Phase3-YR 历史 PASS 均未改写。`NOT_CLOSED` 也不授权 validation-driven tuning。Phase 4-0 尚未获本轮执行结果授权，MLP/GAT/PPO、ID_TEST、V2_TRAIN_POOL solver run 仍未执行。

## A–C. 来源、回归与冻结协议

正式来源是独立 GitHub clone（非外层 MRTA 仓库）：

- origin：`https://github.com/luckyfishanddog/DRL.git`；repository_id：`luckyfishanddog/DRL`。
- 工作目录：`D:\pybullet_test\MRTA_GA\DRL\workspaces\phase3z-validation`。
- remote main 起点：`6049533432fe54927e66df06b39093401a436289`。
- 本地分支：`codex/phase3z-v2-validation`。
- verified HEAD：`43a004c2b40382110b642fa009b7a30ff124470b`；`commit_verified=true`、`worktree_dirty=false`。
- scientific source_tree_hash：`533372e63cef39a1ee967b8cae8afa9dac3b1a259da08f6199d6fc0943f332a7`，与已激活的 Phase3-YR source 相同。
- Phase3-Z protocol hash：`e9767ad446c8ac50f82391638525c7cbe6e8c1eb2dfcd865f3932b7bc07ffde6`。
- Phase3-YR protocol hash：`e2206a3d20a2dddf53912920abbf4ca62844a669db322a7cce51ce3655b99876`。

本地 commit 只新增 orchestration runner、更新三个既有测试文件，没有修改 scientific src。所有正式测量发生于该 commit 完成之后。Git 的 CRLF/LF 转换曾影响旧的 raw-byte identity 检查；正式运行前，在确认规范化后的内容一致后恢复原来源字节，未改变算法文本。生成协议、结果和本报告使用 exact-path local exclude，不忽略 scientific source。没有 push 或上传 GitHub。

独立 clone 初次回归出现 10 failed / 302 passed / 1 error：旧测试把 development label 当作真实 Git commit，在 standalone Git 环境被 provenance 检查拒绝。只修复测试 fixture，使其显式携带 unverified development provenance，不削弱正式 provenance 检查；另补协议 JSON roundtrip 测试。

正式测量前 full regression：**314 passed in 58.95s**。第 165 次停止后 full regression：**314 passed in 61.41s**；恢复前 full regression：**314 passed in 58.61s**；180 次全部调用完成后：**314 passed in 60.53s**。全部输出保存在 artifact 的 regressions/regression_attempts。历史 runner、scope identity、role rejection、resume identity、无重复 seed、no-backfill、完整五 seed 汇总、竞争 Gate 边界、direction observer 恢复等检查均在完整 suite 内。恢复没有改变 source commit、protocol、方法配置或 scientific source。

冻结 identities 保持：

| Identity | Value |
|---|---|
| V1 | `8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9` |
| V1.1 | `5d3323e4445675af362cf6816e46c2f3bb092a28fcfd1d08741ca47c021bd0dc` |
| V2 | `16f6110a7384d585fa539777b059e0a297da4fa394a3b4b545ebe967ece54599` |
| V2 roles | `7a80372eb072b10da3d8044eb8bb330a29ede93fe16c806b1aa0d58bee968a5a` |
| V2 validation set | `2989d8fe15330883a617504d4cc51a6a6cfc547aa9c9813853b6df8fcb2245fe` |

## D–G. 数据、运行与实际失败

使用指定 `D:\pybullet_test\.venv\Scripts\python.exe`，Python 3.11.9，Windows 10 build 19045，Intel64 Family 6 Model 140，8 logical cores。运行串行，不并行比较算法；OMP/MKL/OpenBLAS/NUMEXPR thread environment 均未显式设置。native 初始化计入同一个 60s；checkpoints=5/30/60。执行顺序为 instance-major、seed-major、method offset=(instance_index+seed_index)%3。

五个固定 seed：`20261011, 20261012, 20261013, 20261014, 20261015`，不沿用 development smoke seed。首个正式 batch 时间（Asia/Shanghai）：2026-10-06 19:44:28 至 22:47:57；停止后回归结束于 22:54:57。续跑 batch 于 23:02:07 开始，最终回归及协议复验后于 23:19:19 结束。首个 STOPPED_UNCERTIFIED batch 与执行 blocker 均保留，不因恢复而删除。

下表 Z01–Z12 仅为本报告行别名，不改变 manifest instance_id。所有新角色均为 V2_VALIDATION，旧角色均为 TRAIN_POOL。物理路径 `data/ID_TEST/` 不等于新角色 ID_TEST_SEALED；真实 15 个 sealed workbook 未打开。Z12 的 15 次运行均在续跑 batch 中完成。

| 别名 | tier | N | 完整 instance_id | max X-span (m) | legal X patterns |
|---|---|---:|---|---:|---:|
| Z01 | SMALL | 25 | `data/PPO_TRAIN/seed_0334554054::g16_w025` | 5.254 | 28 |
| Z02 | SMALL | 23 | `data/ID_TEST/seed_0263067416::g17_w023` | 1.577434 | 18 |
| Z03 | SMALL | 24 | `data/ID_TEST/seed_1398320817::g20_w024` | 12.500 | 23 |
| Z04 | SMALL | 23 | `data/PPO_TRAIN/seed_0150777682::g19_w023` | 4.213469 | 20 |
| Z05 | MEDIUM | 55 | `data/PPO_TRAIN/seed_1126479801::g35_w055` | 12.500 | 54 |
| Z06 | MEDIUM | 55 | `data/ID_TEST/seed_1411252158::g22_w055` | 2.129 | 26 |
| Z07 | MEDIUM | 55 | `data/PPO_TRAIN/seed_0414671407::g25_w055` | 2.959141 | 26 |
| Z08 | MEDIUM | 55 | `data/ID_TEST/seed_1674663312::g15_w055` | 2.591 | 48 |
| Z09 | LARGE | 85 | `data/PPO_TRAIN/seed_0455745885::g42_w085` | 4.704260 | 83 |
| Z10 | LARGE | 83 | `data/PPO_TRAIN/seed_0450139711::g43_w083` | 3.1295 | 58 |
| Z11 | LARGE | 85 | `data/PPO_TRAIN/seed_2095476608::g44_w085` | 12.500 | 61 |
| Z12 | LARGE | 85 | `data/ID_TEST/seed_1821446561::g38_w085` | 4.213469 | 51 |

失败 run key：Z11 / `ADAPTED_HGA_V2` / seed `20261015`，run_key_hash=`5e8edea471ab75801160efd21fe850436ba7ff574a11a1d1bb9a7f80fb5ca796`。

- native termination：`INITIALIZATION_FAILED`，best_events=[]，最终 solution/schedule=None，Cmax@5/30/60 全为 NULL。
- 53 candidates，38 direction-infeasible，15 reference calls；15 baseline DEADLOCK，15 remaining DEADLOCK，recovered=0，certifier calls=0，numeric=0。
- 26 iterations，native initialization=23.134686s，actual_runtime=746.775917s，overshoot=686.775917s，scheduler_time=727.630600s。
- X proposals=3，constructed=3，reference=1（首次 reference 开始于 42.305895s）；没有 certified X 或其他 certified candidate。
- HGA 代码仅在 evaluator.best=None 时给出 INITIALIZATION_FAILED；该标签不意味着仅初始构造失败，因为本次也执行了 offspring 搜索。

源代码中 HGA 在迭代边界检查 deadline，CommonBaselineEvaluator 的 reference scheduler 调用不抢占。记录说明存在极端 non-preemptive overshoot，但没有逐次 scheduler 耗时记录，不能精确归因给某一个 candidate。DEADLOCK 是固定 B32 调度策略未返回可行解，不是数学不可行证明。该失败不能通过注入共同 seed、调整 initialization、改变 B32、降低认证要求或重跑同一 seed 来掩盖。

其余 179 次均 final certifier PASS；全部 180 个预定 run key 恰好各一条记录。numeric_failure=0、scheduler/certifier mismatch=0、catalogs_identical=true、checkpoint_no_backfill=true，ALNS 意外 ITERATION_LIMIT=false。首次停止机制因无最终 certified solution 触发，不是因性能差触发。恢复只补齐未执行 run key，不修复或排除第 165 次失败。

## H–M、Q、S. Anytime、五 seed 汇总与竞争性边界

表中 A=ALNS、H=HGA、W=WAG；单位秒。每格必须有同一实例同一方法五个指定 seed 的 certified checkpoint 值，缺任一个即 NULL，不以成功 seed 子集取 median。不使用 overshoot 后的 final_cmax 回填 60s。Z11-H 有 4 次 certified、1 次失败，因此其三个 median 都为 NULL。

| 实例 | 方法 | median Cmax@5 | median Cmax@30 | median Cmax@60 |
|---|---|---:|---:|---:|
| Z01 | A | 2639.162 | 2267.728 | 2031.032 |
| Z01 | H | NULL | 1921.005 | 1913.767 |
| Z01 | W | 2114.779 | 1952.188 | 1919.408 |
| Z02 | A | 1268.545 | 1268.545 | 1268.545 |
| Z02 | H | 1317.342 | 1269.407 | 1231.576 |
| Z02 | W | 1268.545 | 1246.600 | 1230.732 |
| Z03 | A | 3227.527 | 1902.800 | 1661.002 |
| Z03 | H | NULL | 1649.606 | 1645.063 |
| Z03 | W | 2010.100 | 1674.031 | 1639.463 |
| Z04 | A | 1352.730 | 1307.908 | 1292.490 |
| Z04 | H | 1364.769 | 1364.769 | 1331.484 |
| Z04 | W | 1356.635 | 1305.727 | 1305.727 |
| Z05 | A | 4985.501 | 4796.508 | 4639.404 |
| Z05 | H | 4217.374 | 4217.374 | 3744.835 |
| Z05 | W | 4217.374 | 3636.033 | 3623.625 |
| Z06 | A | 2739.794 | 2685.398 | 2685.398 |
| Z06 | H | 2759.476 | 2747.115 | 2733.273 |
| Z06 | W | 2747.115 | 2747.115 | 2710.577 |
| Z07 | A | 3382.085 | 3335.592 | 3330.399 |
| Z07 | H | 3401.665 | 3401.665 | 3401.665 |
| Z07 | W | 3348.203 | 3348.203 | 3348.203 |
| Z08 | A | 2396.134 | 2396.134 | 2396.134 |
| Z08 | H | 2344.014 | 2344.014 | 2343.318 |
| Z08 | W | 2322.335 | 2317.837 | 2278.739 |
| Z09 | A | 6980.069 | 5023.361 | 4556.832 |
| Z09 | H | 4002.955 | 4002.955 | 4002.955 |
| Z09 | W | 3983.376 | 3931.141 | 3925.323 |
| Z10 | A | NULL | 7595.485 | 7169.723 |
| Z10 | H | 4389.154 | 4389.154 | 4389.154 |
| Z10 | W | 4389.154 | 4389.154 | 4389.154 |
| Z11 | A | NULL | 12264.312 | 11921.771 |
| Z11 | H | NULL | NULL | NULL |
| Z11 | W | NULL | 6184.502 | 6159.491 |
| Z12 | A | 4879.953 | 4810.961 | 4762.753 |
| Z12 | H | 4879.953 | 4879.953 | 4879.953 |
| Z12 | W | 4844.916 | 4742.381 | 4736.769 |

早期 certified checkpoint 的缺失数（60 次/方法）：A 的 5/30/60s 为 10/0/0；H 为 10/4/1；W 为 5/0/0。这些不是零成本 incumbent。已完成 SMALL/MEDIUM 行显示 H/W 常较早获得较好解，ALNS 在部分实例仍随时间改善；这只是本轮描述，不是全总体速度或显著性结论。

`R_i = median5(ALNS Cmax@60) / min(median5(HGA Cmax@60), median5(WAG Cmax@60))`：

| 实例 | R_i |
|---|---:|
| Z01 | 1.061274079 |
| Z02 | 1.030724338 |
| Z03 | 1.013137874 |
| Z04 | 0.989862318 |
| Z05 | 1.280321007 |
| Z06 | 0.990710668 |
| Z07 | 0.994682447 |
| Z08 | 1.051517558 |
| Z09 | 1.160880664 |
| Z10 | 1.633509008 |
| Z11 | NULL — HGA 缺 certified seed |
| Z12 | 1.005485771 |

预注册规则未变：180/180 final certified、numeric=0、mismatch=0，且 12-instance median R<=1.10、至少 8/12 个 R<=1.10、各 tier 四实例 median R<=1.15。**竞争 Gate = NOT_EVALUABLE**，不是 PASS，也不把它标成正式 performance FAIL。12-instance median R、正式 count R<=1.10、正式 tier summaries 均为 NULL/未计算；不把十一个可用 R 聚合冒充十二个结果。180 次调用完成不等于 180 次最终认证通过。

仅描述完整 tier 的 SMALL median R=1.021931106、MEDIUM median R=1.023100002；LARGE 四实例 median 不可计算。Z09/Z10 的完整五 seed R 已为 1.160881/1.633509，显示已测 LARGE 仍有 ALNS degradation；Z11 另出现 HGA 无可行 incumbent 的 execution failure。Z11 ALNS/WAG 各自五 seed median 可报告，但不能在 HGA 缺失时按另一种规则生成 R。不能据此调整传统 operator、surrogate、初始化、caps 或 X 优先级。

## N–O、R. Runtime、reference efficiency 与 caps

下列覆盖全部 60 次/方法；H 的统计包括失败调用，不删异常值。

| 方法 | runtime median / max (s) | overshoot median / max (s) | reference median / total | scheduler total (s) | native route-DP total |
|---|---:|---:|---:|---:|---:|
| A | 60.850721 / 68.040526 | 0.850721 / 8.040526 | 100 / 9225 | 1509.710651 | 153884 |
| H | 60.125498 / 746.775917 | 0.125498 / 686.775917 | 110.5 / 8274 | 2435.243353 | 124774 |
| W | 60.185599 / 61.871899 | 0.185599 / 1.871899 | 120.5 / 11219 | 1444.590634 | 104076 |

相同配置 wall-clock 不等于相同 reference 次数或精确相同实际 runtime。逐 run 的 Cmax@60、reference_calls、scheduler_time、runtime、overshoot 配对在 artifact 的 `source_tables.E_reference_efficiency`，不以平均 Cmax/reference 比值替代问题实例差异。巨大 HGA overshoot 与缺解是必须保留的诊断证据，不是额外授予算法的时间 budget。

HGA cap totals：vnd_candidate_cap_hits=12577，vnd_pass_cap_hits=2835，initialization_reference_limit_hits=0，population_survival_events=1393。WAG：factorial_window_cap_hits=12290，factorial_call_cap_hits=0，wag_variant_cap_hits=60，route_combination_cap_hits=4183。ALNS 60 TIME_LIMIT、无 ITERATION_LIMIT，HGA 59 TIME_LIMIT + 1 INITIALIZATION_FAILED，WAG 60 TIME_LIMIT。未因 caps 调任何参数。

冻结参数：ALNS M=64/Kdp=8/Kref=2/Kref_total=4/M_lns=16，construction_budget=5/kinit_ref=5/max_iterations=100000，TWO_OPT_STAR=OFF；HGA mu=20/lambda=10/alpha=20，VND 256 candidates/2 passes；WAG n=10/p=0.5/Imax=500、factorial_window=5、max_windows=8、max_factorial_calls=30720、max_wag_variants=10、max_route_combinations=3。科学配置、native 初始化、Phase3-YR access policies 均原样。

## P. X/Y 描述性使用与 telemetry 语义

以下 X/Y reference rate 为“含该 pattern 的 complete solution 的 reference 次数 / 全部 reference 次数”，不是纯 transition proposal 成功率；含 X/Y 的结构性 move 会重复计数。A 的 exact TARGET family 漏斗另存在 `decision_family_funnel`，不与 legacy presence counters 混用。H/W 冻结代码未提供 exact TARGET family C2 漏斗，相关字段为 NULL+reason，不伪装成零。H/W legacy accepted 在 CommonBaselineEvaluator 中表示 certified candidate，不是必然被当前 population/VNS incumbent 接受，不能跨算法作 acceptance 机制等价比较。

| tier | 方法 | runs | legal X count（四实例，非乘 seed） | X refs / rate | X legacy accepted / refs | final X parent sum | X best updates | Y refs / rate | final Y parent sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMALL | A | 20 | 89 | 1289 / 19.880% | 373 / 1289 | 6 | 96 | 3258 / 50.247% | 44 |
| SMALL | H | 20 | 89 | 1263 / 23.153% | 734 / 1263 | 8 | 25 | 5455 / 100% | 39 |
| SMALL | W | 20 | 89 | 992 / 12.698% | 552 / 992 | 3 | 15 | 7812 / 100% | 40 |
| MEDIUM | A | 20 | 154 | 281 / 14.178% | 57 / 281 | 1 | 5 | 919 / 46.367% | 37 |
| MEDIUM | H | 20 | 154 | 325 / 14.773% | 214 / 325 | 3 | 4 | 1736 / 78.909% | 35 |
| MEDIUM | W | 20 | 154 | 184 / 7.210% | 132 / 184 | 2 | 5 | 2081 / 81.544% | 35 |
| LARGE | A | 20 | 253 | 126 / 16.601% | 32 / 126 | 4 | 18 | 386 / 50.856% | 194 |
| LARGE | H | 20 | 253 | 53 / 8.562% | 19 / 53 | 0（19 个最终解） | 0 | 619 / 100% | 174 |
| LARGE | W | 20 | 253 | 16 / 1.871% | 11 / 16 | 0 | 0 | 855 / 100% | 191 |

artifact 的 legal_pattern_count_sum 逐 run 相加，分别是 445/770/1265；上表去掉固定五 seed 的重复。Y 的 certified/accepted/global-best 原始计数及最终 mandatory/optional Y、WHOLE、X、pattern identities、route directions、robot processing/WAIT/completion 均保存在记录中。

max X-span=12.5m 的三实例：Z03 final X parent 五 seed 总数 A/H/W=2/1/0；Z05=0/2/2；Z11=1/0/0（H 仅四个最终解，第五个是 NULL）。因此长 X-span 不意味着必须保留 X，没有引入长度阈值或 long-weld priority。final X retention 不作为本轮 Gate。

direction_dp_calls 是临时 observer 对 native `_fixed_first_dp` 的实际逐 route 调用计数，含初始化，observer 不改函数返回值/执行顺序/预算，bookkeeping 在 native clock 内，退出后恢复。ALNS certifier_calls 字段采用已有 FEASIBLE-reference 加 final-check 的估计口径，不能当作所有 refinement 内部调用的精确总数；记录给出具体 semantics。额外独立 final certification 属 runner overhead，不回填任何 checkpoint。

## T–V. 保存、交接和下一步

正式 artifact 保留 180 条 records、180 条 attempts、失败 run key、首次停止原因、两个 batch、全部 pre/post 回归输出、完整冻结协议与来源；每次完成均原子保存。FAILED/NULL 不删、不以另一次 seed 结果替换。六组图表源数据直接位于 JSON 的 `source_tables.A_Cmax60/B_R_i/C_anytime/D_tier/E_reference_efficiency/F_X_Y_usage`；D_tier 为空是总体认证不完整导致不可评价，非漏报。未新增 CSV、图片或不必要文件。

独立副本保存正式证据。按用户批准，同步相同 protocol/result/report 至主 DRL 分类目录，覆盖主目录此前 provenance-blocked preparation protocol；主目录副本只是正式结果镜像，不声称主目录是 clean source。runner 与三个测试文件保留供手动上传；没有上传 GitHub。

下一阶段只定位并修复 **“冻结 HGA 在 Z11/seed20261015 下无 certified incumbent，且出现极端不可抢占 scheduler 耗时”** 这一实际 execution blocker。不得把已看到的 performance gap 当作调参依据。用户对本次续跑的授权仅用于补齐原定 15 次，不授权改 scientific domain/B32、注入 oracle、更换初始化或替换失败记录。应先在 development 数据上核查合法的 execution 修复边界与必要证据，再按明确授权处理正式实验记录；不再把 V2_VALIDATION 用于训练或调参。最终 untouched ID_TEST 的显著性/泛化比较、full V2 optimum、MLP/GAT 均不属于本轮结论。
