# FORMAL_SCOPE_GATE_V1 — 2026-09-28

FORMAL_SCOPE_GATE_STATUS = PASS  
FORMAL_SCOPE_V1 = FROZEN  
唯一下一阶段：Phase 3 — Adapted HGA / Adapted WAG common-model。当前轮次到此为止，没有实现 HGA/WAG。

> 2026-09-29 后续审计修复了本报告当时把外层 Git HEAD 写入 `source_commit` 的 provenance 问题，并完成 F4 large-state stress；历史 gate 测量保持不改写，release 结论见 [PRE_PHASE3_RELEASE_AUDIT_20260929](PRE_PHASE3_RELEASE_AUDIT_20260929.md)。

## A. 真实仓库与 baseline

HEAD：`0935ad8a723057084be190e31c6be4733ce3cb62`。已执行 `git status --short`、`git rev-parse HEAD`；DRL 整体为未跟踪目录，不是独立 Git 仓库。外层已有大量用户暂存删除、未暂存修改、未跟踪代码；未执行 reset/checkout/clean，也未提交。source_commit 是当前 HEAD，不声称包含这些未提交实现；运行元数据另记 source_worktree_dirty=true。

用户输入的 `D:\pybullet_test.venv\Scripts\python.exe` 不存在，首次两次调用均未启动 pytest；随后使用实际存在、各 handoff 一致的 `D:\pybullet_test\.venv\Scripts\python.exe`。所有科学运行均在项目目录，未更换 Python 环境。

已阅读根 README、DRL README、唯一主实验方案，以及 Phase 1.1、2A、2B-1、2B-1.5、2B-2 五份交接。根 README 的用户既有改动保持原样，本轮更新 DRL README。

| baseline cwd / command（均为指定 Python 的 -m pytest） | 实测 |
|---|---|
| DRL：-q -p no:cacheprovider | 138 passed in 18.51s |
| 根目录：-q -p no:cacheprovider | 372 passed, 1 skipped, 3 subtests passed in 41.63s |
| 根目录：-q -p no:cacheprovider tests DRL/tests | 510 passed, 1 skipped, 3 subtests passed in 50.67s |

## B. Gate-0 correctness

Mandatory-Y repair：删除 `abs(position2-position)<=1` 的插入前索引预判；完整 placement 后交给既有 canonicalize/scientific validation。新反例精确覆盖 [A,B] → [A,C0,B] → [C1,A,C0,B]，证明结构 alternative 不再被误删。该单元反例隔离 eligibility（真实 mandatory-Y 几何通常要求跨轨）；没有声称同机器人能执行不具备 rail eligibility 的 child。真正 consecutive mandatory-Y collapse 在 development/formal evaluator 都返回 INFEASIBLE，eligibility 未放宽。

Direction refinement：reference FEASIBLE 后先 certify，再将最终有效 status 写入 calls、Nref 四状态、reference_status_sequence 和 diagnostics。认证失败变为 NUMERIC_FAILURE，不留伪 FEASIBLE，不替换 best。Development/formal 两条注入失败测试均通过，formal identity 保留。Initialization 的 status 记录也移到 certification 之后；普通 C4 的既有语义保留。

相关 Gate-0 初次回归：13 passed in 1.55s。后续 formal 用例包含在全量测试中。

## C. Git hygiene 与文件必要性

实际 `git ls-files '*__pycache__*' '*.pyc'` 和 `git ls-tree -r --name-only HEAD` 的对应匹配均为空。因此用户所述“当前 commit 跟踪大量 pyc”不符合此 workspace 真实状态；没有虚构删除或改动 Git index。

根 `.gitignore` 已存在且有 __pycache__/、*.py[cod]、.pytest_cache/；只补 *.egg-info/，保留原有全部忽略规则和用户既有 ppo 输出规则。`git check-ignore` 验证 DRL pyc 与 egg-info 被忽略。Profiler 使用 stdout，不新增输出目录，不忽略源码或 handoff。

新增文件必要性已在交互中逐项提出；复核后取消额外 test/driver 文件，复用现有测试与 profile_scheduler.py。最终只新增本轮指定的：
- DRL/src/mrta_reference/scope.py：唯一 machine-readable scope/identity；
- DRL/docs/FORMAL_SCOPE_V1.md：scope contract；
- DRL/docs/FORMAL_SCOPE_GATE_V1_20260928.md：本交接。

除明确要求的根 .gitignore 外，编辑全部位于 DRL。没有新 dataset、baseline 副本、hash 文件、实验方案副本或算法模块。

## D. Physical/model evidence audit

搜索了整个 D:\pybullet_test workspace 的 URDF、robot_system、motion chain/运动链、rail/gantry、parking/home/initial/terminal、工作空间/可达性/滑轨/悬臂关键词；区分第三方 pybullet_data 示例与实际机器人项目，实际阅读下列相关材料：

- D:/pybullet_test/robot_system/urdf/robot_system.urdf；
- D:/pybullet_test/MRTA/robot_system/urdf/robot_system.urdf；
- D:/pybullet_test/robot/urdf/robot.urdf 与 robot1/urdf/robot1.urdf（不同模型版本，不混同四机器人系统）；
- D:/pybullet_test/MRTA/运动链.docx：读取正文及公式 XML；
- D:/pybullet_test/MRTA/robot_constraints.py 的 base transforms、关节 limits、workspace sampling、IK seeds/restPoses 和轨迹碰撞代码；
- D:/pybullet_test/MRTA/diagnose_workspace.py、generate_welds.py、workspace.py、workspace_boundary.py 及外围 robot_constraints.py。

四机器人 URDF 有 45 links、44 joints，每条分支含滑轨 prismatic、回转、0..3.5 m 悬臂伸缩与机械臂关节。以下为 URDF joint 坐标限位，不直接等于 DRL TCP x 可达域：

| 滑轨 joint | q lower/upper m | base x/y m | 按名义 rpy/axis 推得的 base-frame carriage x 范围 m（近似） |
|---|---|---|---|
| joint1-1 | -6.5 / 16 | 4.5689 / -6.7 | -11.4311 .. 11.0689 |
| joint2-1 | -15.5 / 7 | -4.368 / -6.7 | -11.368 .. 11.132 |
| joint3-1 | -5 / 17.5 | 6.2951 / 6.7 | -11.2049 .. 11.2951 |
| joint4-1 | -15.3 / 7.2 | -4.032236 / 6.7 | -11.232236 .. 11.267764 |

方向由 fixed base yaw ±π/2、slider origin roll π/2 与 local z axis 合成，名义上沿 -x；URDF 近似角导致微小偏差。两份四机器人 URDF 的主要 base z 分别为 1.933 和 2.058 m，不能当成同一已校准模型。Legacy platform 为 [-10,10]×[-6,6]，DRL 为 [0,20]×[0,12]；本轮不声称已完成 TCP/model 坐标标定或从 joint ranges 推导 whole reachability。

运动链文件定义 W/Rm/Em/Pm 坐标变换，不包含 stow/park/退出域规则。Legacy `_reset_all_joints` 设零，`_ik_limits` 的 restPoses 为 zero/midpoint/fixed external sample，属于 IK 初值，不是经过非干涉认证的 parking configuration。URDF 的 effort/velocity 多为 0，不能据此发明实际 retract velocity/time。

| 待证物理结论 | 审计结果 |
|---|---|
| 双轨四分支与可移动关节区间 | 有 URDF/model evidence，见上表；不等同于最终 TCP 可达域 |
| 明确非干涉 home/stow/park 配置 | EVIDENCE_NOT_AVAILABLE |
| inactive robot 可退出本文二维 coordination domain 的验证 | EVIDENCE_NOT_AVAILABLE |
| final POST 后可安全 retract/stow 的路径、时长、顺序约束 | EVIDENCE_NOT_AVAILABLE |
| optional X 独立必要性及完整有限 candidate 规则 | EVIDENCE_NOT_AVAILABLE |

没有因缺少 parking 证据而阻塞项目，也没有把关节伸缩自由度当作安全退出证明。

## E/F/G. F1、F2、F3 最终决定

F1：OPTIONAL_X_SPLIT=EXCLUDED。Formal pattern domain 为 WHOLE/Y_SPLIT；x_up/x_low/Bx 只用于 same-rail spatial/load prior、neighborhood/features。既有模型不存在 hard x ownership boundary，没有发明 Bx lower/center/upper X candidates。

F2：TASK_HORIZON_RELEASE_V1。Final POST 全部计时并占位，包含闭区间结束时刻；之后不加入 terminal WAIT，TCP interference 与 rail-order occupancy 均结束。下层 retract/stow 不进入本文 Cmax。

F3：empty robot 视为未部署到二维 task domain，无 operation/occupancy、completion=0。Non-empty robot 在 t=0 从 first welding block 的显式方向起点合法部署；所有 active robots 一起检查初始干涉及同轨顺序，延迟必须由显式 WAIT 覆盖。

F2/F3 是任务层 scope assumption，不是“机器人消失”的物理事实。不发明 home 坐标或退回速度、时间；论文 limitation 必须说明 lower-level transition 另验。

## H. F4 正式 policy

FORMAL_BOUNDED_DISPATCH_POLICY_V1；baseline 为 optimized deterministic list scheduler。FEASIBLE 立即返回，operations/directions/WAIT/Cmax/canonical timeline 不变。只有 DEADLOCK 才从空 prefix 进行 deterministic DFS，保留 templates、robot-local precedence、optimized ESS、WAIT、连续干涉与同轨约束。

分支按 (ESS,-remaining_processing,-completion,robot_id) 升序，baseline priority branch first。一个 popped prefix 计一个 state，包含 root/dead-end/complete。固定 budget=16；无 RNG、wall-clock 或 algorithm/current/best/history 输入。已访问 complete schedules 以 (Cmax, canonical schedule JSON) 选唯一代表；没有外部 incumbent pruning。

失败保留 DEADLOCK/Cmax=null；NUMERIC_FAILURE 独立。诊断含 baseline DEADLOCK、expanded states、budget、budget/frontier exhaustion；找到解则 source=BOUNDED_DEADLOCK_RECOVERY。Source/provenance 不进入 canonical timeline JSON，因此 baseline timeline 可逐字节比较。

## I. Budget calibration（仅 development fixtures）

| budget | E3 recovery | E3 states / runtime s | E4 status / states / runtime s | baseline DEADLOCK recovery rate |
|---:|---|---|---|---|
| 16 | FEASIBLE | 16 / 0.001442 | DEADLOCK / 7 / 0.001432 | 1/2 |
| 32 | FEASIBLE | 32 / 0.002703 | DEADLOCK / 7 / 0.000833 | 1/2 |
| 64 | FEASIBLE | 60 / 0.004978 | DEADLOCK / 7 / 0.001060 | 1/2 |
| 128 | FEASIBLE | 60 / 0.005224 | DEADLOCK / 7 / 0.000805 | 1/2 |

Known recoverable E3 的恢复率四项均为 1/1；E4 无数学不可行性结论。E1/E2 每项零 recovery states、canonical schedule 保持。选择候选集合中最小的 16；没有使用未来 TEST 或按 ALNS Cmax 选择预算。表中 calibration runtime 仅 recovery 部分；正式调用耗时见 N。

## J/K. Canonical definition 与 scope hash

唯一可执行定义：`DRL/src/mrta_reference/scope.py`。以下是本次实际 `FORMAL_SCOPE_V1.canonical_json`，并非第二份可编辑配置：

```json
{"certifier_policy_id":"INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1","deadlock_policy_id":"FORMAL_BOUNDED_DISPATCH_POLICY_V1","deadlock_state_budget":16,"dispatch_order_id":"DFS_ESS_REMAINING_PROCESS_COMPLETION_ROBOT_V1","empty_route_policy":"UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1","initial_deployment_policy":"FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1","interference_policy_id":"CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1","max_split_per_parent":1,"objective_policy_id":"CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1","open_route_policy_id":"NO_HOME_FIRST_NO_RETURN_HOME_V1","optional_x_split_policy":"EXCLUDED","pattern_domain":["WHOLE","Y_SPLIT"],"recovery_selection_id":"CMAX_THEN_CANONICAL_SCHEDULE_JSON_V1","reference_scheduler_policy_id":"FORMAL_BOUNDED_DISPATCH_POLICY_V1","scope_id":"FORMAL_SCOPE_V1","state_count_policy_id":"POPPED_PREFIX_INCLUDING_ROOT_AND_COMPLETE_V1","terminal_policy":"TASK_HORIZON_RELEASE_V1","y_split_rule_id":"BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1"}
```

SHA-256：`8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9`。

Hash 不包含 timestamp、machine、test counts 或 Git SHA，不复用 ScientificConfig hash。F1/F2/F3/F4 policy/budget mutations 均改变 scope hash；相同 scope 稳定；数值 config 改变仅改变 scientific_config_hash。未实现的 scope/policy 在正式 API 明确拒绝，不静默运行旧语义。

## L. RunScientificIdentity

Frozen structure 包含 scope_id、scope_hash、scientific_config_hash、reference_policy_id、source_commit。Formal run 从实际 HEAD 构造，保存在 result.stats.scientific_identity，profiler 输出时序列化。

Nominal config hash：`791fd398c8819030bfae9ebaa65d11efe57a3dd37b327ad516d37c78310aff0e`。FAST config hash：`e4b9ce937f0ce4742d46fab68a67bb55e60987d6d621111b02cc69b32a1e5ce6`。

## M. Evaluator/certifier/ALNS integration

新 formal APIs：reference_schedule_formal、reference_schedule_from_templates_formal、FormalReferenceEvaluator。保留 reference_schedule/slow/optimized 的 DEVELOPMENT_NO_REPAIR_V1；旧 EXACT_Y_SCOPE_CURRENT_SEMANTICS 不删除。

run_bounded_sa_oi 显式接受 scope 与 source_commit；初始化、candidate C4、refinement、final certify 全部使用同一 FormalReferenceEvaluator/scope。Formal scope 下禁止传入 development callback；memo key 包含 scope hash 和 policy。各次 reference record 保存 scope、policy、status、source、diagnostics 与 expanded states。

Certifier 仍独立重建轨迹，不调用 scheduler 的 ESS/conflict。Formal 增加 identity、X 排除、empty completion/operation、final POST horizon 检查，原有 precedence/duration/initial/order/interference/Cmax 检查保留。新增 template trajectory certifier 只用于 E1–E4 人工 MOVE fixtures，不冒充焊缝 parent coverage 认证。

ALNS M=64、Kdp=8、Kref=2、Kref_total=4、48 atomic+16 LNS、destroy/repair/SA/refinement defaults 均未调参。新增 reference records 为诊断，不参与搜索决策。本轮没有 repair optimization、新增 operators 或 exact solver 优化。

## N. E1–E4 正式结果

| case | baseline | formal | source | states / budget | Cmax | certified | formal call runtime s |
|---|---|---|---|---|---:|---|---:|
| E1 | FEASIBLE | FEASIBLE | BASELINE | 0/16 | 2.0000000000 | true（templates） | 0.000548 |
| E2 | FEASIBLE | FEASIBLE | BASELINE | 0/16 | 14.1622776606 | true（templates） | 0.000645 |
| E3 | DEADLOCK | FEASIBLE | BOUNDED_DEADLOCK_RECOVERY | 16/16 | 15.5287634619 | true（templates） | 0.001961 |
| E4 | DEADLOCK | DEADLOCK | BASELINE partial timeline | 7/16 | null | 不适用 | 0.001338 |

E2 没有借恢复优化已经 FEASIBLE 的 Cmax。E4 frontier exhausted，仍不改为 INFEASIBLE。Budget=1 的强制耗尽反例也保持 DEADLOCK；强制 arithmetic error 独立变成 NUMERIC_FAILURE。

## O. Q1–Q6 formal regression

FAST_CONFIG，使用原 quality fixtures 的 seeds/iterations，算法 defaults 保持：

| case | seed / iterations | development C* | formal initial Cref | formal final Cref | best 来源 | formal search runtime s |
|---|---|---:|---:|---:|---|---:|
| Q1 assignment | 7 / 1 | 7.150000 | 12.840000 | 7.150000 | ATOMIC | 0.016780 |
| Q2 route | 43 / 8 | 11.468136 | 11.545658 | 11.468136 | ATOMIC | 0.151020 |
| Q3 direction | 3 / 20 | 6.271417 | 7.105576 | 6.271417 | DIRECTION_REFINEMENT | 0.163982 |
| Q4 optional Y | 0 / 10 | 4.000000 | 6.000000 | 4.000000 | ATOMIC | 0.207795 |
| Q5 interference | 37 / 12 | 6.271417 | 7.105576 | 6.271417 | ATOMIC | 0.242416 |
| Q6 LNS basin | 1 / 4 | 3.557182 | 19.357912 | 3.557182 | LNS_REPAIRED | 0.083974 |

全部 COMPLETED、best certified。Q4 真实发生 10 次 bounded recovery，其余 quality runs 为 0。C* 仍带 development dispatch+ESS 范围；formal_exact_gap=null，不改名 Cmax_OPT，不把数值相等当成 formal exact certification。micro_gap_decomposition 对 formal run 明确拒绝 development exact gap。

## P. Formal-scope development-family smoke

4 families × N20/50/100 × seeds=20260928/20260929/20260930，每次固定 2 iterations，算法 defaults 不变，无 wall-clock stop。36/36 COMPLETED、final certified，272 search reference calls、128 direction calls、22 LNS acceptances。没有 NUMERIC_FAILURE。总 run runtime=87.868984 s。

初始化是 **12 deterministic instance conditions + seed replay confirmation**，不是 36 个独立 initialization samples。每个条件的 3 次 initial canonical schedule 一致；多 seeds 只用于 stochastic ALNS trajectory 检查，不是正式 benchmark 统计。

下表 seed 列按 20260928、20260929、20260930 排列：

| family | N | runtime s（seed 928/929/930） | formal best Cref（三 seeds） | Nref | direction calls | search DEADLOCK | LNS accepted |
|---|---:|---|---|---|---|---|---|
| load_skew | 20 | 0.325 / 0.332 / 0.347 | 1180.740741 / 1180.740741 / 1180.740741 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| load_skew | 50 | 1.477 / 1.633 / 1.705 | 2749.259259 / 2749.259259 / 2749.259259 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 1 / 0 / 0 |
| load_skew | 100 | 3.161 / 3.428 / 3.554 | 5458.518519 / 5458.518519 / 5458.518519 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| spatial_cluster | 20 | 0.442 / 0.447 / 0.454 | 375.740741 / 375.740741 / 375.740741 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| spatial_cluster | 50 | 1.639 / 1.751 / 1.797 | 960.925926 / 960.925926 / 960.925926 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 2 / 2 |
| spatial_cluster | 100 | 3.815 / 4.328 / 4.236 | 1838.703704 / 1838.703704 / 1838.703704 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 1 / 1 / 0 |
| handover_heavy | 20 | 0.654 / 0.754 / 0.633 | 464.435185 / 470.185185 / 470.185185 | 8 / 6 / 6 | 4 / 2 / 2 | 1 / 3 / 3 | 2 / 1 / 0 |
| handover_heavy | 50 | 2.686 / 2.936 / 3.059 | 1164.564815 / 1164.564815 / 1164.564815 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 2 / 2 / 2 |
| handover_heavy | 100 | 7.742 / 7.294 / 8.083 | 2210.925926 / 2210.925926 / 2210.925926 | 4 / 4 / 4 | 0 / 0 / 0 | 4 / 4 / 4 | 0 / 0 / 0 |
| interference_stress | 20 | 0.439 / 0.444 / 0.460 | 614.074074 / 614.074074 / 614.074074 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 |
| interference_stress | 50 | 1.812 / 1.971 / 1.835 | 1532.592593 / 1532.592593 / 1532.592593 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 2 / 2 |
| interference_stress | 100 | 3.860 / 4.234 / 4.105 | 2910.370370 / 2910.370370 / 2910.370370 | 8 / 8 / 8 | 4 / 4 / 4 | 0 / 0 / 0 | 1 / 1 / 0 |

28 次 baseline deadlock（含 initialization），其中 search-stage 19 次；本次 families 中 recovery=0。特别是 handover_heavy/N100 的 C4 全死锁，未触发 FEASIBLE proposal 的 direction refinement，仍保留 certified initial best。这是有限策略的已知边界，不是隐去的初始化失败。

N100 reference sanity：同一 synthetic solution/方向，slow=6.294270s、optimized=0.175080s、formal=0.203137s；三者 Cmax=3607.314814814816 且 certified，formal/baseline canonical JSON 完全相同。单次计时不构成显著性性能结论。N100 smoke repair 通常仍比 reference 耗时，未优化 repair enumeration/cache/ranker；见附表原始测量。

## Q. Tests 与真实命令

全部 Python 命令前缀：`D:\pybullet_test\.venv\Scripts\python.exe`。根目录为 D:\pybullet_test\MRTA_GA。

```powershell
# DRL cwd
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
# repository root cwd
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests

& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-gate calibration
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-gate quality
& 'D:\pybullet_test\.venv\Scripts\python.exe' DRL/scripts/profile_scheduler.py --formal-scope-gate smoke
```

最终代码版本的实测结果如下；未删除困难测试。

| 最终 cwd / command（指定 Python 的 -m pytest） | 实测结果 | exit code |
|---|---|---:|
| DRL：-q -p no:cacheprovider | 157 passed in 32.09s | 0 |
| 根目录：-q -p no:cacheprovider | 372 passed, 1 skipped, 3 subtests passed in 47.99s | 0 |
| 根目录：-q -p no:cacheprovider tests DRL/tests | 529 passed, 1 skipped, 3 subtests passed in 79.61s | 0 |

最终独立 template certifier 增加 duration 校验后，再次实际执行 calibration 命令，exit code=0。E1/E2 仍为 BASELINE FEASIBLE、0 expanded states；E3 为 BOUNDED_DEADLOCK_RECOVERY FEASIBLE、16 states、Cmax=15.528763461900223、certified=true；E4 为 DEADLOCK、7 states、frontier_exhausted=true。四例 formal 调用实测分别为 0.000586、0.000958、0.002717、0.001481 s。16/32/64/128 重放得到相同 recovery rate 与 schedules，没有重新选择预算。

覆盖 A–M：mandatory placement/collapse；development/formal certifier failure；stable scope hash/F1–F4 mutation/numeric-config separation；formal X rejection；empty/initial/POST/rail/horizon/certifier；E1/E2 exact timeline equality；E3 recovery；E4/budget exhaustion；history/order/algorithm-label-independent replay；formal ALNS fixed-budget reproducibility；初始化/C4/refinement/final 同一 identity；500 ESS/300 solution slow-optimized differentials及 M1–M7/Y/empty cases。

历史 explicit-validator X fixtures 位于 formal domain 之外，保留其 development tests；“baseline FEASIBLE 不变”比较的是同一 WHOLE/Y formal domain，不要求正式接受已排除的 X。

## R. Known limitations

- TASK_HORIZON_RELEASE_V1 是任务层 assumption。无 full physical parking、3D collision-free execution 或机器人全运动学保证。
- Budget 16 包含 root/complete，因此超过 15 个非 WAIT templates 的 DEADLOCK case 无法在该 DFS 内到达 complete leaf。Large family DEADLOCK 是预期保留的状态；要改变此 scientific policy 必须新 scope version，不能按算法或 TEST 临时调预算。
- DFS 不是 mathematical feasibility oracle，E4 失败不证明一般连续时间不可行；FEASIBLE certification 不证明 optimality。
- Exact 仍是 development Y-scope dispatch+ESS generator；Phase 6 的同 scope/final exact validity 未完成。
- Formal scope freeze 不代表 formal datasets 或论文实验已完成。ALNS 参数仍为 development defaults，未进行 ranker、benchmark 或 wall-clock comparison 调参。
- 仓库有既有用户未提交工作，DRL 尚未进入 HEAD；本报告保留实际 source_commit 与 dirty-state 事实，不创建 source-code hash 替代版本管理。

## S. Gate-exit 与交接

F1–F4 均有最终可执行答案；scope identity 实现；evaluator/certifier/ALNS 统一；E1–E4、Q1–Q6、development-family smoke 与全量回归满足本轮条件；所有输出 best certified；Git index 无 pyc/__pycache__。

允许进入 **Phase 3 — Adapted HGA / Adapted WAG common-model**：先 paper-aligned sanity check，再共享 FORMAL_SCOPE_V1、reference evaluator、certifier 与 wall-clock boundary。未完成 Adapted HGA/WAG、LB_LP、MLP/GAT、ranker dataset、formal VALIDATION/ID_TEST/OOD、final exact validation。本轮没有越界实现这些阶段。

## 附：36 次 smoke 的实测耗时分解

所有行 status=COMPLETED、certified=true。Reference time 含初始化及 search；repair/certifier 为运行内实际累计，未以 nominal wall-clock 代替测量。

| family | N | seed | initial Cref | final Cref | runtime s | repair s | reference s | certifier s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| load_skew | 20 | 20260928 | 1180.740741 | 1180.740741 | 0.325296 | 0.083432 | 0.142014 | 0.028915 |
| load_skew | 20 | 20260929 | 1180.740741 | 1180.740741 | 0.331811 | 0.090287 | 0.146103 | 0.029700 |
| load_skew | 20 | 20260930 | 1180.740741 | 1180.740741 | 0.346992 | 0.104360 | 0.151214 | 0.028705 |
| load_skew | 50 | 20260928 | 2749.259259 | 2749.259259 | 1.477166 | 0.751766 | 0.398508 | 0.158512 |
| load_skew | 50 | 20260929 | 2749.259259 | 2749.259259 | 1.633430 | 0.870339 | 0.421436 | 0.163393 |
| load_skew | 50 | 20260930 | 2749.259259 | 2749.259259 | 1.704918 | 0.962664 | 0.410947 | 0.161278 |
| load_skew | 100 | 20260928 | 5458.518519 | 5458.518519 | 3.160764 | 1.524404 | 0.660276 | 0.623681 |
| load_skew | 100 | 20260929 | 5458.518519 | 5458.518519 | 3.428011 | 1.780292 | 0.661093 | 0.617190 |
| load_skew | 100 | 20260930 | 5458.518519 | 5458.518519 | 3.553931 | 1.935189 | 0.655267 | 0.619852 |
| spatial_cluster | 20 | 20260928 | 375.740741 | 375.740741 | 0.441694 | 0.065345 | 0.277222 | 0.029733 |
| spatial_cluster | 20 | 20260929 | 375.740741 | 375.740741 | 0.446542 | 0.065806 | 0.280284 | 0.029904 |
| spatial_cluster | 20 | 20260930 | 375.740741 | 375.740741 | 0.454170 | 0.068099 | 0.290334 | 0.030725 |
| spatial_cluster | 50 | 20260928 | 960.925926 | 960.925926 | 1.638923 | 0.777316 | 0.525903 | 0.162081 |
| spatial_cluster | 50 | 20260929 | 960.925926 | 960.925926 | 1.750521 | 0.877150 | 0.532709 | 0.161662 |
| spatial_cluster | 50 | 20260930 | 960.925926 | 960.925926 | 1.797320 | 0.944602 | 0.529326 | 0.156263 |
| spatial_cluster | 100 | 20260928 | 1838.703704 | 1838.703704 | 3.814516 | 1.538240 | 1.315295 | 0.600134 |
| spatial_cluster | 100 | 20260929 | 1838.703704 | 1838.703704 | 4.327691 | 1.802418 | 1.526780 | 0.627009 |
| spatial_cluster | 100 | 20260930 | 1838.703704 | 1838.703704 | 4.236328 | 1.904185 | 1.304060 | 0.673225 |
| handover_heavy | 20 | 20260928 | 470.185185 | 464.435185 | 0.653633 | 0.152086 | 0.357357 | 0.030175 |
| handover_heavy | 20 | 20260929 | 470.185185 | 470.185185 | 0.753650 | 0.173736 | 0.394149 | 0.018608 |
| handover_heavy | 20 | 20260930 | 470.185185 | 470.185185 | 0.632524 | 0.178227 | 0.331058 | 0.015896 |
| handover_heavy | 50 | 20260928 | 1167.481481 | 1164.564815 | 2.685841 | 1.509312 | 0.762514 | 0.156524 |
| handover_heavy | 50 | 20260929 | 1167.481481 | 1164.564815 | 2.936035 | 1.751953 | 0.764659 | 0.157512 |
| handover_heavy | 50 | 20260930 | 1167.481481 | 1164.564815 | 3.058684 | 1.867422 | 0.771362 | 0.164006 |
| handover_heavy | 100 | 20260928 | 2210.925926 | 2210.925926 | 7.742266 | 3.537374 | 3.563143 | 0.118444 |
| handover_heavy | 100 | 20260929 | 2210.925926 | 2210.925926 | 7.293989 | 3.926087 | 2.715636 | 0.121818 |
| handover_heavy | 100 | 20260930 | 2210.925926 | 2210.925926 | 8.082653 | 3.805421 | 3.631332 | 0.126446 |
| interference_stress | 20 | 20260928 | 614.074074 | 614.074074 | 0.438859 | 0.064026 | 0.278446 | 0.029388 |
| interference_stress | 20 | 20260929 | 614.074074 | 614.074074 | 0.444210 | 0.072384 | 0.275463 | 0.030436 |
| interference_stress | 20 | 20260930 | 614.074074 | 614.074074 | 0.459628 | 0.074753 | 0.287790 | 0.032050 |
| interference_stress | 50 | 20260928 | 1532.592593 | 1532.592593 | 1.811912 | 0.888245 | 0.551676 | 0.173262 |
| interference_stress | 50 | 20260929 | 1532.592593 | 1532.592593 | 1.971340 | 0.998405 | 0.581047 | 0.173993 |
| interference_stress | 50 | 20260930 | 1532.592593 | 1532.592593 | 1.834733 | 0.952439 | 0.542963 | 0.163528 |
| interference_stress | 100 | 20260928 | 2910.370370 | 2910.370370 | 3.859724 | 1.550483 | 1.319071 | 0.619191 |
| interference_stress | 100 | 20260929 | 2910.370370 | 2910.370370 | 4.234184 | 1.762248 | 1.494032 | 0.609391 |
| interference_stress | 100 | 20260930 | 2910.370370 | 2910.370370 | 4.105095 | 1.892954 | 1.258550 | 0.602093 |
