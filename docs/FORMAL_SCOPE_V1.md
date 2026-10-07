# FORMAL_SCOPE_V1

`FORMAL_SCOPE_V1 = FROZEN_HISTORICAL`。任务层 scientific scope contract；主实验方案仍为 [多机器人焊缝分配与排序实验方案](多机器人焊缝分配与排序实验方案.md)。本文件解释 V1 的机器定义与历史结果，完整回归证据见 Gate 交接报告（对应过程文件已清理，结论保留在本报告）。2026-09-29 的 large-state stress 证明 16-prefix F4 与 template depth 结构性耦合；V1 随后由只改变 F4 的 `FORMAL_SCOPE_V1_1` 取代。V1 的定义、hash 与 replay API 保留且不静默变义；active contract 与修正证据见 [V1.1 F4 closure](FORMAL_SCOPE_V1_1_F4_CLOSURE_20260929.md)。

## F1：optional X_SPLIT = EXCLUDED

正式 pattern domain 为 WHOLE 与现有合法 Y_SPLIT。Whole-eligible parent 可保留 WHOLE 或合法 optional Y；其余必须合法 mandatory Y。每 parent 最多 split 一次、最多两个 child；同机器人相邻 child 先 canonicalize 为 WHOLE，随后仍检查 whole eligibility。Y 候选依次由 By 下边界、中心、上边界、合法 midpoint 产生，以 t 容差去重（先出现 identity 保留），再按 `(t, point_id)` 排序。两个 child 均满足 Lmin，child assignment 逐块检查 rail eligibility。

实际审计了四机器人 URDF、运动链与 legacy reachability 代码，但没有形成 optional X 的独立工艺必要性及完整有限 candidate/point/rail/assignment 规则：`EVIDENCE_NOT_AVAILABLE`。URDF 行程或 Bx 的存在本身不足以推出 X split。Formal API 拒绝带 X_SPLIT 的输入，不接受 X provider/validator 绕过；历史 development validator 接口保留。

`x_up/x_low/Bx` 仅用于 same-rail spatial/load prior、neighborhood、feature，不是 hard ownership boundary，也不构成 formal split domain。同轨左右机器人保留 whole candidate。

## F2/F3：TASK_HORIZON_RELEASE_V1

找到了实际机器人几何与伸缩自由度，但非干涉 parking/stow identity、退出域的转移路径、退回时长及验证机制均为 `EVIDENCE_NOT_AVAILABLE`。因此采用任务层 modeling assumption，而不是声称已有物理停车事实：

1. 只建模被分配 welding blocks 的 task-execution horizon。
2. Empty route robot 视为尚未部署到本文二维 theoretical coordination domain：无 operations、completion=0，不参加 initial order、rail order 或 interference。
3. 每个 non-empty robot 在 t=0 已合法部署到 first welding block 的起始位置（由显式方向决定）；所有 active robot 同时检查初始 TCP interference 和同轨顺序。首个 operation 若延迟，必须有从 t=0 开始、位于该位置的显式 WAIT。
4. 从 t=0 到该 robot final POST 结束，MOVE/SETUP/WELD/POST/WAIT 全部参加连续干涉及同轨不越位检查；final POST 的终点时刻采用原有闭区间判据。
5. Final POST 结束之后不增加 terminal WAIT、return-home、parking 或 retract 时间；该 robot 后续不参加两类占位检查。后续真实 retract/stow 属于 lower-level cell control，排除在本文 Cmax 与理论干涉之外。
6. 机器人并未被宣称在物理上消失。论文必须在 scope/limitations 明确说明此任务层 abstraction；下层能否执行部署和退出需要单独工程验证。

保持开放路线：无 home-to-first、无 return-home；保留所有相邻 block 的二维匀速直线空走。Reference slow/optimized 的既有 trajectory 语义不改变；formal certifier 显式检查 scope identity、X 排除、empty route、final POST horizon，独立重建 geometry/timing/WAIT/rail/interference/Cmax。

## F4：FORMAL_BOUNDED_DISPATCH_POLICY_V1

固定每次 evaluation 的 `B_deadlock_states=16`；不使用 wall-clock、算法名、current/best/incumbent、历史或 RNG。

- 首先完整执行 optimized deterministic list policy。任何非 DEADLOCK 结果立即返回；baseline FEASIBLE 的 directions、operations、WAIT、completion、Cmax 和 canonical schedule JSON 完全不变，仅附加 evaluator provenance。
- 只有 baseline DEADLOCK 才从空 dispatch prefix 进行 bounded DFS。Operation templates、机器人局部 precedence、earliest-safe-start、WAIT、连续干涉和 rail order 完全复用既有实现。
- 每个 prefix 枚举当前有限 ESS choices，按 `(ESS, -remaining_processing, -completion, robot_id)` 升序访问；因此 baseline priority branch first。使用 reverse push 的显式 stack 实现 DFS，无递归深度依赖。
- 一次 pop 的 prefix 计一个状态，包含 root、dead end 与 complete schedule；最多 16。预算不重置、不随 algorithm/instance size 调整。每次 expansion 最多四个 choices。
- 在预算内访问到的所有 complete FEASIBLE schedules，以精确 `(Cmax, canonical_schedule_JSON)` 升序选唯一代表。该 JSON 的 operation 顺序为 `(start_time,end_time,robot,sequence,kind,id)`，包含方向，不含 provenance/timestamp/runtime。
- 有解返回 FEASIBLE、source=BOUNDED_DEADLOCK_RECOVERY；否则保留 baseline 部分 schedule、DEADLOCK、Cmax=null，不产生数学 INFEASIBLE 声明。NUMERIC_FAILURE 单独返回，不作为死锁或普通劣解。
- 诊断记录 baseline DEADLOCK、expanded_states、budget、recovery_exhausted（预算用尽且 frontier 尚非空）和 frontier_exhausted。失败 result 的 source=BASELINE 表示保留 baseline timeline，不表示未尝试恢复。

预算仅用 development E1–E4 的 16/32/64/128 校准；16 已恢复 E3，E1/E2 不变，E4 仍 DEADLOCK。之后 Q1–Q6 与 family smoke 验证该固定选择，不反过来按 ALNS Cmax 选 policy。

## Machine identity 与 API

`FormalScope` 是 frozen dataclass，canonical JSON 为 UTF-8、sort_keys、紧凑 separators、allow_nan=False，SHA-256 仅覆盖科学 scope 与 evaluator policy。唯一机器定义如下：

```json
{"certifier_policy_id":"INDEPENDENT_CONTINUOUS_TASK_HORIZON_V1","deadlock_policy_id":"FORMAL_BOUNDED_DISPATCH_POLICY_V1","deadlock_state_budget":16,"dispatch_order_id":"DFS_ESS_REMAINING_PROCESS_COMPLETION_ROBOT_V1","empty_route_policy":"UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1","initial_deployment_policy":"FREE_FIRST_WELD_START_ALL_ACTIVE_LEGAL_AT_ZERO_V1","interference_policy_id":"CLOSED_CONTINUOUS_TCP_AND_SAME_RAIL_ORDER_V1","max_split_per_parent":1,"objective_policy_id":"CMAX_OPTIONAL_SPLITS_PROCESS_SPREAD_EMPTY_WAIT_ID_V1","open_route_policy_id":"NO_HOME_FIRST_NO_RETURN_HOME_V1","optional_x_split_policy":"EXCLUDED","pattern_domain":["WHOLE","Y_SPLIT"],"recovery_selection_id":"CMAX_THEN_CANONICAL_SCHEDULE_JSON_V1","reference_scheduler_policy_id":"FORMAL_BOUNDED_DISPATCH_POLICY_V1","scope_id":"FORMAL_SCOPE_V1","state_count_policy_id":"POPPED_PREFIX_INCLUDING_ROOT_AND_COMPLETE_V1","terminal_policy":"TASK_HORIZON_RELEASE_V1","y_split_rule_id":"BY_LOWER_CENTER_UPPER_LEGAL_MIDPOINT_T_DEDUP_V1"}
```

Scope hash：

```text
8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9
```

`ScientificConfig.scientific_hash` 单独覆盖数值参数；改变 vw 等数值不会改变 scope hash。Timestamp、machine、Git SHA、测试计数均不进入 scope hash。每个运行承载 `RunScientificIdentity(scope_id, scope_hash, scientific_config_hash, reference_policy_id, repository_id, source_commit, source_tree_hash, worktree_dirty, commit_verified)`；实现 provenance 与 scientific scope identity 分离，不进入 scope hash。

`SOURCE_PROVENANCE_POLICY_V1` 只有在实际 Git root 等于 DRL root 且 canonical remote 匹配 `github.com/luckyfishanddog/DRL` 时才自动接受 HEAD；publishable formal result 还要求 clean worktree，并在运行入口重新解析和比对 provenance。嵌套在其他仓库中的 development smoke 必须显式 `allow_unverified_source=True`，输出 `UNVERIFIED_SOURCE_PROVENANCE` 与 `development_only=true`，不得冒充正式结果。

```python
from mrta_reference import FORMAL_SCOPE_V1, reference_schedule_formal, certify_schedule
schedule = reference_schedule_formal(solution, config, scope=FORMAL_SCOPE_V1,
                                     orientations=directions)
certificate = certify_schedule(solution, schedule, config, scope=FORMAL_SCOPE_V1)
# standalone clean luckyfishanddog/DRL checkout only:
# run_bounded_sa_oi(..., scope=FORMAL_SCOPE_V1, formal_result=True)
```

正式 ALNS 的 initial portfolio、candidate C4、direction refinement、final certification 共享一个 scope；显式 scope 下拒绝 development callback。开发历史 API `reference_schedule`/slow/optimized 仍是 `DEVELOPMENT_NO_REPAIR_V1`，`EXACT_Y_SCOPE_CURRENT_SEMANTICS` 保留。未来 HGA/WAG/ranker labels 必须显式使用 formal API 与同一 certifier。

## Scientific model relation 与限制

目标仍为 task-level weld allocation、sequencing、direction、theoretical coordination；保持 Cmax-first 与既有辅助指标词典序。不是 full robot motion planning、full physical parking 或 3D collision-free execution。

有限恢复既非完整可行性判定也非 exact scheduler。16 个 popped prefixes 至多访问深度 15 的 complete schedule；超过 15 个非 WAIT templates 的 baseline DEADLOCK 不可能在此 V1 recovery 内到达 complete leaf。此限制明确接受，不能以后随算法增大预算。大型 handover smoke 的死锁没有被隐瞒，后续科学 policy 变更需要新 scope version。

2026-09-29 large-state development stress 的原始审计曾按 local-plateau 规则保留 V1：31 个 unique baseline-DEADLOCK states 在 16/32/64 的 status/Cmax 相同，16–1024 无恢复，2048 恢复一个 certified handover-heavy/N100 state。后续 closure 明确纠正了科学解释：该 plateau 是 prefix depth censoring，因为 K-template complete leaf 至少需要 K+1 prefix states；它不能证明 16 已稳定。该事实触发 V1.1，而没有改写此处的历史 V1 identity。完整原始证据见 Pre-Phase3 release audit（对应过程文件已清理，结论保留在本报告）。

Development results 不能直接更名为 formal results：F4 policy/identity 改变，必须重新 evaluation/certification，Q4 已实际出现恢复调用。即便数值偶然相同，也分别携带 identity；本轮报告 development C* 与 formal Cref，不伪造正式 exact gap。

`mrta_exact` 仍是 development Y-scope dispatch+ESS micro backbone，保留 `Cmax_OPT_Y_CURRENT`。只有未来 Phase 6 同 final feasible set 并完成 EXACT_SCHEDULER_VALIDITY_GATE，才允许有限定地使用 Cmax_OPT。Adapted HGA/WAG、LB_LP、MLP/GAT、ranker dataset、formal VALIDATION/ID_TEST/OOD 与 final exact validation 均未完成。
