# Pre-Phase3 Release Audit — 2026-09-29

`PRE_PHASE3_RELEASE_STATUS = PASS`
`ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1`
`NEXT_PHASE = Phase 3 — Adapted HGA / Adapted WAG common-model`

本轮只闭合 source provenance 与 F4 large-state stress validation；没有实现 HGA/WAG、扩展 ALNS、改变 F1/F2/F3、优化 exact 或使用未来 VALIDATION/ID_TEST/OOD。

## A. Baseline SHA 与 Git topology

本地实际 Git root 为 `D:/pybullet_test/MRTA_GA`，HEAD 为 `0935ad8a723057084be190e31c6be4733ce3cb62`，origin 为 `https://github.com/luckyfishanddog/MRTA.git`，worktree dirty。`D:/pybullet_test/MRTA_GA/DRL` 不是独立 Git root；从该目录执行 `git rev-parse --show-toplevel` 仍返回外层 MRTA root，DRL 整体在外层状态中为 untracked。

远端 `https://github.com/luckyfishanddog/DRL.git` 可访问；审计时 `main/HEAD = 52e13d23f871cb83ed4d2f93391b8ebff01238dc`。该 SHA 只描述 GitHub 已发布 DRL 仓库，不能冒充当前本地未发布 source tree；外层 MRTA SHA 也不再写入 DRL formal identity。

用户给定的 `D:\pybullet_test.venv\Scripts\python.exe` 不存在；使用实际项目环境 `D:\pybullet_test\.venv\Scripts\python.exe`。修改前 baseline：

| cwd / command | 结果 |
|---|---:|
| DRL：`-m pytest -q -p no:cacheprovider` | 157 passed in 22.86s |
| 外层 root：`-m pytest -q -p no:cacheprovider tests DRL/tests` | 529 passed, 1 skipped, 3 subtests passed in 56.43s |

## B. Source provenance bug

旧 profiler 在当前 nested workspace 直接执行 `git rev-parse HEAD`，得到外层 MRTA commit `0935ad8…`，随后把它写入 DRL `RunScientificIdentity.source_commit`。该值既不是 GitHub DRL HEAD，也不能标识本地 DRL 未提交代码，会污染后续 SA-OI-ALNS/HGA/WAG、ranker labels 与正式表格。

## C. Provenance fix

新增 `src/mrta_reference/provenance.py`，定义 `SOURCE_PROVENANCE_POLICY_V1`、`SourceProvenance`、deterministic source-tree hash 与 fail-closed resolver。

自动接受 commit 必须同时满足：实际 Git root 精确等于调用方给定 DRL root，且至少一个 canonical remote 精确匹配 `github.com/luckyfishanddog/DRL`。Formal result 入口再次从实际执行目录解析 provenance，并要求与传入对象完全相等、`commit_verified=true`、`worktree_dirty=false`；手工伪造 verified dataclass 不能绕过。Wrong remote、outer/nested root、dirty formal checkout 均拒绝 FORMAL_RESULT。

Nested development smoke 只有显式 `allow_unverified_source=True` 且显式 source label 才能运行；本轮使用 `source_commit=UNVERIFIED_LOCAL_TREE`，结果携带 `commit_verified=false`、`worktree_dirty=true`、`development_only=true` 和 `UNVERIFIED_SOURCE_PROVENANCE`。

Source-tree hash 只覆盖按 POSIX relative path 排序的 `src/**/*.py` 与 `pyproject.toml`，以路径长度、路径 bytes、文件长度、文件 bytes 做 SHA-256 framing；排除 docs、corpus、timestamp、machine path、`__pycache__`、pyc 与 pytest cache。最终执行代码的 hash：

```text
eebcd1f5ad0fa86ba4defa8051fb40754db88768ac173c4f668efe58137b52b0
```

## D. DRL `.gitignore`

本地 DRL root 原先没有自身 `.gitignore`，不能依赖外层规则。现已新增且只包含 `__pycache__/`、`*.py[cod]`、`.pytest_cache/`、`*.egg-info/`。外层 `git ls-files '*__pycache__*' '*.pyc'` 无匹配；由于本地 DRL 不是 standalone checkout，这一结果只证明外层 index 未跟踪 bytecode。发布 DRL 时其自身 `.gitignore` 随源码进入仓库，不能把当前外层 index 或 SHA 当成 published DRL 证据。

## E. F4 stress corpus

唯一机器文件为 `data/development/f4_deadlock_stress_corpus.json`，schema=`F4_DEADLOCK_STRESS_CORPUS_V1`，不是 VALIDATION/TEST 数据。先固定运行四个现有 development families 的 N=50/100、seeds 20260928–20260930、每次 2 iterations；真实 unique 不足 30 后，只按固定递增 seed 继续 handover-heavy/N100，未制造样本。

29 个 search runs 产生 43 次 baseline-DEADLOCK observations，按 `solution canonical hash + explicit directions + scientific_config_hash` 去重为 31 entries：handover-heavy/N50 1 个（196 templates），handover-heavy/N100 30 个（每个 396 templates）。其他 families 在这些固定轨迹上没有 baseline DEADLOCK entry。每项保存 canonical solution/pattern/routes、directions、config/hash、baseline diagnostics/wait-for graph/canonical hash、origin，并由 JSON 重建后检查逐字节 canonical round-trip；无 pickle。

Corpus 保留 collection provenance；最终 replay 另记最终 code provenance。Offline replay 固定 solution、assignment、route、directions 与 config，不重新运行 search。

## F. Budgets 16..2048

31 个候选全部实际完成八档 replay。`baseline p50=0.501949 s`，`baseline p95=1.303090 s`。表中 recovery/overall 为 31 candidates 的实测秒数：

| budget | certified FEASIBLE | DEADLOCK | recovery p50 | recovery p95 | overall p50 | overall p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 31 | 0.006846 | 0.007768 | 0.509080 | 1.309927 |
| 32 | 0 | 31 | 0.014105 | 0.016411 | 0.516016 | 1.317508 |
| 64 | 0 | 31 | 0.029884 | 0.057243 | 0.532777 | 1.348177 |
| 128 | 0 | 31 | 0.065304 | 0.236993 | 0.565191 | 1.526955 |
| 256 | 0 | 31 | 0.267290 | 0.969303 | 0.700339 | 2.267579 |
| 512 | 0 | 31 | 1.421869 | 3.366399 | 1.925255 | 4.666092 |
| 1024 | 0 | 31 | 4.762889 | 7.963997 | 5.266276 | 9.193413 |
| 2048 | 1 | 30 | 10.169545 | 17.150127 | 10.672932 | 18.434898 |

所有未恢复结果保持 DEADLOCK；没有 INFEASIBLE、penalty Cmax、big-M 或被吞掉的 NUMERIC_FAILURE。状态随 budget 单调；唯一恢复项之后没有更大已测 budget，所有同时 FEASIBLE 的 budget pairs 满足 Cmax non-worsening；所有 recovered schedules 都通过 formal certifier。

## G. handover-heavy/N100

旧 budget 16 的 30 个 unique handover-heavy/N100 DEADLOCK candidates，在 32、64、128、256、512、1024 中各有 0 个转为 certified FEASIBLE，在 2048 中有 1 个转为 certified FEASIBLE。

该 entry identity 为 `328926b7612633797dbe0ae62b62964bbf0a6765b1ef0d6147305b07a0ec6511`，solution hash 为 `3a5c430bf2e04c163172394debccb4655afe7d59717c69197c49514d30ea953c`，来自 seed 20260932；396 templates，2048 states，Cmax=2298.92129651784，certified=true。其余 29 个 N100 entries 到 2048 仍为 DEADLOCK。结论不是单一的 A/B：16 对这一项确实太小；但扩大到 2048 对绝大多数 corpus 仍无恢复，说明当前 bounded DFS 对 large-state cases 的恢复能力本身有限。

## H. Plateau decision

严格执行给定定义：对 B 比较 B、2B、4B 的逐 candidate status、Cmax 与 certification。Local plateau candidates 为 16、32、64、128、256；最小值 16。16/32/64 对全部 31 项均为同一 DEADLOCK、Cmax=null，因此 `FORMAL_SCOPE_V1` 保持不变，不创建 V1_1，也不把预算偷偷提高到 2048。

这是 depth-censored local plateau，不是 global recovery plateau：每项至少 196 templates，而 V1 的 16 popped-prefix states 不可能到达 complete leaf；2048 出现一个新恢复明确证明更大搜索可以改变个别结果。Release rule 仍选择 16，因为任何 `B* <= 256` 都没有恢复收益，且本轮规则要求最小 local plateau；报告不得把此结论写成“16 足以恢复大型死锁”。未来若改变 policy，必须新 scope version并重评结果。

## I/J. Active formal identity

```text
ACTIVE_FORMAL_SCOPE = FORMAL_SCOPE_V1
scope_hash = 8c8c056c5d22a4f706d62b4b7ce6ae1f522fc67105975fff346b93ede1f344f9
repository_id = luckyfishanddog/DRL
source_provenance_policy = SOURCE_PROVENANCE_POLICY_V1
reference_policy_id = FORMAL_BOUNDED_DISPATCH_POLICY_V1
deadlock_state_budget = 16
```

F1 remains OPTIONAL_X_SPLIT=EXCLUDED；F2 remains TASK_HORIZON_RELEASE_V1；F3 remains UNDEPLOYED_NO_OCCUPANCY_COMPLETION_ZERO_V1。Scope hash 未混入 commit/tree hash。当前本地 release evidence 是 `UNVERIFIED_SOURCE_PROVENANCE` development evidence；正式实验必须在 standalone clean DRL checkout 中重新生成 verified source identity。

## K. Evaluator performance

Stress 表给出了 baseline/recovery/overall p50/p95；budget 16 的 recovery p95 仅 0.007768 s，而 2048 为 17.150127 s，且仅多恢复 1/31，因此没有性能或 recovery 证据支持把 V1 偷升到 2048。

另实际运行 handover-heavy/N100、5 s formal SA-OI-ALNS gate：initialization success，2 个真实 search iterations，Nref=4，最终 certified，actual runtime=8.076377 s，overshoot=3.076377 s；init scheduler p50/p95=0.315626/0.437859 s，search p50/p95=0.834689/1.252713 s，overall p50/p95=0.522060/1.232006 s。5 个 baseline deadlocks、0 recovered、5 remaining；运行仍进入真实 search。Overshoot 来自已存在的 iteration-boundary hard work，本轮未优化 repair enumeration。

## L. E1–E4

最终 V1/16 实测：E1 BASELINE FEASIBLE、Cmax=2.0、0 states、certified；E2 BASELINE FEASIBLE、Cmax=14.16227766056838、0 states、certified；E3 BOUNDED_DEADLOCK_RECOVERY FEASIBLE、Cmax=15.528763461900223、16 states、certified；E4 DEADLOCK、7 states、frontier exhausted。E1/E2 canonical timeline 保持 baseline；E4 没有被写成 INFEASIBLE。

## M. Q1–Q6

每例同 seed 重放两次并比较 best solution JSON、directions 与 schedule JSON；全部 reproducible、COMPLETED、final certified、无 formal exact-gap 声明：

| case | seed / iterations | initial Cref | final Cref | improvement source |
|---|---:|---:|---:|---|
| Q1 assignment | 7 / 1 | 12.84 | 7.15 | ATOMIC |
| Q2 route | 43 / 8 | 11.545657757864154 | 11.468135899374975 | ATOMIC |
| Q3 direction | 3 / 20 | 7.105576065114842 | 6.271416626834708 | DIRECTION_REFINEMENT |
| Q4 optional Y | 0 / 10 | 6.0 | 4.0 | ATOMIC；10 recovery calls |
| Q5 interference | 37 / 12 | 7.105576065114842 | 6.271416626834708 | ATOMIC |
| Q6 LNS basin | 1 / 4 | 19.35791211412712 | 3.557181856472053 | LNS_REPAIRED |

## N. Development-family release smoke

四 families × N=20/50/100 × 3 seeds = 36 runs，均 initial success、2 iterations、COMPLETED、final certified。总 search Nref=272，C4 FEASIBLE=253、DEADLOCK=19；从 initialization 与 search reference records 合计 baseline deadlocks=28、recovered=0、remaining=28。LNS 共 1152 attempts、45 C4、22 accepted。非 handover families 的 baseline deadlocks 均为 0。

重点 handover rows：

| N | seeds | baseline / recovered / remaining（每 seed） | search C4 feasible rate | final certified |
|---:|---|---|---|---|
| 50 | 20260928/29/30 | 1 / 0 / 1 | 1.0 / 1.0 / 1.0 | 3/3 |
| 100 | 20260928/29/30 | 5 / 0 / 5 | 0.0 / 0.0 / 0.0 | 3/3 |

N100 的 best 均来自 certified initialization（Cref=2210.9259259259275）；formal search 的四次 C4 reference calls/seed 均 DEADLOCK，但算法完成 2 iterations，没有 NUMERIC_FAILURE 或 uncertified best。该结果被明确保留，不能用提高 evaluator budget 美化 ALNS Cmax。

## O. Tests 与命令

最终代码实际执行：

```powershell
# cwd D:\pybullet_test\MRTA_GA\DRL
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
# 163 passed in 25.23s

# cwd D:\pybullet_test\MRTA_GA
& 'D:\pybullet_test\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests DRL/tests
# 535 passed, 1 skipped, 3 subtests passed in 58.04s
```

新增覆盖 nested outer repo rejection、matching remote/root auto verify、wrong remote fail-closed、explicit unverified development-only、tree hash stability/source mutation/cache exclusion、完整 result identity、formal-result provenance gate、budget monotonic recovery、Cmax non-worsening、all-recovery certification、plateau detector，以及既有 E1/E2 bypass、E3 recovery、E4 DEADLOCK。没有删除困难测试。

## P. Phase 3 authorization

Provenance 已 fail closed；DRL 自身 `.gitignore` 存在；最终 source tree hash 可复现；31-state large corpus 与八档 offline replay 完成；状态/Cmax 单调且所有 recovery certified；V1/16 按指定 local plateau rule 保留；E1–E4、Q1–Q6、36-run smoke、N100/5s gate 与全量测试通过；F1/F2/F3 未改变。

因此允许进入 **Phase 3 — Adapted HGA / Adapted WAG common-model**。下一阶段先 paper-aligned baseline reconstruction，再接入同一 FORMAL_SCOPE_V1/evaluator/certifier/wall-clock boundary。本轮没有继续实现 HGA/WAG。
