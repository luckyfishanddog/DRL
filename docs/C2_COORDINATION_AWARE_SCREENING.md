# Coordination-aware C2 screening (0.50 m development study)

本报告只使用 `V2_MODEL_DEVELOPMENT_CONSUMED`，当前任务域为 `delta_x=delta_y=0.50 m`。物理目录名不定义数据角色；访问由既有角色清单检查。没有访问 V2_VALIDATION、ORACLE_DEV_CONSUMED 或 ID_TEST_SEALED。生产初始化、M192、Kdp8、Kref2/Kref_total4、原 C4、方向 DP、B32 和 certifier 均未改变。

## 实际代码改动

- 新增纯 C2 几何代理：同轨左右机器人焊缝 X 占据区间（含 `interference_dx/2` 缓冲）的并集交叠长度，以及两侧首任务的左右次序超限量。它们只表示静态协调风险，不判 candidate 非法。
- `COORDINATION_AWARE_C2` 先保留每个非空 decision family 的原启发式最佳候选，再保留原 cheap-score 候选，使最多 2 个名额用于风险候选。风险候选必须不超过原 cheap Top8 的 projected-process-makespan 上限，并在加工负载/交叠/首任务风险上 Pareto 非劣；不足按原 cheap score 补齐。
- `BASELINE_C2` 仍是默认，C4 family explore/exploit 完全不变。指标不调用 direction、B32、certifier，不改 solution，也不消耗随机数。

## S1 的 443 个既有标签：指标是否有区分能力

| 类别 | n | overlap p25 | p50 | p75 | p90 | first-order p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| material | 44 | 5.056 | 6.750 | 8.409 | 15.945 | 0.000 | 0.000 |
| feasible_nonmaterial | 253 | 0.000 | 3.766 | 6.750 | 15.945 | 0.000 | 0.000 |
| deadlock | 125 | 9.362 | 14.107 | 16.234 | 17.232 | 0.000 | 0.500 |

同轨 overlap 对“认证可行风险更低”的 pairwise AUC 为 **0.819**，有实际区分力；首任务次序量 AUC 仅 **0.564**，且前三类中位数均为 0，因此只作 Pareto/tie-break，不给独立大权重。44 个 material candidate 的 overlap 中位数为 6.750 m，明显低于 DEADLOCK 的 14.107 m，指标没有系统性排斥 material candidate。
负载代价并非为零：13 个离线状态中 9 个发生换位，共 15 个名额。与被换出的 baseline 名额相比，换入候选的状态均值 projected-process-makespan 高 1.68%（最大 9.40%），load-spread delta 平均不利 81.3 s，但 overlap 低 2.19 m。因此本方法只保留 2 个风险名额，并用原 Top8 的 projected-process-makespan 上限限制负载代价；不把几何风险当作全局排名目标。
family 分布没有失去访问机会：按 STRUCTURAL/WHOLE/Y/X，S1 为 21/3/4/4→21/3/4/4；额外状态为 43/8/12/9→39/12/12/9。

## S1 四个固定状态重放

| 状态 | pool material | Baseline C2 feasible/material/best | New C2 feasible/material/best | Baseline→New C4 material |
|---|---:|---:|---:|---:|
| 20261081:2 | 10 | 0/0/NA | 1/1/10.94% | 0→0 |
| 20261081:8 | 11 | 1/0/-6.67% | 1/0/-6.67% | 0→0 |
| 20261083:2 | 15 | 3/2/10.13% | 3/2/10.13% | 0→0 |
| 20261083:8 | 8 | 1/1/11.96% | 1/1/11.96% | 1→1 |

合计 C2 认证可行 **5→6**，material **3→4**（443 个池内 material 的 6.82%→9.09%），material-zero 状态 **2→1**。C4 material 仍为 **1→1**。新 C2 在严重 seed 20261081 的 iteration 2 找回一个 10.94% 改善，但原 C4 没有选择它。四状态 C2 选择总耗时为 0.412→5.464 ms。

## 其他 consumed-development 状态

实例在查看 oracle label 前按静态 N、几何与拆分机会固定；状态规则固定为已完成 iteration 2 和 8，不存在 iteration 8 时记录缺失，不用未来状态补齐。

| 实例 | N | workbook / sheet | 捕获迭代 | 缺失 | 轨迹@60 Cmax |
|---|---:|---|---|---|---:|
| N35_S2 | 35 | `data/ID_TEST/seed_0401115467.xlsx` / `g19_w035` | [2, 8] | [] | 1911.268 |
| N35_S3 | 35 | `data/PPO_TRAIN/seed_0194123089.xlsx` / `g17_w035` | [2, 8] | [] | 1465.569 |
| N55_I5 | 55 | `data/PPO_TRAIN/seed_0900775608.xlsx` / `g30_w055` | [2] | [8] | 4028.165 |
| N55_I6 | 55 | `data/PPO_TRAIN/seed_1419626827.xlsx` / `g29_w055` | [2, 8] | [] | 6362.056 |
| N85_I9 | 85 | `data/PPO_TRAIN/seed_0194123089.xlsx` / `g43_w085` | [2] | [8] | 7944.236 |
| N85_I12 | 85 | `data/PPO_TRAIN/seed_0211781140.xlsx` / `g32_w085` | [2] | [8] | 4954.794 |

离线标注 9 个状态、1003 次 B32 reference，用时 319.38 s；它们与在线 60 秒调用分表统计。

| tier | 状态/material状态 | pool material | Baseline C2 feas/material/dead | New C2 feas/material/dead | C2 zero B→C | C4 material B→C | C4 zero B→C |
|---|---:|---:|---:|---:|---:|---:|---:|
| N35 | 4/2 | 6 | 13/2/18 | 19/3/12 | 0→0 | 1→1 | 1→1 |
| N55 | 3/3 | 27 | 6/3/18 | 9/4/15 | 0→0 | 1→1 | 2→2 |
| N85 | 2/1 | 16 | 3/2/12 | 3/2/12 | 0→0 | 0→0 | 1→1 |
| ALL | 9/6 | 49 | 22/7/48 | 31/9/39 | 0→0 | 2→2 | 4→4 |

额外状态中 C2 认证可行率为 **30.56%→43.06%**，material capture 为 **7/49→9/49（14.29%→18.37%）**，DEADLOCK 为 **48/72→39/72（66.67%→54.17%）**。有 material 的状态中，C2 zero capture 为 0→0。各规模 C2 最佳认证改善：N35 1.99%→1.99%，N55 2.05%→18.80%，N85 7.94%→7.94%。原 C4 的 material 仍是 2/49，六个 material-state 中 zero capture 仍为 4；因此新增 C2 收益没有传到 Top2。九状态 C2 选择总耗时为 1.052→125.171 ms。

## 24 次真实 60 秒配对

全部采用 BASELINE initialization；每个实例/seed 两方法的初始认证解相同。C2 风险时间计入 wall clock。下表迭代、reference、FEAS/DEAD 均截在 60 秒，Cmax 不使用 overshoot 后事件。

| instance | seed | Cmax B→C | relative | iter B/C | search ref B/C | FEAS B/C | DEAD B/C | C2 ms B/C | load spread B/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N35_S2 | 20261211 | 1896.523→1896.523 | +0.000% | 14/14 | 44/44 | 25/25 | 19/19 | 1.6/183.3 | 518.3/518.3 |
| N35_S2 | 20261212 | 1891.325→1891.325 | +0.000% | 12/12 | 44/44 | 31/31 | 13/13 | 1.3/149.2 | 473.5/473.5 |
| N35_S3 | 20261211 | 1465.569→1465.569 | +0.000% | 10/9 | 34/34 | 23/26 | 11/8 | 1.9/314.8 | 330.0/330.0 |
| N35_S3 | 20261212 | 1465.569→1465.569 | +0.000% | 10/10 | 36/36 | 24/24 | 12/12 | 2.0/420.1 | 330.0/330.0 |
| N55_I5 | 20261211 | 4112.445→4112.445 | +0.000% | 3/3 | 8/8 | 2/2 | 6/6 | 0.7/7.1 | 376.0/376.0 |
| N55_I5 | 20261212 | 4112.445→4112.445 | +0.000% | 3/3 | 8/8 | 2/2 | 6/6 | 0.8/8.0 | 376.0/376.0 |
| N55_I6 | 20261211 | 6687.390→6687.390 | +0.000% | 6/6 | 18/18 | 9/9 | 9/9 | 1.0/19.3 | 6452.9/6452.9 |
| N55_I6 | 20261212 | 6406.342→6406.342 | +0.000% | 5/5 | 12/12 | 3/3 | 9/9 | 1.2/13.8 | 6167.6/6167.6 |
| N85_I12 | 20261211 | 4954.794→4954.794 | +0.000% | 1/1 | 2/2 | 0/0 | 2/2 | 0.3/3.4 | 1050.5/1050.5 |
| N85_I12 | 20261212 | 4954.794→4954.794 | +0.000% | 1/1 | 2/2 | 0/0 | 2/2 | 0.3/3.8 | 1050.5/1050.5 |
| N85_I9 | 20261211 | 6978.750→6978.750 | +0.000% | 3/3 | 13/12 | 8/8 | 3/2 | 0.5/17.0 | 6353.1/6353.1 |
| N85_I9 | 20261212 | 8577.798→8577.798 | +0.000% | 3/3 | 10/10 | 6/6 | 4/4 | 0.9/9.9 | 8034.9/8034.9 |

| tier | mean Cmax B→C | completed iter B/C | accepted B/C | strict best updates B/C | FEAS B/C | DEAD B/C | C2 time B/C |
|---|---:|---:|---:|---:|---:|---:|---:|
| N35 | 1679.746→1679.746 | 46/45 | 28/29 | 10/10 | 103/106 | 55/52 | 6.8/1067.4 ms |
| N55 | 5329.655→5329.655 | 17/17 | 6/6 | 5/5 | 16/16 | 30/30 | 3.7/48.3 ms |
| N85 | 6366.534→6366.534 | 8/8 | 5/5 | 5/5 | 14/14 | 11/10 | 2.0/34.0 ms |
| ALL | 4458.645→4458.645 | 71/70 | 39/40 | 20/20 | 133/136 | 96/92 | 12.5/1149.7 ms |

24/24 在 60 秒有认证解，NUMERIC_FAILURE 和 certifier error 均为 0。**12/12 配对 Cmax 完全持平**，改善/持平/恶化为 **0/12/0**；N35、N55、N85 均无严重退化。60 秒内严格 Cmax 改善次数为 **20→20**。overshoot 为 1.180–9.777 s；6 个 run 在截止后才得到更优解，均未计入 Cmax@60。总在线 C2 时间为 12.5→1149.7 ms，新增方法占其 12×60 秒预算的 **0.160%**，约 16.42 ms/iteration。它减少 4 个 search DEADLOCK、增加 3 个 FEASIBLE，但没有改变 Cmax。

完整逐 run solution、四机器人 process loads、empty travel、waiting、候选有效率、best events、overshoot 和状态计数保存在既有 `data/development/s1_screening_diagnostic.sqlite` 的 `c2_experiments/search60` 记录。最大 Kdp 为 8、单 iteration reference 上限为 4，预算未扩大。

## 结论与决定

1. 新 C2 是“原 family + cheap-score 主体、最多 2 个负载受限协调风险名额”，没有把 overlap 当成不可行判据。
2. 同轨 overlap 能区分一部分 DEADLOCK；首任务次序代理在这批数据上信号弱。廉价几何只能提供有限信息。
3. S1 找回 1 个 material candidate（3→4），额外状态找回 2 个（7→9）；相对于完整池仍丢失绝大多数改善。
4. 原 C4 没保留新增收益：S1 C4 material 1→1，额外状态 2→2。低 reference 认证率与 C2/C4 送入大量 DEADLOCK 有关，但改进的少量 FEASIBLE 增量没有形成 Cmax收益。
5. 真实 60 秒 12 对 Cmax 全部持平，没有任何规模严重恶化；开销很小但当前不值得切换 production。
6. **production 继续使用 BASELINE_C2**。保留协调风险模式作为开发消融，不把它设为默认。
7. 下一阶段应在新 0.5 m 域按既有 TRAIN/DEV workbook 隔离重新采集正式标签并优先训练 ranking-aligned MLP-C2；同时把真实 C2 shortlist 作为 C4 hard distribution，因为本轮再次显示 C4 会丢掉 C2 增量。当前证据支持采集标签，但本轮未启动训练，也没有理由上 GAT。

## 验证

- 定向回归：138 passed。
- 完整 pytest：320 passed in 36.21 s.
- 新指标纯函数、baseline 默认行为、family access、唯一候选、Kdp8、最多两个 risk slot、加工负载 cutoff 以及不调用 direction/scheduler 均有回归覆盖。
- 旧 S1 的 443 条标签、此前 36 条诊断结果和所有历史实验保持不变。
