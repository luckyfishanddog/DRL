# DRL 已完成工作与实验结果

整理日期：2026-11-03。研究设计见[多机器人焊缝分配与排序实验方案](多机器人焊缝分配与排序实验方案.md)。下表按主题归并旧报告，保留原报告标题、已完成的工作和关键结果。

## 模型、调度与候选池

| 已做工作 | 结果与复用边界 | 原报告标题 |
| --- | --- | --- |
| V1/V1.1 正式域和 F4 校准 | 完成完整 rollout 的有限预算调度、31 状态回归及独立语料的 B32/B64/B128 标定。F4 发布通过，V1/V1.1 现为历史口径。 | 《FORMAL_SCOPE_V1》 · 《FORMAL_SCOPE_V1_1 F4 Closure — 2026-09-29》 · 《Final F4 Release Calibration》 |
| V2 共同模型与搜索可达性 | WHOLE、Y_SPLIT、X_SPLIT 进入统一合法模式目录。3-YR 的 ALNS/HGA/WAG 可达性验证通过。3-Z 实跑 180 次，179 次最终认证，因 1 次 HGA 无认证解保留 FAIL。3-ZR 开发审计 HGA 37/40 认证，历史 746.78 秒长尾未复现。 | 《FORMAL_SCOPE_V2 — Candidate Formal Definition》 · 《Phase 3-YR — V2 Search-Access Closure — 2026-10-06》 · 《Phase 3-Z — V2 Common-Model Validation（2026-10-06）》 · 《Phase 3-ZR — Reference Runtime 与 Certified-Incumbent 审计》 |
| Phase 4 候选池与 MLP 排序（历史 0.2 m 域） | 4-0/0B 评估 oracle 召回、实质改善与筛选损失，推荐 M192，主要剩余损失在 C2/C4。4-1A 完成 72 次 60 秒原生运行并准备训练数据。4-1B/1C 离线均 FAIL，60 秒 MLP smoke 实际为 0 次。1C 最佳 C2 capture 47.10%，C4 中位仍为 0%，pairwise 未进一步提升。 | 《Phase4-0 V2 Candidate-Pool Oracle Recall Audit》 · 《Phase4-0B：候选池规模、materiality 与筛选损失分析》 · 《Phase4-1A：真实60秒生产候选池与MLP数据准备》 · 《Phase4-1B — MLP Candidate Ranker》 · 《Phase4-1C：Top-K aligned MLP ranker》 |
| 0.50 m 范围敏感性、初始化和 S1 筛选 | 24 次范围敏感性运行全部认证，有改善也有退化。初始化修复 I3 串行退化，但 S2 两次恶化 39.67%/21.70%。S1 四个候选池共 44 个认证改善，C2 保留 3 个，C4 仅 1 个。协调风险 C2 增加候选召回，12 个 60 秒配对的 Cmax 全部持平，保留原 heuristic。 | [YX_SPLIT_RANGE_05_CHANGE.md](YX_SPLIT_RANGE_05_CHANGE.md) · 《初始化鲁棒性与S1筛选诊断（0.5 m开发域）》 · 《Coordination-aware C2 screening (0.50 m development study)》 |

当前实验方案记录的任务层口径为拆分窗口 X/Y 均 0.50 m、最短子段 0.20 m、干涉距离 0.50 m。工程配置为 M192（144 atomic + 48 LNS）、Kdp8/Kref2/Kref_total4、heuristic C2/C4、TWO_OPT_STAR 关闭。轨道覆盖属于任务层假设。历史 0.2 m 数据、已消耗 DEV 和独立测试结果分别使用。

## GRC-ALNS、正式比较与协调增强

| 已做工作 | 结果与复用边界 | 原报告标题 |
| --- | --- | --- |
| GAT V05/V06 和 C4 排序 | 首批 18 次 60 秒运行全部认证，GAT 对 heuristic 3 胜 3 负。V06 对 heuristic 为 6/2/4（胜/平/负），中位 Cmax −0.01%。C4-ranked 最终淘汰，保留 heuristic 默认及 V05 学习消融，结束 V07/V08 方向。 | 《GRC-ALNS：第一版 GAT 结果》 · 《GRC-ALNS V2 结果》 · 《GRC-ALNS C4 最终选择》 |
| 统一 0.50 m 五算法比较 | 开发矩阵 24/24，正式矩阵 375/375 均获截止前认证解。15 个 workbook 上，GAT 对 heuristic 8/2/5（中位 −0.06%），对 MLP 13/1/1（−6.49%），对 HGA 2/0/13（+18.28%），对 WAG 0/1/14（+19.37%）。去掉关系边的 C2/C4 capture 仅下降约 0.28/0.56 个百分点。 | 《SCI统一0.5 m最终实验结果》 |
| REGION 初始化、LNS 加速与独立比较 | REGION 开发配对对原 heuristic 为 9/3/0，中位改善 4.56%。LNS 等价加速提高同墙钟搜索机会。后续独立对比支持 REGION_FAST_HEURISTIC 相对原 ALNS 的改进，对 WAG 仍为 7/0/8，高工作量实例仍有明显差距。 | 《区域协同初始化最终结果（0.5 m consumed-development）》 · 《LNS 等价计算加速结果》 · 《Upgraded GRC independent comparison》 |
| CRRR 与 C2 access，JRRB | CRRR 保留默认关闭。C2 access 将 CRRR 选中数从 32 增至 71，C4 仍仅 10 次且 B32 全部 DEADLOCK，零接纳。JRRB 仅 1/6 开发几何有认证改善，24 次 60 秒矩阵未执行。 | 《CRRR 结构性搜索邻域开发结果》 · 《CRRR 协调感知 C2 访问结果》 · 《JRRB joint rail-region rebuilding results》 |
| 同轨再平衡、协调根因与 ROUTE_REORDER | MEDIUM_02/LARGE_04 有合法同轨迁移空间，均衡初值在方向或 B32 阶段失败后回退串行。边界重平衡在 3 个开发几何的 6 个配对降低负载与认证 Cmax，保留默认关闭。最小路线重排随后完成独立验证，20 个 workbook 中 18 个持平，作为确定性 GRC-ALNS 增强保留。完整高工作量配对仍落后 WAG，中位 +47.49%。 | 《Same-Rail Load Rebalance（同轨边界重平衡）开发结果》 · 《协调可行性根因与最小路线修复结果》 · [FINAL_COORDINATION_COMPARISON.md](FINAL_COORDINATION_COMPARISON.md) |

## 固定拆分构造与联合重构

| 已做工作 | 结果与复用边界 | 原报告标题 |
| --- | --- | --- |
| GNN 构造、VNS 修复、教师与计时审计 | 首轮单独学习机器人归属缺少稳定优势。修复 VNS 后，6 个 DEV 上 RULE/GNN 初值各有 6/6 改善。端到端计时已纳入加载模型、构造、修复和认证。17/17 TRAIN 教师严格改善，平均改善 4.58%，但 980 个有选择的块仅 17 个改归属，且全部为同轨。V2 排序/联合初解未稳定超过 RULE。 | 《GNN Construct + VNS Pilot》 · 《VNS 邻域诊断与修复实验报告》 · 《GNN+VNS 端到端计时与固定拆分专家数据审计》 · 《GNN fixed-split initial-solution study V2》 |
| 固定模式瓶颈选区、方向嵌入与 Beam 初解 | 瓶颈选区 320 次尝试、317 个组内唯一完整候选，231 COMPLETED、72 DEADLOCK、14 方向失败，19 个组内改善。方向嵌入在同 RANDOM 移除输入下改善候选由 15 增至 21，尝试级 B32 成功均为 60/96。后续 Beam 对 Greedy 初解 5/0/1，统一 Beam 后 NEW 对 OLD 的严格改善为 36/288 对 32/288，B32 成功为 201/288 对 190/288。 | 《瓶颈驱动固定模式邻域开发实验（2026-10-27）》 · 《方向嵌入联合插入修复开发实验》 · 《联合方向初解与修复后续实验》 |
| JointRepair V1/V2 与父选择消融 | V1 合法补全 288 次，229 次 B32 成功，严格改善 7 次。V2 严格改善 25/288、B32 成功 223/288。V2 对 NEW 的独立候选最佳为 4/1/1，有状态 24 轮最终为 3/1/2。learned 对 MRV-LPT 的六例最佳为 3/1/2，其中一例更换了共同输入，原输入可重放的五例为 2/1/2。 | 《GNN 联合部分解补全首轮实验》 · 《GNN JointRepair V2：扩大 TRAIN 与真实调度反馈实验》 |
| 真实 Cmax 强化学习、WAIT 和直接 blocker | RL 确实更新策略参数，但对 V2 为 0 胜 5 平 1 负，保留原 V2。WAIT 反馈使 B32 成功由 114/144 降至 99/144，接纳 9→10，接纳次数与最终 Cmax 分开判断。直接 blocker 在共同 5 个几何上对 WAIT 为 0/2/3，对 TRAVEL 为 0/0/5，未纳入最终重构。 | 《GNN V2：真实 Cmax 反馈微调》 · 《GNN-V2：调度等待反馈选区实验》 · 《GNN联合重构：最终调度直接blocker选区实验》 |
| 共享联合构造和四候选神经初解 | Shared 纯初解贪心仅 1/6 可调度。Beam+Shared 对 Beam+NEW 的 24 轮最终结果为 2/2/2。Multi4 将 Shared 初解提高至 3/6，V2 仍为 1/6。纯初解失败及无回退条件均计入分母。 | 《共享 GNN 联合初解构造与瓶颈重构》 · 《GNN 四候选联合初解实验》 |

联合重构中，B32 的 DEADLOCK 表示在给定预算内未找到调度。固定部分解的独立补全、24 轮有状态搜索、60 秒端到端求解分别比较。六个 DEV 的候选和轮次重复使用相同几何，统计时以几何作为独立单位。重用已有结果时，优先定位报告里的原始 CSV/JSONL、`DRL/results` 或 `DRL/data/development` 下的 SQLite 数据，再核对口径和数据角色。

## 本次清理

删除 40 份阶段性报告。`YX_SPLIT_RANGE_05_CHANGE.md` 含供脚本读取的嵌入式范围和 SMALL_30_39 原始记录，`FINAL_COORDINATION_COMPARISON.md` 含 blocker 实验使用的 20 个 workbook/sheet 映射，两份仍保留在 `docs`。本次未重跑训练、优化或仿真。
