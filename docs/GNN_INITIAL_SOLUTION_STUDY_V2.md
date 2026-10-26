# GNN fixed-split initial-solution study V2

## A. 实际完成了什么

在既有两层、4 头、hidden=64 GAT 上增加一个轻量排序头；归属头继续使用合法机器人掩码。排序训练只使用同机器人任务对，按路线归一化；推理通过全局拓扑追加序列构造四条路线。固定拆分中的 mandatory Y 两个子块是可并行覆盖任务，当前科学模型没有额外的父子时间边；若传入显式前驱，构造器只从前驱已完成集合选取，因此显式前驱边和四条路线边均沿同一全局序列，不会形成跨机器人环。方向仍由原 DP 决定，随后运行原 B32 和独立 certifier。

旧 `gnn_constructor_v1.pt`、VNS N1/N2/N3、C2/C4、SA、旧 GAT 排序器及科学参数均未修改。新联合 GNN、MLP、去焊缝边 GNN 使用相同 17 个已保存 TRAIN 专家、相同固定 seed、相同 epoch 和损失定义；checkpoint 固定取最后 epoch，未根据 DEV 结果选择。

## 专家排序可学习性检查

17/17 专家通过覆盖、唯一归属、合法机器人和固定拆分核对；17/17 当前路线都与简单 X 升序不同。总同机器人对 7834，其中逆序 394（5.03%）；改变路线 30/68。全部 17 条专家解已在当前代码中重放并经方向 DP、B32 与 certifier 认证。

## B. 核心实验结果（无 VNS、无回退）

| 实例 | 方法 | 状态 | Cmax | 构造秒 | 端到端秒 | 相对 RULE |
|---|---|---:|---:|---:|---:|---:|
| g21_w035 | A RULE / X | CERTIFIED | 2449.298 | 0.0020 | 0.0495 | +0.00% |
| g21_w035 | B GNN V1 / X | CERTIFIED | 3065.201 | 0.0159 | 0.0600 | +25.15% |
| g21_w035 | C RULE / GNN order | CERTIFIED | 2454.380 | 0.0330 | 0.0832 | +0.21% |
| g21_w035 | D GNN assign / GNN order | CERTIFIED | 2521.857 | 0.0357 | 0.0813 | +2.96% |
| g21_w035 | E RULE / MLP order | DEADLOCK | — | 0.0297 | 0.1120 | — |
| g21_w035 | F RULE / no-weld-edge GNN order | CERTIFIED | 2485.692 | 0.0326 | 0.1017 | +1.49% |
| g15_w035 | A RULE / X | CERTIFIED | 1663.724 | 0.0020 | 0.0450 | +0.00% |
| g15_w035 | B GNN V1 / X | CERTIFIED | 2096.493 | 0.0138 | 0.0607 | +26.01% |
| g15_w035 | C RULE / GNN order | CERTIFIED | 1653.687 | 0.0843 | 0.1294 | -0.60% |
| g15_w035 | D GNN assign / GNN order | CERTIFIED | 2101.689 | 0.0425 | 0.0890 | +26.32% |
| g15_w035 | E RULE / MLP order | CERTIFIED | 1714.932 | 0.0269 | 0.0643 | +3.08% |
| g15_w035 | F RULE / no-weld-edge GNN order | CERTIFIED | 1712.617 | 0.0314 | 0.0707 | +2.94% |
| g36_w086 | A RULE / X | CERTIFIED | 3908.021 | 0.0044 | 1.1022 | +0.00% |
| g36_w086 | B GNN V1 / X | CERTIFIED | 3919.599 | 0.0391 | 0.1645 | +0.30% |
| g36_w086 | C RULE / GNN order | CERTIFIED | 3905.831 | 0.1757 | 0.2973 | -0.06% |
| g36_w086 | D GNN assign / GNN order | DEADLOCK | — | 0.1841 | 0.5111 | — |
| g36_w086 | E RULE / MLP order | CERTIFIED | 4020.200 | 0.1559 | 0.4076 | +2.87% |
| g36_w086 | F RULE / no-weld-edge GNN order | DEADLOCK | — | 0.1517 | 0.6033 | — |
| g33_w055 | A RULE / X | CERTIFIED | 4074.442 | 0.0028 | 0.0789 | +0.00% |
| g33_w055 | B GNN V1 / X | CERTIFIED | 4048.139 | 0.0225 | 0.4134 | -0.65% |
| g33_w055 | C RULE / GNN order | CERTIFIED | 4107.448 | 0.0714 | 0.5887 | +0.81% |
| g33_w055 | D GNN assign / GNN order | CERTIFIED | 4029.621 | 0.0774 | 0.1484 | -1.10% |
| g33_w055 | E RULE / MLP order | CERTIFIED | 4107.124 | 0.0606 | 0.2011 | +0.80% |
| g33_w055 | F RULE / no-weld-edge GNN order | CERTIFIED | 4295.020 | 0.0646 | 0.3154 | +5.41% |
| g31_w055 | A RULE / X | CERTIFIED | 3190.602 | 0.0030 | 0.1145 | +0.00% |
| g31_w055 | B GNN V1 / X | CERTIFIED | 3623.945 | 0.0240 | 0.1046 | +13.58% |
| g31_w055 | C RULE / GNN order | CERTIFIED | 3365.711 | 0.0808 | 1.2250 | +5.49% |
| g31_w055 | D GNN assign / GNN order | CERTIFIED | 3801.415 | 0.0991 | 0.2076 | +19.14% |
| g31_w055 | E RULE / MLP order | CERTIFIED | 3142.475 | 0.0891 | 0.1787 | -1.51% |
| g31_w055 | F RULE / no-weld-edge GNN order | CERTIFIED | 3142.345 | 0.0914 | 0.1772 | -1.51% |
| g45_w082 | A RULE / X | CERTIFIED | 5790.590 | 0.0047 | 2.8590 | +0.00% |
| g45_w082 | B GNN V1 / X | CERTIFIED | 6826.015 | 0.0415 | 3.5465 | +17.88% |
| g45_w082 | C RULE / GNN order | CERTIFIED | 5838.860 | 0.2427 | 0.3722 | +0.83% |
| g45_w082 | D GNN assign / GNN order | CERTIFIED | 5838.860 | 0.2382 | 0.3617 | +0.83% |
| g45_w082 | E RULE / MLP order | DEADLOCK | — | 0.2141 | 3.4343 | — |
| g45_w082 | F RULE / no-weld-edge GNN order | CERTIFIED | 5821.621 | 0.1916 | 0.3234 | +0.54% |

| 方法 | 认证 | 胜/平/负 vs A | 配对中位变化 | 平均构造秒 | 平均端到端秒 |
|---|---:|---:|---:|---:|---:|
| A RULE / X | 6/6 | 0/6/0 | +0.00% | 0.0032 | 0.7082 |
| B GNN V1 / X | 6/6 | 1/0/5 | +15.73% | 0.0261 | 0.7249 |
| C RULE / GNN order | 6/6 | 2/0/4 | +0.51% | 0.1147 | 0.4493 |
| D GNN assign / GNN order | 5/6 | 1/0/4 | +2.96% | 0.1128 | 0.2332 |
| E RULE / MLP order | 4/6 | 1/0/3 | +1.84% | 0.0961 | 0.7330 |
| F RULE / no-weld-edge GNN order | 5/6 | 1/0/4 | +1.49% | 0.0939 | 0.2653 |

## C. GNN 实际学到了什么

RULE 归属 + GNN 排序在 DEV 上的直接证据：胜/平/负 2/0/4，配对中位 +0.51%。联合归属+排序为 1/0/4，中位 +2.96%。

MLP 排序中位 +1.84%；去焊缝关系边 GNN 排序中位 +1.49%。三种排序器都没有稳定优于 RULE，且认证成功集合并不完全相同；因此当前 DEV 证据不足以证明焊缝—焊缝图关系优于相同输入的 MLP 或去边 GNN。

## D. 当前失败原因

排序监督虽可重放，但信息密度低：专家实例的 X 对逆序中位仅 4.29%，而 C 的预测路线逆序中位达到 16.02%。加入低权重 X 一致对后已消除 C 的 DEADLOCK，但仍未把排序分数稳定转化为更小 Cmax，说明小样本单一路线标签不足以识别哪些逆序真正改善协调等待。

联合归属的主要失败证据更直接：D 相对 RULE 改变 3–7 个块，并在 6/6 个 DEV 实例上都扩大了四机器人加工负载极差；因此排序头还承受了训练时专家归属与推理时新归属的分布偏移。D 的收益不能归因于工程认证问题，因为其已认证实例中仍为 1 胜 4 负。

MLP 与去边 GNN 也没有稳定改善，无法仅凭完整 GNN 的中位退化略小就宣称图关系有效。直接构造失败与 B32 DEADLOCK 均保留为失败，不解释成数学不可行，也未用修复器或 VNS 掩盖。

## E. 下一步判断

- GNN 排序：当前不值得增加层数或引入 PPO。
- 联合构造：当前不应宣称优于 RULE；若排序有效应保留 RULE+排序。
- 扩大专家数据：若继续 GNN 则必要；必须增加能区分有效/无效逆序和真实归属变化的专家，而不是重复近似 X 顺序。
- 可选 X/Y 拆分：本轮没有足够依据引入；应等固定拆分下图排序价值成立后再研究。

DEV 未出现明确稳定改善迹象，按停止条件未运行 20 组开发阶段几何验证。

完整逐实例负载、空行程、等待、任务数、方向/B32/certifier 时间和失败诊断见 CSV。
