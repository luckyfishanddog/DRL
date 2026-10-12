# GRC-ALNS：第一版 GAT 结果

**结论：保持 HEURISTIC production。** GAT 离线减少错误筛选，但真实60秒为3胜0平3负，N35两个seed均退化；关系边消融不降，当前不推荐切换默认方法。

TRAIN 98 states / 11924 candidates / 590 material positives；DEV 34 / 4184 / 356。17/6 workbook隔离；6个缺失EARLY未回填；旧0.2 m数据未混入。

## 离线完整 C2→direction→C4

| method | tier | C2 material capture | C2 zero | C2 feasible | C4 material capture | C4 zero | C4 feasible |
|---|---|---:|---:|---:|---:|---:|---:|
| HEURISTIC | N35 | 6.52% | 72.73% | 25.00% | 2.17% | 90.91% | 12.50% |
| HEURISTIC | N55 | 18.87% | 27.27% | 50.00% | 6.60% | 63.64% | 66.67% |
| HEURISTIC | N85 | 4.41% | 30.00% | 18.75% | 0.00% | 100.00% | 15.00% |
| HEURISTIC | ALL | 8.99% | 43.75% | 31.99% | 2.25% | 84.38% | 32.35% |
| MLP | N35 | 4.35% | 81.82% | 21.88% | 0.00% | 100.00% | 25.00% |
| MLP | N55 | 12.26% | 54.55% | 41.67% | 2.83% | 81.82% | 75.00% |
| MLP | N85 | 14.22% | 0.00% | 46.25% | 5.39% | 0.00% | 75.00% |
| MLP | ALL | 12.36% | 46.88% | 36.03% | 3.93% | 62.50% | 57.35% |
| GAT | N35 | 15.22% | 63.64% | 28.12% | 10.87% | 63.64% | 33.33% |
| GAT | N55 | 16.98% | 45.45% | 57.29% | 5.66% | 63.64% | 79.17% |
| GAT | N85 | 13.24% | 0.00% | 41.25% | 2.94% | 40.00% | 50.00% |
| GAT | ALL | 14.61% | 37.50% | 42.28% | 4.78% | 56.25% | 54.41% |
| GAT_NO_WELD_EDGES | N35 | 15.22% | 63.64% | 28.12% | 10.87% | 63.64% | 33.33% |
| GAT_NO_WELD_EDGES | N55 | 16.98% | 45.45% | 57.29% | 5.66% | 63.64% | 79.17% |
| GAT_NO_WELD_EDGES | N85 | 13.24% | 0.00% | 41.25% | 2.94% | 40.00% | 50.00% |
| GAT_NO_WELD_EDGES | ALL | 14.61% | 37.50% | 42.28% | 4.78% | 56.25% | 54.41% |

GAT相对heuristic：C2 material capture 8.99%→14.61%，认证率 31.99%→42.28%，zero 43.75%→37.50%；C4 capture 2.25%→4.78%、zero 84.38%→56.25%。C2收益仍有大部分被Top2丢失。
删除weld-weld边后，全部分层Top-K指标相同。本轮不能把GAT相对MLP的增量归因于图关系；顺序编码、机器人资格边和优化差异尚未独立隔离。

## 真实60秒开发 smoke

| tier | seed | initial Cmax | HEURISTIC Cmax@60 | MLP | MLP Δ | GAT | GAT Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| N35 | 20261301 | 2127.430481 | 1900.957819 | 1989.217434 | 4.64% | 1911.269634 | 0.54% |
| N35 | 20261302 | 2127.430481 | 1814.455522 | 1935.667778 | 6.68% | 1964.467833 | 8.27% |
| N55 | 20261301 | 4141.496389 | 4028.164648 | 4031.469182 | 0.08% | 3538.525857 | -12.16% |
| N55 | 20261302 | 4141.496389 | 3990.218625 | 4004.264635 | 0.35% | 3920.096481 | -1.76% |
| N85 | 20261301 | 9831.816611 | 6567.867875 | 7490.650306 | 14.05% | 7007.759036 | 6.70% |
| N85 | 20261302 | 9831.816611 | 9458.574922 | 8124.982008 | -14.10% | 7711.637970 | -18.47% |

胜/平/负：MLP 1/0/5，median Δ=2.50%；GAT 3/0/3，median Δ=-0.61%。N35两次退化，N55两次改善，N85一胜一负；没有跨workbook稳定性证据。

| method | iterations | reference calls | FEASIBLE/DEADLOCK/INFEASIBLE/NUMERIC | strict best@60 | ranker time | overshoot |
|---|---:|---:|---|---:|---:|---|
| HEURISTIC | 71 | 268 | 128/106/0/0 | 30 | 0.000s | 1.964–6.335s |
| MLP | 80 | 340 | 236/70/0/0 | 29 | 10.695s | 1.196–8.285s |
| GAT | 71 | 304 | 209/61/0/0 | 29 | 30.896s | 0.956–11.250s |

## 直接结论

- 训练：MLP C2/C4 rows=11924/2238；GAT encoder梯度L1=11.7263；C4 hard rows=2238。
- 开销：DEV GAT C2/C4=11.600/11.465s，静态attention各6次。在线GAT 30.896s，占360s的 8.58%；build/forward/score=0.094/0.055/22.089s。
- 完整性：18/18有截止前认证Cmax，NUMERIC_FAILURE=0。学习排序降低DEADLOCK、增加FEASIBLE reference，却未稳定降低Cmax。
- 决策：保留MLP/GAT为development选项；production继续BASELINE initialization + HEURISTIC C2/C4，不打开ID_TEST。
- 下一步：扩大development workbook复验N35/N85；在真实Top8分布下优化C4并隔离关系边/顺序编码；批处理动态candidate降低开销，然后冻结方法做独立validation。

固定条件：0.5 m、M192=144+48、Kdp8、Kref2/Kref_total4、BASELINE initialization、原SA/direction/B32/certifier、TWO_OPT_STAR OFF。无额外reference预算。本结果不是SCI正式统计结论。

完整回归：`pytest -q -p no:cacheprovider`，323 passed。
