# Phase4-1C：Top-K aligned MLP ranker

结论：**FAIL**。最佳离线 **V2 / NEW-MLP-C2**；离线Gate **FAIL**；smoke **SKIPPED / 0 runs**；production **HEURISTIC**。

## 直接回答

1. 旧 p×gain 病理严重且真实存在：C2/C4 全体预测负gain为 75.42%/68.11%；280个真实material正样本中 141/102 个预测为负。不可行候选在gain不更好的情况下反超material正样本，分别发生在 23/36、22/36 states。独立阶段Top-K所选slot中出现该反转仅 2、1 states，不能把全部失败单独归因于该交互。
2. Material classification（V2）C2 Top8 median capture：baseline 0.00%→47.10%，增加 47.10 pp；相对V1 30.89% 再增加 16.21 pp。
3. Pairwise 未进一步提高：V3 C2 capture=14.26%，相对V2变化 -32.84 pp。V3 BOTH C4 zero=72.41%，V2 BOTH=62.07%。固定loss权重下没有继续搜索。
4. C4 hard-training使用 3097 个既有candidate（原全部direction-feasible TRAIN 10479），保留716个material positives。局部有用、整体未解决：V2中保持同一新版C2，把heuristic C4替换为新版C4，MEDIUM capture 0.00%→66.74%，但SMALL 6.36%→0.00%，LARGE zero 66.67%→75.00%。未训练‘新objective+旧C4分布’额外对照，不能独立归因于hard subset。
5. ALL C4 Top2 median capture：baseline 0.00%→最佳 0.00%，仍为0；6种新objective/variant均为0。
6. 最佳C4 zero capture：79.31%→62.07%，下降 17.24 pp；29个material states中最终捕获改善 11 个。
7. 最佳variant的 SMALL/MEDIUM/LARGE：SMALL C2 50.65%、C4 6.36%、C4 zero 45.45%；MEDIUM C2 88.83%、C4 0.00%、C4 zero 83.33%；LARGE C2 0.00%、C4 0.00%、C4 zero 66.67%。完整四组结果见下表。
8. LARGE仍系统失败：最佳C2/C4 median capture均0；12个material states中 8 个在C2已无改善可送入C4，0 个在C4丢光。V2 C2 LARGE TRAIN capture=73.12%、DEV=0，TRAIN仅作in-sample诊断。LARGE zero相对baseline 75.00%→66.67% 的改善不代表泛化问题已解决。
9. 最佳variant的 EARLY/MID/LATE：EARLY C2 52.10%、C4 30.68%、C4 zero 40.00%；MID C2 1.26%、C4 0.00%、C4 zero 80.00%；LATE C2 6.36%、C4 0.00%、C4 zero 66.67%。
10. 最佳描述性variant：V2 / NEW-MLP-C2。按ALL C4 capture median降序、zero升序、normalized regret升序选取；与V2 BOTH的capture/zero并列时，C2-only regret更低。无variant取得smoke资格。
11. 离线Gate：FAIL。最佳variant满足zero下降≥10pp、LARGE zero不恶化，但ALL median capture没有增加≥10pp。标准未降低。
12. 真实60秒Cmax：SKIPPED / 0 runs；离线失败后停止在线实验，未测量Cmax收益。
13. MLP开销：现有状态真实API重放的feature extraction+scaler+forward+ranking中位数 97.76 ms/state，模型load在计时外。216次比较与离线选择一致，新增reference/direction=0。没有60秒在线run，因此ranker/total time比例未测量。
14. 本轮能确认的失败位置：ALL 10/29 states在C2无捕获，另 8/29 states在C4丢光。既有LARGE跨workbook问题仍在，C4在SMALL/MEDIUM的取舍也没有形成整体收益。在线未执行，无法声称原因是overhead、trajectory shift、diversity或SA；这些须未来有在线证据才判断。
15. Production不切换：继续M192 + heuristic C2/C4；新版模型保留为失败实验和对照证据。
16. GAT：NOT YET。本轮没有实现/训练。LARGE失败和TRAIN/DEV差距不足以证明缺少weld-robot-route关系表达；尤其pairwise本身未改善、C4分组效果不一致，不能直接推出需要GAT。
17. 下一步：保持heuristic，下一轮先用现有labels分析LARGE跨workbook错误与hard-negative覆盖，再决定是否更多LARGE TRAIN states或关系建模。本轮停止，不新增数据、模型搜索、operator或在线运行。

## 旧score审计：36个DEV states

每个数列依次为 median / p25 / p75 / p90。FEASIBLE_MATERIAL=≥0.5%；FEASIBLE_NONMATERIAL包括零改善、负改善与<0.5%改善。C4禁止direction-infeasible输入，因此该组为0行，不给伪造统计。审计只使用旧checkpoint、旧scaler和现有DEV labels。

| stage | group | n | negative gain | p_feasible | predicted_gain | p×gain |
|---|---|---:|---:|---|---|---|
| C2 | FEASIBLE_MATERIAL | 280 | 141 | 0.565477 / 0.050637 / 0.909666 / 0.972418 | -0.000577 / -0.050713 / 0.037735 / 0.089314 | -0.000045 / -0.029404 / 0.010900 / 0.042797 |
| C2 | FEASIBLE_NONMATERIAL | 1868 | 1484 | 0.713094 / 0.433284 / 0.944505 / 0.980776 | -0.053806 / -0.107093 / -0.011686 / 0.034020 | -0.031338 / -0.060201 / -0.002347 / 0.014685 |
| C2 | DEADLOCK | 1935 | 1453 | 0.374114 / 0.131745 / 0.637369 / 0.819023 | -0.045455 / -0.103570 / -0.000176 / 0.039927 | -0.014391 / -0.040364 / -0.000004 / 0.007710 |
| C2 | DIRECTION_INFEASIBLE | 51 | 40 | 0.772584 / 0.396749 / 0.935290 / 0.975679 | -0.071387 / -0.137846 / -0.039857 / 0.032620 | -0.042708 / -0.059349 / -0.016468 / 0.027047 |
| C4 | FEASIBLE_MATERIAL | 280 | 102 | 0.297416 / 0.140586 / 0.659407 / 0.872674 | 0.031481 / -0.043513 / 0.080985 / 0.115622 | 0.004906 / -0.016225 / 0.019609 / 0.042071 |
| C4 | FEASIBLE_NONMATERIAL | 1868 | 1418 | 0.709923 / 0.515526 / 0.877845 / 0.945748 | -0.049770 / -0.106221 / -0.001719 / 0.041089 | -0.028870 / -0.058698 / -0.001042 / 0.031880 |
| C4 | DEADLOCK | 1935 | 1261 | 0.262800 / 0.137163 / 0.499113 / 0.749706 | -0.042068 / -0.113652 / 0.024905 / 0.084128 | -0.008318 / -0.036872 / 0.003572 / 0.019066 |
| C4 | DIRECTION_INFEASIBLE | 0 | 0 | N/A | N/A | N/A |

审计中的selected-inversion统计为独立阶段全pool Top8/Top2，用来量化病理；C4该诊断并非串联Top8→Top2选择率，最终结论只用下方真实串联离线表。

## 固定训练配置与C4分布

网络：C2 87→128→64、C4 130→128→64，各自独立，两head feasibility/material；不保留gain head，无attention/GAT。material=(status==FEASIBLE_CERTIFIED and (Cs−Ccandidate)/Cs≥0.005)，NUMERIC_FAILURE排除。score仅sigmoid(material_logit)，tie使用原heuristic。C2全valid unique M192 candidates；C4只用direction-feasible hard subset。

Loss V2 = feasibility BCE + TRAIN-pos-weight material BCE；V3另加权重1的softplus(-(sp−sn))。每positive最多4个hard negatives，交替取heuristic、V1得分最靠前的negative，始终同state；pairwise对有pair的state等权平均，每个classification minibatch加同一有限pair集合的平均loss。没有DEADLOCK人工Cmax penalty。

AdamW lr=1e-3、weight_decay=1e-4、batch512、max100、patience10；ReLU，stage seeds C2=20261123/C4=20261124。V2/V3相同初始化，无超参搜索。直接复用V1的TRAIN-only scaler，不覆盖旧文件。C4 hard subset为 baseline C2 Top8、新V3 C2 Top8、V1 C2 Top8、所有material-positive direction-feasible、并集外heuristic C3/rerank额外8个的去重并集。

为隔离pairwise，V2/V3 C4共享一次构造的V3-C2 hard subset，DEV epoch也共享baseline/V3-C2两种固定Top8 contexts。C4只在这两种Top8内用原family policy选Top2，两context等权，改善分母仍为完整pool；C2 epoch直接评价完整pool Top8。最后四组串联消融中，各objective使用自己的C2。C4 epoch没有从百余direction-feasible候选直接选Top2。

| objective | stage | TRAIN rows | positives | negatives | pos_weight | pairs | selected epoch | epochs run |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | C2 | 10759 | 716 | 10043 | 14.026536 | 0 | 6 | 16 |
| V2 | C4 | 3097 | 716 | 2381 | 3.325419 | 0 | 1 | 11 |
| V3 | C2 | 10759 | 716 | 10043 | 14.026536 | 2864 | 5 | 15 |
| V3 | C4 | 3097 | 716 | 2381 | 3.325419 | 2864 | 1 | 11 |

原split完全保留：17 TRAIN/6 DEV workbook，98/36 states、10,759/4,134 candidates，numeric failure=0；C4 hard rows=3,097、material positives=716、direction-infeasible=0；只有TRAIN计数计算pos_weight。原8 ORACLE_DEV_CONSUMED的结果不用于训练/模型选择，V2_VALIDATION及最终ID_TEST未访问，无candidate随机拆分。

## 同预算四组离线结果

C2每非空family先选1个、再全局补至8；C4保留global-best slot和seed+iteration family rotation/fallback slot，不改成global Top2。全部M192 / Kdp8 / Kref2 / Kref_total4，TWO_OPT_STAR OFF。V1只重放，不重新训练，全部group指标与Phase4-1B原JSON完全相同。

Capture = selected best捕获的改善 / 全pool最佳改善；material state仍要求pool存在≥0.5%改善。Zero capture指没有捕获任何正改善。Capture/zero/regret仅汇总material states，state等权；direct normalized regret只在selected feasible存在时定义，无feasible为NULL，JSON保留defined counts和fallback regret。Feasibility统计该group全部selected candidates。

| objective | variant | group | material states | C2 capture | C2 zero | C2 regret | C2 feasible | C4 capture | C4 zero | C4 regret | C4 feasible |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | BASELINE | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 79.31% | 9.76% | 40.28% |
| V1 | BASELINE | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 72.73% | 13.03% | 66.67% |
| V1 | BASELINE | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 20.83% |
| V1 | BASELINE | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 75.00% | 5.90% | 33.33% |
| V1 | BASELINE | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 0.00% | 70.00% | 6.89% | 54.17% |
| V1 | BASELINE | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 80.00% | 12.62% | 41.67% |
| V1 | BASELINE | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 8.93% | 25.00% |
| V1 | MLP-C2 | ALL | 29 | 30.89% | 48.28% | 6.92% | 60.07% | 0.00% | 68.97% | 7.63% | 55.56% |
| V1 | MLP-C2 | SMALL | 11 | 46.12% | 18.18% | 7.11% | 86.46% | 46.12% | 27.27% | 7.11% | 83.33% |
| V1 | MLP-C2 | MEDIUM | 6 | 66.74% | 33.33% | 1.59% | 46.88% | 0.00% | 100.00% | N/A | 37.50% |
| V1 | MLP-C2 | LARGE | 12 | 0.00% | 83.33% | 8.12% | 46.88% | 0.00% | 91.67% | 8.12% | 45.83% |
| V1 | MLP-C2 | EARLY | 10 | 63.11% | 20.00% | 7.11% | 52.08% | 23.25% | 50.00% | 7.63% | 41.67% |
| V1 | MLP-C2 | MID | 10 | 0.00% | 70.00% | 5.63% | 65.62% | 0.00% | 80.00% | 6.88% | 66.67% |
| V1 | MLP-C2 | LATE | 9 | 0.00% | 55.56% | 7.43% | 62.50% | 0.00% | 77.78% | 8.03% | 58.33% |
| V1 | MLP-C4 | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 72.41% | 6.89% | 45.83% |
| V1 | MLP-C4 | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 54.55% | 9.25% | 75.00% |
| V1 | MLP-C4 | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 33.33% |
| V1 | MLP-C4 | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 75.00% | 4.90% | 29.17% |
| V1 | MLP-C4 | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 25.15% | 50.00% | 6.89% | 58.33% |
| V1 | MLP-C4 | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 80.00% | 7.78% | 41.67% |
| V1 | MLP-C4 | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 6.74% | 37.50% |
| V1 | MLP-BOTH | ALL | 29 | 30.89% | 48.28% | 6.92% | 60.07% | 0.00% | 68.97% | 9.56% | 63.89% |
| V1 | MLP-BOTH | SMALL | 11 | 46.12% | 18.18% | 7.11% | 86.46% | 20.96% | 27.27% | 12.62% | 95.83% |
| V1 | MLP-BOTH | MEDIUM | 6 | 66.74% | 33.33% | 1.59% | 46.88% | 0.00% | 100.00% | N/A | 50.00% |
| V1 | MLP-BOTH | LARGE | 12 | 0.00% | 83.33% | 8.12% | 46.88% | 0.00% | 91.67% | 8.12% | 45.83% |
| V1 | MLP-BOTH | EARLY | 10 | 63.11% | 20.00% | 7.11% | 52.08% | 8.53% | 50.00% | 12.82% | 58.33% |
| V1 | MLP-BOTH | MID | 10 | 0.00% | 70.00% | 5.63% | 65.62% | 0.00% | 90.00% | 10.37% | 70.83% |
| V1 | MLP-BOTH | LATE | 9 | 0.00% | 55.56% | 7.43% | 62.50% | 0.00% | 66.67% | 8.03% | 62.50% |
| V2 | BASELINE | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 79.31% | 9.76% | 40.28% |
| V2 | BASELINE | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 72.73% | 13.03% | 66.67% |
| V2 | BASELINE | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 20.83% |
| V2 | BASELINE | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 75.00% | 5.90% | 33.33% |
| V2 | BASELINE | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 0.00% | 70.00% | 6.89% | 54.17% |
| V2 | BASELINE | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 80.00% | 12.62% | 41.67% |
| V2 | BASELINE | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 8.93% | 25.00% |
| V2 | MLP-C2 | ALL | 29 | 47.10% | 34.48% | 3.94% | 57.64% | 0.00% | 62.07% | 7.11% | 56.94% |
| V2 | MLP-C2 | SMALL | 11 | 50.65% | 9.09% | 3.94% | 78.12% | 6.36% | 45.45% | 12.62% | 79.17% |
| V2 | MLP-C2 | MEDIUM | 6 | 88.83% | 16.67% | 1.32% | 53.12% | 0.00% | 83.33% | 0.00% | 41.67% |
| V2 | MLP-C2 | LARGE | 12 | 0.00% | 66.67% | 6.53% | 41.67% | 0.00% | 66.67% | 4.90% | 50.00% |
| V2 | MLP-C2 | EARLY | 10 | 52.10% | 10.00% | 3.40% | 55.21% | 30.68% | 40.00% | 5.52% | 58.33% |
| V2 | MLP-C2 | MID | 10 | 1.26% | 50.00% | 5.27% | 62.50% | 0.00% | 80.00% | 12.62% | 54.17% |
| V2 | MLP-C2 | LATE | 9 | 6.36% | 44.44% | 3.87% | 55.21% | 0.00% | 66.67% | 7.56% | 58.33% |
| V2 | MLP-C4 | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 79.31% | 8.12% | 45.83% |
| V2 | MLP-C4 | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 63.64% | 11.67% | 70.83% |
| V2 | MLP-C4 | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 37.50% |
| V2 | MLP-C4 | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 83.33% | 8.12% | 29.17% |
| V2 | MLP-C4 | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 10.45% | 50.00% | 7.90% | 50.00% |
| V2 | MLP-C4 | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 100.00% | 10.37% | 41.67% |
| V2 | MLP-C4 | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 9.42% | 45.83% |
| V2 | MLP-BOTH | ALL | 29 | 47.10% | 34.48% | 3.94% | 57.64% | 0.00% | 62.07% | 7.83% | 58.33% |
| V2 | MLP-BOTH | SMALL | 11 | 50.65% | 9.09% | 3.94% | 78.12% | 0.00% | 63.64% | 12.62% | 75.00% |
| V2 | MLP-BOTH | MEDIUM | 6 | 88.83% | 16.67% | 1.32% | 53.12% | 66.74% | 33.33% | 1.59% | 50.00% |
| V2 | MLP-BOTH | LARGE | 12 | 0.00% | 66.67% | 6.53% | 41.67% | 0.00% | 75.00% | 7.55% | 50.00% |
| V2 | MLP-BOTH | EARLY | 10 | 52.10% | 10.00% | 3.40% | 55.21% | 29.94% | 40.00% | 7.37% | 54.17% |
| V2 | MLP-BOTH | MID | 10 | 1.26% | 50.00% | 5.27% | 62.50% | 0.00% | 90.00% | 8.12% | 54.17% |
| V2 | MLP-BOTH | LATE | 9 | 6.36% | 44.44% | 3.87% | 55.21% | 0.00% | 55.56% | 9.68% | 66.67% |
| V3 | BASELINE | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 79.31% | 9.76% | 40.28% |
| V3 | BASELINE | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 72.73% | 13.03% | 66.67% |
| V3 | BASELINE | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 20.83% |
| V3 | BASELINE | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 75.00% | 5.90% | 33.33% |
| V3 | BASELINE | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 0.00% | 70.00% | 6.89% | 54.17% |
| V3 | BASELINE | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 80.00% | 12.62% | 41.67% |
| V3 | BASELINE | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 8.93% | 25.00% |
| V3 | MLP-C2 | ALL | 29 | 14.26% | 41.38% | 5.57% | 55.56% | 0.00% | 62.07% | 7.78% | 59.72% |
| V3 | MLP-C2 | SMALL | 11 | 47.10% | 18.18% | 6.21% | 80.21% | 30.89% | 36.36% | 10.67% | 91.67% |
| V3 | MLP-C2 | MEDIUM | 6 | 28.23% | 50.00% | 3.98% | 50.00% | 0.00% | 83.33% | 0.00% | 45.83% |
| V3 | MLP-C2 | LARGE | 12 | 0.00% | 58.33% | 5.63% | 36.46% | 0.00% | 75.00% | 7.78% | 41.67% |
| V3 | MLP-C2 | EARLY | 10 | 38.43% | 30.00% | 7.11% | 50.00% | 23.55% | 50.00% | 7.27% | 50.00% |
| V3 | MLP-C2 | MID | 10 | 1.26% | 50.00% | 5.92% | 59.38% | 0.00% | 60.00% | 6.51% | 58.33% |
| V3 | MLP-C2 | LATE | 9 | 0.85% | 44.44% | 4.22% | 57.29% | 0.00% | 77.78% | 8.96% | 70.83% |
| V3 | MLP-C4 | ALL | 29 | 0.00% | 62.07% | 5.77% | 37.50% | 0.00% | 79.31% | 10.72% | 43.06% |
| V3 | MLP-C4 | SMALL | 11 | 73.64% | 27.27% | 2.42% | 68.75% | 0.00% | 63.64% | 11.67% | 79.17% |
| V3 | MLP-C4 | MEDIUM | 6 | 0.00% | 100.00% | N/A | 17.71% | 0.00% | 100.00% | N/A | 37.50% |
| V3 | MLP-C4 | LARGE | 12 | 0.00% | 75.00% | 7.50% | 26.04% | 0.00% | 83.33% | 6.89% | 12.50% |
| V3 | MLP-C4 | EARLY | 10 | 52.45% | 40.00% | 3.44% | 38.54% | 10.45% | 50.00% | 7.11% | 54.17% |
| V3 | MLP-C4 | MID | 10 | 0.00% | 70.00% | 9.39% | 40.62% | 0.00% | 100.00% | 12.62% | 37.50% |
| V3 | MLP-C4 | LATE | 9 | 0.00% | 77.78% | 5.44% | 33.33% | 0.00% | 88.89% | 10.72% | 37.50% |
| V3 | MLP-BOTH | ALL | 29 | 14.26% | 41.38% | 5.57% | 55.56% | 0.00% | 72.41% | 8.90% | 59.72% |
| V3 | MLP-BOTH | SMALL | 11 | 47.10% | 18.18% | 6.21% | 80.21% | 0.00% | 81.82% | 13.03% | 79.17% |
| V3 | MLP-BOTH | MEDIUM | 6 | 28.23% | 50.00% | 3.98% | 50.00% | 28.23% | 50.00% | 2.45% | 54.17% |
| V3 | MLP-BOTH | LARGE | 12 | 0.00% | 58.33% | 5.63% | 36.46% | 0.00% | 75.00% | 7.55% | 45.83% |
| V3 | MLP-BOTH | EARLY | 10 | 38.43% | 30.00% | 7.11% | 50.00% | 0.00% | 80.00% | 13.44% | 41.67% |
| V3 | MLP-BOTH | MID | 10 | 1.26% | 50.00% | 5.92% | 59.38% | 0.00% | 80.00% | 8.12% | 70.83% |
| V3 | MLP-BOTH | LATE | 9 | 0.85% | 44.44% | 4.22% | 57.29% | 0.00% | 55.56% | 7.50% | 66.67% |

## LARGE泛化与严格Gate

| objective | variant | TRAIN LARGE C2 capture | DEV LARGE C2 capture | TRAIN LARGE C4 capture | DEV LARGE C4 capture | DEV no C2 capture | DEV C4 dropped all |
|---|---|---:|---:|---:|---:|---:|---:|
| V2 | MLP-C2 | 73.12% | 0.00% | 65.93% | 0.00% | 8/12 | 0/12 |
| V2 | MLP-C4 | 8.93% | 0.00% | 0.00% | 0.00% | 9/12 | 1/12 |
| V2 | MLP-BOTH | 73.12% | 0.00% | 35.21% | 0.00% | 8/12 | 1/12 |
| V3 | MLP-C2 | 66.47% | 0.00% | 55.65% | 0.00% | 7/12 | 2/12 |
| V3 | MLP-C4 | 8.93% | 0.00% | 3.07% | 0.00% | 9/12 | 1/12 |
| V3 | MLP-BOTH | 66.47% | 0.00% | 4.41% | 0.00% | 7/12 | 2/12 |

TRAIN为in-sample诊断，不用于epoch/variant选择。

| objective | variant | ALL capture +10pp | ALL zero −10pp | LARGE zero not worse | gate |
|---|---|---|---|---|---|
| V2 | MLP-C2 | False | True | True | FAIL |
| V2 | MLP-C4 | False | False | False | FAIL |
| V2 | MLP-BOTH | False | True | True | FAIL |
| V3 | MLP-C2 | False | True | True | FAIL |
| V3 | MLP-C4 | False | False | False | FAIL |
| V3 | MLP-BOTH | False | False | True | FAIL |

## 开销与验证

下表是现有DEV accepted states上的真实接口重放中位数（ms/state）。包含feature extraction、scaler、forward、ranking；load排除。只跑一次重放，受当前硬件/进程预热影响，不外推60秒在线占比。heuristic阶段计为0 ranker time；不是整轮solver总时间。

| objective | variant | group | C2 ms | C4 ms | total ms |
|---|---|---|---:|---:|---:|
| V2 | MLP-C2 | ALL | 97.760 | 0.000 | 97.760 |
| V2 | MLP-C2 | SMALL | 64.850 | 0.000 | 64.850 |
| V2 | MLP-C2 | MEDIUM | 91.598 | 0.000 | 91.598 |
| V2 | MLP-C2 | LARGE | 136.312 | 0.000 | 136.312 |
| V2 | MLP-C4 | ALL | 0.000 | 9.150 | 9.150 |
| V2 | MLP-C4 | SMALL | 0.000 | 7.639 | 7.639 |
| V2 | MLP-C4 | MEDIUM | 0.000 | 9.125 | 9.125 |
| V2 | MLP-C4 | LARGE | 0.000 | 12.473 | 12.473 |
| V2 | MLP-BOTH | ALL | 96.460 | 8.975 | 105.953 |
| V2 | MLP-BOTH | SMALL | 64.605 | 7.529 | 71.832 |
| V2 | MLP-BOTH | MEDIUM | 90.802 | 8.754 | 99.288 |
| V2 | MLP-BOTH | LARGE | 134.995 | 12.289 | 147.121 |
| V3 | MLP-C2 | ALL | 96.088 | 0.000 | 96.088 |
| V3 | MLP-C2 | SMALL | 67.018 | 0.000 | 67.018 |
| V3 | MLP-C2 | MEDIUM | 90.502 | 0.000 | 90.502 |
| V3 | MLP-C2 | LARGE | 137.014 | 0.000 | 137.014 |
| V3 | MLP-C4 | ALL | 0.000 | 9.294 | 9.294 |
| V3 | MLP-C4 | SMALL | 0.000 | 7.604 | 7.604 |
| V3 | MLP-C4 | MEDIUM | 0.000 | 8.942 | 8.942 |
| V3 | MLP-C4 | LARGE | 0.000 | 12.523 | 12.523 |
| V3 | MLP-BOTH | ALL | 95.324 | 9.141 | 104.233 |
| V3 | MLP-BOTH | SMALL | 64.761 | 7.691 | 72.664 |
| V3 | MLP-BOTH | MEDIUM | 90.241 | 8.775 | 98.759 |
| V3 | MLP-BOTH | LARGE | 138.639 | 12.429 | 151.881 |

281 passed; final full DRL suite 35.16s; initial exact python -B -m pytest -q -p no:cacheprovider from DRL 35.24s

真实model API重放36 states × 2 objectives × 3 variants = 216次比较，mismatch=0，新增reference/direction=0；禁用scheduler与trajectory入口执行重放。必要回归包括0.499%/0.500%边界、pairwise优化方向、C4 subset包含Top8并排除direction-infeasible、material score不受feasibility负gain交互影响、family coverage/rotation、Kdp/Kref预算、推理不调用scheduler、workbook隔离。

没有生成新candidate dataset、trajectory或label，没有新protocol/hash/scope/manifest，没有改M/Kdp/Kref、X/Y、scheduler/B32、SA、operator或production默认。历史Phase4-1B FAIL、0 MLP smoke runs、旧模型/scaler/核心报告和JSON保持不变。四个新checkpoint直接保存最佳epoch，无临时checkpoint或cache，清理删除0文件。仅新增已确认的4个模型、1个结果JSON和本报告。
