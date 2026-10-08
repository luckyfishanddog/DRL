# Phase4-1B — MLP Candidate Ranker

结论：**FAIL**。推荐 **BASELINE**；M192 / Kdp8 / Kref2 / Kref_total4 不变。GAT：NOT YET。

完成日期：2026-10-08。报告文件名沿用此前批准的名称。

## 清理结果

删除 53 个无当前依赖的旧过程文件：旧 Phase3/4 runner、profiling、候选 oracle SQLite、重复中间 JSON 和 handoff。通用 typed reader、accepted-state capture、存储与数据准备移入正式 `mrta_ranker.dataset`，新流程不依赖历史 runner。删除明细及字节数保存在结果 JSON。没有创建 archive/backup，没有提交或上传 GitHub。

保留科学 src/tests、FORMAL_SCOPE_V2、实验方案、正式数据角色与固定 MLP split，以及解释当前算法的 Phase4-0B、Phase4-1A 等最终报告；清理后的文档链接已修复。退役104项仅服务旧runner/protocol/历史机械判定的过程测试，科学scheduler/certifier/search测试保留；清理后261项，加10项当前dataset/ranker及失败定位回归后271项通过。清理前已向用户说明DRL未被当前外层Git跟踪，用户明确授权仍直接删除；没有声称这些本地过程文件已存入当前Git历史。

## 训练数据与隔离

使用既有 17 TRAIN / 6 DEV workbook 白名单，每个选1个静态代表实例。TRAIN 为6/6/5、DEV为2/2/2 SMALL/MEDIUM/LARGE；只依据 N 静态选择，不读取 solver 结果。原8 ORACLE_DEV_CONSUMED 不参与本轮训练、epoch或variant选择；V2_VALIDATION和真正 sealed ID_TEST未访问。物理目录名不作为数据角色。
固定 seeds 20261121/20261122，46条保留的原生 M192 60s trajectory。延续 Phase4-1A 机制 smoke 的初始化 portfolio 配置 construction_budget=5 / kinit_ref=5 / bootstrap=1；baseline与MLP一致，正式 SearchConfig 默认未改。5/30/60s context 取时点前最后一个完整 iteration boundary 的 accepted-current，未用 global-best 代替、未从未来回填。
用户确认：初始化晚于5s时保留真实缺失，不强行补足138。实际 contexts=134，stage分布={'EARLY': 42, 'LATE': 46, 'MID': 46}；缺失时点与首次boundary时间逐条保存在JSON。

| split | states | candidates | FEASIBLE_CERTIFIED | DEADLOCK | DIRECTION_INFEASIBLE | INFEASIBLE |
|---|---:|---:|---:|---:|---:|---:|
| MLP_TRAIN | 98 | 10759 | 6351 | 4128 | 280 | 0 |
| MLP_DEV | 36 | 4134 | 2148 | 1935 | 51 | 0 |

Numeric failure=0；无未标注candidate、无C4 direction-infeasible row、无不可行样本的改善回归标签；全部C4记录的前87维与生成时C2记录逐字节一致，存储往返未改变公共特征。原生trajectory actual runtime median/max=62.86/69.49s。每个状态只生成一次144 atomic+48 LNS attempt pool，所有valid unique candidate做direction，direction-feasible才做B32 reference与认证。

采集错误如实保留：首批提交46条，采集器因缺EARLY抛错，shutdown等待其余任务却未收集结果，仅4条落盘；42条未保存的运行被重新执行。修复后每条worker独立事务落盘，已存4条复用。首批丢失结果未用于任何模型选择，未按质量选择性重试；实际 invocation 数因此超过46。

## 模型、特征与选择

独立 C2(87)→128→64 和 C4(130)→128→64 ReLU MLP，各有feasibility logit和normalized improvement两个head。C2使用全部valid unique候选；C4使用整个pool的全部direction-feasible候选，未限制为baseline Top8。
C2没有candidate direction/reference特征；C4只增加direction阶段信息。当前已知state Cmax合法，candidate reference Cmax/WAIT/DEADLOCK/certifier仅作标签。特征纯函数不调用scheduler、不修改state、不消耗RNG。TRAIN-only分别拟合mean/std，零方差std=1，DEV只apply。存储与在线提取都先使用相同float32原始特征语义，再标准化。
AdamW lr=1e-3、weight_decay=1e-4、batch512、最多100 epochs、patience10；BCE + feasible-only SmoothL1=1:1。不可行improvement=NULL，numeric failure排除。score=sigmoid(logit)×pred_gain，允许负gain。固定配置，没有超参搜索。
Epoch只按6个DEV workbook的material-state capture、zero capture和normalized regret选取。C2看完整pool的Top8，C4看完整direction-feasible pool的Top2；最终消融再串联C2→C4。

| model | training rows | selected epoch | epochs run | DEV material capture | zero capture |
|---|---:|---:|---:|---:|---:|
| C2 | 10759 | 5 | 15 | 30.89% | 48.28% |
| C4 | 10479 | 5 | 15 | 0.00% | 68.97% |

固定损失与预测的描述性诊断：
C2 selected epoch TRAIN BCE=0.398520、SmoothL1=0.003512；TRAIN feasible样本正改善比例=16.63%；DEV预测正gain比例=24.58%，predicted gain median=-0.047477。这些数字仅解释固定模型，不用于追加调权或选feature。
C4 selected epoch TRAIN BCE=0.338585、SmoothL1=0.002962；TRAIN feasible样本正改善比例=16.63%；DEV预测正gain比例=31.89%，predicted gain median=-0.043038。这些数字仅解释固定模型，不用于追加调权或选feature。

## 四组离线消融

主统计为state equal weight，仅在完整M192 pool存在≥0.5%改善机会的material states上汇总capture/zero/regret。C2和C4分母均为完整pool可利用改善。无可行selected candidate时capture=0、direct normalized regret=NULL；主regret=(selected best−pool best)/Cs（非负），不把更差候选截到incumbent。NULL不参与regret median，JSON记录regret_defined_states及保留incumbent的辅助fallback regret。Feasibility rate统计该group全部selected candidates，未限定material states。
C2每个非空family先选1个，再全局填满8；C4保留global-best + seed/iteration family explore及真实轮转fallback。四种variant只改score。Heuristic重放用原始双精度排序键，避免float32特征舍入改变基线。

| variant | group | material | C2 capture | C2 zero | C2 regret | C4 capture | C4 zero | C4 regret | C4 feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | ALL | 29 | 0.00% | 62.07% | 5.77% | 0.00% | 79.31% | 9.76% | 40.28% |
| BASELINE | SMALL | 11 | 73.64% | 27.27% | 2.42% | 0.00% | 72.73% | 13.03% | 66.67% |
| BASELINE | MEDIUM | 6 | 0.00% | 100.00% | — | 0.00% | 100.00% | — | 20.83% |
| BASELINE | LARGE | 12 | 0.00% | 75.00% | 7.50% | 0.00% | 75.00% | 5.90% | 33.33% |
| BASELINE | EARLY | 10 | 52.45% | 40.00% | 3.44% | 0.00% | 70.00% | 6.89% | 54.17% |
| BASELINE | MID | 10 | 0.00% | 70.00% | 9.39% | 0.00% | 80.00% | 12.62% | 41.67% |
| BASELINE | LATE | 9 | 0.00% | 77.78% | 5.44% | 0.00% | 88.89% | 8.93% | 25.00% |
| MLP-C2 | ALL | 29 | 30.89% | 48.28% | 6.92% | 0.00% | 68.97% | 7.63% | 55.56% |
| MLP-C2 | SMALL | 11 | 46.12% | 18.18% | 7.11% | 46.12% | 27.27% | 7.11% | 83.33% |
| MLP-C2 | MEDIUM | 6 | 66.74% | 33.33% | 1.59% | 0.00% | 100.00% | — | 37.50% |
| MLP-C2 | LARGE | 12 | 0.00% | 83.33% | 8.12% | 0.00% | 91.67% | 8.12% | 45.83% |
| MLP-C2 | EARLY | 10 | 63.11% | 20.00% | 7.11% | 23.25% | 50.00% | 7.63% | 41.67% |
| MLP-C2 | MID | 10 | 0.00% | 70.00% | 5.63% | 0.00% | 80.00% | 6.88% | 66.67% |
| MLP-C2 | LATE | 9 | 0.00% | 55.56% | 7.43% | 0.00% | 77.78% | 8.03% | 58.33% |
| MLP-C4 | ALL | 29 | 0.00% | 62.07% | 5.77% | 0.00% | 72.41% | 6.89% | 45.83% |
| MLP-C4 | SMALL | 11 | 73.64% | 27.27% | 2.42% | 0.00% | 54.55% | 9.25% | 75.00% |
| MLP-C4 | MEDIUM | 6 | 0.00% | 100.00% | — | 0.00% | 100.00% | — | 33.33% |
| MLP-C4 | LARGE | 12 | 0.00% | 75.00% | 7.50% | 0.00% | 75.00% | 4.90% | 29.17% |
| MLP-C4 | EARLY | 10 | 52.45% | 40.00% | 3.44% | 25.15% | 50.00% | 6.89% | 58.33% |
| MLP-C4 | MID | 10 | 0.00% | 70.00% | 9.39% | 0.00% | 80.00% | 7.78% | 41.67% |
| MLP-C4 | LATE | 9 | 0.00% | 77.78% | 5.44% | 0.00% | 88.89% | 6.74% | 37.50% |
| MLP-BOTH | ALL | 29 | 30.89% | 48.28% | 6.92% | 0.00% | 68.97% | 9.56% | 63.89% |
| MLP-BOTH | SMALL | 11 | 46.12% | 18.18% | 7.11% | 20.96% | 27.27% | 12.62% | 95.83% |
| MLP-BOTH | MEDIUM | 6 | 66.74% | 33.33% | 1.59% | 0.00% | 100.00% | — | 50.00% |
| MLP-BOTH | LARGE | 12 | 0.00% | 83.33% | 8.12% | 0.00% | 91.67% | 8.12% | 45.83% |
| MLP-BOTH | EARLY | 10 | 63.11% | 20.00% | 7.11% | 8.53% | 50.00% | 12.82% | 58.33% |
| MLP-BOTH | MID | 10 | 0.00% | 70.00% | 5.63% | 0.00% | 90.00% | 10.37% | 70.83% |
| MLP-BOTH | LATE | 9 | 0.00% | 55.56% | 7.43% | 0.00% | 66.67% | 8.03% | 62.50% |

总体离线指标最优（capture/zero/regret依次比较）：MLP-C2。MLP-BOTH是否最优：False。
MLP-C2：C2 capture变化+30.89个百分点；C4 capture变化+0.00个百分点；C4 zero capture下降+10.34个百分点。达到可进入native smoke的明确改善要求：False。
MLP-C4：C2 capture变化+0.00个百分点；C4 capture变化+0.00个百分点；C4 zero capture下降+6.90个百分点。达到可进入native smoke的明确改善要求：False。
MLP-BOTH：C2 capture变化+30.89个百分点；C4 capture变化+0.00个百分点；C4 zero capture下降+10.34个百分点。达到可进入native smoke的明确改善要求：False。

C4 median capture至少提高10个百分点且zero capture至少下降10个百分点：False。满足条件的最佳variant：None。

## 真实60s development

未执行（0 runs）：没有任何variant同时满足C4 median capture提高至少10个百分点、zero capture下降至少10个百分点。因此按预定条件跳过36次native smoke；没有Cmax@60、inference开销或LARGE在线收益结论。

## 结果解释

本轮完成数据与两个固定模型训练；FAIL指学习排序没有达到采用要求。离线未达条件时按约定不执行native smoke，当前没有真实60s MLP收益或inference开销结论。

MLP-C2：ALL material Top8 capture 0.00%→30.89%；MEDIUM 0.00%→66.74%。MEDIUM 6个material states中Top8有4个捕获改善，Top2最终有0个。

MLP-C2最终C4：SMALL capture 0.00%→46.12%；LARGE capture 0.00%→0.00%，zero capture 75.00%→91.67%。分stage结果见同预算表，不用局部收益替代整体要求。

总体描述性最优=MLP-C2，MLP-BOTH是否最优=False。selected feasibility改善不等于Top2 material improvement capture改善。

未发现label pipeline错误或泄漏：数值、公共特征往返、目标公式和C4阶段范围检查均为0异常。固定模型的Top-K效果不足，现有结果不能单独证明feature不足。

Selected-epoch BCE/SmoothL1数值比：C2 113.5倍 / C4 114.3倍。TRAIN feasible候选正改善比例=16.63%。逐candidate平均loss与少数material候选的Top-K目标存在可能不匹配，尚不能据此断言因果。

DEV predicted gain median：C2 -0.04748、C4 -0.04304。规定p×gain在gain<0时，降低p会使分数靠近0；校准与排序交互值得检查。本轮仍保留规定score与1:1 loss。

Direct regret只在selected feasible存在时有定义，各variant定义样本数不同，需结合JSON中的regret_defined_states与zero capture阅读。可选ranker接口已接入原生ALNS并通过预算/policy回归，生产默认仍为heuristic。


## 判断与下一步

最大描述性离线C4 capture增量：46.12个百分点，MLP-C2 / SMALL。SMALL/MEDIUM/LARGE、EARLY/MID/LATE与联合/单阶段替换的效果见相同预算表，不预设MLP-BOTH最好。

尚未证明可推荐的在线MLP收益。保留两个模型用于消融复查，生产继续heuristic。固定逐candidate BCE+SmoothL1并不直接优化Top-K capture，training loss下降不等于ranking有效。当前结果不足以断言特征缺失或标签错误；先检查目标与Top-K指标、负gain/尺度及score分布，不直接跳到GAT。

下一步：保持 M192 heuristic；下一轮限定改进 LARGE 的 C2 跨workbook泛化和 C4 state-level Top2 排序目标，复用既有TRAIN/DEV，不进入GAT。

没有修改M192、Kdp/Kref、科学scope、X/Y、scheduler/B32或算子；没有访问真正最终测试、训练GAT或重新划分workbook。


## 固定模型失败定位（继续分析）

仅复用已有TRAIN/DEV标签与两个已选checkpoint；没有新增trajectory、reference调用、训练epoch或文件，没有改变score、family policy或生产配置。DEV四组结果逐项与原消融核对一致。TRAIN结果只作拟合程度的描述，未用于选模型。
C4条件capture以C2已捕获改善为分母，只在C2捕获到改善的material states上计算。Oracle C4只作事后上界：用已知标签重排相同Top8并保留相同family policy，不能用于实际搜索。

| split | variant | tier | material | C2有改善 | C4有改善 | 未进C2 | C4丢光 | 条件C4 capture | oracle C4/full-pool capture |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP_TRAIN | BASELINE | ALL | 79 | 46 | 23 | 33 | 23 | 0.62% | 9.05% |
| MLP_TRAIN | BASELINE | SMALL | 29 | 12 | 5 | 17 | 7 | 0.00% | 0.00% |
| MLP_TRAIN | BASELINE | MEDIUM | 29 | 21 | 11 | 8 | 10 | 1.24% | 21.25% |
| MLP_TRAIN | BASELINE | LARGE | 21 | 13 | 7 | 8 | 6 | 3.07% | 8.93% |
| MLP_TRAIN | MLP-C2 | ALL | 79 | 53 | 33 | 26 | 20 | 96.35% | 47.40% |
| MLP_TRAIN | MLP-C2 | SMALL | 29 | 16 | 7 | 13 | 9 | 0.00% | 1.78% |
| MLP_TRAIN | MLP-C2 | MEDIUM | 29 | 19 | 11 | 10 | 8 | 100.00% | 41.96% |
| MLP_TRAIN | MLP-C2 | LARGE | 21 | 18 | 15 | 3 | 3 | 100.00% | 92.24% |
| MLP_TRAIN | MLP-C4 | ALL | 79 | 46 | 34 | 33 | 12 | 100.00% | 9.05% |
| MLP_TRAIN | MLP-C4 | SMALL | 29 | 12 | 8 | 17 | 4 | 100.00% | 0.00% |
| MLP_TRAIN | MLP-C4 | MEDIUM | 29 | 21 | 17 | 8 | 4 | 100.00% | 21.25% |
| MLP_TRAIN | MLP-C4 | LARGE | 21 | 13 | 9 | 8 | 4 | 77.29% | 8.93% |
| MLP_TRAIN | MLP-BOTH | ALL | 79 | 53 | 36 | 26 | 17 | 69.93% | 47.40% |
| MLP_TRAIN | MLP-BOTH | SMALL | 29 | 16 | 8 | 13 | 8 | 0.78% | 1.78% |
| MLP_TRAIN | MLP-BOTH | MEDIUM | 29 | 19 | 17 | 10 | 2 | 100.00% | 41.96% |
| MLP_TRAIN | MLP-BOTH | LARGE | 21 | 18 | 11 | 3 | 7 | 68.96% | 92.24% |
| MLP_DEV | BASELINE | ALL | 29 | 11 | 6 | 18 | 5 | 27.03% | 0.00% |
| MLP_DEV | BASELINE | SMALL | 11 | 8 | 3 | 3 | 5 | 0.00% | 73.64% |
| MLP_DEV | BASELINE | MEDIUM | 6 | 0 | 0 | 6 | 0 | — | 0.00% |
| MLP_DEV | BASELINE | LARGE | 12 | 3 | 3 | 9 | 0 | 100.00% | 0.00% |
| MLP_DEV | MLP-C2 | ALL | 29 | 15 | 9 | 14 | 6 | 100.00% | 30.89% |
| MLP_DEV | MLP-C2 | SMALL | 11 | 9 | 8 | 2 | 1 | 100.00% | 46.12% |
| MLP_DEV | MLP-C2 | MEDIUM | 6 | 4 | 0 | 2 | 4 | 0.00% | 66.74% |
| MLP_DEV | MLP-C2 | LARGE | 12 | 2 | 1 | 10 | 1 | 50.00% | 0.00% |
| MLP_DEV | MLP-C4 | ALL | 29 | 11 | 8 | 18 | 3 | 100.00% | 0.00% |
| MLP_DEV | MLP-C4 | SMALL | 11 | 8 | 5 | 3 | 3 | 87.60% | 73.64% |
| MLP_DEV | MLP-C4 | MEDIUM | 6 | 0 | 0 | 6 | 0 | — | 0.00% |
| MLP_DEV | MLP-C4 | LARGE | 12 | 3 | 3 | 9 | 0 | 100.00% | 0.00% |
| MLP_DEV | MLP-BOTH | ALL | 29 | 15 | 9 | 14 | 6 | 21.72% | 30.89% |
| MLP_DEV | MLP-BOTH | SMALL | 11 | 9 | 8 | 2 | 1 | 66.26% | 46.12% |
| MLP_DEV | MLP-BOTH | MEDIUM | 6 | 4 | 0 | 2 | 4 | 0.00% | 66.74% |
| MLP_DEV | MLP-BOTH | LARGE | 12 | 2 | 1 | 10 | 1 | 50.00% | 0.00% |

LARGE逐workbook复核（MLP-C2；物理路径只标识来源，角色按固定split）：

| split | workbook | material | Top8 capture median | Top8有改善 | Top2有改善 |
|---|---|---:|---:|---:|---:|
| MLP_TRAIN | data/ID_TEST/seed_0357172400.xlsx | 4 | 91.58% | 3 | 3 |
| MLP_TRAIN | data/ID_TEST/seed_1846918101.xlsx | 6 | 92.19% | 6 | 5 |
| MLP_TRAIN | data/PPO_TRAIN/seed_0514719396.xlsx | 1 | 0.00% | 0 | 0 |
| MLP_TRAIN | data/PPO_TRAIN/seed_0765477955.xlsx | 6 | 86.95% | 5 | 3 |
| MLP_TRAIN | data/PPO_TRAIN/seed_1783157644.xlsx | 4 | 96.42% | 4 | 4 |
| MLP_DEV | data/PPO_TRAIN/seed_0834257622.xlsx | 6 | 0.00% | 0 | 0 |
| MLP_DEV | data/VALIDATION/seed_0151520473.xlsx | 6 | 0.00% | 2 | 1 |

固定head误差（candidate级描述，不代替state equal weight capture）：

| split | head | tier | 真改善候选 | 其中预测负gain | 真改善候选gain平均偏差 | 真改善候选p中位数 | 不可行候选p中位数 |
|---|---|---|---:|---:|---:|---:|---:|
| MLP_TRAIN | C2 | ALL | 1048 | 54.20% | -0.03144092454214661 | 73.89% | 25.93% |
| MLP_TRAIN | C2 | SMALL | 252 | 55.56% | -0.05573583228062943 | 65.52% | 44.98% |
| MLP_TRAIN | C2 | MEDIUM | 336 | 63.10% | -0.03535457605855549 | 78.30% | 8.75% |
| MLP_TRAIN | C2 | LARGE | 460 | 46.96% | -0.015272873108209519 | 75.91% | 30.56% |
| MLP_TRAIN | C4 | ALL | 1048 | 37.31% | -0.01085845813278267 | 74.79% | 16.19% |
| MLP_TRAIN | C4 | SMALL | 252 | 55.56% | -0.04657047983370057 | 63.36% | 25.09% |
| MLP_TRAIN | C4 | MEDIUM | 336 | 37.50% | -0.002195046889154963 | 87.42% | 7.36% |
| MLP_TRAIN | C4 | LARGE | 460 | 27.17% | 0.0023774620645486337 | 78.97% | 27.68% |
| MLP_DEV | C2 | ALL | 360 | 53.06% | -0.0588962384144084 | 58.73% | 38.39% |
| MLP_DEV | C2 | SMALL | 182 | 46.15% | -0.04870390965791075 | 91.83% | 88.79% |
| MLP_DEV | C2 | MEDIUM | 52 | 76.92% | -0.10409736570603721 | 57.77% | 40.91% |
| MLP_DEV | C2 | LARGE | 126 | 53.17% | -0.05496405757724868 | 3.93% | 14.53% |
| MLP_DEV | C4 | ALL | 360 | 40.28% | -0.048722903332821164 | 43.30% | 26.28% |
| MLP_DEV | C4 | SMALL | 182 | 33.52% | -0.02958929358678085 | 63.93% | 49.00% |
| MLP_DEV | C4 | MEDIUM | 52 | 25.00% | -0.030836201720537245 | 22.90% | 18.51% |
| MLP_DEV | C4 | LARGE | 126 | 56.35% | -0.0837421529964569 | 21.35% | 34.27% |

实际负gain乘积倒序：selected不可行候选的预测gain不高于池中真改善候选，且二者gain均负，但前者p×gain更高。该统计说明固定模型分数与所选集合的关系；heuristic阶段仅作对照，不能归因于该score。不是因果实验，也不是新score选择。

| DEV variant | C2存在倒序的states | C4存在倒序的states | C4倒序pairs |
|---|---:|---:|---:|
| BASELINE | 19 | 0 | 0 |
| MLP-C2 | 2 | 0 | 0 |
| MLP-C4 | 19 | 0 | 0 |
| MLP-BOTH | 2 | 0 | 0 |

TRAIN标准化后|z|>5的特征比例（每state先平均、再取中位数）：

| split | stage | ALL | SMALL | MEDIUM | LARGE |
|---|---|---:|---:|---:|---:|
| MLP_TRAIN | C2 | 0.02% | 0.02% | 0.04% | 0.09% |
| MLP_TRAIN | C4 | 0.02% | 0.01% | 0.03% | 0.06% |
| MLP_DEV | C2 | 0.07% | 0.25% | 0.02% | 0.11% |
| MLP_DEV | C4 | 0.05% | 0.18% | 0.01% | 0.08% |

分布尾部差异只说明观测特征的尺度/分布，不证明泛化失败由某个feature导致；候选相关性和workbook差异仍存在。所有事后分析均未用于追加选epoch、改分数或重复训练。

进一步结论：LARGE既有C2跨workbook泛化差距，MEDIUM又有独立C4丢解。仅改善C4不能找回LARGE已被C2丢掉的候选。所检查的C4负gain乘积倒序为0个states，不能据理论性质直接断言它是本次失败主因。

可用 `python -B scripts/train_mlp_ranker.py diagnose` 重复本节；`finish` 更新本报告。下一轮应限定为现有TRAIN上的state-level ranking目标与跨workbook泛化改进，DEV继续只用于模型选择，保持两层MLP、预算与family policy不变；当前没有依据进入GAT或native smoke。

LARGE MLP-C2 Top8 capture median: TRAIN 92.24% vs DEV 0.00%; DEV 10/12 material states missed all improvement before C4. This is consistent with a cross-workbook generalization gap; TRAIN is in-sample and DEV has only two LARGE workbooks.

MEDIUM DEV: MLP-C2 captured improvement in 4/6 material states, and C4 dropped all improvement in 4 of them. The same-policy oracle C4 ceiling is 66.74% capture vs the actual 0.00%.

LARGE true-improving candidate predicted feasibility median: TRAIN 75.91% vs DEV 3.93%; DEV infeasible candidates median 14.53%. This is descriptive discrimination failure on this sample, not a standalone calibration metric or proven cause.

MLP-BOTH C4 had 0 states with the defined selected infeasible negative-gain inversion. The theoretical p*negative_gain issue alone does not explain the observed C4 failure; no alternative score was selected or tested.

Do not change M192, family access, Kdp/Kref, data split or architecture based on these diagnostics. The next bounded learning revision should address state-level ranking and cross-workbook generalization on existing TRAIN, with DEV used for selection; no GAT or native smoke is justified yet.


## 验证

完整回归：271 passed in 34.06s。真实模型API重放36 states×4 variants=144次比较，mismatch=0，新增direction/reference调用=0。SQLite核验：quick_check OK; foreign_key_check empty; 14893 candidates after compaction。当前candidate总数=14893。
