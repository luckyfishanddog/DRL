# Phase4-1B — MLP Candidate Ranker

结论：**FAIL**。推荐 **BASELINE**；M192 / Kdp8 / Kref2 / Kref_total4 不变。GAT：NOT YET。

完成日期：2026-10-08。报告文件名沿用此前批准的名称。

## 清理结果

删除 53 个无当前依赖的旧过程文件：旧 Phase3/4 runner、profiling、候选 oracle SQLite、重复中间 JSON 和 handoff。通用 typed reader、accepted-state capture、存储与数据准备移入正式 `mrta_ranker.dataset`，新流程不依赖历史 runner。删除明细及字节数保存在结果 JSON。没有创建 archive/backup，没有提交或上传 GitHub。

保留科学 src/tests、FORMAL_SCOPE_V2、实验方案、正式数据角色与固定 MLP split，以及解释当前算法的 Phase4-0B、Phase4-1A 等最终报告；清理后的文档链接已修复。退役104项仅服务旧runner/protocol/历史机械判定的过程测试，科学scheduler/certifier/search测试保留；清理后261项，加9项当前dataset/ranker回归后270项通过。清理前已向用户说明DRL未被当前外层Git跟踪，用户明确授权仍直接删除；没有声称这些本地过程文件已存入当前Git历史。

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

下一步：保持 M192 heuristic；优先检查 C4 Top2 排序目标、score校准与 LARGE 跨workbook泛化，暂不进入 GAT。

没有修改M192、Kdp/Kref、科学scope、X/Y、scheduler/B32或算子；没有访问真正最终测试、训练GAT或重新划分workbook。


## 验证

完整回归：270 passed in 34.00s。真实模型API重放36 states×4 variants=144次比较，mismatch=0，新增direction/reference调用=0。SQLite核验：quick_check OK; foreign_key_check empty; 14893 candidates after compaction。当前candidate总数=14893。
