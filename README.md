# Multi-Robot Weld Allocation and Sequencing

FORMAL_SCOPE_V2 active: WHOLE / Y_SPLIT / X_SPLIT.

- Production ALNS: M192 (144 atomic + 48 LNS), Kdp8 / Kref2 / Kref_total4; heuristic C2/C4.
- Candidate generation material loss: closed relative to the audited finite pool; M192 passed native 60-second development selection.
- Current phase: Phase4-1C completed; Top-K aligned material/pairwise objectives evaluated. Offline gate **FAIL**; native MLP smoke **SKIPPED / 0 runs**. Production stays heuristic.
- Phase4-1C best: **V2 / NEW-MLP-C2**. ALL C2 Top8 capture 0% → 47.10%; C4 Top2 median capture stays 0%. C4 zero capture 79.31% → 62.07%. LARGE C2/C4 median capture stays 0%; its zero capture improves 75% → 66.67%, while cross-workbook ranking failure persists.
- Pairwise V3 C2 capture is 14.26%, below V2 47.10%. C4 hard training uses 3,097 existing candidates; V2 BOTH helps MEDIUM Top2 capture (0% → 66.74%) but does not improve ALL median capture. No loss/architecture search followed the failure.
- Historical Phase4-1B remains **FAIL**. Its checkpoints, scaler, result JSON and report remain unchanged; 0 native MLP smoke runs.
- Historical V1 MLP-C2 material Top8 capture median: 0% → 30.89%. End-to-end C4 Top2 capture median: 0% for every variant. Native MLP smoke was skipped under the specified offline criterion.
- Fixed workbook split: 17 TRAIN / 6 DEV / 8 ORACLE_DEV_CONSUMED. Dataset: 98 TRAIN / 36 DEV states, 14,893 labeled candidates, numeric failure 0. Four missing TRAIN EARLY states remain missing; no future backfill.
- Historical V1 fixed-model follow-up: LARGE MLP-C2 Top8 capture median is 92.24% in TRAIN vs 0% in DEV; 10/12 DEV material states miss improvement before C4. MEDIUM captures improvement in 4/6 states at Top8, then loses all four at C4. Both stages need attention; GAT: NOT YET.
- TWO_OPT_STAR remains OFF. V2_VALIDATION is consumed; final ID_TEST remains sealed.

Scientific rules: [FORMAL_SCOPE_V2](docs/FORMAL_SCOPE_V2.md) and [实验方案](docs/多机器人焊缝分配与排序实验方案.md).
Current evidence: [offline pool/materiality](docs/PHASE4_0B_POOL_SCALING_AND_MATERIALITY_20261007.md), [native production pool and MLP split](docs/PHASE4_1A_PRODUCTION_POOL_AND_MLP_DATA_PREP_20261007.md), [V1 failure and cleanup](docs/PHASE4_1B_MLP_CANDIDATE_RANKER_20261007.md), [Top-K objective results](docs/PHASE4_1C_MLP_TOPK_RANKER_20261008.md), [Phase4-1C result JSON](data/development/phase4_1c_mlp_results.json).
Historical formal Phase3-Z remains FAIL / 179 of 180 certified; no final-test superiority claim is made.

## Run locally

From `D:\pybullet_test\MRTA_GA\DRL`, use `D:\pybullet_test\.venv\Scripts\python.exe`.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider
```

The Phase4-1C workflow uses the existing `scripts/train_mlp_ranker.py`: `audit` reads V1 checkpoints and DEV score distributions; `rank` trains fixed V2/V3, evaluates all four combinations and runs native smoke only if all three strict gate conditions pass; `rank-finish` writes the single result JSON/report summary. The completed run failed the gate and stopped before native smoke. Training uses the original whole-workbook whitelist and unchanged TRAIN-only scaler. No trajectory, scheduler, candidate generator or reference label collection is part of the offline workflow.

Models: V1 `models/mlp_c2.pt` / `models/mlp_c4.pt` remain failure evidence. V2 classification checkpoints are `models/mlp_c2_material_v2.pt` / `models/mlp_c4_material_v2.pt`; V3 pairwise checkpoints are `models/mlp_c2_rank_v2.pt` / `models/mlp_c4_rank_v2.pt`. All reuse `models/mlp_scalers.npz`. `LearnedRanker(variant, model_version="material_v2")` selects V2; `model_version="rank_v2"` selects V3; legacy `"v1"` remains the optional interface default. None is recommended for production after the failed gate.

Verification: `pytest -q -p no:cacheprovider` passed 281 tests. Existing-state native API replay matched 216 offline selections with 0 new direction/reference calls. Best V2 C2 feature extraction/scaler/forward/ranking median was 97.76 ms/state in replay; its share of a real 60-second run is unmeasured.

Scientific modules remain independent of PyTorch. Optional training dependencies are in the `ranker` extra. Workbook paths are relative to `../ppo`; physical directory names do not define data roles.
