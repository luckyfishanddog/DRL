# Multi-Robot Weld Allocation and Sequencing

FORMAL_SCOPE_V2 active: WHOLE / Y_SPLIT / X_SPLIT.

- Production ALNS: M192 (144 atomic + 48 LNS), Kdp8 / Kref2 / Kref_total4; heuristic C2/C4.
- Candidate generation material loss: closed relative to the audited finite pool; M192 passed native 60-second development selection.
- Current phase: Phase4-1B completed. Both independent MLPs trained; offline adoption condition **FAIL**.
- MLP-C2 material Top8 capture median: 0% → 30.89%. End-to-end C4 Top2 capture median: 0% for every variant. Native MLP smoke was skipped under the specified offline criterion.
- Fixed workbook split: 17 TRAIN / 6 DEV / 8 ORACLE_DEV_CONSUMED. Dataset: 98 TRAIN / 36 DEV states, 14,893 labeled candidates, numeric failure 0. Four missing TRAIN EARLY states remain missing; no future backfill.
- Remaining work: C4 ranking objective/calibration and LARGE generalization. GAT: NOT YET.
- TWO_OPT_STAR remains OFF. V2_VALIDATION is consumed; final ID_TEST remains sealed.

Scientific rules: [FORMAL_SCOPE_V2](docs/FORMAL_SCOPE_V2.md) and [实验方案](docs/多机器人焊缝分配与排序实验方案.md).
Current evidence: [offline pool/materiality](docs/PHASE4_0B_POOL_SCALING_AND_MATERIALITY_20261007.md), [native production pool and MLP split](docs/PHASE4_1A_PRODUCTION_POOL_AND_MLP_DATA_PREP_20261007.md), [trained MLP results and cleanup](docs/PHASE4_1B_MLP_CANDIDATE_RANKER_20261007.md).
Historical formal Phase3-Z remains FAIL / 179 of 180 certified; no final-test superiority claim is made.

## Run locally

From `D:\pybullet_test\MRTA_GA\DRL`, use `D:\pybullet_test\.venv\Scripts\python.exe`.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
& 'D:\pybullet_test\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider
```

The reusable workflow is `scripts/build_mlp_dataset.py` (`trajectories`, `pools`, `labels`) and `scripts/train_mlp_ranker.py` (`train`, `offline`, conditional `smoke`, `finish`). Existing dataset rows are resumed, not regenerated. Training uses the original whole-workbook whitelist and TRAIN-only scalers; do not resplit candidate rows.

Models: `models/mlp_c2.pt`, `models/mlp_c4.pt`, `models/mlp_scalers.npz`. The optional `LearnedRanker` interface is connected to native ALNS and covered by budget/policy regression; heuristic ranking remains the production default after the failed offline adoption condition.

Scientific modules remain independent of PyTorch. Optional training dependencies are in the `ranker` extra. Workbook paths are relative to `../ppo`; physical directory names do not define data roles.
