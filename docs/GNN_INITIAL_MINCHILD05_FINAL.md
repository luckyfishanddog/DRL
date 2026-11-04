# 最后一次0.50 m GNN初始构造实验

仅使用现有V2和Shared checkpoint；无训练、无网络或权重修改。历史CONFIG=0.20保留，实验显式使用CONFIG_05=ScientificConfig()，min_child_length=0.50。
两checkpoint均为hidden=64、heads=4、normalize_inputs=True，动态输入41维/父输入12维与现有特征实现一致；checkpoint未存ScientificConfig快照。当前教师脚本默认0.50不能证明历史训练域：V2已有294条轨迹中30条、Shared已有40条中4条含小于0.50 m的子段，最短0.277058 m。因此本次属于冻结既有权重在0.50合法目录上的迁移评价，不能称为0.50域专门训练的模型。没有重训或更新归一化。
固定父模式、FORMAL_SCOPE_V2、insertion_limit=8、同一B32。Multi4为Greedy及seed+1/+2/+3，温度1，不补采；完整路线和方向去重后最多4次B32。RULE/BEAM保持原Beam宽度并设max_reference_calls=4。Greedy复用Multi4第一候选。
travel为MOVE总耗时（秒），wait为WAIT总耗时（秒）；负载为PRE+WELD+POST。失败无惩罚Cmax。

已记录135/135行（9×3×5）。

|方法|FEASIBLE/总次数|成功Cmax平均|成功Cmax中位|平均耗时秒|B32合计|
|---|---:|---:|---:|---:|---:|
|RULE/BEAM|27/27|3448.6855232866405|3399.247032236728|7.347|108|
|GNN-V2-Greedy|0/27|None|None|1.534|27|
|GNN-V2-Multi4|0/27|None|None|6.239|108|
|GNN-Shared-Greedy|6/27|6840.549727726397|6840.549727726397|1.313|27|
|GNN-Shared-Multi4|6/27|6840.549727726397|6840.549727726397|6.281|108|

## GNN-V2-Multi4

```json
{
  "method": "GNN-V2-Multi4",
  "success": 0,
  "total": 27,
  "baseline_success": 27,
  "geometry_improvement_pct": {},
  "median_improvement_pct": null,
  "winning_tiers": [],
  "keep": false
}
```

## GNN-Shared-Multi4

```json
{
  "method": "GNN-Shared-Multi4",
  "success": 6,
  "total": 27,
  "baseline_success": 27,
  "geometry_improvement_pct": {
    "N35_02": -94.19205917606224,
    "N55_01": -79.25943978605198
  },
  "median_improvement_pct": -86.72574948105711,
  "winning_tiers": [],
  "keep": false
}
```

结论：正式淘汰GNN初始分配与排序。停止该方向训练、架构调整、增加采样、温度调整和PPO初始分配；不影响独立评价GNN局部联合修复。
