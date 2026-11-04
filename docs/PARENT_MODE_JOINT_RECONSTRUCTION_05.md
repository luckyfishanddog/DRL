# 0.50 m父焊缝模式与方向感知联合重构2×2实验

已记录162/162次运行。
ScientificConfig()、FORMAL_SCOPE_V2、同一合法WHOLE/Y_SPLIT/X_SPLIT目录、同一B32，30秒截止前结果。四种联合修复复用每个几何/seed同一个确定性共同初解及方向；共同初解实际构造耗时在四种方法中等额计入预算，GNN加载时间计入本方法预算。每次只接受截止前完成且严格改善Cmax的候选。
六种方法均关闭独立certify_schedule，生产默认认证不变。B32调用可跨截止时间返回，但跨时结果不纳入best；实际运行时间如实记录。模式变化为联合方法的已接受单父模式提案次数；WAG/HGA记录初始到最终不同模式的父数，不等同于搜索接受次数。
稳定收益判据在比较前固定：可行次数不降低、几何胜数大于负数、几何中位改善为正且至少两个规模有胜例；GNN须同时在MODE和FIXED组满足。

|方法|FEASIBLE/总次数|成功Cmax平均|成功Cmax中位|平均运行秒|平均模式变化|
|---|---:|---:|---:|---:|---:|
|FIXED_NEW|27/27|3252.3529412343073|3341.5493317425853|30.770|0.000|
|FIXED_GNN|27/27|3215.920765351258|3324.8419026655174|30.790|0.000|
|MODE_NEW|27/27|3197.4658294379724|3341.5493317425853|30.761|0.741|
|MODE_GNN|27/27|3194.114153368325|3341.5493317425853|30.853|0.963|
|WAG|27/27|3411.290350686937|3352.864856057165|30.447|0.222|
|HGA|27/27|3586.392547116285|3369.182184080884|30.650|0.148|

Cmax均值/中位仅描述成功运行；成功数与失败数必须同时阅读。几何配对先取每个几何各种子Cmax算术平均，仅使用双方全部种子成功的几何；不完整几何单列，不补值。相对改善为(基线−方法)/基线，正值表示改善。

|比较|完整几何数|胜/平/负|平均改善%|中位改善%|不完整几何|
|---|---:|---:|---:|---:|---|
|MODE_NEW vs FIXED_NEW|9|5/1/3|2.1593699925730294|0.5169629644612882|无|
|MODE_GNN vs FIXED_GNN|9|3/1/5|0.9501470486896576|-0.03017068182931023|无|
|MODE_GNN vs MODE_NEW|9|5/1/3|0.10326632724745427|0.0014439492097007608|无|
|FIXED_GNN vs FIXED_NEW|9|6/2/1|1.3189429235985564|0.09645198787220559|无|
|MODE_NEW vs WAG|9|8/0/1|6.221257068232262|0.6649393296197845|无|
|MODE_NEW vs HGA|9|8/0/1|10.570562719142176|8.006894448760328|无|
|MODE_GNN vs WAG|9|8/0/1|6.327207059468259|0.8932018996997485|无|
|MODE_GNN vs HGA|9|9/0/0|10.6648954084218|8.384282619021267|无|

|几何|方法|成功/次数|平均Cmax|平均四机器人加工负载/s|平均MOVE/s|平均WAIT/s|
|---|---|---:|---:|---|---:|---:|
|N35_01|FIXED_NEW|3/3|2400.088708817689|[1777.726, 2076.871, 1989.444, 2017.61]|275.43608295744457|287.00634492486256|
|N35_01|FIXED_GNN|3/3|2208.867520916813|[1830.697, 2023.9, 1935.802, 2071.252]|248.93463656372134|152.61300435626933|
|N35_01|MODE_NEW|3/3|2219.60538924239|[2035.881, 1885.383, 2017.063, 1989.991]|239.1543439119067|87.121541739552|
|N35_01|MODE_GNN|3/3|2218.250115563867|[2066.928, 1871.002, 1969.652, 2054.069]|250.73923621475646|161.89859611161748|
|N35_01|WAG|3/3|2532.773861180055|[1998.414, 1931.499, 1973.588, 2008.15]|219.71450119830286|569.2570116709079|
|N35_01|HGA|3/3|2485.233266874649|[2219.284, 1686.832, 2060.294, 1911.908]|257.6737260465186|431.43931417577795|
|N35_02|FIXED_NEW|3/3|2555.582569440141|[2319.973, 2436.999, 1682.698, 1666.585]|217.16875782644524|162.447434131999|
|N35_02|FIXED_GNN|3/3|2555.582569440141|[2319.973, 2436.999, 1682.698, 1666.585]|217.16875782644524|162.447434131999|
|N35_02|MODE_NEW|3/3|2542.3711540299073|[2352.381, 2367.016, 1634.364, 1785.829]|247.6188755620786|108.45079783126812|
|N35_02|MODE_GNN|3/3|2532.034584676331|[2319.973, 2414.658, 1749.322, 1638.97]|213.31881169029538|138.89944936818893|
|N35_02|WAG|3/3|2555.5825694401415|[2319.973, 2436.999, 1682.698, 1666.585]|195.7902136678568|179.7292588546004|
|N35_02|HGA|3/3|3144.7583339562684|[2256.856, 2500.117, 2405.006, 944.278]|282.7622752359179|201.52046918478723|
|N35_03|FIXED_NEW|3/3|2390.015125098138|[2083.929, 1867.263, 1689.805, 1804.384]|206.97403897598267|299.40504888305276|
|N35_03|FIXED_GNN|3/3|2359.6568254872736|[2279.384, 1671.808, 1761.848, 1732.34]|243.21150620270578|51.251128670853724|
|N35_03|MODE_NEW|3/3|2165.429243183359|[2009.357, 1975.168, 1821.405, 1672.784]|261.5418828111878|119.5161105182719|
|N35_03|MODE_GNN|3/3|2166.388375258379|[2123.212, 1861.314, 1871.423, 1656.099]|246.7921252982801|17.681562421335638|
|N35_03|WAG|3/3|2905.1743054903322|[1592.267, 2375.592, 1732.646, 1761.543]|272.0794534396924|573.002656680737|
|N35_03|HGA|3/3|3045.649477834137|[1767.197, 2252.174, 1886.45, 1539.559]|201.793503073169|1128.3150702377945|
|N55_01|FIXED_NEW|3/3|4193.715768585728|[3818.163, 3878.196, 2195.267, 2430.927]|381.2396850485238|194.33596792510207|
|N55_01|FIXED_GNN|3/3|4107.19320227686|[3835.1, 3861.26, 2326.989, 2299.205]|340.9767346681576|156.9566293003677|
|N55_01|MODE_NEW|3/3|4125.356036182162|[3769.933, 3912.396, 2283.042, 2390.516]|396.7760371636894|93.53774533369597|
|N55_01|MODE_GNN|3/3|4108.432370470034|[3787.376, 3909.702, 2305.391, 2370.085]|360.7890480994667|149.61384333637534|
|N55_01|WAG|3/3|4065.090571757235|[3755.103, 3941.257, 2263.746, 2362.448]|293.5580125651689|0|
|N55_01|HGA|3/3|4484.418708839393|[3816.294, 3855.815, 2433.074, 2234.037]|465.99000894742625|563.2740782660554|
|N55_02|FIXED_NEW|3/3|2904.789915052971|[2776.823, 2849.805, 2546.061, 2623.238]|393.66972330106023|0|
|N55_02|FIXED_GNN|3/3|2905.0466013894084|[2776.823, 2849.805, 2567.966, 2601.334]|424.1926324800847|25.960607992575586|
|N55_02|MODE_NEW|3/3|2882.4450831355853|[2776.823, 2809.395, 2604.821, 2621.556]|441.70247166511416|2.2666144354201756|
|N55_02|MODE_GNN|3/3|2882.826425859415|[2776.823, 2809.395, 2555.488, 2687.555]|417.58653425827634|8.447449517940678|
|N55_02|WAG|3/3|2901.7398929269234|[2776.823, 2849.805, 2546.061, 2623.238]|336.2277460963864|0|
|N55_02|HGA|3/3|2951.516581260368|[2776.823, 2849.805, 2546.061, 2623.238]|627.4892299687448|0|
|N55_03|FIXED_NEW|3/3|3340.643992917922|[3191.109, 3137.931, 3173.915, 3124.462]|360.3857910167299|68.60574422129159|
|N55_03|FIXED_GNN|3/3|3324.6818272710248|[3191.109, 3137.931, 3126.036, 3172.34]|341.4412974771439|63.160019065295195|
|N55_03|MODE_NEW|3/3|3345.259111217939|[3191.109, 3131.878, 3160.942, 3143.488]|358.4665494569053|59.70196674484934|
|N55_03|MODE_GNN|3/3|3338.2423992776967|[3191.109, 3117.889, 3134.783, 3200.302]|341.2540024997083|32.683691139300514|
|N55_03|WAG|3/3|3368.3283722870906|[3151.603, 3188.994, 3093.818, 3193.001]|539.9908977431909|0|
|N55_03|HGA|3/3|3338.536511791712|[3151.603, 3177.437, 3126.036, 3172.34]|498.0573468788783|0|
|N85_01|FIXED_NEW|3/3|4060.0142405296724|[3895.627, 3862.13, 3796.646, 3871.687]|420.78224701369584|56.65213320551226|
|N85_01|FIXED_GNN|3/3|4060.0142405296724|[3895.627, 3862.13, 3796.646, 3871.687]|420.78224701369584|56.65213320551226|
|N85_01|MODE_NEW|3/3|4060.0142405296724|[3895.627, 3862.13, 3796.646, 3871.687]|420.78224701369584|56.65213320551226|
|N85_01|MODE_GNN|3/3|4060.0142405296724|[3895.627, 3862.13, 3796.646, 3871.687]|420.78224701369584|56.65213320551226|
|N85_01|WAG|3/3|4070.6240536258165|[3895.627, 3862.13, 3796.646, 3871.687]|662.0419202887813|0|
|N85_01|HGA|3/3|4073.6740536258167|[3895.627, 3862.858, 3796.646, 3870.96]|652.3570916004627|0|
|N85_02|FIXED_NEW|3/3|4078.181359747879|[2104.182, 2309.851, 3986.636, 3942.864]|339.990585211424|0|
|N85_02|FIXED_GNN|3/3|4074.2478727573684|[2055.912, 2358.121, 3959.382, 3970.117]|373.96880423313183|0|
|N85_02|MODE_NEW|3/3|4087.4824199425025|[2038.027, 2392.674, 3983.873, 3962.293]|396.02221312354925|0|
|N85_02|MODE_GNN|3/3|4091.657442378342|[2326.72, 2087.313, 4003.459, 3959.374]|378.03087010822736|0|
|N85_02|WAG|3/3|4949.804739157656|[2104.182, 2309.851, 3104.563, 4841.603]|272.41876357808974|60.77354472030439|
|N85_02|HGA|3/3|5299.576792825192|[1816.952, 2455.413, 3788.627, 4299.207]|384.5473686966836|993.125251874124|
|N85_03|FIXED_NEW|3/3|3348.1447909186268|[3253.622, 3243.491, 3116.516, 3254.917]|416.24083859829545|0|
|N85_03|FIXED_GNN|3/3|3347.996228092762|[3253.622, 3243.491, 3128.692, 3242.741]|431.96974415245876|0|
|N85_03|MODE_NEW|3/3|3349.229787478234|[3253.622, 3243.491, 3124.149, 3247.284]|423.5407246248955|0|
|N85_03|MODE_GNN|3/3|3349.1814263011865|[3237.699, 3243.491, 3149.106, 3254.917]|421.0273220712102|0|
|N85_03|WAG|3/3|3352.4947903171824|[3222.897, 3128.676, 3247.515, 3269.458]|332.91697636947015|98.28179871440852|
|N85_03|HGA|3/3|3454.1691970390307|[3253.622, 3243.491, 3116.516, 3254.917]|739.8824730678361|0|

## 最终算法选择

```json
{
  "method": "MODE_NEW",
  "fixed": "FIXED_NEW",
  "gnn_stable": false,
  "gnn_pairwise_positive": true,
  "mode_stable": true,
  "selection_rule": "Pairwise checks: no worse feasibility, positive geometry median, wins>losses, wins in >=2 tiers. Final joint method also requires its own MODE-versus-FIXED benefit. This final development selection precedes independent geometry generation; search parameters are unchanged."
}
```
若mode_stable=false，不宣称父模式调整有稳定收益；保留MODE对FIXED正式独立比较以确认负结果，不增加模块。

最终选择也检查所选联合路线自身的模式消融。虽然GNN两项配对的方向检查为正，但不能把它等同于有实际幅度的稳定联合收益：本轮MODE_GNN对MODE_NEW中位改善仅0.00144%，且MODE_GNN对FIXED_GNN为3胜1平5负、中位改善−0.03017%。因此保留模式调整有正收益的MODE_NEW；GNN在固定模式组的局部收益如实保留，不作为主方法模块。该选择发生在独立几何生成之前，没有改动搜索参数。

## 困难实例路线与B32时间线诊断

选择仍落后WAG的N55_01/N85_01；不足两个时按实际落后幅度补充。只读取本轮保留的原始ScheduleResult信息，不二次调度。方向0=start→end，方向1=end→start。完整PRE/WELD/POST/MOVE/WAIT操作及坐标存于CSV payload.details.final_state.operations。

### N55_01（几何平均相对WAG改善 -1.483%）

#### MODE_NEW, seed=20261031

Cmax=4213.747s，由R1决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3650.704|81.467|0.000|3732.171|
|R1|4045.656|96.356|71.736|4213.747|
|R2|2263.746|57.714|0.000|2321.460|
|R3|2362.448|74.739|0.000|2437.187|

R0执行顺序（block[方向]）：

inst0014|group=888-BK304A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0003::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[1] → inst0023|group=145-LB17A.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[0] → inst0027|group=804-FR315A|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0] → inst0025|group=143-FR65A.dxf|row=0003::whole[1] → inst0025|group=143-FR65A.dxf|row=0004::whole[0] → inst0019|group=826BK312A|row=0000::whole[0] → inst0021|group=312-GR2C|row=0000::0[1] → inst0017|group=183-CM2B.dxf|row=0000::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::whole[1] → inst0005|group=145-LB20A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[0] → inst0015|group=804-FR315A|row=0000::whole[1]

R0 WAIT起止秒：

无正时长WAIT

R1执行顺序（block[方向]）：

inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0002::whole[1] → inst0002|group=143-FR76A.dxf|row=0003::whole[0] → inst0004|group=143-GR14A|row=0000::0[0] → inst0022|group=312-GR2C|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0000::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[1] → inst0020|group=685-BK21A.dxf|row=0000::whole[1] → inst0012|group=144-DK1B.dxf|row=0000::whole[0]

R1 WAIT起止秒：

2954.604–3026.340(inst0012|group=144-DK1B.dxf|row=0000::whole)

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0000::whole[0] → inst0011|group=804-FR314A.dxf|row=0001::whole[0] → inst0011|group=804-FR314A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0021|group=312-GR2C|row=0000::1[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0] → inst0018|group=143-FR76A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0005::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1] → inst0018|group=143-FR76A.dxf|row=0003::whole[0]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0010|group=804-FR314A|row=0000::whole[1] → inst0009|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[1] → inst0000|group=143-FR65A|row=0000::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[1] → inst0002|group=143-FR76A.dxf|row=0006::whole[0] → inst0004|group=143-GR14A|row=0000::1[0] → inst0001|group=826BK312A|row=0000::whole[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[0] → inst0006|group=183-CM2B.dxf|row=0000::whole[0]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[]
```

相对WAG：最大加工负载差 +5.054s；MOVE总耗时差 +39.942s；WAIT总耗时差 +71.736s；Cmax差 +59.349s。

决定完工的R1：加工4045.656s、MOVE 96.356s、WAIT 71.736s；WAG决定完工的R1：加工4040.602s、MOVE 113.796s、WAIT 0.000s。全局最大加工负载不能替代这条实际完成时间线。

R1长WAIT 2954.604–3026.340s，下一块inst0012|group=144-DK1B.dxf|row=0000::whole；同期同轨R0操作：WELD inst0017|group=183-CM2B.dxf|row=0000::whole 2818.472–2963.975s; POST inst0017|group=183-CM2B.dxf|row=0000::whole 2963.975–2993.975s; MOVE None 2993.975–2997.290s; SETUP inst0028|group=183-CM2B.dxf|row=0000::whole 2997.290–3017.290s; WELD inst0028|group=183-CM2B.dxf|row=0000::whole 3017.290–3162.793s。这是同轨操作重叠的时间线证据，不将时间重叠本身当作独立因果证明。

#### WAG, seed=20261031

Cmax=4154.398s，由R1决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3655.758|32.044|0.000|3687.802|
|R1|4040.602|113.796|0.000|4154.398|
|R2|2263.746|54.163|0.000|2317.909|
|R3|2362.448|70.331|0.000|2432.780|

R0执行顺序（block[方向]）：

inst0012|group=144-DK1B.dxf|row=0000::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[1] → inst0023|group=145-LB17A.dxf|row=0001::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[0] → inst0023|group=145-LB17A.dxf|row=0003::whole[0] → inst0027|group=804-FR315A|row=0000::whole[0] → inst0014|group=888-BK304A|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0000::whole[0] → inst0019|group=826BK312A|row=0000::whole[1] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0] → inst0025|group=143-FR65A.dxf|row=0003::whole[1]

R0 WAIT起止秒：

无正时长WAIT

R1执行顺序（block[方向]）：

inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0002::whole[1] → inst0002|group=143-FR76A.dxf|row=0003::whole[0] → inst0020|group=685-BK21A.dxf|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::0[0] → inst0022|group=312-GR2C|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[0] → inst0008|group=143-FR69B.dxf|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[0] → inst0015|group=804-FR315A|row=0000::whole[0] → inst0005|group=145-LB20A|row=0000::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::whole[0] → inst0017|group=183-CM2B.dxf|row=0000::whole[1] → inst0021|group=312-GR2C|row=0000::0[0] → inst0025|group=143-FR65A.dxf|row=0004::whole[1]

R1 WAIT起止秒：

无正时长WAIT

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0000::whole[0] → inst0011|group=804-FR314A.dxf|row=0001::whole[0] → inst0011|group=804-FR314A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1] → inst0018|group=143-FR76A.dxf|row=0005::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[1] → inst0018|group=143-FR76A.dxf|row=0003::whole[1] → inst0018|group=143-FR76A.dxf|row=0002::whole[0] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0021|group=312-GR2C|row=0000::1[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0006|group=183-CM2B.dxf|row=0000::whole[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[1] → inst0001|group=826BK312A|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::1[1] → inst0002|group=143-FR76A.dxf|row=0006::whole[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[0] → inst0000|group=143-FR65A|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[1] → inst0009|group=642-BL14A.dxf|row=0000::whole[1] → inst0010|group=804-FR314A|row=0000::whole[0]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[]
```
#### MODE_NEW, seed=20261032

Cmax=4133.714s，由R0决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3748.019|176.818|208.877|4133.714|
|R1|3859.952|114.188|0.000|3974.140|
|R2|2527.070|105.159|0.000|2632.229|
|R3|2287.513|86.604|0.000|2374.117|

R0执行顺序（block[方向]）：

inst0023|group=145-LB17A.dxf|row=0003::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[1] → inst0020|group=685-BK21A.dxf|row=0000::whole[0] → inst0015|group=804-FR315A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[0] → inst0014|group=888-BK304A|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0000::whole[1] → inst0027|group=804-FR315A|row=0000::whole[0] → inst0019|group=826BK312A|row=0000::whole[1] → inst0025|group=143-FR65A.dxf|row=0003::whole[1] → inst0025|group=143-FR65A.dxf|row=0004::whole[0] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0] → inst0021|group=312-GR2C|row=0000::0[1] → inst0017|group=183-CM2B.dxf|row=0000::whole[0] → inst0005|group=145-LB20A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[1] → inst0002|group=143-FR76A.dxf|row=0000::whole[0]

R0 WAIT起止秒：

678.662–886.587(None); 3906.223–3907.176(inst0023|group=145-LB17A.dxf|row=0000::whole)

R1执行顺序（block[方向]）：

inst0012|group=144-DK1B.dxf|row=0000::whole[1] → inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0002::whole[1] → inst0002|group=143-FR76A.dxf|row=0003::whole[0] → inst0008|group=143-FR69B.dxf|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::0[0] → inst0004|group=143-GR14A|row=0000::0[1] → inst0022|group=312-GR2C|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[1]

R1 WAIT起止秒：

无正时长WAIT

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0000::whole[0] → inst0011|group=804-FR314A.dxf|row=0001::whole[0] → inst0011|group=804-FR314A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0021|group=312-GR2C|row=0000::1[1] → inst0028|group=183-CM2B.dxf|row=0000::1[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0] → inst0018|group=143-FR76A.dxf|row=0003::whole[0] → inst0018|group=143-FR76A.dxf|row=0005::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1] → inst0001|group=826BK312A|row=0000::0[0]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0010|group=804-FR314A|row=0000::whole[1] → inst0009|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[1] → inst0000|group=143-FR65A|row=0000::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[0] → inst0002|group=143-FR76A.dxf|row=0006::whole[0] → inst0001|group=826BK312A|row=0000::1[1] → inst0004|group=143-GR14A|row=0000::1[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[0] → inst0006|group=183-CM2B.dxf|row=0000::whole[0]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[
  [
    "inst0001|group=826BK312A|row=0000",
    {
      "parent_id": "inst0001|group=826BK312A|row=0000",
      "kind": "WHOLE",
      "t": null,
      "point_id": null,
      "mandatory": false,
      "rail": null
    },
    {
      "parent_id": "inst0001|group=826BK312A|row=0000",
      "kind": "X_SPLIT",
      "t": 0.5,
      "point_id": "MIDPOINT",
      "mandatory": false,
      "rail": "LOWER"
    }
  ],
  [
    "inst0028|group=183-CM2B.dxf|row=0000",
    {
      "parent_id": "inst0028|group=183-CM2B.dxf|row=0000",
      "kind": "WHOLE",
      "t": null,
      "point_id": null,
      "mandatory": false,
      "rail": null
    },
    {
      "parent_id": "inst0028|group=183-CM2B.dxf|row=0000",
      "kind": "Y_SPLIT",
      "t": 0.39252854352567196,
      "point_id": "BY_OUTER_UPPER",
      "mandatory": false,
      "rail": null
    }
  ]
]
```

相对WAG：最大加工负载差 -64.145s；MOVE总耗时差 +169.098s；WAIT总耗时差 +208.877s；Cmax差 +71.557s。

决定完工的R0：加工3748.019s、MOVE 176.818s、WAIT 208.877s；WAG决定完工的R1：加工3924.097s、MOVE 138.059s、WAIT 0.000s。全局最大加工负载不能替代这条实际完成时间线。

R0长WAIT 678.662–886.587s，该WAIT未携带block_id；同期同轨R1操作：WELD inst0012|group=144-DK1B.dxf|row=0000::whole 20.000–1177.407s。这是同轨操作重叠的时间线证据，不将时间重叠本身当作独立因果证明。

R0长WAIT 3906.223–3907.176s，下一块inst0023|group=145-LB17A.dxf|row=0000::whole；同期同轨R1操作：WELD inst0008|group=143-FR69B.dxf|row=0001::whole 3733.770–3944.140s。这是同轨操作重叠的时间线证据，不将时间重叠本身当作独立因果证明。

#### WAG, seed=20261032

Cmax=4062.157s，由R1决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3772.263|51.117|0.000|3823.380|
|R1|3924.097|138.059|0.000|4062.157|
|R2|2263.746|54.163|0.000|2317.909|
|R3|2362.448|70.331|0.000|2432.780|

R0执行顺序（block[方向]）：

inst0012|group=144-DK1B.dxf|row=0000::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0003::whole[0] → inst0027|group=804-FR315A|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0000::whole[0] → inst0019|group=826BK312A|row=0000::whole[1] → inst0025|group=143-FR65A.dxf|row=0004::whole[1] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0]

R0 WAIT起止秒：

无正时长WAIT

R1执行顺序（block[方向]）：

inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0002::whole[1] → inst0002|group=143-FR76A.dxf|row=0003::whole[0] → inst0020|group=685-BK21A.dxf|row=0000::whole[1] → inst0022|group=312-GR2C|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::0[1] → inst0008|group=143-FR69B.dxf|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[0] → inst0015|group=804-FR315A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[1] → inst0005|group=145-LB20A|row=0000::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::whole[1] → inst0017|group=183-CM2B.dxf|row=0000::whole[0] → inst0021|group=312-GR2C|row=0000::0[1] → inst0014|group=888-BK304A|row=0000::whole[1] → inst0025|group=143-FR65A.dxf|row=0003::whole[0]

R1 WAIT起止秒：

无正时长WAIT

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0000::whole[0] → inst0011|group=804-FR314A.dxf|row=0001::whole[0] → inst0011|group=804-FR314A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1] → inst0018|group=143-FR76A.dxf|row=0005::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[1] → inst0018|group=143-FR76A.dxf|row=0003::whole[1] → inst0018|group=143-FR76A.dxf|row=0002::whole[0] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0021|group=312-GR2C|row=0000::1[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0006|group=183-CM2B.dxf|row=0000::whole[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[1] → inst0001|group=826BK312A|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::1[1] → inst0002|group=143-FR76A.dxf|row=0006::whole[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[0] → inst0000|group=143-FR65A|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[1] → inst0009|group=642-BL14A.dxf|row=0000::whole[1] → inst0010|group=804-FR314A|row=0000::whole[0]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[]
```
#### MODE_NEW, seed=20261033

Cmax=4028.607s，由R0决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3911.075|117.532|0.000|4028.607|
|R1|3831.582|90.060|0.000|3921.642|
|R2|2058.310|50.322|0.000|2108.632|
|R3|2521.588|139.368|0.000|2660.956|

R0执行顺序（block[方向]）：

inst0023|group=145-LB17A.dxf|row=0003::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[0] → inst0015|group=804-FR315A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[0] → inst0014|group=888-BK304A|row=0000::whole[0] → inst0027|group=804-FR315A|row=0000::whole[1] → inst0025|group=143-FR65A.dxf|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0] → inst0025|group=143-FR65A.dxf|row=0003::whole[1] → inst0025|group=143-FR65A.dxf|row=0004::whole[0] → inst0019|group=826BK312A|row=0000::whole[1] → inst0017|group=183-CM2B.dxf|row=0000::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::whole[1] → inst0005|group=145-LB20A|row=0000::whole[1] → inst0021|group=312-GR2C|row=0000::0[0]

R0 WAIT起止秒：

无正时长WAIT

R1执行顺序（block[方向]）：

inst0012|group=144-DK1B.dxf|row=0000::whole[1] → inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0003::whole[1] → inst0002|group=143-FR76A.dxf|row=0002::whole[0] → inst0022|group=312-GR2C|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0000::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[1] → inst0020|group=685-BK21A.dxf|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::0[0]

R1 WAIT起止秒：

无正时长WAIT

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0001::whole[1] → inst0011|group=804-FR314A.dxf|row=0000::whole[1] → inst0011|group=804-FR314A.dxf|row=0002::whole[0] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0] → inst0018|group=143-FR76A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0003::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[0] → inst0018|group=143-FR76A.dxf|row=0005::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0010|group=804-FR314A|row=0000::whole[0] → inst0009|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[1] → inst0000|group=143-FR65A|row=0000::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[1] → inst0004|group=143-GR14A|row=0000::1[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[1] → inst0002|group=143-FR76A.dxf|row=0006::whole[0] → inst0001|group=826BK312A|row=0000::whole[1] → inst0006|group=183-CM2B.dxf|row=0000::whole[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[1] → inst0021|group=312-GR2C|row=0000::1[1]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[
  [
    "inst0004|group=143-GR14A|row=0000",
    {
      "parent_id": "inst0004|group=143-GR14A|row=0000",
      "kind": "Y_SPLIT",
      "t": 0.15445950791565735,
      "point_id": "BY_CENTER",
      "mandatory": true,
      "rail": null
    },
    {
      "parent_id": "inst0004|group=143-GR14A|row=0000",
      "kind": "Y_SPLIT",
      "t": 0.2731265800281537,
      "point_id": "BY_OUTER_LOWER",
      "mandatory": true,
      "rail": null
    }
  ]
]
```

相对WAG：最大加工负载差 +52.003s；MOVE总耗时差 +100.614s；WAIT总耗时差 +0.000s；Cmax差 +49.890s。

决定完工的R0：加工3911.075s、MOVE 117.532s、WAIT 0.000s；WAG决定完工的R1：加工3859.072s、MOVE 119.645s、WAIT 0.000s。全局最大加工负载不能替代这条实际完成时间线。

#### WAG, seed=20261033

Cmax=3978.717s，由R1决定。

|机器人|加工/s|MOVE/s|WAIT/s|完成/s|
|---|---:|---:|---:|---:|
|R0|3837.288|52.529|0.000|3889.817|
|R1|3859.072|119.645|0.000|3978.717|
|R2|2263.746|54.163|0.000|2317.909|
|R3|2362.448|70.331|0.000|2432.780|

R0执行顺序（block[方向]）：

inst0012|group=144-DK1B.dxf|row=0000::whole[0] → inst0023|group=145-LB17A.dxf|row=0004::whole[1] → inst0023|group=145-LB17A.dxf|row=0002::whole[1] → inst0008|group=143-FR69B.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0001::whole[0] → inst0023|group=145-LB17A.dxf|row=0003::whole[0] → inst0027|group=804-FR315A|row=0000::whole[0] → inst0014|group=888-BK304A|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0000::whole[0] → inst0025|group=143-FR65A.dxf|row=0001::whole[1] → inst0025|group=143-FR65A.dxf|row=0002::whole[0] → inst0025|group=143-FR65A.dxf|row=0003::whole[1] → inst0025|group=143-FR65A.dxf|row=0004::whole[0]

R0 WAIT起止秒：

无正时长WAIT

R1执行顺序（block[方向]）：

inst0024|group=888-BK304A|row=0000::whole[0] → inst0007|group=888-BK304A|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0001::whole[0] → inst0002|group=143-FR76A.dxf|row=0000::whole[0] → inst0002|group=143-FR76A.dxf|row=0002::whole[1] → inst0002|group=143-FR76A.dxf|row=0003::whole[0] → inst0020|group=685-BK21A.dxf|row=0000::whole[1] → inst0022|group=312-GR2C|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::0[1] → inst0008|group=143-FR69B.dxf|row=0000::whole[1] → inst0008|group=143-FR69B.dxf|row=0003::whole[0] → inst0015|group=804-FR315A|row=0000::whole[1] → inst0023|group=145-LB17A.dxf|row=0000::whole[1] → inst0005|group=145-LB20A|row=0000::whole[0] → inst0008|group=143-FR69B.dxf|row=0002::whole[0] → inst0028|group=183-CM2B.dxf|row=0000::whole[0] → inst0017|group=183-CM2B.dxf|row=0000::whole[1] → inst0021|group=312-GR2C|row=0000::0[0] → inst0019|group=826BK312A|row=0000::whole[0]

R1 WAIT起止秒：

无正时长WAIT

R2执行顺序（block[方向]）：

inst0011|group=804-FR314A.dxf|row=0000::whole[0] → inst0011|group=804-FR314A.dxf|row=0001::whole[0] → inst0011|group=804-FR314A.dxf|row=0002::whole[1] → inst0018|group=143-FR76A.dxf|row=0004::whole[1] → inst0018|group=143-FR76A.dxf|row=0005::whole[0] → inst0018|group=143-FR76A.dxf|row=0006::whole[1] → inst0018|group=143-FR76A.dxf|row=0003::whole[1] → inst0018|group=143-FR76A.dxf|row=0002::whole[0] → inst0018|group=143-FR76A.dxf|row=0001::whole[1] → inst0021|group=312-GR2C|row=0000::1[1] → inst0026|group=801-FR322A|row=0000::whole[1] → inst0018|group=143-FR76A.dxf|row=0000::whole[0]

R2 WAIT起止秒：

无正时长WAIT

R3执行顺序（block[方向]）：

inst0006|group=183-CM2B.dxf|row=0000::whole[1] → inst0003|group=642-BL14A.dxf|row=0000::whole[1] → inst0001|group=826BK312A|row=0000::whole[0] → inst0004|group=143-GR14A|row=0000::1[1] → inst0002|group=143-FR76A.dxf|row=0006::whole[1] → inst0002|group=143-FR76A.dxf|row=0004::whole[1] → inst0002|group=143-FR76A.dxf|row=0005::whole[0] → inst0000|group=143-FR65A|row=0000::whole[0] → inst0013|group=642-BL14A.dxf|row=0000::whole[0] → inst0016|group=685-BK21A.dxf|row=0000::whole[1] → inst0009|group=642-BL14A.dxf|row=0000::whole[1] → inst0010|group=804-FR314A|row=0000::whole[0]

R3 WAIT起止秒：

无正时长WAIT

父模式从共同/原生初解到最终的变化：

```json
[]
```

诊断阶段不新增搜索算子；上述等待与路线差异是观察证据，不能只凭加工负载判定协调效果。未根据本次诊断调参或增加WAIT/CRRR/JRRB。

## 主方法确定（独立几何生成前）

```json
{
  "method": "MODE_NEW",
  "fixed": "FIXED_NEW",
  "gnn_stable": false,
  "gnn_pairwise_positive": true,
  "mode_stable": true,
  "selection_rule": "Pairwise checks: no worse feasibility, positive geometry median, wins>losses, wins in >=2 tiers. Final joint method also requires its own MODE-versus-FIXED benefit. This final development selection precedes independent geometry generation; search parameters are unchanged."
}
```
固定沿用既有瓶颈选择轮换、q=4/8、插入限制8、插入评价上限4096、direction_versions=2、严格Cmax接受和B32；不随正式测试结果修改。
