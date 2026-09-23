# 第 3 天：天象计算运行记录

## 计算范围

- 目标：M42（SIMBAD 标准名称 `M 42`）
- 目标坐标：ICRS 赤经 83.8201 度，赤纬 -5.3876 度
- 观测点：北京，经度 116.4074 度，纬度 39.9042 度，高程暂按 0 米
- 本地时段：2026-01-10 18:00 至 2026-01-11 02:00
- UTC 时段：2026-01-10 10:00 至 18:00
- 采样间隔：10 分钟，包含起止点，共 49 个样本

## 计算契约

生产结果由 Astropy 7.2.0 生成。输入目标坐标按 ICRS 解释，观测点使用 `EarthLocation`，时间使用 UTC 尺度的 `Time`，再统一转换到同一个 `AltAz` 坐标系。

每个采样点计算：

- M42 的几何高度角和方位角；
- 太阳的几何高度角；
- 月亮的几何高度角；
- M42 与月亮在同一 AltAz 坐标系中的角距离。

所有内部角量使用 Astropy Quantity 携带单位，只有在写出边界才转为度。`AltAz` 的气压设置为 0 hPa，因此结果不包含大气折射。当前输入模型没有观测点海拔字段，计算暂按海拔 0 米处理。

## 离线数据策略

`iers.auto_download` 固定为 `False`，`iers.auto_max_age` 在计算期间设为 `None`（两者由 `starskill.astropy_offline.offline_iers()` 统一提供，星图、可见性规划与关系计算共用同一策略）。计算期间使用临时可写的 Astropy 缓存目录，使闰秒表和 IERS 检查只读取随依赖安装的离线数据，不访问远程 URL；观测日期超出随包预测范围时按表末数值外推，而不是抛异常。元数据记录 Astropy 版本、UTC 时间尺度、AltAz 坐标系、折射设置和 IERS 下载策略。

这项策略优先保证课堂演示、测试和评审复现的一致性。长期运行时应定期升级 `astropy-iers-data`；如果观测日期超出随包数据覆盖范围，需要人工评估精度或更新离线数据。

## 实际结果抽样

| 本地时间 | M42 高度角 | M42 方位角 | 太阳高度角 | 月亮高度角 | 月亮角距 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-01-10 18:00 +08:00 | 13.213937° | 108.765534° | -9.895488° | -60.246024° | 109.332300° |
| 2026-01-10 22:00 +08:00 | 44.181934° | 169.376318° | -54.975220° | -30.651790° | 111.528854° |
| 2026-01-11 02:00 +08:00 | 23.862364° | 239.778385° | -62.470534° | 12.151843° | 113.133021° |

完整结果位于：

- `runs/day3_m42/intermediate/ephemeris.csv`
- `runs/day3_m42/intermediate/ephemeris.json`

CSV 固定包含本地时间、UTC、目标高度角、目标方位角、太阳高度角、月亮高度角和月亮角距 7 列；角度统一保留 6 位小数。

## 独立交叉验证

验证脚本使用 Skyfield 1.53、`skyfield-data` 6.0.0 和本地 DE421 星历，独立计算 18:00、22:00、02:00 三个本地时刻。每个时刻检查目标高度角、目标方位角、太阳高度角、月亮高度角和月亮角距，共 15 个角量。

- 允许绝对差值：0.001 度（3.6 角秒）
- 实际最大差值：0.000226558 度（约 0.82 角秒）
- 通过数量：15/15

验证明细位于 `runs/day3_m42/verification/skyfield_crosscheck.csv`。两套引擎在太阳系星历、地球定向参数和坐标变换实现上并不完全相同，因此不要求逐浮点位一致；当前差异在预先记录的容差内。

## 复现命令

```powershell
python -m starskill ephemeris examples\observation_m42_beijing.json `
  --target-file runs\day2_m42\intermediate\target_resolved.json `
  --output runs\day3_m42\intermediate\ephemeris.csv `
  --metadata runs\day3_m42\intermediate\ephemeris.json

python -m pip install -e ".[validation]"
python scripts\verify_day3_skyfield.py `
  examples\observation_m42_beijing.json `
  runs\day2_m42\intermediate\target_resolved.json `
  runs\day3_m42\verification\skyfield_crosscheck.csv
```

## 当前限制

- 输出是几何高度角，不代表目视时受温度、气压和折射影响后的表观高度角。
- M42 坐标未包含自行、视差或径向速度；对该深空目标和当前教学时段影响很小。
- 月相照明比例和观测窗口判定属于第 4 天任务，本阶段只提供月亮高度与角距证据。
- 本阶段不结合天气、云量、地平线遮挡和设备视场作最终可观测性结论。
