# 单帧宿主开销的定位与消除（2026-09-18）

自适应路线的一帧是宿主瓶颈，不是设备瓶颈。本记录是测量、四项改动和验收。

测量环境：witwin2 / RTX 5080 / Ryzen 7 9800X3D / Torch 2.10.0+cu128 / Windows，共享桌面。
Channel 使用既有受校验开发二进制，指纹
`183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`，本轮未重建。
Radar 原生库因 `64c36e8` 改动 CUDA 源而失效，按 `scripts/build_radar_cuda_prebuilt.py` 重建，
新指纹 `6ffb654e876f420ef3d8c502a543891554f0a7e910d0c7bf574d2c17ce5d9f6c`。

## 一、瓶颈在哪里

fixture 是 `tools/validate_frame_streaming.py` 的 3TX×4RX、128 chirp × 256 ADC、10 fps walker，
LOS、空世界、`Motion.adaptive()` 默认容差。

CUPTI 在这台机器上不产生 kernel 事件（chrome trace 里只有 `cpu_op`），因此设备时间用 CUDA event
直接卡在两个原生调用两端测得，不取 profiler 的归因：

| 项 | ms/帧 |
| --- | ---: |
| `interpolate_path_rows` | 1.75 |
| `synthesize_fmcw_observations` | 1.49 |
| **原生合计** | **3.23（占 99 ms 帧的 3.1%）** |

同一帧的诊断：`discovery_count=1`、`compile_count=1`、`evaluations=27`、`accepted_intervals=13`、
`observation_count=98304`。发现和编译各只发生一次，所以三面墙多径那条"发现占 69%"的结论
对本 fixture 不成立。

其余 96 ms 是宿主按**每观测**粒度搬运一张 98304 项的调度表，而该表的不同取值只有十几个。
逐项实测（独立复现，非 profiler 归因）：

| 位置 | ms/帧 | 做的事 |
| --- | ---: | --- |
| `_adaptive_trace` 接受分支 | ~29 | 98304 次 dict 写入，每次重建 `tuple(indices[::2])` |
| `_adaptive_trace` 尾部 | ~13 | 98304 次单行 numpy 赋值灌进 `node_index` |
| `_adaptive_echo` 批次索引 + H2D | ~16 | 118 万行的 numpy 展开与上传，5 批 |
| `_open_session` | ~7.5 | 98304 个 Python float 建 `offsets` |
| `traces()` | ~4 | 再平移一遍，又 98304 个 |
| `Result.from_frames` | ~4 | 对已是 float 的值再 `float()` 一遍 |

一帧共 7253 次 ATen 调度、35.1 ms 宿主自用时间。

## 二、四项改动

均不触碰容差、分区判据或物理。

1. **接受的分区按区间写，不按观测写。** `partitions` 从 `{观测: 节点}` 字典改为
   `[(left, right, nodes)]` 列表，`node_index[left:right+1] = group` 一次切片。区间端点本身
   永远是探针，随后的自指赋值 `node_index[ordered] = ...` 会覆盖相邻区间共享的行，因此与
   字典版逐位等价。
2. **观测调度改 `float64` 数组。** `offsets` 在 `_open_session` 里一次向量化构造，运算次序与
   逐元素 Python 写法相同，因此逐位相同。`Result.sample_times_s` 与 `Paths.sample_times` 随之
   发布数组而非 Python float 元组——这是本轮唯一的公开 API 变更。
3. **echo 只合成本 slot 自己的发射机。** `PAIR_RANK_LAYOUT` 下 pair 的发射机秩是
   `pair % num_tx`，一个 TDM 观测只落在一个 slot、只听一个发射机；帧尾的 slot gather 本来就把
   其余 `1 - 1/num_tx` 丢掉了。改为只展开活跃 pair 的行，行数从 1179648 降到 393216。
4. **pair 行界按被求解的观测存。** `pair_offsets` 本身就是观测内各 pair 的排他前缀，不需要再
   cumsum；且插值观测永不读自己的行，所以 `[观测, pairs]` 的表除探针行外全是零。改为
   `bounds[求解序号, pair]` 加一张 `rank` 映射，省掉每帧两个 9.4 MB 的零数组。

## 三、验收

**逐位等价。** 两个 fixture——上限绑定的 MIMO walker 和相位检验绑定的转子点——每一步改动后
`torch.equal` 对比改动前保存的 cube，全部 True；调度表 `np.array_equal` 全部 True。

**帧延迟**（完整 `Radar.simulate`，7 次取中位）：

| 阶段 | walker ms/帧 | 累计 |
| --- | ---: | ---: |
| 改动前 | 104.9 | 1.00× |
| 分区切片 | 75.9 | 1.38× |
| 数组调度 | 72.4 | 1.45× |
| 发射机限定 echo | 46.8 | 2.24× |
| 数组化发布调度 | 41.0 | **2.56×** |

**序列**（`tools/validate_frame_streaming.py --frames 8 32 128`）：

| 帧数 | 流式峰值 | 堆叠峰值 | 比值 | 流式每帧 | 堆叠每帧 | 不一致帧 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 56.2 MiB | 93.1 MiB | 1.66× | 39.4 ms | 39.9 ms | 0 |
| 32 | 56.2 MiB | 300.6 MiB | 5.35× | 40.1 ms | 40.5 ms | 0 |
| 128 | 56.2 MiB | 1230.3 MiB | 21.91× | 38.3 ms | 39.4 ms | 0 |

10 fps 下 128 帧（12.8 秒数据）耗时 **4.90 秒**，改动前为 13.86 秒；1 分钟该会话的数据约 23 秒。
流式峰值从 67.4 MB 降到 56.2 MB，来自第 4 项去掉的逐观测 pair 表。

**精度未变，也未提升。** `tools/validate_adaptive_motion.py` 同会话测得 IQ 相对 L2
2.80e-4／2.26e-4／2.49e-4／5.68e-4，与 2026-09-17 表格逐位相同。
`tools/validate_doppler_motion.py` 三个 fixture 的 IQ/STFT 误差同样在既有量级。

该工具的**加速比**列不要当成本轮的结果，正反都不要。它的四个 fixture 是 512 和 2048 个观测
对 97 和 19 个探针，量的是探针路径而不是本轮改的逐观测簿记；同会话报出 96.1×／94.7×／95.9×／
143.5×，2026-09-17 那行是 120.4×／119.1×／92.5×／141.3×，这是共享桌面上该规模的单次运行离散。
本轮的结论只立在上面的帧延迟表和序列表上。

**测试。** `pytest tests --gpu` 全绿：1453 passed, 1 skipped。

## 四、公开面变更与连带修改

`Result.sample_times_s` 和 `Paths.sample_times` 改为每帧一个 `float64` 数组。
理由是这份 metadata 的成本高于它记录的计算：ADC 帧每 chirp／发射机／采样各一个瞬时，
物化成 Python float 每帧约 2.4 MB 对象，而值本来就是 numpy 算出来的。
需要 float 的调用方对所需帧调 `.tolist()`；比较两份调度用 `numpy.array_equal` 而不是 `==`。

连带更新：`ci/public-api-snapshot.json` 按 `ci/write_public_api_snapshot.py` 重新生成（diff 仅
这两个签名）；`tests/test_adaptive_motion.py`、`tests/test_frame_streaming.py` 改用
`np.array_equal`；`tests/test_dynamic_motion_sampling.py` 与两个 validate 工具在把调度时刻喂给
`torch.tensor` 前显式取 `float`，否则 dtype 会被 `np.float64` 推成 float64。

## 五、剩余边界

改动后 walker 一帧 41 ms，其中：

- `_adaptive_echo` 约 17 ms，是 393216 行的 numpy 展开与 H2D。逐项实测：节点行索引 3.3 ms、
  `basis[observation]` 的 `[行, 2]` float64 gather 3.2 ms、`adc_time` 1.65 ms、段展开 1.14 ms。
- `_adaptive_trace` 约 3 ms 自用，加 27 个探针的 Channel 回放。
- 原生设备时间约 1.5 ms。

再往下要动原生签名：`interpolate_path_rows` 与 `synthesize_fmcw_observations` 目前按**行**接收
`basis` 和 `adc_time`，若改为按观测接收再由 kernel 展开，可去掉上述 gather 与约三分之二的 H2D。
那需要改 ABI、重建扩展、更新 `ci/native-binding-manifest.json`，本轮未做。
