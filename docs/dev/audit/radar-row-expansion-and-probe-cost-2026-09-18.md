# 行展开搬到设备、腿窄化改为按需（2026-09-18）

承接 [单帧宿主开销](radar-frame-host-cost-2026-09-18.md)。那一轮把逐观测簿记降下来之后，
剩下两个限制，**分属不同 regime**，所以两项都得做：观测多的帧卡在行展开，探针多的帧卡在每探针。

测量环境与上一轮相同：witwin2 / RTX 5080 / Ryzen 7 9800X3D / Torch 2.10.0+cu128 / Windows，
共享桌面。基线是 `f00bd8e`。

## 一、两个 regime 的实测

三个 fixture，`Motion.adaptive()` 默认容差，CUDA event 直接测原生调用：

| fixture | 观测 | 探针 | 一帧 | `_adaptive_echo` | `_adaptive_trace` | 原生设备 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIMO walker 3×4，128×256 | 98304 | 27 | 47.3 ms | 27.1 ms (57%) | 19.5 ms (41%) | 1.48 ms (3.1%) |
| 转子点 1×1，128×128 | 16384 | 155 | 71.2 ms | 1.7 ms (2%) | 68.2 ms (96%) | 1.85 ms (2.6%) |

左边这类由 echo 的行展开决定，右边这类由每探针固定开销决定，**每探针 455 µs**。

## 二、改动一：行展开在设备上做

`_adaptive_echo` 原本在主机上把紧凑的逐观测列展开成每行一个整数，再把结果传过去。
主机手里只有几十个区间和一组逐观测列，这些设备已经有了。

独立测量（本帧的 393216 行，索引逐位相同）：

```
numpy 展开 + H2D    9.23 ms
设备端展开          0.64 ms     14.4×
```

H2D 从每帧约 15 MB 降到约 5 MB，且改为每帧上传一次而不是每批上传。

**展开本身不引入同步，但上传仍然同步——这一条最初写错了。**
每个 `repeat_interleave` 都用主机自己的行总数声明 `output_size`，所以展开不需要读回设备。
`_adaptive_echo` 对设备也确实是**零次**宿主读取，`tests/test_import_boundary.py` 的围栏按原样通过。

但 pageable 的主机到设备拷贝本身是同步的，所以每次 `upload` 就是一次同步 CUDA 操作：
改动后**每帧 6 次**，改动前是**每批 5 次**。用 `torch.cuda.set_sync_debug_mode` 加正负对照
（`.item()` 与无 `output_size` 的 `repeat_interleave` 命中；有 `output_size` 的与 `cumsum` 不命中）
独立复现。多批次的帧上这是大幅减少，**单批次的帧反而多一次**。

本文件与 `PERFORMANCE.md` 的初版都写成了"不引入同步"，并说这条由
`tests/test_import_boundary.py` 保证——两句都是错的。那个测试是对四个属性名
（`cpu`／`numpy`／`tolist`／`item`）的 AST 扫描，看不见上传、`int(tensor)`、`torch.nonzero`，
也看不见漏掉 `output_size` 的 `repeat_interleave`。它保证的是"不读取设备"，不是"不同步"。

**一个 dtype 陷阱。** 设备上 `int64 张量 * Python float` 提升到**默认 dtype**，即 float32，
会把 ADC 瞬时悄悄降成单精度。显式 `.double()` 才与 numpy 的 int64→float64 提升一致。
这是这项改动里唯一可能静默改变数值的地方，逐位比对就是为了兜住它。

## 三、改动二：批处理组不再急切窄化 slot

`RadarLegBatch.slot(...)` 构造一个重新校验过的单 slot 批次：16 次 narrow 切片加约 25 次
`require_tensor`（dtype、rank、shape、contiguous）。回放循环对**每个观测**构造两个。

但批处理组只合成一次，之后不从单个 slot 读任何东西——它要的行已经从
`batched_paths` 里切出来了。那些窄化只服务于记录里的 `legs` 成员，而 `legs`
在自适应路线上**每帧只有一个观测**会发布（帧的收尾观测，进 `last_propagation`）。
非批处理分支（`len(group) == 1`）里 `legs` 就是 replay 本身，窄化是空操作。

改为存一个返回 legs 的可调用对象，两个消费点（`observations()` 和 `traces()`）调用它。
实测（155 探针的转子帧）：

```
simulate: 310 次 slot() → 2 次
trace:    310 次 slot() → 2 次
```

保留语义不变：闭包捕获的是 `replay`，而窄化视图本来也让父张量存活，所以 `Paths` 的
retention 契约没变。

## 四、验收

**逐位等价。** 三个 fixture 在每一步改动后 `torch.equal` 对比 `f00bd8e` 的 cube，全部 True；
调度表 `np.array_equal` 全部 True。

**这句话的范围是 cube。** 两处不在其内，都要说明：

- `Result.adaptive_diagnostics[...]["synthesis_batches"]` **会变**。行预算不再为非活跃发射机的行
  买单，一批因此能覆盖更多观测：walker 帧从 5 批变成 2 批。这是公开字段。
- **梯度在本改动两侧都不是逐位可复现的**：反向用原子累加，同一版本跑两次就差约 1.8e-7 绝对值。
  前向 primal 与 JVP 切线是逐位一致的。所以这里的等价不延伸到梯度。

| fixture | `f00bd8e` | 设备端展开后 | 按需窄化后 | 累计 |
| --- | ---: | ---: | ---: | ---: |
| MIMO walker | 50.3 ms | 23.7 ms | **21.6 ms** | 2.33× |
| 转子点 | 124.2 ms | 68.3 ms | **48.3 ms** | 2.57× |
| 双散射点 3×4，32×64 | 20.3 ms | 10.2 ms | **10.1 ms** | 2.01× |

**序列**（`tools/validate_frame_streaming.py --frames 8 32 128`）：

| 帧数 | 流式峰值 | 堆叠峰值 | 比值 | 流式每帧 | 堆叠每帧 | 不一致帧 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 76.4 MiB | 113.4 MiB | 1.48× | 18.4 ms | 19.8 ms | 0 |
| 32 | 75.4 MiB | 293.6 MiB | 3.89× | 20.2 ms | 22.7 ms | 0 |
| 128 | 76.4 MiB | 1223.4 MiB | 16.01× | 19.6 ms | 20.1 ms | 0 |

10 fps 下 128 帧（12.8 秒数据）**2.51 秒**，上一轮是 4.90 秒，两轮之前是 13.86 秒。
1 分钟该会话的数据约 **12 秒**。

**峰值显存上升了，要说清楚。** 流式峰值从 56.2 MiB 涨到 76.4 MiB：行映射现在是设备张量，
一个批次内有若干个同时存活。它**仍与帧数无关**——那是 `stream` 存在的理由，没有破坏；
`row_budget` 仍然是约束它的那个旋钮，只是常数变大了。这笔交易是 20 MiB 的恒定分配换一半延迟。

**精度未变。** `validate_adaptive_motion` 的 IQ 相对 L2 为 2.8020e-4／2.2612e-4／2.4938e-4／
5.6830e-4，`validate_doppler_motion` 为 1.144e-6／1.845e-4／4.085e-4，
`validate_heavy_multipath` 的四个最强峰仍在图像几何的一个距离/速度 bin 内——
每一个都与改动前同样的数字。

**测试。** `pytest tests --gpu` 全绿：1453 passed, 1 skipped。

## 五、评估过但否决的，以及一处顺带修掉的

**前端无相位噪声时仍上传时间戳。** `clock` 的上传原本以"存在前端"为条件，而
`_apply_path_phase_rows` 在 `has_phase_noise` 为假时原样返回输入——`frontend.py` 自己的
docstring 就写着"没有相位噪声的链路不应该每帧分配时间戳张量"。改为以相位噪声为条件，
省掉一次上传（即一个同步点）和一次逐行 gather。输出逐位不变。


**逐探针 `pair_offsets.tolist()` 的 memo。** 一帧 155 次设备同步，读的是同一个 composer
拥有的同一个张量。按张量 identity memo 掉，实测转子帧 47.6 → 47.0 ms，**1.3%**。
八行代码加一个 `id()` 键的缓存换 1.3%，不划算，已回退。记在这里是为了不再有人重做一遍。

## 六、一条两侧共有的未检查前提

`_adaptive_echo` 用**节点 0** 的 pair 行界去读**所有**节点的行（`local_row` 由
`bounds[owner, active]` 造出，再作用到每个 `starts[nodes[observation, k]]` 上）。
前提是"一个被接受的区间，其各节点共享拓扑恒等因而共享 pair 布局"——`simulation.py` 的注释写了它，
但没有任何代码检查它。构造一张节点间 pair 布局不一致的表，两侧都会**从错误的行返回一个有限的
cube 而不报错**；差得更多时两侧都会触发设备端断言。

改动前后行为相同，不是本轮引入的，但这条注释目前是唯一的守卫。

## 七、剩余边界

转子帧 48.3 ms，每探针约 310 µs，剩下的成本不在 Radar：

| 位置 | 次数/帧 | cumtime/帧 |
| --- | ---: | ---: |
| Channel `consumer.rediscovery_required` | 924 | 10.3 ms |
| Channel `consumer.of` | 1246 | 7.3 ms |
| Core `scene._state_version` | 1236 | — |
| `dataclasses.replace` | 637 | — |

`rediscovery_required` 是每探针的陈旧性轮询，语义上是必需的：自适应控制器必须知道两次探针之间
拓扑有没有变。把它降频需要一个正确性论证，不是性能改动，本轮不动。

walker 帧 21.6 ms，其中原生设备约 1.5 ms。再往下要动原生签名（`basis` 和 `adc_time`
按观测而非按行传入），需要改 ABI、重建扩展、更新 `ci/native-binding-manifest.json`。
