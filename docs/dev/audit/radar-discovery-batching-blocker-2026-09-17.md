# 多径拓扑发现的批处理：收益已量化，阻塞点在 Channel 接口（2026-09-17）

结论：**Radar 侧无法实现。** 批处理值 3.8 倍发现时间（重多径帧约 2.4 倍），
但消费一次打包发现需要 Channel 提供按探针切分或按配对限制的能力，
现有 `PropagationRequest` 两者都没有。本轮不伪造 Radar 侧绕路，
只把收益、阻塞点和所需接口写清楚。

## 发现成本占比

三面墙、每程 depth 2、64 条往返路径、32 chirp × 64 ADC，`motion_sampling="adaptive"`。
单帧 cProfile（仅用于定位）：总 467 ms，其中 `_rediscover` 401 ms，
`ChannelPropagationAdapter.freeze` 38 次共 361 ms，`consumer.evaluate` 38 次共 324 ms。
**发现占整帧 69%**，19 个探针 × 2 条腿 = 38 次独立 Channel 调用。

非可证明世界里每个探针都必须自己发现一次——这就是探针的定义。
可优化的不是发现次数，而是每次调用的固定开销。

## 每次调用的固定开销支配总量

`tools/validate_discovery_batching.py` 把 P 对端点打进一次 `freeze`。
Channel 的 `PropagationRequest` 没有配对限制，因此一次打包调用会求解完整的 P×P 交叉积。

| 打包探针数 | 实际求解配对 | 一次调用耗时 | 等量独立调用 | 批处理加速 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 9.49 ms | 9.49 ms | 1.00× |
| 2 | 4 | 10.06 ms | 18.97 ms | 1.89× |
| 4 | 16 | 14.05 ms | 37.94 ms | 2.70× |
| 8 | 64 | 21.94 ms | 75.88 ms | 3.46× |
| 19 | 361 | 43.48 ms | 180.22 ms | **4.15×** |
| 38 | 1444 | 79.47 ms | 360.45 ms | 4.54× |

拟合出的固定开销约 **9.4 ms/次**，边际约 **0.05 ms/配对**。
即使白算 19×19 的交叉积，一次打包调用仍比 19 次独立调用快 4 倍。
按发现占 69% 折算，重多径帧端到端约 2.4 倍。

这些数字来自空间上互不相同的 1 mm 步进端点，不是真实探针轨迹；
它们衡量的是调用结构的成本，不是某条具体轨迹的发现难度。

## 为什么 Radar 侧到此为止

`consumer.PropagationRequest` 只有 `sources`/`sinks` 两个 `EndpointBatch`，
没有配对掩码、没有 slot 分组。`topology_mode="discover"` 求解全交叉积，
返回一个 `PropagationTopology`，再由 `consumer.prepare_fixed_topology` 变成
`PreparedFixedTopology` 句柄。生产路径用 `reevaluate`/`reevaluate_slots` 消费该句柄。

要按探针使用一次打包发现，需要以下两者之一，二者都在 Channel：

1. **配对限制**：`PropagationRequest` 接受一个配对索引或 slot 分组，只求解
   第 i 个源与第 i 个汇，避免交叉积，并使返回的行天然按探针分组。
2. **句柄切分**：一个把 `PreparedFixedTopology` 按 `(source_id, sink_id)` 子集切成
   若干独立句柄的接口。Radar 能看到 `source_id`/`sink_id`，但不能切 `prepared`。

单靠现有接口不行的具体原因：不同探针的路径族行数不同（这正是探针要检测的），
所以打包后的拓扑在探针维度上是不规则的，既不能用 `_for_slots` 复制，
也不能在 Radar 侧安全地重建每个探针的 `RadarLegBatch`——那需要 Channel 的 `prepared` 句柄。

另外 8.4 ms 的单次成本分布在 Channel 的 Python 编排里
（`evaluate_path_fields` 2.9 ms、`_evaluate_reflection_fields` 1.9 ms、
`evaluated_paths_from_block` 1.6 ms），也不是 Radar 能优化的位置。

## 不做的事

没有把两条腿合并成一次发现：那会产生 `(site, site)` 的零长度自配对和
共址收发时的 `(TX, RX)` 零长度配对，是退化几何，且仍然需要按腿切分句柄。
没有修改 Channel 的原生库：那会产生新的 Channel 构建指纹，
使本仓库每一份引用 `183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`
的验收记录失效，并需要满足 Channel 自己的测试与发布策略。这是独立的跨仓库工作。

## 复跑

```powershell
python tools/validate_discovery_batching.py
```

证据：`output/discovery-batching/results.json`。
环境：witwin2 / RTX 5080 / Ryzen 7 9800X3D / Torch 2.10.0+cu128 / Windows，共享桌面，
Channel 开发二进制指纹 `183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`。
