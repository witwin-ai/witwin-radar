# 长序列生成六项改进的总验收（2026-09-17）

六项按依赖顺序实施：区间上限先于插值阶数，因为上限强制时阶数买不到任何区间长度。
四项落为代码，两项落为量化结论——其中一项实测无余量，一项阻塞在 Channel 接口。
所有数字来自实际执行；不宣称任何未执行的运行。

## 六项结果

| # | 内容 | 结果 | 提交 |
| --- | --- | --- | --- |
| 1 | 高阶插值 | K 节点 Lagrange 原生插值，默认 5 节点。转子帧 76–82 ms → **19.0 ms**，探针 73–101 → **9** | `f32c13b` |
| 2 | 区间上限与二分 | 上限仅在拓扑不可证明完整时强制；强制时从最粗允许分区起步。MIMO 探针 65 → 17 | `596744e` |
| 3 | 跨帧热启动 | **实测无余量**：可回收探针 0/69、0/36、13/23511（0.055%）。不实现 | `d79c333` |
| 4 | 多径发现批处理 | **阻塞在 Channel**：批处理值 4.15×（整帧 2.4×），但需要配对限制或句柄切分 | `3ff8cd9` |
| 5 | 长序列流式出口 | `Radar.stream(...)`，峰值显存与帧数脱钩，逐帧逐位一致 | `25fb800` |
| 6 | 完整性标志语义 | `path_set_complete` 与 `motion_sampling_exhaustive` 分开发布 | `36abdb9` |

## 端到端效果

witwin2 / RTX 5080 / Ryzen 7 9800X3D / Torch 2.10.0+cu128 / Windows，共享桌面。
Channel 使用既有受校验开发二进制，指纹
`183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`，本轮未重建。

### 公开场景入口，每帧

| 场景 | 本轮前 | 本轮后 | 探针 |
| --- | ---: | ---: | ---: |
| 转子点，128 chirp × 128 ADC，4 MHz | 75.8–81.6 ms | **19.0 ms** | 73–101 → 9 |
| MIMO walker，3TX×4RX，128 chirp × 256 ADC，10 fps | 132–148 ms | **108.3 ms** | 65 → 9 |
| 三面墙多径，64 路径，32 chirp × 64 ADC | 1.580 s | **0.332 s** | 37 → 19 |

同一会话内对逐 ADC 穷举路线的加速：radial 196×、rotor 134×、双散射点 218×、
heavy 多径 129×；本轮前的同一工具记录为 2.93×／2.91×／3.08×／64.14×。
两组绝对秒数来自负载不同的桌面，不可直接相比；加速比可比，因为各自在同一会话内测得。

### 128 帧真实 MIMO 序列

| 帧数 | 流式峰值 | 堆叠峰值 | 比值 | 流式每帧 | 堆叠每帧 | 不一致帧 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 67.4 MB | 102.3 MB | 1.52× | 110.0 ms | 117.9 ms | 0 |
| 32 | 67.4 MB | 301.3 MB | 4.47× | 107.3 ms | 106.6 ms | 0 |
| 128 | 67.4 MB | 1204.2 MB | 17.88× | 108.3 ms | 109.9 ms | 0 |

10 fps 下 128 帧（12.8 秒数据）耗时 13.86 秒，接近实时。
流式峰值与帧数无关，因此不再有"单次调用约 2500 帧"的显存上限；
堆叠路线的上限仍在，只是现在有了替代出口。

### 精度

精度向声明的容差靠拢，而不是超出它。默认 0.02 rad 下，对同一会话的逐 ADC 参考：
转子 8.5e-3 → **2.7e-3**（更快且更准），双散射点代理 2.2e-4 → 4.5e-3，
三面墙多径 5.66e-4 → 5.68e-4。全部在容差内，也都通过原有 1.5%／2.5% 断言。

`tools/validate_adaptive_tolerance.py` 给出了原验收文档指出缺失的标定：
实际 IQ 相对 L2 约为控制器实测最大每路径相位残差的 0.55–0.66 倍。
要把 IQ 误差压到 X，相位容差取约 1.6X。该系数在单主径、无相干零陷的 fixture 上测得，
不是普遍界。需要恢复此前更严精度的调用方据此调低容差即可。

## 验收执行

- 完整 GPU：**1555 passed、12 skipped、0 failed**，291 秒。12 项跳过为 11 个缺 SMPL 资产
  和 1 个 nightly 共存证据；无缺失 Channel 导致的跳过。
- CPU quick tier：**701 passed、866 skipped**，覆盖率 56%，全部门禁通过。
- Ruff 格式与检查：223 个文件通过。
- 九项静态门禁加原生绑定清单：全部通过。
- 原生 ABI 7 → 8，38 个算子、14 个 AD 组，开发构建指纹
  `f5579a04d83aceaf83bc8b71aa8a4112e76326ffb0e40940cc6aa5f366b0efba`。
- 上一阶段唯一失败的 Channel 反向耗时比门禁，本轮多次完整运行均通过
  （单独复核测得 1.3457，预算 2.00），与该报告"受并发 GPU 负载影响"的结论一致。

本机 `%TEMP%\pytest-of-Asixa` 存在 ACL 拒绝，`tmp_path` 用例需重定向 TEMP 才能运行。
这是环境问题不是被测代码失败：未重定向时 89 个用例报 `PermissionError` 而非断言失败。

复跑：

```powershell
python tools/validate_adaptive_motion.py --output output/adaptive-quartic
python tools/validate_adaptive_tolerance.py --output output/adaptive-tolerance-quartic
python tools/validate_adaptive_probe_efficiency.py
python tools/validate_discovery_batching.py
python tools/validate_frame_streaming.py --frames 8 32 128
pytest tests/ --gpu -q
python ci/run_ci_tier.py quick
```

## 未证明的范围

本轮没有重新执行 MATLAB 对照，因此 `docs/dev/audit/radar-matlab-material-motion-performance-2026-09-16.md`
的所有对外精度与性能限定继续成立，包括"公开动态场景入口慢于 MATLAB 点目标入口"这一条——
该比较需要重跑才能更新。没有执行 Linux、发行 wheel 或远程工作流。
没有测试移动网格、粗糙面漫反射、穿透或绕射下的收益。
中点误差测试是采样界而非区间内最大误差的严格上界，实测乐观 3%–11%。
