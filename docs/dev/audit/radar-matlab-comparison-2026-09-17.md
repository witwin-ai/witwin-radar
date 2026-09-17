# MATLAB 对照复跑：动态场景性能结论反转（2026-09-17）

本报告**取代** [2026-09-16 扩展对照报告](radar-matlab-material-motion-performance-2026-09-16.md)
的"轨迹和场景到 beat IQ"性能结论，并更新其精度表。材质与已知路径两节结论不变。
那份报告保留为历史记录，因为它描述的是当时的实现。

两端在同一次会话内重新执行，不复用任何一侧的旧数字。

## 结论

上一版写的"本组动态场景 WiTwin 约慢 21～53 倍，故当前不具备普遍端到端性能优势"**已不成立**。
同一 fixture、同一机器、同一次会话重测：WiTwin 在四个基准动态场景上比 MATLAB 快 **1.6～2.9 倍**。

这个反转来自本轮六项长序列改进（`docs/dev/audit/radar-long-sequence-program-acceptance-2026-09-17.md`），
不是测量口径变化：fixture、采样参数、对齐规则、重复次数与上一版一致。

**仍然成立的限定**：4 倍过采样的旋转场景 WiTwin 慢 0.75 倍（MATLAB 更快）；
该场景下 MATLAB 的解析参考误差 0.291% 仍低于 WiTwin 的 0.476%。
这仍是部署入口比较，不是同硬件、同精度、同算法工作量的比较。

## 环境与边界

- Windows，witwin2，Torch 2.10.0+cu128，RTX 5080，Ryzen 7 9800X3D。共享桌面，未暂停用户程序。
- MATLAB **25.2.0.3150157 (R2025b) Update 4**，Radar Toolbox 25.2、Phased Array System Toolbox 25.2，
  两者实际安装并调用（`radarTransceiver` 解析到 `D:\Softwares\MATLAB\toolbox\radar\radar\radarTransceiver.m`）。
- WiTwin 原生 complex64；MATLAB FMCWWaveform／radarTransceiver 输出为 CPU double。
  未执行 MATLAB GPU 或单精度版本，不宣称本次基线是 MATLAB 的最快实现。
- Channel 使用既有受校验开发二进制，指纹
  `183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`，本轮未重建。
- Radar 原生 ABI 8，开发构建指纹
  `f5579a04d83aceaf83bc8b71aa8a4112e76326ffb0e40940cc6aa5f366b0efba`。
- 与上一版的**唯一协议差异**：上一版把 `static_os4`／`rotor_os4` 作为"精度控制组"单次执行、
  不计性能；本轮对全部六个运动场景统一 1 次预热 + 3 次测量，因此这两行首次有可比的耗时。
  其余场景同为 1 预热 + 3 测量，地面四路径组同为单次精度组。

工具与复跑：

```powershell
python tools/compare_matlab_scenarios.py motion --output output/matlab-2026-09-17/motion --cases static acceleration rotor limbs static_os4 rotor_os4
python tools/compare_matlab_scenarios.py motion --output output/matlab-2026-09-17/ground --cases ground_concrete ground_metal --accuracy-only
python tools/compare_matlab_scenarios.py materials --output output/matlab-2026-09-17
python tools/compare_matlab_scenarios.py performance --output output/matlab-2026-09-17
# 地面 _os4 控制组是同一输入交给 MATLAB 用 4 倍内部采样率重算，复制输入文件后再启动：
# copy ground_concrete-motion-input.mat ground_concrete_os4-motion-input.mat （metal 同理）
.\tools\run_matlab_comparison.ps1 -OutputDirectory 'output/matlab-2026-09-17/motion' -ComparisonFunction compare_matlab_scenarios
.\tools\run_matlab_comparison.ps1 -OutputDirectory 'output/matlab-2026-09-17/ground' -ComparisonFunction compare_matlab_scenarios
.\tools\run_matlab_comparison.ps1 -OutputDirectory 'output/matlab-2026-09-17' -ComparisonFunction compare_matlab_scenarios
python tools/compare_matlab_scenarios.py analyze --output output/matlab-2026-09-17
python tools/compare_matlab_scenarios.py plot --output output/matlab-2026-09-17
```

三次 MATLAB 启动的 `launch-status.json` 均为 `exit_code=0`、`comparison_executed=true`、
`fresh_results` 等于 `expected_cases`（6／4／6）。原始复数输出、日志、图和
`comparison.json` 保存在 `output/matlab-2026-09-17`。

## 动态场景性能：结论反转

两端均为 77 GHz、1 W、1 m² 各向同性点 RCS、理想无噪声接收机。
基线 4 MHz、128 ADC、32 us chirp；静态 128 chirp，动态 1024 chirp。
`_os4` 把采样率与 ADC 数同时乘 4（512 ADC × 1024 chirp = 524288 个观测），保持扫频斜率与时长。
MATLAB 按 chirp 更新位姿、通道内使用速度模型；WiTwin 按相位容差探测并插值到 ADC。

| 场景 | WiTwin 中位数 s | 实测范围 | MATLAB 中位数 s | 实测范围 | MATLAB/WiTwin |
| --- | ---: | --- | ---: | --- | ---: |
| static | **0.00777** | 0.00776–0.00827 | 0.02211 | 0.02178–0.02359 | **2.85×** |
| acceleration | **0.10274** | 0.10227–0.10278 | 0.25273 | 0.24557–0.27561 | **2.46×** |
| rotor | **0.13363** | 0.13202–0.13380 | 0.21681 | 0.21182–0.22918 | **1.62×** |
| limbs | **0.16927** | 0.14885–0.19943 | 0.29514 | 0.27818–0.31549 | **1.74×** |
| static_os4 | **0.01707** | 0.01143–0.01837 | 0.03253 | 0.03169–0.03270 | **1.91×** |
| rotor_os4 | 0.45180 | 0.42250–0.50178 | 0.33910 | 0.30536–0.38939 | **0.75×** |

与上一版同一 fixture 的对照（MATLAB 侧两次会话的差异属于机器负载，不是实现变化）：

| 场景 | WiTwin 2026-09-16 | WiTwin 2026-09-17 | 提升 | 上一版 WiTwin/MATLAB 耗时比 | 本版 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0.01401 | 0.00777 | 1.8× | 0.59 | 0.35 |
| acceleration | 6.47887 | 0.10274 | **63×** | 21.45 | 0.41 |
| rotor | 13.24748 | 0.13363 | **99×** | 52.83 | 0.62 |
| limbs | 11.87191 | 0.16927 | **70×** | 34.66 | 0.57 |

rotor_os4 是唯一 MATLAB 更快的场景：524288 个观测下 WiTwin 的 ADC 合成批次成为主导，
探针数已经不是瓶颈。这条不能推广成"高采样率下 MATLAB 总是更快"，
也不能用其余五行推广成"WiTwin 在所有动态场景更快"。

![Measured performance boundaries](../../../output/matlab-2026-09-17/performance-comparison.png)

## 精度：动态场景全面改善，地面多径略降

独立 NumPy oracle 在每个 ADC 时刻按同样的 float32 作者位姿计算连续延迟标量雷达方程。
两端统一剔除最大传播延迟加 0.8 us 的前沿和首个 chirp。全部为未经全局幅相校准的原始差异。

| 场景 | 两端原始 IQ 差异 | WiTwin／连续 oracle | （上一版） | MATLAB／连续 oracle | 两端 micro-Doppler 功率差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| acceleration | 2.193% | **0.244%** | 0.814% | 2.184% | 3.311% |
| limbs | 2.221% | **0.458%** | 0.564% | 2.193% | 3.283% |
| rotor | 2.229% | **0.460%** | 0.721% | 2.192% | 3.330% |
| rotor_os4 | 0.487% | **0.476%** | 0.752% | 0.291% | 0.334% |
| static | 2.181% | 0.046% | 0.046% | 2.177% | 3.316% |
| static_os4 | 0.144% | 0.046% | 0.046% | 0.139% | 0.207% |

WiTwin 对连续 oracle 的误差在四个动态场景上都下降（加速点 3.3 倍）。
原因是这些 fixture 的帧观测跨度 32.8 ms 远大于原 2 ms 区间上限：
上一版被上限强制切成 17 段线性插值，本版由相位检验用四次插值决定分段。
MATLAB 列与上一版逐位一致，符合预期——MATLAB 侧没有变化。

**仍然不能宣称 WiTwin 普遍更准**：`rotor_os4` 下 MATLAB 的 0.291% 仍优于 WiTwin 的 0.476%。
差距从 2.58 倍收窄到 1.64 倍，但方向没有变。MATLAB 过采样降低其分数延迟插值误差，
WiTwin 的残差主要受自适应容差与 float32 路径精度限制。

![Actual micro-Doppler comparison](../../../output/matlab-2026-09-17/microdoppler-comparison.png)

STFT 两端统一窗口、时间和功率基准；有限窗造成谱带展宽。

## 地面动态多径：四路径组合

雷达和加速点目标相对地面高度均为 30 m，初始水平距离 30 m，H 极化；
512 ADC、512 chirp、20 MHz、10 MHz 扫频。MATLAB 用 twoRayChannel 正向加两个回程通道
相干相加四种组合，反射系数由其 SurfaceReflectionCoefficient 随掠射角独立计算。
WiTwin 由有限大平面真实发现 LOS／反射并组合四条往返路径，已断言 `path_count=4`。

| 场景／MATLAB 内部采样 | 两端原始 IQ 差异 | WiTwin／连续四路径 oracle | （上一版） | MATLAB／连续四路径 oracle |
| --- | ---: | ---: | ---: | ---: |
| ground_concrete / 20 MHz | 17.037% | 0.762% | 0.511% | 17.001% |
| ground_concrete_os4 / 80 MHz | 0.929% | 0.762% | 0.511% | 0.538% |
| ground_metal / 20 MHz | 32.062% | 0.682% | 0.486% | 32.041% |
| ground_metal_os4 / 80 MHz | 1.130% | 0.682% | 0.486% | 0.922% |

20 MHz 的 17%／32% 仍然主要来自 MATLAB 的级联分数延迟滤波：80 MHz 控制组把它降到
0.538%／0.922%，而 WiTwin 的值不随 MATLAB 采样率变化。

WiTwin 这里**变差了**，0.511%→0.762%、0.486%→0.682%。这是预期的代价而不是缺陷：
地面场景有反射几何，拓扑不可证明完整，因此区间上限仍然强制、插值阶数固定为 2，
而本轮改动让初始分区取上限允许的最粗等分，段数比上一版少。误差仍远在 0.02 rad 容差内。
代价是 80 MHz 控制组下 concrete 的胜负翻转：MATLAB 0.538% 现在优于 WiTwin 0.762%；
metal 仍是 WiTwin 0.682% 优于 MATLAB 0.922%。需要更小误差的调用方按标定调低
`phase_error_rad`（实测 IQ 相对 L2 ≈ 0.55–0.66 × 实测最大每路径相位残差）。

## 材质与已知路径：结论不变

材质反射 360 个组合（10／24／77 GHz × 5／15／30／45／60／85 度 × H／V ×
1000 m 有损半空间与 3 mm 薄板，五组 (eps_r, sigma)）：**最大复系数绝对误差 1.9348e-6**，
与上一版的 1.94e-6 一致。Channel 的反射场本轮未改动，这一致性正是预期。

已知路径到 beat IQ，静态整采样延迟、单位理想硬件，两端完成相同尺寸 IQ：

| N/C/P | 原始 IQ 相对误差 | WiTwin 含主机传输中位数 ms | MATLAB CPU double 中位数 ms | 本机延迟比 |
| --- | ---: | ---: | ---: | ---: |
| n128_c64_p1 | 1.185e-07 | 0.2058 | 12.9898 | 63.1 |
| n512_c256_p1 | 1.147e-07 | 0.2830 | 84.6786 | 299.2 |
| n512_c256_p16 | 1.002e-07 | 0.3502 | 167.8257 | 479.2 |
| n512_c256_p64 | 1.530e-07 | 0.5902 | 549.9557 | 931.8 |
| n512_c256_p256 | 3.491e-07 | 1.5334 | 2167.1026 | 1413.3 |

这条路径本轮未改动；上一版记录的是 0.2407–2.2516 ms 对 16.671–2637.869 ms。
两版差异属于机器状态，不是实现变化。这些仍是不同设备、不同原生精度的部署数字，
不是同硬件算法加速比。

## 未证明的范围

没有真实材料测量、真实雷达采集、完整人体网格散射、商业 mesh 求径的同场景对照、
完整相噪硬件对照，也没有同精度同 GPU 的 MATLAB 基准。
Fresnel／窄带点散射与连续运动 oracle 是模型内物理参考，不是 Maxwell 全波求解或实测真值。
本次没有测试粗糙漫反射、穿透层、绕射或遮挡拓扑跳变。
双肢体是解析材料点代理，不是 SMPL 人体模型。
性能中位数不代表独占 GPU、稳定尾延迟或跨机器保证；原始最小／最大／p95 保留在
`output/matlab-2026-09-17/comparison.json`。

官方接口：
[radarTransceiver](https://www.mathworks.com/help/radar/ref/radartransceiver-system-object.html)、
[twoRayChannel](https://www.mathworks.com/help/radar/ref/tworaychannel-system-object.html)、
[SurfaceReflectionCoefficient](https://www.mathworks.com/help/radar/ref/surfacereflctioncoefficient.html)。
