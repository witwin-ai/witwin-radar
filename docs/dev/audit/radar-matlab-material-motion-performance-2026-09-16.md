# MATLAB 材质、动态场景与性能实测对照

> **历史记录。** 本报告描述 2026-09-16 当时的实现。其"轨迹和场景到 beat IQ"性能结论
> 与动态场景精度表已被 [2026-09-17 复跑](radar-matlab-comparison-2026-09-17.md) 取代：
> 同一 fixture 重测后 WiTwin 在四个基准动态场景上比 MATLAB 快 1.6～2.9 倍，
> 而不是慢 21～53 倍。材质与已知路径两节的结论未变。
> 本文保留原样，不回填新数字。

日期：2026-09-16。阶段前置提交 dc41203 已完成七组指定路径对照。本报告扩展到真实材质反射 API、
完整运动点目标、地面四路径组合及重复性能测量；生产 Radar／Channel 代码未在本阶段修改。

## 结论

WiTwin 在已测的 Fresnel／有限厚度板材模型下与外部参考高度一致；其连续延迟合成避免了低采样率
分数延迟滤波的大误差，但默认自适应运动仍有约 0.5%～0.8% 的模型内 IQ 误差。
不能据此宣称 WiTwin 在所有采样率和容差下都比 MATLAB 准确：提高 MATLAB 采样率后，旋转点的
解析参考误差低于当前 0.02 rad 容差的 WiTwin 自适应结果。

性能必须按入口区分：指定路径到 IQ 的 CUDA 合成很快，但当前公开动态场景入口明显慢于本机
MATLAB 的解析点目标入口。原来的 heavy 自适应相对自身 ADC 参考加速，不能改写成对 MATLAB 的优势。

## 环境、边界与复现

- Windows，witwin2，Torch 2.10.0+cu128，RTX 5080；CPU 为 Ryzen 7 9800X3D（8 核／16 线程）。
- MATLAB 25.2.0.3150157，R2025b Update 4；实际安装并调用 Radar Toolbox 与 Phased Array System Toolbox。
- WiTwin 使用原生 CUDA complex64；本次 MATLAB FMCWWaveform／radarTransceiver 输出为 CPU double。
  没有执行 MATLAB GPU 或单精度优化版本，不宣称本次基线是 MATLAB 的最快实现。
- 性能计时包含返回 IQ；同时保留 GPU 驻留与主机输入／输出传输的数字。路径合成预热 3 次，
  WiTwin 重复 31 次、MATLAB 重复 11 次。场景预热 1 次、重复 3 次。精度控制组仅单次 WiTwin 执行，
  不作为重复性能证据。MATLAB 静态和指定路径场景采用 NumRepetitions 一次提交整帧。
- 最终用于性能表的仿真进程在本任务内串行执行，但运行于共享桌面；系统存在 Unreal Editor 等
  GPU 进程，未暂停用户应用。曾观察到静态场景的严重耗时波动，保留原日志并单独重测静态场景。
  表内中位数不代表独占 GPU、稳定尾延迟或跨机器保证；原始最小／最大／p95 及全部采样均保留。
- 两端已构造世界与雷达对象。WiTwin 公开 simulate 内部的编译／会话／路径工作计入场景耗时；
  MATLAB 复用 System object。动态 MATLAB 按 chirp 更新位姿、通道内使用速度模型；WiTwin 根据
  相位容差探测并插值到 ADC。因此这是部署入口比较，不是同硬件、同精度、同算法工作量的比较。

工具：tools/compare_matlab_scenarios.py、tools/compare_matlab_scenarios.m，通用启动器
tools/run_matlab_comparison.ps1 的 -ComparisonFunction compare_matlab_scenarios 参数。
原始输入／外部输出／环境／日志／图和逐项误差在 output/matlab-scenarios；同名 JSON 证据文件
包含运行身份、比较指标、检查结果和文件 SHA-256。所有数值来自实际执行，不以解析 oracle 冒充 MATLAB。

## 材质反射：360 个组合

扫描 10／24／77 GHz、5／15／30／45／60／85 度掠射角及 H／V 极化，
每种参数同时测试 1000 m 厚度的有损半空间极限和 3 mm 薄板。
材质名称仅表示固定参数的类比，不是实测、随频率变化的玻璃／混凝土／土壤数据库。
五组 (eps_r, sigma[S/m]) 分别为 (2.1,0.001)、(6.31,0.01)、(5.24,0.0462)、(15,1)、(1,5.8e7)。
参数不能直接用于声称真实建筑材料在全部频段的预测精度。

WiTwin 值来自原生 Channel 反射场，除以相同镜像距离的 LOS 场以去除传播与载频相位。
极化明确采用 s 和 p=s×k；入射与反射 TM 基底方向不同，不能把全局 z 极化当作同一个有符号系数。
MATLAB 通过 radar.scenario.SurfaceReflectionCoefficient/reflectionCoefficient 独立计算界面系数。
半空间的 180 个组合直接对照该工具箱模型；另 180 个薄板组合用 MATLAB 界面系数加独立 Airy
内部往返级数参考。后者是组合参考，不冒称 Radar Toolbox 原生完成了有限厚度薄板求解。

| 材质参数组／厚度 | 组合数 | 最大复系数绝对误差 | 相对 L2 |
|---|---:|---:|---:|
| concrete_like_1000m | 36 | 1.45e-07 | 9.356e-08 |
| concrete_like_0.003m | 36 | 1.638e-06 | 7.152e-07 |
| copper_like_1000m | 36 | 2.413e-07 | 1.099e-07 |
| copper_like_0.003m | 36 | 2.413e-07 | 1.099e-07 |
| glass_like_1000m | 36 | 1.821e-07 | 1.292e-07 |
| glass_like_0.003m | 36 | 1.299e-06 | 7.1e-07 |
| low_loss_dielectric_1000m | 36 | 1.819e-07 | 1.03e-07 |
| low_loss_dielectric_0.003m | 36 | 5.393e-07 | 3.11e-07 |
| wet_soil_like_1000m | 36 | 1.96e-07 | 1.077e-07 |
| wet_soil_like_0.003m | 36 | 1.935e-06 | 9.874e-07 |

全表最大绝对误差约 1.94e-6。复系数接近 Brewster 零点时，相对误差会被放大，故同时报告绝对误差。
没有把材质系数从 MATLAB 复制给 WiTwin 来制造一致。
官方接口：[SurfaceReflectionCoefficient](https://www.mathworks.com/help/radar/ref/surfacereflctioncoefficient.html)、
[reflectionCoefficient](https://www.mathworks.com/help/radar/ref/surfacereflectioncoefficient.reflectioncoefficient.html)。

## 动态与 micro-Doppler

两端均为 77 GHz、1 W 发射、1 m² 各向同性点 RCS、理想无噪声接收机。WiTwin 使用 from_rcs 构造
散射响应，MATLAB 使用 radarTransceiver 的真实 Position／Velocity 点目标入口，由工具箱独立计算
传播与散射；没有把 WiTwin 的路径或幅度喂给 MATLAB。位姿接口统一为 float32，MATLAB 内部波形仍为 double。
基线 4 MHz、128 ADC、32 us chirp；静态 128 chirp，动态 1024 chirp。对齐剔除最大传播延迟加 0.8 us
的前沿和首个 chirp。静态／旋转控制将采样率与 ADC 数同时乘 4，保持波形斜率与时长。

加速点：x=30+0.3t+15t² m；旋转点：30 m 中心、3 mm 半径、80 Hz；双肢体代理使用两个正弦
运动散射点，不是人体网格或完整 SMPL。WiTwin 显式启用 adaptive，容差 0.02 rad、幅度 0.02、最大间隔 2 ms。
独立 NumPy oracle 在每个 ADC 时刻计算相同 float32 作者位姿下的连续延迟标量雷达方程。

| 场景 | 两端原始 IQ 差异 | WiTwin／连续 oracle | MATLAB／连续 oracle | 两端 micro-Doppler 功率差异 |
|---|---:|---:|---:|---:|
| acceleration | 2.273% | 0.814% | 2.184% | 3.312% |
| limbs | 2.262% | 0.564% | 2.193% | 3.392% |
| rotor | 2.298% | 0.721% | 2.192% | 3.357% |
| rotor_os4 | 0.776% | 0.752% | 0.291% | 0.388% |
| static | 2.181% | 0.046% | 2.177% | 3.316% |
| static_os4 | 0.144% | 0.046% | 0.139% | 0.207% |

全部均报告原始差异；另存全局复数校准后的指标，但不拿它隐藏幅度、相位或符号问题。
旋转控制说明两端误差来源不同：MATLAB 过采样降低插值误差，WiTwin 的残差主要受自适应容差／
位姿和传播数值精度限制。不能把某一个采样率下的胜负推广为普遍物理优劣。

![Actual micro-Doppler comparison](../../../output/matlab-scenarios/microdoppler-comparison.png)

STFT 两端统一窗口、时间和功率基准，白色虚线是瞬时理论 Doppler；有限窗造成谱带展宽。

## 动态地面多径与材质组合

WiTwin 由有限大平面真实发现 LOS／反射并组合四条往返路径，已断言最终 path_count=4。
雷达和加速点目标相对地面高度均为 30 m，初始水平距离 30 m；H 极化。
两组材质为上述 concrete_like 和 copper_like 的半空间极限。512 ADC、512 chirp、20 MHz、10 MHz 扫频。
MATLAB 使用 twoRayChannel 正向传播和两个回程通道，再相干相加全部四种组合；反射系数由其
SurfaceReflectionCoefficient 随掠射角独立计算。只给它世界坐标、速度和材质参数，没有复用 Channel 求径结果。
额外控制只把 MATLAB 内部采样率提高到 80 MHz，保存完整高采样结果并抽取与原 WiTwin 相同的时刻。

| 场景／MATLAB 内部采样 | 两端原始 IQ 差异 | WiTwin／连续四路径 oracle | MATLAB／连续四路径 oracle |
|---|---:|---:|---:|
| ground_concrete / 20 MHz | 16.999% | 0.511% | 17.001% |
| ground_concrete_os4 / 80 MHz | 0.703% | 0.511% | 0.538% |
| ground_metal / 20 MHz | 32.037% | 0.486% | 32.041% |
| ground_metal_os4 / 80 MHz | 1.003% | 0.486% | 0.922% |

20 MHz 的大差异没有被作为“通过”处理；80 MHz 控制与独立连续参考共同支持其主要来自级联
分数延迟滤波及多路径相干叠加，而非发现了 WiTwin 的 17%／32% 物理错误。
四路径解析 oracle 包含各自总路程、Fresnel 复相位、距离衰减及两条混合路径；无全局幅相拟合。
这仍是平地／平滑界面的模型对照，不是任意 mesh、粗糙漫反射、穿透层、绕射或遮挡拓扑跳变的商业验收。
参考：[twoRayChannel](https://www.mathworks.com/help/radar/ref/tworaychannel-system-object.html)。

## 性能：两个入口得出不同结论

### 已知路径到 beat IQ

静态整采样延迟路径、单位理想硬件。两端完成相同尺寸 IQ，原始 IQ 相对误差均小于 3.5e-7。
MATLAB 单次提交整帧；WiTwin 表列计入主机到 GPU 输入和输出回主机。N 为 ADC，C 为 chirp，P 为路径数。

| N/C/P | WiTwin 含传输中位数 ms | MATLAB CPU double 中位数 ms | 本机延迟比 MATLAB/WiTwin |
|---|---:|---:|---:|
| n128_c64_p1 | 0.2407 | 16.6714 | 69.3 |
| n512_c256_p1 | 0.3115 | 113.8002 | 365.3 |
| n512_c256_p16 | 0.4746 | 219.9849 | 463.5 |
| n512_c256_p64 | 0.7021 | 710.9538 | 1012.6 |
| n512_c256_p256 | 2.2516 | 2637.8686 | 1171.6 |

这些是所测产品入口在此 CPU／GPU 和各自原生精度下的部署数字，不能作为同硬件、同精度算法加速比。
只测了 beat，不能据此推断默认 spectrum、AD、噪声硬件或完整动态场景也有同样加速。

### 轨迹和场景到 beat IQ

| 场景 | WiTwin 场景中位数 s | MATLAB 场景中位数 s | WiTwin/MATLAB 耗时比 |
|---|---:|---:|---:|
| static | 0.01401 | 0.02369 | 0.59 |
| acceleration | 6.47887 | 0.30201 | 21.45 |
| rotor | 13.24748 | 0.25074 | 52.83 |
| limbs | 11.87191 | 0.34254 | 34.66 |

静态在最终独立重测中略快，但共享桌面曾出现很大波动，不作稳健优势承诺。
本组动态场景 WiTwin 约慢 21～53 倍，故当前不具备普遍端到端性能优势。
MATLAB 使用针对自由空间点目标的解析通道，WiTwin 的通用场景路径机制仍进入重发现流程。
两端精度、运动采样策略与内部工作量不同，不能把这个比例反向推广为全部复杂场景的性能结论。

![Measured performance boundaries](../../../output/matlab-scenarios/performance-comparison.png)

## 已定位的瓶颈与下一步

独立 cProfile 旋转点记录（只用于定位，不混入基准中位数）：simulate 15.42 s，自适应部分 15.28 s；
677 次路径重发现约 7.98 s，1354 次两侧 Channel freeze 约 5.83 s；263062 次 torch.full_like 自身约 2.16 s。
evaluate_many 累计 9.59 s，19,526 次 torch.tensor 自身约 2.87 s。累计时间相互包含，不能把它们相加。
原始 profile 与调用表保存在 output/matlab-scenarios/profile。

建议按以下顺序继续优化，并维持现有相位／拓扑／AD 约束：

1. 对空场景 LOS 等可证明拓扑固定的情况，整帧批处理几何与运动，跳过所有无意义的重发现。
2. 将“需要增加相位插值采样点”和“需要重新搜索拓扑”分离。已知拓扑先批量重求；只有遮挡、
   可见性、边界余量或场景事件需要时才重发现。不能用低频轮询来虚称不会漏掉任意短暂路径。
3. 把逐 ADC 创建插值权重、full_like 和对象组装改为设备上的区间索引与批量计算，复用缓冲区。
4. 在这些改动后，以同一连续 oracle／真实 MATLAB 输出重新验精度，并重测公开 simulate 入口。
   先消除会话／发现／Python 调度开销，再考虑进一步优化已很快的 CUDA 相位求和。

本报告提出优化方向，没有在这轮比较中悄悄更改生产算法或放宽验收阈值。
测试新增了“空气边界应消失”以及“功率一致／全局校准不能掩盖相位错误”的诊断不变量，
连同此前外部结果缺失与错误 Doppler 拒绝，共 4 个工具测试通过。生产全 GPU 验收仍引用上一阶段
1531 passed / 12 skipped 的精确记录，本阶段不虚称重新跑过全套。新的 Ruff 与架构／治理门禁结果在证据 JSON。

## 未证明的范围

没有真实材料测量、真实雷达采集、完整人体网格散射、商业 mesh 求径的同场景对照、完整相噪硬件对照，
也没有同精度同 GPU 的 MATLAB 基准。Fresnel／窄带点散射与连续运动 oracle 是模型内物理参考，
不是 Maxwell 全波求解或实测真值。当前结果支持已测模型实现的正确性与误差来源定位，不支持全面商业等价。

MATLAB 官方区分了波形、通道与测量级多径能力；本报告不把 radarDataGenerator 的多径检测
输出当作 radarTransceiver 的一般 mesh IQ 参考：
[RF Propagation Models](https://www.mathworks.com/help/radar/ug/rf-propagation-models.html)。
