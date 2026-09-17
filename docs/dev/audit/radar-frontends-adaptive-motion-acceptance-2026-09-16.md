# 前端物理顺序、自适应运动与 MATLAB 对照验收

日期：2026-09-16。环境：Windows、witwin2、Torch 2.10.0 + CUDA 12.8、RTX 5080。

## 实现与提交

| 阶段 | 提交 | 结果 |
| --- | --- | --- |
| 前端域顺序 | 9f5f99e | 有硬件模型时先合成 beat、执行接收链，再转换成请求的 spectrum；理想接收机保留原生直接频谱路径 |
| 同源相位噪声 | 52f50bc | 所有路径共享连续时间 Wiener 振荡器，按真实 ADC 时刻与传播延迟查询相位差；包含 chirp 空闲间隔 |
| 自适应运动 | e6210eb | 原生可微相位运输插值、拓扑一致的 Channel 合批、快时间合批合成，以及误差／拓扑事件细分 |

默认 ADC 模式仍是逐观测重发现的严格参考。快速模式显式选择：

    from witwin.radar.simulation import AdaptiveMotionSpec
    result = radar.simulate(
        scene, times=(0.0,), response=response, sites=sites,
        motion_sampling="adaptive",
        adaptive_motion=AdaptiveMotionSpec(
            phase_error_rad=0.02, relative_amplitude_error=0.02,
            max_interval_s=0.002, max_evaluations=8192,
            batch_observations=256,
        ),
    )

## 物理与数值约束

- 硬件非线性、热噪声和量化作用于时域样本；beat／spectrum 的相同硬件测试覆盖静态、运动、自适应、ADC、AGC、热噪声及组合模型。
- 同源混频相位是同一振荡器过程在接收时刻和发射时刻的差。零延迟时抵消；不同延迟路径的协方差由重叠时间区间决定。统计测试覆盖延迟抑制谱、空闲间隔、查询顺序和帧时间。
- 振荡器目前是白频率噪声对应的 Wiener 模型，不是包含多个斜率区段的完整器件相噪掩模。Wiener 时间／延迟导数不存在，因此显式拒绝；固定查询的信号导数仍支持。
- 插值先将两端 Channel 复数系数运输到查询延迟对应的载波相位，再混合包络，避免直接插值高频复数导致的假相消。原生前向、VJP、JVP 与独立双精度公式核对。
- 控制器检查区间端点、四分点、中点的完整路径身份、有效性、相位和幅度。身份或有效性变化时细分到相邻 ADC 点；发现预算不足则报错。移动反射面在编译句柄被下一次更新作废前求值。
- 这是采样误差控制，不是任意运动的数学全局保证。探测点之间极短的遮挡或高频运动可能未被捕获。未枚举所有 ADC 点时，结果发布 path_set_complete=False。相干零陷附近不能由单路径容限直接推出相对 IQ 误差界。
- 自适应拓扑和区间选择作为离散决策固定，梯度沿选中的原生插值传播；控制器读回是显式记账的主机决策。

## 实测 ADC 对照

工具：tools/validate_adaptive_motion.py。完整 public simulation 耗时，预热后各测一次；
不含首次原生加载，不代表百分位延迟或实时保证。均无噪声，统一处理和归一化。

| 场景 | ADC / 自适应耗时 | 加速 | IQ 相对 L2 误差 | RD 功率相对 L2 误差 | 重发现次数 |
| --- | --- | --- | --- | --- | --- |
| 径向点 | 8.464 / 2.888 s | 2.93× | 0.02601% | 0.004237% | 512 → 193 |
| 转子点 | 8.555 / 2.935 s | 2.91× | 0.02014% | 0.006526% | 512 → 193 |
| 双散射点运动代理 | 8.853 / 2.871 s | 3.08× | 0.02150% | 0.007299% | 512 → 193 |
| 三面墙 heavy 多径 | 101.330 / 1.580 s | 64.14× | 0.05659% | 0.006488% | 2048 → 37 |

Heavy 配置为 32 chirps × 64 ADC、每条单程最多二次反射、64 条往返路径。
未改变多径等效距离：同一物体的不同总路程仍出现在不同距离上，区别仅在求值方法。
原始复数 cube、误差和耗时保存在 output/doppler-repair/adaptive。
最终代码对四组自适应场景的独立重放均与保存的测量 cube 逐位一致，记录见 final-replay.json。

![Exact and adaptive heavy multipath comparison](../../../output/doppler-repair/adaptive/heavy-comparison.png)

图使用矩形窗，弱条纹包含有限观测窗旁瓣；不能把每条条纹都解释为独立物理路径。

## 验收记录

最终完整 GPU 回归：1531 passed、12 skipped、0 failed，309.44 秒。
跳过项为 11 个缺少 SMPL 资产的用例和 1 个需要 nightly 共存证据的用例；
没有缺失 Channel 引起的跳过。CPU quick 全流程通过：697 passed、846 skipped，覆盖率 57%。
Ruff 对 215 个文件检查通过，九项额外架构／文档／治理／发布声明检查通过。
完整日志保存在 output/doppler-repair/frontend-adaptive-full-gpu-final.log、
frontend-adaptive-quick-final.log 和 frontend-adaptive-static.log。
完整数值摘要、运行身份及本地证据文件 SHA-256 索引保存在
docs/dev/audit/radar-frontends-adaptive-motion-evidence-2026-09-16.json。

已完成的阶段测试：
阶段一 44 项；阶段二最终重编译后 58 项；阶段三架构／梯度／运动相关 89 项，
以及扩展的前端域与自适应组合 24 项。各组有重叠，不应相加为独立覆盖总数。

首轮完整 GPU 回归为 1517 passed、12 skipped、5 failed；失败均为新增能力的架构清单未同步，
不是数值断言失败。已精确更新计数和自适应主机读回作用域，未放宽物理误差或性能阈值。
上述最终完整回归是修正后的独立重跑；不能把首轮称为通过。

本轮原生 ABI 6，35 个算子、13 个 AD 组。Radar 构建指纹：
92d3fcc85bd2a2bfa6ec7b50ae629ce35ea0311676a3753a87b873f88dc08e5e。
Channel 使用已有开发二进制，指纹：
183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f。
本轮没有重编译 Channel，也没有完成新的 Linux／wheel／远程发布矩阵。

## MATLAB Radar Toolbox：真实执行完成，指定路径对照通过

本节更新替代此前的启动受阻结论；原始超时证据保留在 output/doppler-repair/matlab。
用户登录后，批处理恢复正常，MATLAB 自身确认两个工具箱许可证均可用，但最初缺少 Radar Toolbox 文件。
经用户批准，使用 MathWorks 官方签名的 MPM 安装 R2025bU4 Radar_Toolbox。
文件安装完成后，安装器在 Windows 注册子进程处报告 system:740（需要提升权限），未正常退出。
随后保留原有 MATLAB 路径、合并新默认工具箱路径、刷新缓存并保存路径；
新的独立 MATLAB 进程成功发现工具箱并完成全部七组 radarTransceiver 执行。
因此确认工具箱在 MATLAB 中可用，但不把 MPM 的失败退出写成成功。

实际版本为 25.2.0.3150157 (R2025b) Update 4，Radar Toolbox 与 Phased Array System Toolbox 均为 25.2。
最新身份、原始复数输出、退出状态及指标保存在 output/doppler-repair/matlab-final。
launch-status.json 记录 exit_code=0、expected_cases=7、fresh_results=7；
脚本逐个检查本次启动后的文件时间，旧结果文件不能冒充本次成功。

### 实验边界与误差

使用 77 GHz 载频、10 MHz 扫频、256 个 chirp，基线采样率 20 MHz、每 chirp 512 点。
静止单径总长 60 m；径向单径总长 60 m、总路程变化率 2 m/s；
三径总长为 60/180/300 m、变化率 2/1/-1 m/s、实数幅度 1/0.6/0.3。
两端使用完全相同的 float32 可表示延迟和延迟变化率，以及无噪声、单位增益硬件。
正 beat 定义为 tx*conj(rx)，等效距离 L/2，正向远离速度 (dL/dt)/2。
MATLAB 由 radarTransceiver 生成接收信号，再调用 dechirp；未以 Python 解析值代替 MATLAB。

下面是未经全局幅相校准的相对 L2 误差。两端统一剔除最大传播延迟加 0.8 us 的
chirp 起始区，以及第一个 chirp；范围写入每组 JSON。四倍采样保持扫频带宽和 chirp 时长。
整采样控制使用 2^24 Hz 采样、恰好 4 个采样间隔的静态延迟，以消除分数延迟插值。

| 场景 | 原始 IQ 相对误差 | RD 功率相对误差 |
|---|---:|---:|
| static | 0.0050169497 | 0.005864795 |
| radial | 0.0058274904 | 0.0067705408 |
| multipath | 0.01069665 | 0.0092813756 |
| static_os4 | 0.0014923431 | 0.0019087613 |
| radial_os4 | 0.0017307589 | 0.0022042839 |
| multipath_os4 | 0.0031013278 | 0.0029896841 |
| static_integer | 9.8861483e-08 | 2.0179611e-08 |

七组中的各路径均得到相同的距离／多普勒 FFT 峰值网格。三径峰值分别对应约
30/90/150 m，以及 +1/+0.5/-0.5 m/s。这验证的是指定路径输入；三条路径不是由本次
MATLAB 实验的实体几何自动发现，不能作为同一物体在某个具体场景中必有这三条路径的证明。

![Actual MATLAB and WiTwin multipath comparison](../../../output/doppler-repair/matlab-final/range-doppler-comparison.png)

图使用相同 Hann 窗和同一功率基准，右图为功率差的绝对值；红叉为指定路径的预期位置。

### 差异解释与验收范围

对已安装 R2025bU4 实现的只读检查发现，radarTransceiver 的路径通道采用线性分数延迟插值；
chirp 内保持包络延迟，并施加指定载频 Doppler。WiTwin 直接计算连续延迟的解析 beat 相位，
包括 chirp 内延迟变化。独立的线性采样延迟诊断 oracle 与真实 MATLAB 输出相差不超过
2.4e-11；整采样静态控制下 WiTwin 与 MATLAB 的原始 IQ 误差约 9.9e-8，
四倍采样使三组原始 IQ 误差均下降。这些证据支持主要残差来自采样延迟模型差异，
没有发现本组实验中的载频相位、多普勒符号或二倍路程系数错误。
诊断 oracle 只解释误差，不替换外部结果，也不回写生产代码以模仿 MATLAB 的插值误差。

本套明确的工程验收界限为普通指定路径原始 IQ 误差小于 2%、整采样控制小于 1e-6、
MATLAB 与声明的线性延迟诊断模型误差小于 1e-8、各路径峰值网格一致，且四倍采样误差下降。
这些阈值是在本轮诊断后为可重复回归设置的，不属于事先注册的盲测标准。
七组均满足界限；分析器在缺失外部输出、数值超限或峰值不一致时失败。
额外两个合成文件测试验证缺失结果拒绝与错误 Doppler 拒绝；这些是对照工具测试，
不计为真实 MATLAB 实验。原有生产 GPU 验收仍是前文记录的 1531 passed / 12 skipped，
本次未修改生产代码或重新声称执行整套 GPU 测试。

本次对照覆盖理想硬件与指定路径波形，不覆盖两端 mesh 求径、拓扑发现、micro-Doppler 动画、
材料散射、完整硬件相噪模型或商业实时性能。此前 rotor／limbs／heavy 自适应对照仍属于
WiTwin 自身逐 ADC 参考验收，不能改称 MATLAB 的 micro-Doppler 或 heavy 场景验收。
两端计时边界不同，已记录时间但不据此给出性能倍数，也不宣称与商业仿真器全面等价。

复跑：先在 witwin2 执行 tools/compare_matlab_radar.py --output <目录>，
再执行 tools/run_matlab_comparison.ps1 -OutputDirectory <同一目录>，
最后执行 Python 工具 --analyze --plot --output <同一目录>。
启动器默认继承用户登录配置；只有显式 -IsolatedPreferences 时才使用隔离配置。
官方依据：[radarTransceiver](https://www.mathworks.com/help/radar/ref/radartransceiver-system-object.html)、
[dechirp](https://www.mathworks.com/help/phased/ref/dechirp.html)、
[MPM 安装](https://www.mathworks.com/help/install/ug/mpminstall.html)。

后续实际材质、完整动态场景、地面多径和重复性能对照见
[扩展报告](radar-matlab-material-motion-performance-2026-09-16.md)。
其中完整动态入口仍明显慢于本次 MATLAB 点目标基线，不能由原生合成或自身 ADC 对照推断商业性能优势。
