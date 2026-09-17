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

## MATLAB Radar Toolbox：执行受阻，未通过对比

本机安装树为 D:/Softwares/MATLAB，VersionInfo 报告 R2025b Update 4。
普通批处理及无 JVM 启动均未产生有效输出；最后一次实际对比启动在 120 秒后超时，
本次启动的进程已清理。output/doppler-repair/matlab/launch-status.json 明确记录
completed=false、comparison_executed=false。

安装树中找到了 Phased Array System Toolbox 文件，未找到 radarTransceiver。
最终工具箱和许可证状态尚未由可运行的 MATLAB 自身确认。Phased Array System Toolbox
不能直接冒充 Radar Toolbox 的实机对比。

已交付可复跑材料：

- tools/compare_matlab_radar.py：导出静态、径向和三条指定路径的匹配 FMCW 输入及 WiTwin 复数结果；导出已运行并核对每组 [512,256] 形状。
- tools/compare_matlab_radar.m：实际调用 radarTransceiver，记录 MATLAB／工具箱身份并保存结果。由于启动失败，此脚本尚未完成 MATLAB 语法／运行验证。
- tools/run_matlab_comparison.ps1：限定等待时间、记录退出／超时状态，只结束本次启动的进程。
- Python --analyze 只读取真实 MATLAB 输出；同时报告原始 IQ 误差及单一全局复数校准后的误差，不会默默消除偏差。传播滤波启动区的排除范围写入结果。

这套对照首先覆盖指定路径的波形合成，不能代替 mesh 场景求解、多径发现、完整器件相噪、
散射材料或商业实时性能验收。两端计时边界不同，脚本不据此发布性能倍数。
官方依据：[radarTransceiver](https://www.mathworks.com/help/radar/ref/radartransceiver-system-object.html)、
[FMCW 仿真与处理示例](https://www.mathworks.com/help/radar/ug/simulate-an-automotive-4d-imaging-mimo-radar.html)。

恢复条件是该 MATLAB 能正常完成批处理，并提供 Radar Toolbox／Phased Array System Toolbox。
恢复后运行 PowerShell 工具，再运行 Python --analyze；得到实际输出前，不宣称与 MATLAB 等价。
