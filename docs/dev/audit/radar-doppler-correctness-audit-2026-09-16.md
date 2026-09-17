# Radar Doppler / micro-Doppler 正确性审计

日期：2026-09-16。历史记录：本文描述修复前 e0c79ad 的代码审计、定向 GPU 测试与独立数值探针。后续修复和当前验收见 [修复验收记录](radar-doppler-correctness-acceptance-2026-09-16.md)。

## 结论

当前实现具有正确的固定拓扑双程合成基础，但尚不能把公开的 `Radar.simulate()` 当作完整的动态、多径、micro-Doppler 仿真入口。最优先的问题是运动数据没有接入生产入口、环境运动与参数 JVP 的语义混淆、多径方向图方向错误，以及冻结模型缺少可执行的误差边界。

没有发现生产链通过“相邻帧第 k 条路径的距离相减”来计算速度。它使用冻结拓扑上的几何 JVP，因此常见的路径排序变化导致速度爆炸，并不是这里已经确认的实现机制。真正的风险在于：遗漏新的路径、跨 epoch 身份缺少连续性契约、离散几何/可见性改变导致复数回波突变，以及错误地把这些突变解释成运动。

以下将“已经复现的缺陷”“明确存在但未受约束的近似/能力缺口”“尚未完成验证的风险”分开。注释承认限制，不等于这个限制已满足用户所需的动态雷达功能。

## 审计环境与证据

- Radar 源码 HEAD：`e0c79adb05fce528e461d1dbb837fa5634244e61`。
- Channel 源码 HEAD：`33cce0ea49260b02377c1604a00d07dc4e557464`。
- Core 源码 HEAD：`46c826d969654912461d2e18b6cb87df9cc3df8f`。
- 正式验证环境：`C:/Users/Asixa/miniconda3/envs/witwin2/python.exe`，Torch 2.10.0，CUDA 12.8，RTX 5080。
- Core / Channel / RayD 通过当前 monorepo 源码路径导入，没有安装或修改环境包。
- Channel 使用受 loader 验证的本地 developer binary，fingerprint 为 `183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`。
- Radar 使用源码目录中的 packaged native binary，fingerprint 为 `f289166a52825ccbf6c393b358372ef6837976305c542aea3beeae25ce92278b`，loader 校验了源码指纹。
- 重要边界：Channel binary 自报构建 Git 为 `f88c806caa38c371af5d0d043a9abca801d4c6ae` 且 dirty；本次没有重新构建当前 Channel HEAD。因此测试证明的是所记录的源码包装层 + 已验证本地二进制组合，不是当前 Channel 全部 native 源码的重新构建认证。

原始证据目录：[output/doppler-audit-2026-09-16](E:/Code/witwin-platform/radar/output/doppler-audit-2026-09-16)。其中保留 `build-info.log`、两批测试日志、静态检查日志、`probes.py`、`probe-results.json`、`probes.log` 和重运行脚本。

| 验证 | 结果 | 含义 |
|---|---|---|
| Doppler、micro-Doppler、kinematics、slot batching、FMCW、processing 等定向 GPU 测试 | 191 passed / 3 skipped | 所测固定路径和既有契约通过 |
| invalidation、rediscovery、公开入口、方向图、join、OFDM、pulsed、frontend、scene-leaf AD 等另一批测试 | 199 passed | 与上一批无重复测试文件 |
| 7 组独立数值探针 | 均运行并记录数值 | 复现下文问题；不是“正确性测试全部通过” |
| Ruff format / Ruff check / duplicate-code / architecture | 全部通过 | 结构和风格检查，不是物理正确性证明 |

三个 skip 均是缺少 SMPL 模型文件，不是缺少 Channel。额外的 skip 核查重跑复用了一个已经通过的测试，不计入 390。初次核查命令漏配 native override，导致该测试 loader refusal；该失败保留在 `skip-details.log`，补齐配置后通过，见 `skip-details-verified.log`。未运行整个仓库全量 suite、Linux、发布矩阵或 benchmark。

## 物理判断基准

对一条在当前时刻存在且可微的几何路径，令所有节点位置为 `x_i(t)`，单位 m；速度为 `v_i(t)`，单位 m/s；段方向为 `u_i=(x_(i+1)-x_i)/|x_(i+1)-x_i|`。非零段长下：

```text
L_p = sum_i |x_(i+1) - x_i|                       [m]
dL_p/dt = sum_i u_i · (v_(i+1) - v_i)              [m/s]
tau_p = L_p / c                                   [s]
f_D,p = -f_ref * d(tau_p)/dt                       [Hz]
```

这里使用仓库的 Channel 约定：时间因子 `exp(+j 2πft)`，传播因子 `exp(-j 2πf tau)`，接近为正 Doppler。双程应把去程和回程的 delay-rate 相加；一般多径不能把目标对雷达直线方向的径向速度统一乘二。双基地两腿 Doppler 相加亦见 [MathWorks 的双基地仿真说明](https://www.mathworks.com/help/radar/ug/simulating-a-bistatic-radar-with-two-targets.html)。

上式是同一时刻、非相对论几何光学模型。它不对路径出生/死亡瞬间定义普通导数。对静止光滑镜面的内部反射点，镜面点沿面的滑动项会按驻相条件抵消；不能把“镜面点在网格上移动”直接视为“材料在移动”。移动镜面/变形表面则必须提供其真实运动或相应几何时间切线。

更完整的单路径复场为 `H_p(t)=a_p(t) exp(j psi_p(t)) exp(-j 2πf_ref tau_p(t))`，其中实幅度 `a_p>=0`，`psi_p` 包括材料、极化和散射相位。其瞬时频率还包含 `d(psi_p)/dt / (2π)`；只推进 delay-rate 会遗漏这些相位随姿态的变化。总信号先相干求和 `z=sum_p H_p`，不能先平均各路径速度。`arg(z)` 在相消零点附近的尖峰不等于真实材料速度无限大。

## 已确认问题与修复建议

### A1 — P1：公开入口没有接入运动速度

位置：[simulation.py:835](E:/Code/witwin-platform/radar/witwin/radar/simulation.py:835)、[channel.py:731](E:/Code/witwin-platform/radar/witwin/radar/channel.py:731)、[fmcw.py:240](E:/Code/witwin-platform/radar/witwin/radar/synthesis/fmcw.py:240)。

`simulate_scene` 每帧只做 `slot_count=1` 的两腿 replay，默认 `ad_mode='none'`。因此 `_delay_rate` 返回 `None`，join 继续发布 `None`，synthesis 将其转换为零速度。帧与帧之间更新目标位置，并不等于帧内的 chirp 已包含运动。

复现：结构 anchor 以 1 m/s 沿视线远离、LOS-only，帧时间为 0 和 1 ms。`delay_rate=None`，同一帧所有 chirp 的最大差异为 **0**，帧间信号最大差异非零。中心共址单站参考 delay-rate 应约为 `6.67128e-9 s/s`，对应约 `-513.69 Hz`。

附带能力缺口：默认结构站点取 `rigid_motion.translation`，不是随肢体/叶片运动的多个材料点。纯转动而 anchor 不变时，这个站点没有转动 micro-Doppler；explicit positions 又没有随时间求值的站点轨迹接口。Radar TX/RX 来自 `radar.tx_pos/rx_pos`，也没有将 Core endpoint trajectories 自动映射为该阵列的逐时刻姿态。

建议：在 scene-session 内显式构建 TX、RX、scatter-site 的位置和速度，保留稳定材料点身份。运动量应独立于用户选择的 AD 模式；缺失运动数据不能自动解释成静止。为 rigid / deformation / 显式轨迹提供 concept-owned 的站点采样契约，纯静态场景仍可明确给零速度。第一条验收必须从公开 `Radar.simulate()` 进入。

### A2 — P1：移动反射面的速度没有自动进入 delay-rate

位置：[propagation.py:1306](E:/Code/witwin-platform/radar/witwin/radar/propagation.py:1306)、[channel.py:540](E:/Code/witwin-platform/radar/witwin/radar/channel.py:540)、[test_phase7_moving_structures.py:535](E:/Code/witwin-platform/radar/tests/test_phase7_moving_structures.py:535)。

现有 `two_way_duals` 只处理 TX / site / RX。用 `DynamicScene` 移动环境墙面时，compiled scene 的顶点没有自动携带时间切线；即使 endpoint dual 正确，仍会漏掉环境项。这对“静止目标 + 移动车辆/反射墙面”尤其严重。

复现：墙沿法向以 4 m/s 移动，端点全部静止。endpoint-only JVP 给出 **0 Hz**；同一冻结路径的两侧快照距离差分 oracle 给出最大 **4088.612 Hz**。手工对墙面顶点注入实际速度后，现有 native 路径给出 **4088.590 Hz**，所有行相对该差分 oracle 的最大绝对差为 **0.145 Hz**。

这说明当前底层已具备可利用的 vertex JVP，不能直接沿用旧测试注释中“需要新增 native 顶点切线通道”的判断。[test_phase9_scene_leaf_ad.py:210](E:/Code/witwin-platform/radar/tests/test_phase9_scene_leaf_ad.py:210) 的墙面 JVP 测试本次也通过。

建议：先复用现有 vertex JVP，将结构平移、旋转、变形速度接入 compiled geometry 的时间方向；与 endpoint 运动在同一个时间导数计算中求和。避免另写镜像法作为生产 velocity fallback。旋转和变形叠加时需在 Core 定义的坐标系中组合刚体项与变形项，避免漏旋转或双算。

### A3 — P1：多径方向图按目标直线方位计算

位置：[sensors.py:1018](E:/Code/witwin-platform/radar/witwin/radar/sensors.py:1018)、[sensor_weight.cu:498](E:/Code/witwin-platform/radar/witwin/radar/cuda/sensor_weight.cu:498)。

`RoundTripPatternStage.apply` 对每条 composed path 都传入同一个 target site 作为 `site_in/site_out`，native 随后用 `site-tx` 和 `site-rx` 查方向图。含环境反射的路径应使用 TX 到第一交互点的发射方向，以及 RX 朝最后交互点的接收方向。当前路径 delay 可以正确，但路径幅度权重错误；不同 Doppler 分量的相对强度、相消位置和 micro-Doppler 包络都会改变。

独立镜像几何 oracle：12 条含反射路径的最大幅度增益相对误差 **6.336%**；LOS 对照误差约 `6e-8`。方向图较窄或反射路径偏离目标方位更大时，没有证据保证误差仍只有这个量级。既有方向图测试虽然通过，但它的 oracle 也使用 `site-tx/site-rx`，不能暴露这个物理错误。

建议：Channel leg contract 明确发布 departure / arrival direction，或提供第一/最后交互点；Radar 消费路径自身的方向。不要在 sensors 中重新求解反射点。同时补齐 scatter-site 处的入射/出射方向：目前 `AspectScatterResponse` 在 outbound depth > 0 时明确拒绝，无法覆盖完整双腿多径 micro-Doppler。这种拒绝是诚实边界，应通过扩展方向契约解决。

### A4 — P1：参数 JVP 的方向改变了仿真的 primal

位置：[channel.py:731](E:/Code/witwin-platform/radar/witwin/radar/channel.py:731)、[paths.py:809](E:/Code/witwin-platform/radar/witwin/radar/paths.py:809)、[simulation.py:843](E:/Code/witwin-platform/radar/witwin/radar/simulation.py:843)。

`_delay_rate` 把任何 delay 的 forward tangent 解释成时间导数。composer 虽有 `include_delay_rate=False` 用于参数方向导数，公开 scene driver 却始终采用默认 true。这样 `ad_mode='jvp'` 同时承担“物理速度”和“参数扰动”两个不同语义。

复现：静态世界和相同的 primal 站点，仅将位置参数 tangent seed 从 0 改为 +x 单位方向；公开入口输出的 primal cube 相对峰值最大变化 **2.0302**。对于同一个函数的普通参数 JVP，primal 本应与种子无关。若用户有意把 seed 作为物理速度，这一变化有物理意义，但该调用就不能再被当作同一个静态模拟函数的参数 JVP。

建议：明确分离物理时间切线与优化参数 AD。时间导数由 kinematics 构造并作为显式、带来源的 delay-rate 发布；参数 JVP 不得隐式覆盖它。不能仅把公开入口默认改为 `ad_mode='jvp'` 来修 A1，这会固化 A4。加入“改变参数 tangent seed 不改变 primal”和“同一运动在 none/vjp/jvp 下 primal 一致”的回归检查。

### A5 — P1：冻结模型不足以支持一般 micro-Doppler，且无误差门限

位置：[assembly.py:890](E:/Code/witwin-platform/radar/witwin/radar/synthesis/assembly.py:890)、[simulation.py:654](E:/Code/witwin-platform/radar/witwin/radar/simulation.py:654)、[test_phase7_slot_batching.py:355](E:/Code/witwin-platform/radar/tests/test_phase7_slot_batching.py:355)。

即便修复 A1，现有 frozen 模式仍只使用 `tau(t)=tau0+tau_dot0*t`，复数 weight、散射响应和可见性在帧内冻结。横向运动、转子、肢体加速、移动反射面、路径进出遮挡都可能违反这个模型。`max_unambiguous_speed` 只限制采样混叠，不限制相位线性化误差。

本次通过的既有测试已经验证：77 GHz、约 2.1 m 距离、12 m/s 横向运动、约 0.975 ms CPI，即使径向速度未超混叠界限，帧末路径相位误差仍超过 0.05 rad；模块文档记录的量级约 0.1 rad。micro-Doppler 测试主要使用 `tests/support` 中逐 slot replay 后的 Channel 复数传输系数，而不是公开入口产生的完整 range-gated FMCW spectrogram；短时间的双转子侧带也不足以证明完整 blade-flash 周期、幅度调制和遮挡行为。

建议：把逐 slot 的世界采样、propagation、scattering、pattern、visibility 更新形成生产能力；按实际 TDM 时间取值，生成刷新权重的批量合成输入，并保证 carrier 不重复推进。端点运动 + 静态环境可优先复用现有 `reevaluate_slots`；移动结构还需要明确每个 slot 所属几何状态，不能把一个 compiled snapshot 重复用于所有时刻。

冻结模式可保留为有界加速模式。令 `a_L=d²L/dt²`，单位 m/s²，则载频相位误差的一阶上界为 `pi*f_ref*|a_L|*T²/c`。为其设置显式预算，并同时约束姿态/幅度变化和可见性事件。单站匀速横穿最近点附近，`a_L≈2*v_transverse²/R`。预算只控制连续路径，不能包办路径出生死亡。超限时缩短 block 或切换逐 slot 模式，不能只写注释。

### A6 — P2：新路径出生不会触发 replay 的重发现提示

位置：[propagation.py:804](E:/Code/witwin-platform/radar/witwin/radar/propagation.py:804)、[propagation.py:825](E:/Code/witwin-platform/radar/witwin/radar/propagation.py:825)、[channel.py:381](E:/Code/witwin-platform/radar/witwin/radar/channel.py:381)。

这是当前固定 winner replay 的明确设计限制，而不是 row validity 的实现错误。replay 可以判定旧路径死亡，却发现不了冻结时不存在的新路径。`motion_event_period_frames=None` 没有出生恢复期限；endpoint-only 运动也不会改变世界 geometry version。

复现：静态平面墙、目标从 `(2,2.4,0)` 移到 `(2,0.6,0)`。旧拓扑只返回 **1 条**路径，且全部 valid；重新发现得到 **4 条**双程组合；`rediscovery_required` 返回 **None**。这会遗漏新增 Doppler 分量并改变干涉，而不是仅改变数组长度。

默认移动结构 + `world_motion='frozen_world'` 会逐帧重发现，因此不能笼统说“所有默认模拟都漏路径”。该风险主要落在 endpoint-only motion、固定 winner replay、无出生 cadence，以及帧内部仍被冻结的时段。

建议：动态会话应显式声明出生检测策略和最大允许延迟，优先使用实际时间/位移/几何事件界限，而不是只有 frame count。`row_valid` 与“路径集合完整性”应为不同诊断。允许 conservative candidate superset 或定期重发现，但必须测量遗漏时长和回波误差。仅增加死亡路径检查不能修复出生缺口。

### A7 — P2：micro-Doppler 工具没有继承 waveform 的符号元数据

位置：[range_doppler.py:410](E:/Code/witwin-platform/radar/witwin/radar/processing/range_doppler.py:410)、[range_doppler.py:520](E:/Code/witwin-platform/radar/witwin/radar/processing/range_doppler.py:520)。

`range_doppler_map` 读取 `ProcessingAxes.doppler_sign`，对 FMCW beat 的共轭约定做频率反向。`microdoppler_spectrogram` 只接收 raw tensor 和 slot period，直接 STFT；它既不知道 phasor，也不能判断调用者传入的是 Channel transfer 还是 FMCW range-bin 序列。其模块文字却声明远离目标为负频率。

复现：FMCW 远离目标 2 m/s，规范 Channel Doppler 为 **-1027.38 Hz**，原始 beat 慢时间序列直接传给公开 spectrogram 后峰值为 **+1041.67 Hz**，差额量级内含 FFT bin 量化及 FMCW ramp 项；关键错误是符号相反。这个函数作为原始数学 STFT 并没有算错，而是不能自动满足它旁边声明的跨波形物理符号约定。

建议：提供接受 typed range-gated slow-time product 的入口，携带 phasor、时间轴、TX slot offset、range gate 和 gap 信息；复用已有 Doppler sign owner 做规范化。原始 STFT 若保留，应清楚声明输入约定。跨帧有 idle gap 时不能简单展平后当成均匀 PRF 数据；非均匀采样应明确拒绝或进入单独的处理策略。

### A8 — P2：FMCW 没有 chirp 内连续运动的 Doppler 项

位置：[fmcw_beat.cu:195](E:/Code/witwin-platform/radar/witwin/radar/cuda/fmcw_beat.cu:195)、[fmcw_spectrum.cu:1](E:/Code/witwin-platform/radar/witwin/radar/cuda/fmcw_spectrum.cu:1)。

两个 kernel 使用 `tau=tau0+tau_dot*t_slot`，同一个 chirp 的所有 fast-time 样本共享 tau；carrier drift 也只包含 t_slot。这是 chirp 内 stop-and-hop 近似，因此首 chirp 中运动目标与静止目标在相同 tau0 下完全相同，缺少连续运动带来的 fast-time Doppler 和部分 range-Doppler coupling。

独立 time-varying-delay dechirp oracle：77 GHz、3.7 m、2 m/s、slope=60 MHz/us、256 点、5 MSPS、ADC start=6 us，native 首 chirp 相位斜率为 **1,481,024.583 Hz**，连续延迟 oracle 为 **1,482,102.375 Hz**，相差约 **1,077.79 Hz**。该数值包括载频 Doppler 和 sweep 下的连续 range walk，不能把它全称为常数 f_D。固定模型中的 moving/still 首 chirp 差异严格为 0。

这不是双程 factor-of-two 错误，而是未设置适用界限的信号近似。FMCW 发射、接收、dechirp 的基本相位关系可参见 [TI 系统模型](https://www.ti.com/content/dam/videos/external-videos/en-us/2/3816841626001/5675916489001.mp4/subassets/Mmwave_webinar_Dec2017.pdf)；这里的连续运动差值由探针直接推导，不由该资料的静止近似提供。

建议：先规定 fast-time 运动相位预算 `2*pi*|f_D|*T_ADC` 和距离偏差预算。对于精确模式，在统一的同一 chirp 收发模型内使用 `tau(t_slot+t_start+t_m)`，同时确认当前 Channel reference phase 的基准时刻，防止重算。完整延迟代入会使样本相位含二次项，原有单纯 Dirichlet 闭式不再普遍精确；应明确选择有误差界限的线性化修正或支持二次相位的 native synthesis，不能只在 spectrum 路径给 bin 加一个常数就宣称完全修复。

## 路径离散跳变：哪些已处理，哪些还没有

| 情况 | 当前证据 | 应采用的策略 |
|---|---|---|
| 路径行重新排序 | join 使用 endpoint / component / depth / primitive / material 身份；相关测试通过 | 继续以身份组合，不按 row index 做跨帧差分 |
| 镜面点跨同材质共面三角形 | 实测 primitive 1→0，冻结 replay 保留旧标签；fresh 与 replay 的 delay 和 complex transfer 完全相同 | 利用已有共面支持；不要把三角形标签变化当成物理运动 |
| 新路径出生 | 实测 replay 1 条 vs fresh 4 条，poll 无提示 | 显式出生检测/重发现策略和完整性诊断 |
| 遮挡导致旧路径死亡 | 相关 validity/AD 测试通过，dead row 归零 | 保留事件，不对 dead row 的零 delay 与上一帧做差分 |
| 曲面离散为非共面 facets | 本次未完成网格细化和边界全扫描验证 | 几何/法向和驻点求解的收敛研究；必要时使用物理表面表示或经验证的过渡模型 |
| RGB-D 每帧重建/重采样点集 | 点索引不天然等于材料点身份 | 建立可靠 correspondence，或用显式 scene-flow/运动模型；无对应关系时标为未知 |
| 多条复场接近相消 | 数学上总场相位率可在低幅度处剧烈变化 | 区分 per-path 几何率与总场估计；在低 SNR / 低幅度处降低估计置信度 |

共面探针只覆盖一个平面共享边场景，不能推论所有网格边界都稳定。跨 epoch 的 primitive/material 序列也不能直接当作持久 path ID：一个物理表面可换三角形标签，真实路径也可能在出生/死亡处无连续延续。

建议的生产原则是：在每个可微路径分支上从运动学求导，在路径事件上记录离散变化。对于时序诊断，可用端点 ID + 交互类型 + 稳定物理 surface/patch ID + epoch/generation 记录 lineage；不得把仅空间近邻的两个不同反射路径强行拼成一条。信号本身始终按该时刻存在的路径相干求和，新路径可直接使用几何决定的绝对相位，并不需要强行延续旧路径相位。

不建议以 clamp velocity、低通路径距离、对总 IQ 相位无条件 unwrap 或随机相位拼接作为修复。它们会隐藏拓扑错误，并可能抹除真实 micro-Doppler。若需要平滑几何光学中的硬可见性，应声明额外物理近似，并以衍射/有限孔径或更高精度参考验证其过渡宽度，不能把任意 cross-fade 当作真实散射。

## 建议实施顺序和验收门槛

1. **先修动力学入口与语义。** 统一 TX/RX/site/structure 的时刻、速度来源与稳定 ID；分离 temporal derivative 和 parameter JVP。验收静态、匀速径向、移动阵列、移动墙、刚体转动、刚体+变形组合；参数 seed 不得改变 primal。A1/A2/A4 同一阶段解决。
2. **修路径方向契约。** Channel 发布每腿两端方向；Radar pattern 和 aspect response 消费它。加入“一个 site 的 LOS 与反射对应不同天线增益”的独立镜像 oracle；包含 outbound reflection。
3. **把逐 slot 动态流水线投入生产。** 以真实 TDM 时间更新路径、复数散射、方向图和可见性；同时保留受误差预算限制的 frozen 模式。用完整转子周期、非对称肢体动作和加速目标验证 range-gated spectrogram，不只验证瞬时速度或短窗口双侧峰。
4. **建立路径事件验证。** 连续扫描共享边、曲面 facets、遮挡边、birth/death 和多次反射；比较 freeze/replay 与每时刻 fresh discovery。分别测路径集合、per-path delay-rate、复场、谱底和谱脊。记录最大出生延迟；换网格分辨率应检查波形收敛。
5. **统一 DSP 与连续运动模型。** typed micro-Doppler 输入继承波形符号和真实时间轴；补 fast-time 动态模型及预算。跨波形验证接近/远离符号，beat/spectrum 验证同一物理模型下的一致性。

建议初始验收使用独立的 float64 几何/时变延迟 oracle：静态无噪声的 delay-rate 必须为零；光滑分支上的 Doppler 使用绝对 Hz 误差与相对误差双门限（近零不使用纯相对误差）；相位误差预算可以先用 0.05 rad 做工程起点，最终按应用需要确定。FFT 峰值只适用于 bin 级检查，不能替代相位率检查。深衰落区使用绝对误差/峰值归一化，避免逐 cell 相对误差失真。

还应增加高坐标值、小位移、长路径、grazing angle、非有限输入和非均匀时间采样等测试。当前 positions/delay 为 float32，后端 double 累积不能恢复前端量化掉的微位移；需要时由 Core/Channel 管理局部坐标或受控的高精度几何。当前 `velocity_mps=lambda*f_D/2` 在多径/双基地中是等效单站速度坐标，不能直接标作目标三维速度真值。

## 本次未证明的范围

本次没有完成任意复杂网格、多 bounce 的全面时序压力测试，也没有真实人体 SMPL 模型、实测雷达、全波求解器对照或长序列统计。已有 tests 中的多径 oracle 主要基于光滑平面和有限几何；通过它们不能证明非共面曲面网格、粗糙/漫反射、衍射、动态材料相位或任意网格重建的正确性。

当前证据足以确定上述优先修复项，并明确指出已有正确基础：固定分支 JVP、双程求和、carrier 单次归属、TDM slot 时序、规范 Range-Doppler 符号转换、共面边界 replay，以及失效行归零。应保留这些基础，在生产时序与运动数据链上补齐缺口。
