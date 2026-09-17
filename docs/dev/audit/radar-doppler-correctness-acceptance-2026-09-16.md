# Radar Doppler 修复与验收记录

日期：2026-09-16。环境：witwin2 / Python 3.11.14 / Torch 2.10.0 / CUDA 12.8 / Windows / RTX 5080。

本记录对应 `radar-doppler-correctness-audit-2026-09-16.md` 的 A1–A8。
原审计描述的是修复前 `e0c79ad`；当前行为以本记录、代码和可执行测试为准。
此次没有修改用户的 RGBD notebook 或 recording.mkv。

## 修复结果

| 问题 | 当前处理 | 主要验证 |
| --- | --- | --- |
| A1：公开入口缺少运动 | 轨迹在实际波形时间上取样；动态 FMCW 默认逐 ADC，OFDM/pulsed 逐 symbol/pulse | 运动目标、TDM 顺序、移动雷达 endpoint 映射 |
| A2：遗漏运动反射面 | 每个观测时刻更新 Core 世界和 Channel 几何，复场相位直接包含反射面运动 | 4 m/s 运动墙面，独立镜像路程和相位 oracle |
| A3：多径方向错误 | 方向图使用真实首末线段；出射散射方向取目标至首反射点 | 镜像方向、原生方向归一化 JVP/VJP、有限差分 |
| A4：参数 JVP 改变 primal | 参数切向量与物理 delay-rate 分离；公开场景入口不把 JVP 当速度 | 0、1、−2 倍 seed 的 primal 一致；正反模式伴随一致 |
| A5：冻结线性模型无法描述一般微动 | 动态默认逐 ADC 更新位置、幅度、相位；低层线性模型及显式 chirp 近似保留明确边界 | 径向、转子、双肢体散射点代理的独立 IQ/STFT |
| A6：路径出生被遗漏 | 动态默认每观测重发现；延长 cadence 时输出 path_set_complete=false | 新反射路径出生、完整与稀疏发现结果差异 |
| A7：微多普勒符号与时间不明确 | SlowTimeSignal 携带 phasor 和真实时间；拒绝非均匀/非法时间；统一远离为负 | 两种 phasor、时间间隙和 STFT 绝对中心时间 |
| A8：chirp 内运动遗漏 | 原生 FMCW 使用 tau=tau0+rate*(slot+u)；移动谱为完整归一化 DFT | beat/spectrum、静止/正负速度、primal/JVP/VJP 独立 oracle |

`frame_synthesis()` 直接提供已仿真帧和输出域元数据，后处理不再次合成、不从 shape 猜输出域。
静态路径仍使用 Dirichlet；动态 per-ADC 场景输出由实际 beat 样本经处理层的归一化 range FFT 得到。

### 路径离散化与速度

没有引入“相邻帧第 k 行路径长度差分”的速度估计。几何路径排序、数量变化不构成物体速度。
共面三角形 primitive winner 切换实验中，ID 发生变化，但延迟和复场逐位一致；几何物理
delay-rate 在 1e-15 s/s 容差内一致。微多普勒来自连续材料点轨迹的复场时间序列。

可见性出生/消失和真实散射幅度变化本来就可能产生宽带能量；不能为了让谱平滑而抹除它们。
非共面低分辨率网格代表的就是一个有棱角的几何体，本次不把真实离散几何替换成未声明的光滑表面。
需要保持散射点数量、次序和材料点身份；仅设置 angular_velocity 标签而不更新位置不等于真实旋转。

## 分阶段提交

1. `3cd4578`：参数 JVP 与物理运动分离。
2. `15b876d`：波形时间上的动态场景、轨迹、endpoint 和路径重发现。
3. `db74db7`：多径方向图和散射出射方向，含 native AD companions。
4. `94c3a83`：chirp 内连续运动、逐 ADC 默认、typed micro-Doppler。
5. `5f82fe9`：完整回归、独立物理实验、性能基线核对与验收证据。
6. 后续独立提交：按用户追加要求验证多径密集场景的 Range–Doppler 热度变化。

## 已执行验收

- CPU quick tier：697 passed，813 skipped，124.53 s。跳过的是该 tier 不执行的 GPU 等项，不计作通过。
- 全量 `pytest tests/ --gpu`：1499 passed，12 skipped，291.55 s，零失败。
  11 项因为没有 SMPL 模型文件，1 项缺少 nightly coexistence 证据；零 missing-Channel skips。
- Ruff：209 个 Python 文件格式检查通过，lint 通过。
- architecture、duplicate code、documentation surface、governance inventory、no compatibility、
  public API manifest、release claims、required Channel coverage、workflow references、native bindings：全部通过。
- native 注册：ABI 4，31 operators，11 AD groups；源文件和共享相位头都纳入 loader source fingerprint。

全套之后只清理了旧公式注释及文档，并重建 native；重建后的最终全量回归再次得到 **1499 passed、12 skipped、零失败，304.08 s**，记录在 `gpu-rebuilt.log/xml`。
日志与数值原始产物保存在 `output/doppler-repair/`，包括 `quick-final.log`、`gpu-final.log/xml`、
`static-final.log/json`、`native-final-build.log` 和 `environment-final.json`。
核心数值、最终二进制身份、全量测试统计与逐项跳过原因同时提交于
`radar-doppler-correctness-evidence-2026-09-16.json`，不依赖本地未跟踪日志来保存结论。

### 独立数值实验

由 `tools/validate_doppler_motion.py` 执行，生产入口逐 ADC 仿真；oracle 使用独立自由空间往返
路程与 FMCW 相位。每个案例只在首观测校准一个常数复增益，没有逐采样拟合相位或速度。

| 场景 | IQ 相对 RMS 误差 | STFT 相对误差 | 观测/发现次数 | 实测总耗时 |
| --- | ---: | ---: | ---: | ---: |
| 径向 0.2 m/s | 1.144e-6 | 1.221e-6 | 512/512 | 21.78 s |
| 8 Hz、3 mm 半径转子 | 1.845e-4 | 1.805e-4 | 512/512 | 10.19 s |
| 双频双肢体散射点代理 | 4.085e-4 | 4.098e-4 | 512/512 | 9.78 s |

首例包含冷启动，各耗时不能直接用于比较场景复杂度。三例均 `motion_sampling=adc`、
`path_set_complete=true`。径向 Doppler 理论 −102.738 Hz，实测峰 −109.375 Hz，误差小于
15.625 Hz 的 STFT bin。双肢体是解析材料点代理，不能作为真实 SMPL 人体模型验收。

77 GHz、3.7 m、2 m/s 的 chirp 内运动实验：实测 fast-time 频率 1482102.376482 Hz，
独立 oracle 1482102.375491 Hz；最大 IQ 误差 2.647e-7。
图：`output/doppler-repair/experiments/microdoppler.png`；数值：同目录 `results.json`。

### 多径密集场景：一个目标，多个距离热区

追加实验由 `tools/validate_heavy_multipath.py --chirps 32 --samples 64` 执行。
三面静态墙位于 x=6 m、y=±2 m，单程最多两次反射；一个目标从 (2, 0.35, 0) m
以 (0.8, 0.3, 0) m/s 移动。比较 t=0 与 0.5 s 两帧，77 GHz、5 MSPS、
60 THz/s、500 us chirp 周期。每帧 2048 次 ADC 观测和发现，均为完整策略。
这里墙壁提供传播反射，散射目标只有一个；没有加入墙面自身的独立静态杂波点。

两帧均得到 8 条 inbound × 8 条 outbound = **64 条双程组合**，合并互易路径后
为 36 个 image-pair 组，部分组继续在当前分辨率内重合。独立镜像几何的单程长度
最大误差分别为 9.552e-7 m、9.569e-7 m。两帧最强四个分离峰均匹配镜像预测，
误差小于一个距离 bin（0.19518 m）和一个速度 bin（0.12167 m/s）。

| 热区 | t=0 实测距离 bin | t=0.5 实测距离 bin | 峰值功率变化（统一参考） |
| --- | ---: | ---: | ---: |
| 主径 | 1.95 m | 2.54 m | 0 → −3.06 dB |
| 一侧墙混合路径 | 3.32 m | 3.71 m | −3.79 → −12.63 dB |
| 另一侧墙混合路径 | 3.12 m | 3.32 m | −8.69 → −4.59 dB |
| 约 6 m 的重合多径区 | 6.05 m | 6.05 m | −9.03 → −12.02 dB |

主径连续几何距离是 2.037 → 2.458 m，表中的 bin 跳转不是目标瞬移。
约 6 m 区域包含后墙混合路径及未分辨贡献，不能把整个 bin 唯一归属给一条路径。
这里速度轴采用 closing-positive；后墙混合路径的预测等效速度约 −0.025/−0.031 m/s，
尽管物体实际仍以 (0.8, 0.3, 0) m/s 运动。

原因是每条双程路径 q 有自己的总长 Lq，图上等效距离 Rq=Lq/2，物理 Doppler
fq=−(fc/c) dLq/dt。目标移动会同时改变多个 Lq、方向投影、反射幅度与相干叠加。
因此同一个物体会在多个距离处产生移动、增强或减弱的热区，而且不同路径可以有不同甚至
相反的 Doppler 符号。不可将每个多径峰直接解释为另一个实体，也不可把强度变化直接当作速度。
这些是可分辨多径的物理 ghost；当延迟/Doppler 差小于分辨率时，主要体现为同一单元内的相干
增强/衰落，而非独立峰。独立文献核对：[MathWorks 多径 ghost 的实际路径说明](https://www.mathworks.com/help/driving/ug/radar-ghost-multipath.html)
及 [RF propagation：ghost 与 fading 的分辨条件](https://www.mathworks.com/help/radar/ug/rf-propagation-fundamental-concepts.html)。

图使用统一参考功率，range Hann 窗、Doppler 矩形窗；弱竖条包含有限 CPI 的窗旁瓣，
不是每条弱条纹都代表一条路径。白叉是独立镜像预测。两帧耗时 117.64/101.20 s。
这次验证没有执行随机大场景统计或更高阶反射的全面覆盖。

图与原始 tensor：`output/doppler-repair/heavy-multipath/range-doppler.png`、`rd-0.0.pt`、`rd-0.5.pt`。
路径数据：`results.json`；峰值独立验证：`peak-validation.json`。这些数值也进入本记录的
已提交 evidence JSON；通过 `--analyze-only` 可以重新检查并绘图而无需重新仿真。

## 性能与适用边界

原提交 `e0c79ad` 独立导出并重建，在同一个 witwin2 环境测得 DSP pipeline 3.6963 ms、
静态场景边际每帧 8.66685 ms。旧 2.899/5.044 ms 限制在未修复版本同样失败。
现在按实测基线的 3.70/8.67 ms 保持原 1.30 倍余量，内存及操作数量限制不变。
完整 GPU 首次验收实测 3.8831/9.8595 ms，均通过 4.81/11.271 ms 上限。
这不是普遍性能提升的宣称。前后日志保存在 baseline-performance.log 和 stage4-migration.log。

逐 ADC 发现和更新是正确性优先模式，开销随 chirps × TX × ADC samples 增长；动态 mesh 还需要
更新编译几何。低层移动路径的有限和 spectrum 是 O(N²)，静态 Dirichlet 是 O(N)。
`motion_sampling="chirp"` 是调用者明确选择的 fast-time 冻结近似；较长发现周期可能漏掉路径出生，
不能把该模式的结果声明为完整路径集合。`path_set_complete` 仅表示配置深度/组件与发现策略内的完整性，
不是无限反射、所有物理散射机制的完备证明。

验证边界：

- 保留 Channel 的准静态几何光学模型；未实现完整时延电磁场、相对论或无限阶多次散射。
- FMCW 默认动态处理到 ADC 时间；OFDM/pulsed 仍在 symbol/pulse 内冻结，不声称任意块内快速运动均精确。
- 运动传感器中心可绑定 Core endpoint；天线姿态仍按显式传感器配置，未自动支持任意旋转波束轨迹。
- 固定拓扑 AD 的导数不跨越离散可见性事件；一阶 AD 通过不代表二阶 AD 已支持。
- SMPL 资产测试与 nightly coexistence 没有执行；Linux、发行 wheel、远程 CI 没有执行。
- Channel 使用既有受校验 developer binary，未重建当前 Channel HEAD；结论针对记录的二进制组合。

Native 运行身份：

- 最终 Radar fingerprint：`4035dcdd9978c926c7036b7b1972ab1ab677efd130268f2b2a075daee5fc0964`。
- 最终 Radar binary SHA256：`e1d673d92fa1df2775ac382e7ffb32c00859eb5a3a21ece66249971c33254fb0`。
- 首次全套/独立微动实验 Radar fingerprint：`b170ee1e35e359e7af421cda79385efd57315229caee002db61468b83acb90f4`。
- Channel fingerprint：`183118e96d75856e71df2f90c621fa191d6151d2b63ec12fd5f71062e49ed73f`。
- Channel binary 构建 Git：`f88c806caa38c371af5d0d043a9abca801d4c6ae`，dirty；当前包装层 HEAD 为 `33cce0e`。
