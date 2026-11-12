# Stage-II Phase 4 入口准备（2026-07-25）

本文档记录 Phase 4 启动前的现状盘点、源码级 smoke 证据和入口清单状态。
它不修改 Radar production source，符合 Stage-I exit gate 的约束。

## 1. Radar 现状盘点

### Dr.Jit / Torch-physics 归属（Phase 5 工作项 12 的删除对象）

| 文件 | 行数 | drjit 引用 | 处置 |
|---|---:|---:|---|
| `witwin/radar/trace.py` | 441 | 52 | production 调用 → 删除，改 consumer adapter |
| `witwin/radar/_rayd_bridge.py` | 304 | 21 | RayD Dr.Jit 桥 → 删除 |
| `witwin/radar/material.py` | 29 | 4 | → `witwin.core.PhysicalMaterial` 映射 |
| `witwin/radar/solvers/common.py` | 302 | 0 | Torch path geometry/amplitude → native owner |

drjit 导入仅存在于上述前三个文件；`sigproc/`（cfar/music/pointcloud）与
`solvers/solver_dirichlet.py` 属 Torch/DSP 例外域。`cuda/` 已有 prebuilt
扩展骨架（`witwin_radar` native），可作 `_radar_native` 的构建基座。

### 测试基线

`168 passed, 89 skipped`（skip 为 GPU/prebuilt 门控），前置修复：4 个测试文件
的 `Material` 导入按 ADR-036 迁移为 `PhysicalMaterial as Material`
（radar `62a96f8` + tests 提交）。

## 2. 源码级消费 smoke（非 artifact-pinned gate）

`witwin.radar` 与 `witwin.channel.propagation.consumer` 同进程共存，
consumer contract v2 从消费侧验证：

- `capabilities()`：`fixed_topology_components == {los, reflection}`，
  `row_valid` 覆盖两者，polarimetric 三种 AD 全开；
- `evaluate → prepare_fixed_topology → reevaluate(ad_mode='jvp')` 全链路，
  **forward-only dual**（无 requires_grad，ADR-038）直接产出逐行
  `delay_s` 切线；
- **Doppler 解析对照**：77 GHz、sink 以 12 m/s 横向运动，
  LoS 行 f_D = −1378.4 Hz（解析 −1377.5），反射行 −219.6 Hz
  （镜像源解析 −219.5），均 <1%；
- 消费过程未导入任何 Channel solver 模块（solver 中立性从消费侧成立）。

结论：**Section 8.3 的 Doppler `delay_rate` 路线已被完整证实可用**，
Radar adapter 无需 FD、无需 requires_grad 约定。

## 3. 入口发现：`witwin.core.Mesh` 的 `recenter=True` 默认值

Smoke 首跑时反射行发布 LoS 长度 + 零 hit + 零场。经五轮变量隔离
（法线朝向、faces dtype、材质形式、显式 ID、三角剖分模式均排除）后
定位：`Mesh` 默认 `recenter=True`，**静默把作者给定的世界坐标重心平移**
——x=4 的墙被移到 x=0 穿过收发端，下游一切输出对"实际几何"均正确。

这是 Stage-II 的高危陷阱：Radar 场景构造若直接用世界坐标建 `Mesh`
而不写 `recenter=False`，会得到静默错误的物理而非任何报错。测试世界
的 `make_mesh_structure` 已显式传 `recenter=False`，印证内部作者早已
绕行。**建议在 Phase 4 的 Radar-side ADR 批次中提案 Core 变更**：默认
改 `False` 或强制显式选择（属 breaking Core 决策，需要 owner 拍板；
Radar 资产加载器可能依赖现默认值，改动前需盘点）。

## 4. Phase 4 入口清单状态

| 前置 | 状态 |
|---|---|
| Stage-I 源码合并至 main 并推送 | ✅ core `7791ce2` / channel `fb23078` |
| Python matrix 决策 | ✅ Core `>=3.10,<3.15`；radar pin 已提 `>=0.4,<0.5`（`4825ae9`） |
| consumer 契约满足 Doppler/Jones 需求 | ✅ v2 + ADR-037 修正案 + ADR-038，本文档 §2 证据 |
| windows-smoke workflow | 🔄 运行中（首跑 Core wheel 构建缺陷已修 `0b98900`） |
| manylinux_2_28 full artifacts | ⬜ 待 smoke 绿后 dispatch `scope=full` |
| Radar required-consumer CI pin release artifacts | ⬜ 依赖上一项 |
| Radar 全量 baseline / factor traces / Nsight | ⬜ Phase 4 entry 第一项，未开始 |
| Radar-side ADR 批次（§Phase 4 工作项 3） | ⬜ 草案未开始；应含 recenter 提案 |

## 5. 建议的 Phase 4 前几步

1. artifacts 发布后：把 §2 的 smoke 固化为 artifact-pinned 版本，作为
   required-consumer CI 的第一个测试；
2. 纵向 AD spike（Phase 4 工作项 5）以 §2 链路为骨架扩展：加一个可微
   scatter site + 标量 target response + `_radar_native` FMCW synthesis
   原语，端点/site 扰动 → round-trip phase 梯度；
3. Radar-side ADR 草案批次并行起草，`recenter` 提案一并送审。
