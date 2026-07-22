# inference 模块说明

`inference` 目录负责训练完成后的多层 PDGCN + 厚度方向 1D FDM 推理。该模块只承担推理、输出和推理配置校验职责；单层训练逻辑、TBPTT、checkpoint 保存等仍属于 `training`。

## 主要职责

- 从推理 JSON 配置读取 `inference` 段，并通过 `training_config` 复用训练阶段的尺度参数、模型超参数和 checkpoint。
- 读取单个 HDF5 输入文件，按时间帧构造单层曲面 PyG 图。
- 每步根据 checkpoint 中的 `use_explicit_heat_source` 决定是否先对顶层施加显式热源；取消时顶层 PDGCN 直接依据 `q*` 和物理 `ΔT_Q*` 预测完整增量。
- 在面内输运之后叠加厚度方向 Backward Euler 隐式 1D FDM 层间导热，并按配置对顶层追加当前步热源增量补偿，最后重新钳制面内与底层恒温边界。
- 输出多层温度场 HDF5；VTK 文件只由离线渲染入口生成。

## 主要文件

- `config.py`：定义 `InferenceRunConfig`，校验层数、层间距、步数、warmup、VTK 输出等配置。
- `fdm.py`：导出厚度方向 FDM 系数、显式层间温度增量诊断函数和 Backward Euler 隐式 FDM 步进函数，系数为 `k_ratio * dt_star * inverse_pe / layer_spacing_star^2`。
- `multilayer.py`：实现 `rollout_multilayer_fdm(...)`，输入单层图或图工厂，输出形状为 `[time, layer, node, 1]` 的温度序列。
- `io.py`：负责从配置运行推理、加载 checkpoint、构造图、写 HDF5、离线写 VTK 和 metadata。
- `infer_entry.py`：推理命令行入口，只输出 HDF5，默认读取 `configs/pdgcn_infer.example.json`。
- `render_entry.py`：离线渲染入口，从已生成的多层 HDF5 输出合并三维云图 VTK。
- `tests/`：覆盖配置校验、FDM 公式、多层 rollout 和推理输出。

## 推理约定

- `layer=0` 为顶层，`layer=num_layers-1` 为底层模具恒温边界。
- 热源信息固定只作用于顶层：显式模式在顶层预加温升，取消显式推进时仅顶层保留 `q*` 和物理 `ΔT_Q*` 输入；下层能量只能来自厚度 FDM。
- `post_fdm_source_compensation_alpha` 仅在 FDM 后把当前步原始 `delta_T_source` 的指定比例补到顶层，默认 `0.0` 保持旧行为，取值范围为 `[0, 1]`。
- `post_fdm_output_layer_compensations` 是唯一的固定输出温差补偿接口，每层独立配置 `temperature`，层号从 1 开始且不得包含底层；所有层复用输入帧 `dynamic/Q` 生成的连续权重场。最高 `post_fdm_output_q_region_percent` 百分比为完整补偿核心区，随后 `post_fdm_output_q_transition_percent` 百分比用 smoothstep 衰减到零。修正后的副本用于 HDF5/返回值，未经修正的 `T_next` 继续 rollout。
- 多层状态张量形状固定为 `[layer, node, 1]`。
- 输出序列形状固定为 `[time, layer, node, 1]`。
- CUDA 推理默认按较小层批量前向，避免 30 层等大规模场景一次性构造完整多层图导致显存溢出。
- 厚度方向当前使用隐式 FDM，`C_n` 仅作为诊断指标；`allow_unstable_fdm` 为兼容旧显式 FDM 配置的保留字段。
- VTK 输出从真实 `edge_index` 恢复 Gmsh 三角网格面，并写成相邻层连接的 `UNSTRUCTURED_GRID` wedge 体单元；拓扑渲染必须使用全节点，不支持按节点数降采样。

## 与其他模块关系

- 依赖 `training.run_config` 读取训练配置和派生 `PDGCNConfig`。
- 依赖 `training.train_entry` 中的 HDF5 时间步推导与文件发现工具。
- 依赖 `data` 读取 HDF5 并构造图。
- 依赖 `models.PDGCN` 恢复训练好的单层网络。
- 依赖 `visualization` 写 ParaView VTK 文件。

本代码库的 Python 环境为 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```
