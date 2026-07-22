# PDGCN

PDGCN 是面向曲面铺放热场预测的图神经网络训练框架。项目目标是构建曲面内 PDGCN 神经网络与曲面差分技术，用于预测多层堆叠铺放曲面的温度场。

## 项目特点

- 使用 PyTorch Geometric 表示曲面局部窗口图。
- 节点特征默认保持 `[x*, y*, z*, fx, fy, fz, T*]`；可通过配置追加 `q*` 和 `ΔT_Q*`，让 PD-GCN 感知当前步热源分布。
- 显式表面热源模块将 HDF5 中的表面热流 `q''` 转换为顶层温升。
- 边特征编码局部几何、扫描方向和纤维各向异性关系。
- 使用无源曲面内输运 residual、出流边界约束和图梯度平滑正则训练 PD-GCN。
- 支持 FEM 温度监督训练：`fem/temperature` 只作为标签读取，训练时无量纲化后计算监督损失，不进入节点特征。
- 支持目录级 HDF5 切片训练：同一目录内多个 `.h5` 文件按自然升序训练，每个文件是独立序列。
- 支持共享静态缓存：一个训练集目录复用同一份拓扑缓存，动态帧数据从各 HDF5 文件读取。

## 环境

本项目使用 conda 中已配置好的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```

建议训练和测试命令都使用该 Python 解释器。

## 数据约定

用于生成静态缓存的首个 HDF5 文件需要包含：

```text
dynamic/xyz
dynamic/fiber
dynamic/normal
dynamic/Q
edge_index
boundary_nodes/upwind
boundary_nodes/downwind
boundary_nodes/side
path/heat_center_step_distance
```

若启用 FEM 监督训练，HDF5 还需要包含监督温度字段：

```text
fem/temperature          [T, N, 1]
fem/valid_mask           [T, N, 1]  可选
fem/temperature_unit     scalar string，可选，支持 degC、C、K
```

`fem/temperature` 必须与 `dynamic/Q` 在帧数、节点数和节点顺序上对齐。若 `fem/valid_mask` 缺失或配置为 `null`，训练时默认所有节点有效。监督温度按真实物理温度保存，进入训练循环后转换为无量纲温度：

$$
T_{\mathrm{FEM}}^*
=
\frac{T_{\mathrm{FEM}}-T_{\mathrm{amb}}}{\Delta T_0}.
$$

代码只校验 `fem/temperature_unit` 是否为 `degC`、`C` 或 `K`，不会自动做摄氏/开尔文转换；配置中的 `T_amb` 必须与 FEM 温度使用同一温标。

HDF5 原始切片采用生成程序的原生单位：几何为 `mm`，速度为 `mm/s`，`dynamic/Q` 为面热流 `W/mm^2`。PDGCN 预处理阶段会强制转换为 SI 后再无量纲化：

- `dynamic/xyz`: `mm -> m`
- `dynamic/normal`: 曲面逐节点单位法向，参与速度方向向曲面切平面的投影。
- `velocity_speed`: `mm/s -> m/s`
- `velocity_direction_local`: 局部坐标中的速度方向；若 HDF5 坐标系为 `nip_local_velocity_side_normal`，默认 `[1, 0, 0]`。
- `path/heat_center_step_distance`、`path/slice_path_length`: `mm -> m`
- `dynamic/Q`: `W/mm^2 -> W/m^2`，作为表面热流 `q''` 保存在图对象的 `q_surface_star` 字段中；若启用 `include_q_in_features`，还会以 `q* = q'' / Q0` 追加到节点特征。

边特征中的 `cos_theta` 使用接收节点处的切向速度方向计算：先将 `velocity_direction_local` 投影到 `dynamic/normal` 给出的节点切平面，再与边方向取点积。旧静态缓存不包含法向和速度方向，升级数据后需要删除并重建 `datasets[].cache_dir`。

因此训练配置中的 `datasets[].scale` 必须使用 SI：`L0` 为 `m`，`v0` 为 `m/s`，`Q0` 为表面热流标尺 `W/m^2`，`K0` 为 `W/(m·K)`，`rho` 为 `kg/m^3`，`Cp` 为 `J/(kg·K)`。`heat_source_effective_thickness` 为必填字段，单位 `m`，用于显式热源温升公式；`heat_source_absorptivity` 默认为 `1.0`。

热源节点特征为可选消融项，默认关闭以兼容旧 checkpoint。开启 `include_q_in_features=true` 和 `include_delta_t_source_in_features=true` 后，节点特征布局为 `[x*, y*, z*, fx, fy, fz, T*, q*, ΔT_Q*]`。其中 `ΔT_Q* = heat_source_absorptivity * source_coefficient * dt_star * q*`。默认 `use_explicit_heat_source=true` 时先显式叠加该温升，PD-GCN 只学习无源输运增量；设为 `false` 时不预加热模型输入，但仍保留 `q*`、物理 `ΔT_Q*` 和 PDE 源项，由 PD-GCN 学习包含热源贡献的完整 `ΔT*`。

详细配置说明见 [configs/README.md](configs/README.md)。

## 训练配置

训练示例配置位于：

```text
configs/pdgcn_train.example.json
```

关键字段：

- `datasets[0].h5_dir`：训练输入 HDF5 目录。
- `datasets[0].cache_dir`：该训练集共享的静态缓存目录。
- `datasets[0].scale`：SI 无量纲化标尺与 PDE 系数派生参数。
- `supervision`：FEM 温度监督配置，默认关闭；启用后 v1 使用 `teacher_forcing`。
- `hyperparameters.training.lr_scheduler`：可选学习率调度，支持固定学习率、warmup+cosine 衰减和 plateau 自动降学习率。
- `hyperparameters.training.resume_from_checkpoint`：分阶段训练时设为 `true`，可从已有 checkpoint 继续训练。

静态缓存默认启用。缓存缺失时，训练入口会使用目录内排序后的第一个 HDF5 文件生成；缓存存在时直接复用。

推理示例配置位于：

```text
configs/pdgcn_infer.example.json
```

推理配置通过 `training_config` 引用训练配置，复用其中的数据集、尺度参数、模型超参和默认 checkpoint 路径。

## 快速开始

运行核心测试：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe -m unittest training.tests.test_static_topology training.tests.test_run_config training.tests.test_monitor
```

使用示例配置训练：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

使用训练后的 checkpoint 执行多层 PD-GCN + 1D FDM 推理：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_infer.example.json
```

默认输出：

```text
runs/pdgcn/checkpoint.pt
runs/pdgcn/history.json
runs/pdgcn/cache/case_3/
runs/pdgcn/multilayer_prediction.h5
runs/pdgcn/multilayer_prediction_vtk/
```

多层单文件推理输出 HDF5 包含：

- `temperature`：真实温度，形状 `[time, layer, node, 1]`。
- `temperature_star`：无量纲温度，形状 `[time, layer, node, 1]`。
- `metadata`：JSON 字符串数据集，同时写入根属性副本；记录 checkpoint、源 HDF5、层数、层间距、纤维旋转角、法向偏移方向、`dt_star`、FDM 系数、层分块大小、VTK 输出间隔、无量纲化参数和推理/渲染耗时。

多层批量推理会先把每个原始 HDF5 复制到 `output_dir/<output_prefix><原文件名>`，再在副本内追加/替换 `prediction/pdgcn_multilayer` 预测组；组内包含 `temperature`、`temperature_star` 和 `metadata`，形状与单文件模式一致。原始 HDF5 保持不变。

层索引约定为 `layer=0` 是顶层，`layer=L-1` 是底层恒温模具边界。多层推进顺序固定为：顶层显式表面热源、所有层无源 PD-GCN 面内输运、厚度方向 1D FDM、底层恒温钳制。下层不读取热源，能量只能由厚度 FDM 从上层传入；若启用热源节点特征，下层的 `q*` 和 `ΔT_Q*` 特征会置零。下层几何沿节点曲面法向偏移，纤维方向按 `layer_fiber_angles_deg` 绕节点法向旋转。

多层批量模式在 `write_vtk=true` 时，会在每个 HDF5 推理完成后按 `cloud_interval` 自动输出 ParaView 可读取的 legacy `.vtk` 合并三维拓扑云图；单文件模式仍可使用 `render_entry.py` 离线渲染。文件名形如
`temperature_step_000000.vtk`。每个 VTK 从真实 `edge_index` 恢复 Gmsh 三角网格面，并在相邻层之间生成 `UNSTRUCTURED_GRID` wedge 体单元，可在 ParaView 中用 `temperature` 或 `temperature_star` 着色。拓扑渲染必须使用全节点，不能按节点数降采样。

训练监控快照不再生成 PNG 热力图。需要查看单层曲面温度/残差时，运行：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\visualize_monitor.py --monitor-data runs\pdgcn\metrics\monitor_data.h5
```

该命令会导出 VTK 文件，默认位于 `runs/pdgcn/figures/vtk/`。

## 训练语义

1. 读取 JSON 配置并发现 `h5_dir` 下所有 `.h5`/`.hdf5` 文件。
2. 按文件名自然升序遍历切片，保证训练顺序可复现。
3. 缓存缺失时用第一个文件生成共享静态缓存；后续文件不读取或比较拓扑数据。
4. 每个 HDF5 文件作为独立样本序列训练，不与前一个文件共享温度状态。
5. 每个文件开始时重新初始化温度；若 `warmup_steps > 0`，先显式施加当前帧热源再使用无源 PD-GCN 做连续前向 warmup，不反向传播、不更新参数。
6. 同一 run 内模型参数和优化器状态持续更新。
7. 若启用 `resume_from_checkpoint`，训练入口会先加载已有模型权重；默认也恢复 Adam 状态，并把学习率重设为当前配置中的 `lr`，再按当前阶段的 `lr_scheduler` 调度。
8. epoch loss 统计为该 epoch 内所有文件所有 TBPTT 窗口损失的平均值；恢复训练时 epoch 编号从 checkpoint 的下一轮继续。
9. 若启用 `supervision.enabled=true`，训练按 `supervision.mode` 选择 FEM 监督语义：`teacher_forcing` 使用 FEM 当前帧做单步监督，`rollout` 从窗口起点 FEM 温度初始化后自回归推进并逐帧对齐 FEM，`mixed` 同时计算两类监督。监督模式忽略 `warmup_steps` 对训练输入温度的影响。

FEM 监督损失为：

$$
\mathcal L_T
=
\frac{
\sum_i m_i
\left(
T_{\mathrm{pred},i}^*
-
T_{\mathrm{FEM},i}^*
\right)^2
}{
\sum_i m_i+\varepsilon
}.
$$

总损失为：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{physics}}
+
\lambda_T\mathcal L_T.
$$

history 和 monitor 中会额外记录 `loss_physics`、`loss_supervised`、`loss_temperature`、`loss_teacher_forcing_temperature`、`loss_rollout_temperature`、`fem_temperature_rmse`、`fem_temperature_mae`、`fem_temperature_max_error` 和 rollout FEM 指标。监控快照在监督启用时可包含 `fem_temperature`、`pred_temperature` 和 `temperature_error`。

## 目录说明

```text
configs/      训练配置示例与说明
data/         HDF5 数据读取、SI 转换、无量纲化、特征构建、静态缓存
models/       PDGCN 模型、编码器、处理器、解码器
pde/          无源输运 residual、显式表面热源、边界条件、FDM 和物理损失
training/     训练入口、TBPTT、单层 rollout、checkpoint、监控和 VTK 快照导出
inference/    独立多层 PD-GCN + 1D FDM 推理入口、HDF5/VTK 输出和单元测试
visualization/ 通用 VTK 导出工具
DesignPlan/   研究设计、技术方案和实验方案文档
PIGNN/        PIGNN 参考仓库，只读
runs/         训练输出，不进入 Git
```

### 单层推理诊断

若只想检查单层 PD-GCN 训练效果、不启用多层 FDM，可运行：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\single_layer_infer_entry.py --config configs\pdgcn_single_layer_infer.example.json
```

单层入口默认输出 `runs/pdgcn/single_layer_prediction.h5`，输出文件会保留原始 HDF5 内容，并在 `prediction/pdgcn_single_layer/temperature` 中按帧写入预测真实温度，形状与 `fem/temperature` 保持一致为 `[time, node, 1]`；原始输入 HDF5 不会被原地修改。

单层入口也支持批量推理：设置 `single_layer_inference.batch_mode=true` 后，会按自然升序遍历 `h5_dir` 下直属 `.h5`/`.hdf5` 文件，并把增强后的 HDF5 写入 `output_dir/pre_<原文件名>`；对应 VTU 目录写入同一个 `output_dir` 下的 `pre_<原stem>_vtu/`。批量中单个文件失败时会继续处理其他文件，并在命令行汇总成功和失败清单。
