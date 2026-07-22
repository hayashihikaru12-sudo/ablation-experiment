# PDGCN 配置说明

训练和推理使用不同的示例配置文件管理：

- `configs/pdgcn_train.example.json`：训练、监控、数据集、模型/物理损失超参和训练产物路径。
- `configs/pdgcn_infer.example.json`：多层推理参数，并通过 `training_config` 引用训练配置。

训练入口命令为：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe training\train_entry.py --config configs\pdgcn_train.example.json
```

多层推理入口命令为：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\infer_entry.py --config configs\pdgcn_infer.example.json
```

配置文件中的相对路径均以各自配置文件所在目录为基准解析。例如 `pdgcn_infer.example.json` 中的 `training_config` 相对推理配置文件所在目录解析；训练配置中的 `datasets[].h5_dir` 和 `outputs.checkpoint_path` 继续相对训练配置文件所在目录解析。

## 顶层结构

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `monitoring` | object | 训练过程监控配置，控制是否记录 loss、温度场快照和 VTK 可视化数据。 |
| `supervision` | object | 可选节点级 FEM 温度监督配置，默认关闭。 |
| `peak_supervision` | object | 可选峰值温升 FEM 监督配置，默认关闭；与 `supervision` 互斥。 |
| `outputs` | object | 训练产物输出路径，包括 checkpoint 和 history JSON。 |
| `datasets` | array | 训练数据集列表。当前训练入口一次只支持使用第一个数据集，但一个数据集目录内可以包含多个 `.h5`/`.hdf5` 切片文件。 |
| `hyperparameters` | object | 模型结构、物理损失和训练超参数。 |

## `monitoring`

示例：

```json
"monitoring": { "enabled": true }
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | 是否启用训练监控。启用后会写入 `monitor_data.h5`，并按间隔保存温度和残差快照。 |
| `interval_epochs` | integer | `10` | 监控快照记录间隔。示例文件未显式写出时使用默认值。 |
| `temperature_frame_index` | integer 或 `null` | `null` | 指定用于温度场快照的帧索引；为 `null` 时使用首个 HDF5 文件的中间帧。 |
| `figures_dir` | string 或 `null` | `null` | 监控图像输出目录；为 `null` 时使用 `history_path` 同级的 `figures/`。 |
| `metrics_path` | string 或 `null` | `null` | 监控 HDF5 文件路径；为 `null` 时使用 `history_path` 同级的 `metrics/monitor_data.h5`。 |

## `supervision`

示例：

```json
"supervision": {
  "enabled": false,
  "temperature_dataset": "fem/temperature",
  "valid_mask_dataset": "fem/valid_mask",
  "lambda_temperature": 1.0,
  "lambda_rollout_temperature": 1.0,
  "rollout_window": null,
  "mode": "teacher_forcing"
}
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | 是否启用 FEM 温度监督。关闭时保持原无监督物理残差训练行为。 |
| `temperature_dataset` | string | `fem/temperature` | FEM 监督温度字段路径，形状必须为 `[T, N, 1]`，并与 `dynamic/Q` 对齐。 |
| `valid_mask_dataset` | string 或 `null` | `fem/valid_mask` | 可选有效节点 mask 字段。缺失或设为 `null` 时使用全 1 mask。 |
| `lambda_temperature` | number | `1.0` | 温度监督损失权重，必须非负。 |
| `lambda_rollout_temperature` | number | `1.0` | `rollout` / `mixed` 模式下自回归路径温度监督损失权重，必须非负。 |
| `rollout_window` | integer 或 `null` | `null` | `rollout` / `mixed` 模式下的路径监督窗口长度；为 `null` 时使用 `training.tbptt_window`。 |
| `mode` | string | `teacher_forcing` | 显式选择监督模式，可为 `teacher_forcing`、`rollout` 或 `mixed`。 |

监督温度按真实温度读取，并在训练中转为无量纲温度：

```text
T_fem* = (T_fem - T_amb) / delta_T0
```

FEM 温度不会进入额外节点特征通道。PDGCN 默认节点输入仍为 `[x*, y*, z*, fx, fy, fz, T*]`；若模型配置启用热源节点特征，FEM 温度只替换其中的 `T*` 列。

## `peak_supervision`

示例：

```json
"peak_supervision": {
  "enabled": false,
  "temperature_dataset": "fem/temperature",
  "valid_mask_dataset": "fem/valid_mask",
  "lambda_peak": 0.001,
  "topk": 30,
  "warmup_epochs": 30,
  "rollout_window": null
}
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | 是否启用峰值温升监督。启用时会读取 FEM 温度，但只使用峰值统计，不计算节点级 MSE。 |
| `temperature_dataset` | string | `fem/temperature` | FEM 温度字段路径，形状必须为 `[T, N, 1]`，并与 `dynamic/Q` 对齐。 |
| `valid_mask_dataset` | string 或 `null` | `fem/valid_mask` | 可选有效节点 mask。缺失或设为 `null` 时使用全 1 mask。 |
| `lambda_peak` | number | `0.001` | 峰值温升监督损失权重，必须非负。 |
| `topk` | integer | `30` | 使用温度最高的 `topk` 个有效节点均值近似峰值，必须为正整数。 |
| `warmup_epochs` | integer | `30` | 峰值监督权重从 0 线性升至 `lambda_peak` 的 epoch 数；设为 `0` 时不 warmup。 |
| `rollout_window` | integer 或 `null` | `null` | 峰值监督 rollout 窗口长度；为 `null` 时使用 `training.tbptt_window`。 |

`peak_supervision` 与 `supervision` 不能同时启用。峰值监督使用 FEM 第 0 帧初始化每个 HDF5 序列，之后跨 TBPTT 窗口持续携带模型预测温度状态；窗口边界只截断反传计算图，不再把温度重置为 FEM。损失在无量纲温度上计算：

```text
loss_peak_temperature_rise = (TopKMean(T_pred_next*) - TopKMean(T_fem_next*))^2
loss_total = loss_physics + lambda_peak(epoch) * loss_peak_temperature_rise
```

其中：

```text
lambda_peak(epoch) = lambda_peak * min(1, epoch / warmup_epochs)
```

启用后 history 和 monitor 会记录 `loss_peak_temperature_rise`、`peak_temperature_rise_pred`、`peak_temperature_rise_fem`、`peak_temperature_rise_error`、`peak_temperature_rise_abs_error`、`peak_temperature_rise_rmse` 和 `lambda_peak_temperature_rise`。此外会按完整 HDF5 case 记录 `case_peak_temperature_*`、`case_peak_temperature_rise_*` 和 `case_peak_topk_temperature_rise_*`，用于监控完整自回归路径上的峰值偏差。

## `outputs`

示例：

```json
"outputs": {
  "checkpoint_path": "../runs/pdgcn/checkpoint.pt",
  "history_path": "../runs/pdgcn/history.json"
}
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `checkpoint_path` | string | 模型权重、优化器状态和元数据保存路径。 |
| `history_path` | string 或 `null` | 训练历史 JSON 保存路径。为 `null` 时会使用 checkpoint 路径派生。 |

## `datasets[]`

示例：

```json
"datasets": [
  {
    "name": "case_3",
    "h5_dir": "../case_3_HDF",
    "cache_dir": "../runs/pdgcn/cache/case_3",
    "scale": {
      "L0": 0.051764991760253906,
      "v0": 0.08,
      "T_amb": 120,
      "delta_T0": 230,
      "Q0": 465000.0,
      "K0": 5.9,
      "rho": 1575,
      "Cp": 1600.0,
      "heat_source_effective_thickness": 0.00015,
      "heat_source_absorptivity": 1.0
    }
  }
]
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | 数据集名称，仅用于记录和区分实验。 |
| `h5_dir` | string | 训练输入 HDF5 目录。目录内 `.h5`/`.hdf5` 文件按自然升序遍历，每个文件作为一个独立序列训练。 |
| `cache_dir` | string | 静态拓扑缓存目录。缓存不存在时，训练入口会用目录内排序后的第一个 HDF5 文件生成缓存。 |
| `scale` | object | 无量纲化和 PDE 系数派生所需的物理标尺参数。详见下一节。 |
| `scan_velocity` | number 或 `null` | 可选真实扫描速度，单位 `m/s`。若设置，必须与 HDF5 根属性 `velocity_speed` 转换到 SI 后一致。示例文件未写出该字段时使用 HDF5 属性。 |

静态缓存只保存拓扑、边界节点、节点类型、节点数、边数和特征维度等静态信息。动态帧数据始终从各 HDF5 文件读取，并在读取阶段转换为 SI 单位。

## `datasets[].scale`

`scale` 必须使用 SI 单位。HDF5 原始数据使用生成程序的原生单位，进入模型和 PDE 前会做如下转换：

| HDF5 字段 | 原始单位 | 进入模型/PDE 前的转换 |
| --- | --- | --- |
| `dynamic/xyz` | `mm` | 乘 `1e-3` 转为 `m`，再除以 `L0` 得到无量纲坐标。 |
| `dynamic/fiber` | 无量纲方向向量 | 在代码内归一化。 |
| `dynamic/normal` | 无量纲单位法向 | 在代码内归一化，并用于把速度方向投影到接收节点切平面。 |
| `dynamic/Q` | 面热流 `W/mm^2` | 乘 `1e6` 转为表面热流 `W/m^2`，保存在图对象 `q_surface_star` 中，供显式热源模块使用。 |
| `path/heat_center_step_distance` | `mm` | 乘 `1e-3` 转为 `m`，用于自动派生 `dt` 和 `dt_star`。 |
| `path/slice_path_length` | `mm` | 乘 `1e-3` 转为 `m`，用于校验路径长度。 |
| 根属性 `velocity_direction_local` | 无量纲方向向量 | 作为速度基准方向；会逐节点投影到曲面切平面。若局部坐标系为 `nip_local_velocity_side_normal` 且该属性缺失，使用 `[1, 0, 0]`。 |
| 根属性 `velocity_speed` | `mm/s` | 乘 `1e-3` 转为 `m/s`。 |

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| `L0` | `m` | 特征长度。坐标、边位移和边距离都会除以该值。 |
| `v0` | `m/s` | 特征速度。扫描速度会除以该值进入模型和 PDE。 |
| `T_amb` | `degC` | 温度基准。无量纲温度定义为 `(T - T_amb) / delta_T0`。 |
| `delta_T0` | `degC` 或 `K` 温差 | 特征温升。只作为温差尺度使用，数值上摄氏温差和开尔文温差相同。 |
| `Q0` | `W/m^2` | 表面热流标尺。`dynamic/Q` 转为 `W/m^2` 后会除以该值，得到 `q_surface*`。 |
| `K0` | `W/(m·K)` | 特征导热系数，通常取纤维方向主导热系数。用于派生 `inverse_pe`。 |
| `rho` | `kg/m^3` | 密度。用于派生 PDE 热扩散和热源系数。 |
| `Cp` | `J/(kg·K)` | 比热容。用于派生 PDE 热扩散和热源系数。 |
| `heat_source_effective_thickness` | `m` | 显式表面热源作用的等效热容量厚度。该值越小，同一表面热流造成的温升越大。 |
| `heat_source_absorptivity` | 无量纲 | 可选热源吸收率，默认 `1.0`。 |
| `eps` | 无量纲 | 可选数值下界，默认 `1e-12`，用于距离、归一化和除法稳定性。 |

训练入口会根据 `scale` 和首个 HDF5 文件自动派生：

```text
dt = heat_center_step_distance / velocity_speed
dt_star = dt / (L0 / v0)
inverse_pe = K0 / (rho * Cp * v0 * L0)
source_coefficient = Q0 * L0 / (rho * Cp * v0 * heat_source_effective_thickness * delta_T0)
```

不建议在配置中手动写入 `inverse_pe`、`source_coefficient`、`pi_q` 或 `dt_star`。即使写入模型配置，也会被训练入口自动覆盖。`pi_q` 仅作为兼容别名写入 checkpoint metadata，显式热源模块使用 `source_coefficient`。

## `hyperparameters.model`

示例：

```json
"model": {
  "hidden_size": 64,
  "message_passing_num": 3,
  "gamma_upwind": 0.8,
  "dropout": 0.0,
  "layer_norm": true
}
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `hidden_size` | integer | Encoder、Processor 和 Decoder 的隐空间维度。值越大表达能力越强，但显存和训练时间也会增加。 |
| `message_passing_num` | integer | 消息传递层数。增加层数可以扩大图上的信息传播范围，但也更容易过平滑或训练变慢。 |
| `gamma_upwind` | number | 上风项权重相关参数，用于加强流向方向上的信息传播偏置。 |
| `include_q_in_features` | boolean | 是否把无量纲表面热流 `q*` 追加到节点特征，默认 `false`。 |
| `include_delta_t_source_in_features` | boolean | 是否把当前步显式源温升 `ΔT_Q*` 追加到节点特征，默认 `false`。 |
| `use_explicit_heat_source` | boolean | 是否在 PD-GCN 前把 `ΔT_Q*` 显式叠加到温度状态，默认 `true`。设为 `false` 时保留热源特征并在 PDE residual 中扣除源增量，使网络学习完整 `ΔT*`。 |
| `dropout` | number | MLP dropout 比例。当前示例为 `0.0`，表示不使用 dropout。 |
| `layer_norm` | boolean | 是否在 MLP 中使用 LayerNorm。通常有助于稳定训练。 |

模型默认输入输出约定：

| 项 | 维度 | 说明 |
| --- | --- | --- |
| 节点特征 | `7` / `8` / `9` | 默认 `[x*, y*, z*, fx, fy, fz, T*]`；开启两个热源开关后为 `[x*, y*, z*, fx, fy, fz, T*, q*, ΔT_Q*]`。 |
| 边特征 | `7` | `[dx*, dy*, dz*, d*, cos_theta, cos_phi, cos_phi_sq]`，其中 `cos_theta` 为接收节点切向速度方向与边方向的夹角余弦。 |
| 全局特征 | `1` | 无量纲扫描速度大小 `v_scan / v0`。 |
| 模型输出 | `1` | 无量纲温度增量 `delta_T*`。 |

升级到包含 `dynamic/normal` 的数据后，需要删除旧 `cache_dir` 并重建静态缓存；旧 checkpoint 的 `cos_theta` 物理语义也不同，正式实验建议重新训练。

## `hyperparameters.physics_loss`

示例：

```json
"physics_loss": {
  "k_ratio": 0.05,
  "lambda_outflow": 1.0,
  "gradient_regularization": 0.001,
  "residual_time_scheme": "explicit",
  "convection_scale": 2.0
}
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `k_ratio` | number | 横向或厚度方向导热系数与主导热系数的比值，即 `K_perp / K_parallel`。示例 `0.05` 表示横向导热明显弱于纤维方向。 |
| `lambda_outflow` | number | 出流边界 Neumann 软约束损失权重。越大越强调 downwind 边界法向温度梯度接近零。 |
| `gradient_regularization` | number | 图梯度平滑损失权重，作用于边界钳制后的预测温度 `T_next_bc*`，抑制相邻内部节点的高频温度振荡。推荐从 `1e-4` 到 `1e-2` 调参；过大可能抹平热峰。 |
| `residual_time_scheme` | string | PDE 空间项的时间离散方式。可选 `"explicit"` 或 `"backward"`。 |
| `convection_scale` | number | 对流项缩放补偿因子，默认 `2.0`，用于补偿带符号边方向对流项在典型三角网格上的投影缩放。 |
| `adaptive_pde_node_weight_enabled` | boolean | 是否启用 PDE residual 内部节点动态权重；默认 `true`。 |
| `adaptive_pde_node_weight_scheme` | string | 动态权重方案。可选 `"heat_flux"`、`"temperature_continuous_clamped"`、`"temperature_hard_threshold"` 或 `"none"`；默认 `"heat_flux"`，保持旧版行为。 |
| `adaptive_pde_node_weight_min` | number | `"heat_flux"` 方案中零热流节点的最低 PDE 权重，范围 `[0, 1]`，默认 `0.2`。 |
| `temperature_pde_node_weight_beta` | number | 方案 B 连续截断的温度权重斜率，默认 `0.5`。 |
| `temperature_pde_node_weight_max` | number | 方案 B 连续截断的权重上限，默认 `8.0`。 |
| `temperature_pde_node_weight_clamp_enabled` | boolean | 方案 B 是否启用 `temperature_pde_node_weight_max` 上限截断；默认 `true` 保持旧行为，设为 `false` 时权重不再按上限截断。 |
| `temperature_pde_node_weight_threshold` | number | 方案 C 硬阈值的无量纲温度阈值，默认 `1.0`。 |
| `temperature_pde_node_weight_high` | number | 方案 C 硬阈值中热区节点权重，默认 `4.0`。 |
| `adaptive_pde_node_weight_warmup_enabled` | boolean | 是否对 B/C 温度权重启用 epoch 级 warmup；默认 `false`。 |
| `adaptive_pde_node_weight_warmup_epochs` | integer | warmup 阶段长度，默认 `50`。启用后前 `E` 个 epoch 使用均匀权重，`E` 到 `2E` 线性引入完整温度权重。 |

当前总损失为：

```text
loss_total = loss_pde + lambda_outflow * loss_outflow + gradient_regularization * loss_smooth
```

若启用节点级 FEM 监督，则总损失变为：

```text
loss_physics = loss_pde + lambda_outflow * loss_outflow + gradient_regularization * loss_smooth
loss_temperature = mean_masked((T_pred_next* - T_fem_next*)^2)
loss_total = loss_physics + lambda_temperature * loss_temperature
```

若启用峰值温升监督，则总损失变为：

```text
loss_physics = loss_pde + lambda_outflow * loss_outflow + gradient_regularization * loss_smooth
loss_peak_temperature_rise = (TopKMean(T_pred_next*) - TopKMean(T_fem_next*))^2
loss_total = loss_physics + lambda_peak(epoch) * loss_peak_temperature_rise
```

`loss_smooth` 为内部边上的一阶图梯度平方均值：

```text
loss_smooth = mean_edges(((T_i* - T_j*) / d_ij*)^2)
```

PDE residual 的主要形式为：

```text
residual_transport =
  (T_next* - T_source_applied*) / dt_star
  + convection
  - inverse_pe * diffusion
```

其中 `convection` 采用带符号边方向贡献方法。对边 `sender=j -> receiver=i`，`cos_theta` 为接收节点切向速度方向与 `sender -> receiver` 边方向的点积，接收节点内使用距离倒数权重归一化：

```text
alpha_ij = (d_ij*)^-1 / sum_k (d_ik*)^-1
convection_i += convection_scale * alpha_ij * v_scan* * cos_theta_ij * (T_i* - T_j*) / d_ij*
```

该写法保留正负方向贡献，不使用 `ReLU(cos_theta)` 截断反方向邻居，也不再强制纯对流项全局求和为 0。

当 `adaptive_pde_node_weight_enabled = true` 时，`loss_pde` 会在内部节点上按 `adaptive_pde_node_weight_scheme` 使用归一化节点权重统计 residual MSE；`loss_outflow`、`loss_smooth` 和 FEM 监督损失不受影响。`"heat_flux"` 使用 `q_surface_star`，`"temperature_continuous_clamped"` 使用方案 B：

```text
w = clamp(1 + beta * max(0, T_source_applied*), 1, W_max)
```

若 `temperature_pde_node_weight_clamp_enabled = false`，方案 B 改为：
```text
w = max(1 + beta * max(0, T_source_applied*), 1)
```

`"temperature_hard_threshold"` 使用方案 C：

```text
w = W_high if T_source_applied* > threshold else 1
```

三类动态权重为互斥方案，不叠乘。权重计算结果会 `detach()`，只改变 residual MSE 的统计权重。

默认训练采用算子分裂：先用显式表面热源得到 `T_source_applied*`，再把该温度输入无源 PD-GCN。若 `use_explicit_heat_source=false`，则 `T_source_applied*=T_current*`，模型直接预测包含热源与输运贡献的完整增量；此时 residual 瞬态项为 `(T_next* - T_current* - ΔT_Q*) / dt*`。无论是否显式推进，`q*` 和物理 `ΔT_Q*` 特征的数值保持一致。当 `residual_time_scheme = "explicit"` 时，空间项使用模型输入温度评估；当为 `"backward"` 时使用 `T_next*` 评估。

## `hyperparameters.training`

示例：

```json
"training": {
  "lr": 0.0001,
  "lr_scheduler": "warmup_cosine",
  "min_lr": 0.000001,
  "lr_warmup_epochs": 30,
  "lr_patience": 30,
  "lr_factor": 0.5,
  "epochs": 1000,
  "tbptt_window": 5,
  "warmup_steps": 30,
  "grad_clip_norm": null,
  "resume_from_checkpoint": false,
  "resume_checkpoint_path": null,
  "resume_optimizer_state": true,
  "loss_threshold": 0.02,
  "device": null,
  "seed": 42
}
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `lr` | number | Adam 学习率。 |
| `lr_scheduler` | string | 学习率调度策略。支持 `"none"`、`"warmup_cosine"` 和 `"plateau"`；默认 `"none"` 保持固定学习率。 |
| `min_lr` | number | 调度器允许的最小学习率。`warmup_cosine` 会从 warmup 后的 `lr` 逐渐衰减到该值；`plateau` 降学习率时不会低于该值。 |
| `lr_warmup_epochs` | integer | `warmup_cosine` 的线性学习率 warmup epoch 数。注意它不同于温度场伪时间 `warmup_steps`。 |
| `lr_patience` | integer | `plateau` 策略中 loss 连续多少个 epoch 未改善后触发降学习率。 |
| `lr_factor` | number | `plateau` 触发时的学习率缩放因子，新学习率为 `max(current_lr * lr_factor, min_lr)`。 |
| `epochs` | integer | 最大训练轮数。 |
| `tbptt_window` | integer | 截断反向传播的时间窗口长度。窗口越大，跨时间步梯度越完整，但显存消耗越高。 |
| `warmup_steps` | integer | 每个 HDF5 文件开始训练前的伪时间 warmup 步数。`0` 表示直接从冷态初温开始。 |
| `grad_clip_norm` | number 或 `null` | 梯度裁剪阈值。为 `null` 时不裁剪。 |
| `resume_from_checkpoint` | boolean | 是否从已有 checkpoint 恢复模型权重并继续训练。默认 `false`，即每次从随机初始化开始。 |
| `resume_checkpoint_path` | string 或 `null` | 可选恢复来源 checkpoint 路径。为 `null` 且 `resume_from_checkpoint=true` 时，使用 `outputs.checkpoint_path` / legacy `data.checkpoint_path`。相对路径按训练配置文件所在目录解析。 |
| `resume_optimizer_state` | boolean | 是否同时恢复 Adam 优化器状态。恢复后仍会把优化器学习率重设为当前配置中的 `lr`，便于分阶段调小学习率。 |
| `loss_threshold` | number 或 `null` | 提前停止阈值。epoch 平均 loss 低于该值时停止训练；为 `null` 时禁用该规则。 |
| `device` | string 或 `null` | 训练设备。为 `null` 时自动选择 CUDA，若 CUDA 不可用则使用 CPU。也可显式写 `"cpu"` 或 `"cuda"`。 |
| `seed` | integer | Python、NumPy 和 PyTorch 共用的非负随机种子，默认 `42`。A0/A1 对比必须使用相同值。 |

训练语义：

- 一个 HDF5 文件是一条独立序列，文件之间不继承温度状态。
- 同一 HDF5 文件内，温度状态会随 frame 自回归推进。
- 每个文件开始时先初始化温度；若 `warmup_steps > 0`，会用当前模型前向传播若干步生成伪初温，不反向传播、不更新参数。
- 若 `lr_scheduler = "warmup_cosine"`，训练入口会在每个 epoch 开始时设置本轮学习率；history 和 monitor 会记录该 epoch 实际使用的 `lr`。
- 若 `lr_scheduler = "plateau"`，训练入口会根据 epoch 平均 loss 在 epoch 结束后调整下一轮学习率。
- epoch loss 是该 epoch 内所有文件、所有 TBPTT 窗口损失的平均值。
- 分阶段训练时，将 `resume_from_checkpoint` 设为 `true` 即可从上一阶段 checkpoint 继续；新阶段的 epoch 编号会从已加载 checkpoint 的下一轮开始，历史记录也会合并保存。

## `inference`

推理配置位于 `configs/pdgcn_infer.example.json`。示例：

```json
{
  "training_config": "pdgcn_train.example.json",
  "inference": {
    "num_layers": 4,
    "layer_spacing": 0.00015,
    "output_path": "../runs/pdgcn/multilayer_prediction.h5",
    "dataset_index": 0,
    "h5_path": null,
    "h5_dir": "../case_3_HDF",
    "output_dir": "../runs/pdgcn/multilayer_batch",
    "output_prefix": "pre_",
    "prediction_group_path": "prediction/pdgcn_multilayer",
    "batch_mode": false,
    "steps": null,
    "warmup_steps": null,
    "bottom_temperature_star": 0.0,
    "allow_unstable_fdm": false,
    "layer_fiber_angles_deg": [0.0, 45.0, -45.0, 90.0],
    "normal_offset_sign": -1,
    "write_vtk": false,
    "use_pdgcn_inplane": true,
    "pdgcn_inplane_top_layer_only": false,
    "post_fdm_source_compensation_alpha": 0.0,
    "post_fdm_output_layer_compensations": [
      {"layer": 1, "temperature": 125.0},
      {"layer": 2, "temperature": 102.0},
      {"layer": 3, "temperature": 20.0}
    ],
    "post_fdm_output_q_region_percent": 2.5,
    "post_fdm_output_q_transition_percent": 5.0,
    "cloud_interval": 20,
    "layer_batch_size": null,
    "delta_smoothing_alpha": 0.2,
    "delta_smoothing_steps": 1,
    "cloud_max_nodes_per_layer": null,
    "vtk_output_dir": null
  }
}
```

`training_config` 指向训练配置文件，推理入口会从该文件读取 `datasets`、`outputs`、`hyperparameters` 和训练 `warmup_steps/device` 默认值。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `training_config` | string | 被引用的训练配置路径，相对推理配置文件所在目录解析。 |
| `num_layers` | integer | 多层堆叠层数，必须大于等于 `2`。 |
| `layer_spacing` | number | 层间距，单位 `m`。内部会转为 `layer_spacing / L0`。 |
| `output_path` | string | 多层推理输出 HDF5 路径。 |
| `dataset_index` | integer | 使用 `datasets[]` 中的第几个数据集，示例为 `0`。 |
| `h5_path` | string 或 `null` | 可选输入 HDF5 文件。为 `null` 时使用所选数据集目录中自然升序的第一个文件。 |
| `batch_mode` | boolean | 是否启用多层批量推理。启用后遍历 `h5_dir` 下直属 `.h5`/`.hdf5` 文件。 |
| `h5_dir` | string 或 `null` | 多层批量模式输入目录；为 `null` 时使用 `datasets[dataset_index].h5_dir`。 |
| `output_dir` | string 或 `null` | 多层批量模式输出目录；每个原始 HDF5 会先复制为 `<output_dir>/<output_prefix><原文件名>`，再在副本中写入预测组。 |
| `output_prefix` | string | 多层批量输出文件名前缀，默认 `pre_`。 |
| `prediction_group_path` | string | 多层批量模式写入增强副本的预测组路径，默认 `prediction/pdgcn_multilayer`。组内包含 `temperature`、`temperature_star` 和 `metadata`。 |
| `steps` | integer 或 `null` | 推理步数。为 `null` 时使用输入 HDF5 的全部帧。 |
| `warmup_steps` | integer 或 `null` | 推理前 warmup 步数。为 `null` 时沿用训练配置中的 `warmup_steps`。 |
| `bottom_temperature_star` | number | 底层恒温边界的无量纲温度。`0.0` 对应真实温度 `T_amb`。 |
| `allow_unstable_fdm` | boolean | 兼容旧显式 FDM 配置的保留字段。当前厚度方向使用 Backward Euler 隐式 FDM，不再依赖该字段跳过稳定性检查。 |
| `layer_fiber_angles_deg` | number array 或 `null` | 每层相对第 0 层纤维方向的旋转角，单位为度；长度需等于 `num_layers`，第 0 项必须为 `0.0`。为 `null` 时所有层使用 `0.0`。 |
| `normal_offset_sign` | integer | 法向偏移方向，只能为 `-1` 或 `1`。默认 `-1` 表示 `pos_i = pos_0 - i * layer_spacing * normal`。 |
| `write_vtk` | boolean | 多层批量模式是否在每个 HDF5 推理完成后自动生成 VTK；单文件模式仍使用 `render_entry.py` 离线渲染。 |
| `use_pdgcn_inplane` | boolean | 是否启用无源 PD-GCN 面内输运增量。设为 `false` 时执行“显式热源 + 厚度 FDM only”对照推理。 |
| `pdgcn_inplane_top_layer_only` | boolean | 当 `use_pdgcn_inplane=true` 时，是否只在第 0 层启用 PD-GCN 面内输运；下层面内增量置零，仅由厚度 FDM 传热。 |
| `post_fdm_source_compensation_alpha` | number | FDM 后顶层热源补偿系数，范围 `[0, 1]`。补偿量为当前步原始 `delta_T_source` 的该比例；默认 `0.0` 保持旧行为。 |
| `post_fdm_output_layer_compensations` | object array | 逐层补偿配置。每项只包含 `layer`（从 1 开始的非底层层号）和 `temperature`（该层补偿温差，`°C`）。层号不得重复，数组长度可扩展；示例初值为第 1/2/3 层 `125/102/20°C`。设为空数组可关闭固定输出补偿。所有层共享输入帧 Q 值最高百分比节点掩码。 |
| `post_fdm_output_q_region_percent` | number | 输入帧 `dynamic/Q` 高热流区域百分比，范围 `(0, 100]`，默认 `10.0`；选取 Q 值最高的指定百分比节点作为所有配置层的补偿区域。 |
| `post_fdm_output_q_transition_percent` | number | 高 Q 核心区外的平滑过渡比例，默认 `5.0`，必须非负且与核心区比例之和不超过 `100`。过渡区补偿权重按 Q 值通过 smoothstep 从 `1` 连续衰减到 `0`；设为 `0.0` 恢复二值硬边界。 |
| `cloud_interval` | integer | 合并三维云图输出间隔。默认 `20` 表示输出第 `0, 20, 40, ...` 帧。 |
| `layer_batch_size` | integer 或 `null` | 每次模型前向处理的层数；为 `null` 时 CUDA 自动使用较小层批量以降低显存。 |
| `delta_smoothing_alpha` | number | 推理端网络增量图低通强度，范围 `[0, 1]`。默认 `0.2`；设为 `0` 可关闭平滑。 |
| `delta_smoothing_steps` | integer | 对 `delta_T_net` 执行的图低通迭代次数，必须非负。默认 `1`；设为 `0` 可关闭平滑。 |
| `cloud_max_nodes_per_layer` | integer 或 `null` | 兼容旧配置的保留字段；拓扑 wedge 渲染必须使用全节点，该字段不会被自动应用。 |
| `vtk_output_dir` | string 或 `null` | 单文件离线渲染的 VTK 输出目录。多层批量模式固定使用 `<output_dir>/<output_prefix><原stem>_vtk/`。 |

直接运行 `inference/infer_entry.py` 时，会优先使用 JSON 中的 `inference.batch_mode`。只有显式传入命令行参数 `--batch` 或 `--no-batch` 时，才会覆盖 JSON 配置。

- 单文件模式：设置 `batch_mode=false`。此时入口使用 `h5_path` 作为输入 HDF5，并把结果写入 `output_path`；若 `h5_path=null`，入口会退回到训练配置 `datasets[dataset_index].h5_dir` 中自然排序的第一个 HDF5 文件。
- 批量模式：设置 `batch_mode=true`。此时入口使用 `h5_dir` 作为输入目录，非递归遍历直属 `.h5`/`.hdf5` 文件，把每个原始 HDF5 复制到 `output_dir/<output_prefix><原文件名>`，并在副本内追加/替换 `prediction_group_path` 预测组。若 `write_vtk=true`，还会按 `cloud_interval` 自动写出 `output_dir/<output_prefix><原stem>_vtk/temperature_step_*.vtk`；每个文件的返回结果和 HDF5 metadata 会记录 `vtk_written`、`vtk_output_dir`、`rendered_steps` 与 `render_seconds`。原始 HDF5 不会被修改。单个文件失败时会继续处理其他文件，并在命令行汇总成功和失败清单。

多层推理中厚度方向 FDM 系数为：

```text
C_n = dt_star * inverse_pe * k_ratio / layer_spacing_star^2
layer_spacing_star = layer_spacing / L0
```

当前厚度方向采用 Backward Euler 隐式 FDM，`C_n` 仍记录到推理 metadata 中作为诊断指标，但不再作为显式格式的稳定性报错阈值。

多层推理的温度更新为：

```text
T_src[0] = T_current[0] + delta_T_source
T_src[k>0] = T_current[k]
T_inplane = T_src + delta_T_inplane
T_fdm = implicit_fdm_step(T_inplane)
T_fdm[0] = T_fdm[0] + post_fdm_source_compensation_alpha * delta_T_source
T_next = apply_boundary(T_fdm)
T_output = T_next
for item in post_fdm_output_layer_compensations:
    k = item.layer - 1
    weight = smooth_high_q_weight(q, core_percent, transition_percent)
    T_output[k] += weight * item.temperature / delta_T0
T_rollout_next = T_next
```

其中 `delta_T_source` 只由显式表面热源模块作用于顶层；若启用热源节点特征，多层图中也只有顶层保留 `q*` 和 `ΔT_Q*`，非顶层这两列置零。`delta_T_inplane` 由无源 PD-GCN 计算，若 `use_pdgcn_inplane=false` 则全层置零，若 `pdgcn_inplane_top_layer_only=true` 则仅第 0 层保留 PD-GCN 增量、下层置零；`implicit_fdm_step` 由厚度方向 Backward Euler 隐式 1D FDM 基于 `T_inplane` 计算。网络增量会先按当前层内图拓扑进行可选低通平滑，再进入 FDM 步骤；该平滑只作用于网络增量，不直接平滑最终温度。逐层固定输出补偿以输入帧 `dynamic/Q` 最高 `post_fdm_output_q_region_percent` 百分比节点为完整补偿核心区，并在随后的 `post_fdm_output_q_transition_percent` 百分比节点中按 Q 值使用 smoothstep 将权重从 1 平滑衰减到 0，再乘各层配置的 `temperature / delta_T0`；所有层共享同一连续权重场，只修改当帧写入 HDF5/返回值的副本，不修改 `T_rollout_next`，因此不会逐步累计，也不会改变 PD-GCN/FDM 动力学。

## 调参注意事项

- `Q0` 是表面热流无量纲化标尺，不是直接削弱真实热输入的旋钮。真实热输入主要由 HDF5 中的 `dynamic/Q`、`heat_source_effective_thickness`、`rho`、`Cp`、`dt` 和吸收率决定。
- `heat_source_effective_thickness` 越小，同一 `q''` 造成的显式温升越大。若温度明显偏高，应优先检查该值是否等于真实受热厚度或是否还需要调整吸收率。
- `warmup_steps` 会在每个 HDF5 文件开始时先显式加热、再用当前无源 PD-GCN 生成伪初温。较大的 warmup 可能显著抬高初始温度。
- `residual_time_scheme = "backward"` 通常比 `"explicit"` 更稳定，但训练代价和收敛行为可能不同，需要结合监控结果判断。
- `delta_smoothing_alpha` 和 `delta_smoothing_steps` 用于抑制推理端自回归高频误差。若云图锯齿明显，可在 `0.1~0.3` 和 `1~2` 次迭代内消融；若热峰被抹平，应降低强度或关闭。

## `single_layer_inference`

单层推理配置位于 `configs/pdgcn_single_layer_infer.example.json`。该入口用于检查单层 PD-GCN 自身推理效果，不启用多层 FDM。输出 HDF5 会先复制原始输入文件，再把预测数据写入 `prediction/pdgcn_single_layer`，因此原始输入 HDF5 不会被原地修改。

运行命令：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe inference\single_layer_infer_entry.py --config configs\pdgcn_single_layer_infer.example.json
```

### 单文件与批量模式选择

直接运行 `inference/single_layer_infer_entry.py` 时，会优先使用 JSON 中的 `single_layer_inference.batch_mode`。只有显式传入命令行参数 `--batch` 或 `--no-batch` 时，才会覆盖 JSON 配置。

- 单文件模式：设置 `batch_mode=false`。此时入口使用 `h5_path` 作为输入 HDF5，并把增强后的 HDF5 写入 `output_path`；若 `h5_path=null`，入口会退回到训练配置 `datasets[dataset_index].h5_dir` 中自然排序的第一个 HDF5 文件。
- 批量模式：设置 `batch_mode=true`。此时入口使用 `h5_dir` 作为输入目录，非递归遍历直属 `.h5`/`.hdf5` 文件，并把增强后的 HDF5 写入 `output_dir/<output_prefix><原文件名>`；对应 VTU 写入 `output_dir/<output_prefix><原stem>_vtu/`。

为避免模式混淆，批量模式建议保持 `h5_path=null`，单文件模式建议明确填写 `h5_path`。如果看到程序访问训练配置中的 `../../autodl-tmp/...` 等目录，通常说明当前实际进入了单文件模式，且 `h5_path` 为空。

关键字段：

| 参数 | 说明 |
| --- | --- |
| `output_path` | 单文件模式的增强 HDF5 输出路径。输出文件保留原 HDF5 内容，并新增预测组。 |
| `h5_path` | 单文件模式输入 HDF5；为 `null` 时使用所选数据集目录中自然升序的第一个文件。 |
| `batch_mode` | 是否启用批量推理。启用后遍历 `h5_dir` 下直属 `.h5`/`.hdf5` 文件。 |
| `h5_dir` | 批量模式输入目录；为 `null` 时使用 `datasets[dataset_index].h5_dir`。 |
| `output_dir` | 批量模式输出目录。增强 HDF5 和对应 VTU 目录都写入该目录。 |
| `output_prefix` | 批量输出文件名前缀，默认 `pre_`，例如 `case1.h5` 输出为 `pre_case1.h5`。 |
| `prediction_group_path` | 预测结果组路径，默认 `prediction/pdgcn_single_layer`。 |
| `mode` | 兼容字段，可为 `"autoregressive"`、`"teacher_forcing"` 或 `"both"`；HDF5 落盘仅保存自回归预测温度。 |
| `write_vtu` | 是否在推理后写出单层 `.vtu` 曲面文件。 |
| `vtu_interval` | VTU 输出帧间隔，从第 0 帧开始采样。 |
| `vtu_output_dir` | 单文件模式 VTU 输出目录；为 `null` 时使用 `<output_path stem>_vtu/`。批量模式固定使用 `<output_dir>/<output_prefix><原stem>_vtu/`。 |
| `fem_temperature_dataset` | FEM 温度数据集路径，默认 `fem/temperature`。 |
| `fem_valid_mask_dataset` | FEM mask 数据集路径，默认 `fem/valid_mask`。 |

`prediction/pdgcn_single_layer` 内仅新增 `temperature` 数据集，形状为 `[time, node, 1]`，按帧写入预测真实温度，组织方式与 `fem/temperature` 保持一致。
