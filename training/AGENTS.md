# training 模块说明

`training` 目录实现 PDGCN 的训练、推理、运行配置、checkpoint 和监控逻辑，是从配置文件到模型产物的主流程模块。

## 主要训练流程

本模块支持两类训练路径：

- 通用图序列训练：`trainer.py` 和 `tbptt.py` 面向已经构建好的 PyG 图序列。默认按“显式表面热源 -> 无源 PD-GCN 面内输运”推进；取消显式推进时模型输入不预加热源，并由带源 residual 约束 PD-GCN 学习完整温度增量。
- 固定拓扑高速训练：`static_topology.py` 配合 `data/static_cache.py` 的缓存，把静态拓扑常驻设备，在每帧只更新动态节点特征、表面热流字段和边特征，减少大规模训练时的数据搬运开销。
- FEM 监督训练：固定拓扑路径支持 `supervision.mode = "teacher_forcing" | "rollout" | "mixed"`；`teacher_forcing` 约束单步预测，`rollout` 约束自回归路径，`mixed` 同时启用两者。

## 主要文件

- `train_entry.py`：命令行训练入口，默认读取 `configs/pdgcn_train.example.json`，完成缓存准备、模型构建、训练、checkpoint 保存和 history 写入。
- `run_config.py`：读取并校验 JSON 运行配置，支持 legacy 和分类式 schema；包含 `SupervisionRunConfig`，并从尺度参数派生 `inverse_pe`、`source_coefficient` 和 `dt_star`；`pi_q` 仅作为兼容别名写入模型配置。
- `config.py`：定义 `TrainConfig`，管理学习率、epoch、TBPTT 窗口、warmup、梯度裁剪、提前停止、设备和随机种子配置。
- `static_topology.py`：实现固定拓扑训练所需的 `StaticGraphState`、`GpuFeatureBuilder`、`train_static_topology` 和 FEM 监督路径。
- `tbptt.py`：实现 TBPTT 窗口切分、窗口内自回归 rollout 和窗口损失计算。
- `trainer.py`：提供基于图序列的常规训练循环。
- `warmup.py`：使用当前 PDGCN 权重进行伪时间松弛，生成训练初始温度。
- `inference.py`：提供训练后自回归推理接口，并可将无量纲温度还原为真实温度。
- `checkpoint.py`：保存和恢复模型、优化器与元信息。
- `monitor.py`：训练过程中的损失记录与历史写入；监督启用时记录 `loss_temperature`、`loss_rollout_temperature`、`fem_temperature_rmse` 等 FEM 指标。
- `graph_utils.py`：封装图对象温度字段、边界节点、物理热源温升、显式推进增量、residual 源增量和设备迁移等工具函数。
- `tests/`：覆盖配置、checkpoint、推理、静态拓扑、TBPTT 和训练循环。

## 与其他模块的关系

`training` 从 `configs` 读取实验配置，从 `data` 获取图数据、缓存和 FEM 监督标签，从 `models` 构建 PDGCN，并调用 `pde` 计算物理损失。监督温度损失在 `training/static_topology.py` 中与物理损失合并。训练输出默认写入 `runs`。

本代码库的 Python 环境为 conda 的 `PIGNN` 环境；运行训练或测试时应使用该环境中的 Python。
