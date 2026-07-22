# pde 模块说明

`pde` 目录实现 PDGCN 训练和推理所需的物理算子，包括默认无源输运及消融用带源残差、表面热源温升、厚度方向 FDM、Dirichlet 边界处理、出流边界软约束和总损失函数。

## 主要文件

- `residual.py`：实现 `compute_pde_residual`；默认计算源项已显式处理后的无源输运残差，也可接收 `delta_t_source_star` 约束完整增量预测。
- `source.py`：实现显式表面热源温升，将 `q''` 或 `q_surface*` 转换为 `delta_T_Q*`。
- `fdm.py`：对外导出厚度方向 FDM 系数、显式层间温度增量诊断函数和 Backward Euler 隐式 FDM 步进函数。
- `loss.py`：实现边界条件和损失聚合，包括 `apply_dirichlet_boundary`、`compute_outflow_loss` 和 `total_loss`。
- `tests/`：包含 PDE 残差和损失函数的单元测试，用于验证形状广播、边界处理和损失分量。
- `__init__.py`：对外导出 PDE 模块的核心接口。

## 物理建模约定

边特征布局遵循 `data` 模块生成的 `[dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]`。其中 `cos_theta` 表示接收节点切向扫描方向与 `sender -> receiver` 边方向的夹角余弦，用于带符号边方向对流项；残差按接收节点聚合距离倒数归一化后的邻居贡献，不使用 `ReLU(cos_theta)` 截断反方向邻居。`cos_phi_sq` 和 `k_ratio` 用于构造沿纤维方向的各向异性导热权重。

`upwind` 和 `side` 节点会被钳制到 Dirichlet 温度，默认无量纲值为 `0.0`。`downwind` 节点用于出流边界 Neumann 软约束。PDE 残差支持单步张量和 TBPTT 窗口张量，返回形状会与输入温度保持一致；默认不含热源项，取消显式推进时通过可选 `delta_t_source_star` 在瞬态项中扣除当前步源温升。

## 与训练模块的关系

`training/tbptt.py` 和 `training/static_topology.py` 根据 `use_explicit_heat_source` 选择显式源—输运分裂或完整增量模式，并把匹配模式的源增量传给 `total_loss`。FEM 温度监督损失不在 `pde.total_loss` 中实现，而是在固定拓扑训练循环中与物理损失合并。多层推理通过 `inference/fdm.py` 使用 Backward Euler 隐式厚度 FDM。
