# models 模块说明

`models` 目录实现 PDGCN 神经网络主体。该模型改造自 PIGNN 风格的图网络结构；默认预测无源曲面内输运增量，取消显式热源推进时预测包含热源贡献的完整温度增量。

## 网络结构

PDGCN 由三段组成：

- `Encoder`：将原始节点特征和边特征编码到统一隐藏维度。
- `Processor`：堆叠多层图网络消息传递块，在边消息和节点状态之间传播曲面热场信息。
- `Decoder`：将节点隐藏状态解码为温度残差或温度增量预测。

## 主要文件

- `pdgcn.py`：定义顶层 `PDGCN` 模型，串联 encoder、processor 和 decoder。
- `config.py`：定义 `PDGCNConfig`，集中管理输入维度、隐藏维度、消息传递层数、物理损失系数和时间离散参数。
- `encoder.py`：实现节点和边特征编码，并可将全局扫描速度拼接到每个节点。
- `decoder.py`：实现节点级输出头，默认输出单通道 `delta_T*`。
- `mlp.py`：构建模块内复用的 MLP。
- `processor/edge_block.py`：实现边消息计算，包含迎风门控和基于纤维方向的各向异性门控。
- `processor/node_block.py`：对入边消息进行聚合，并更新节点隐藏状态。
- `processor/__init__.py`：定义 `GnBlock` 和 `Processor`，负责多层消息传递与残差连接。

## 输入输出约定

模型输入是 PyTorch Geometric `Data` 对象，通常包含：

- `x`：节点特征，默认形状 `[N, 7]`，布局为 `[x*, y*, z*, fx, fy, fz, T*]`；启用热源特征开关后可为 `[N, 8]` 或 `[N, 9]`，追加 `q*` 和/或 `ΔT_Q*`。
- `edge_index`：图边索引，形状 `[2, E]`。
- `edge_attr`：边特征，形状 `[E, 7]`。
- `global_attr`：全局工艺条件，当前主要是无量纲扫描速度。

模型输出为形状 `[N, 1]` 的无量纲温度增量。`use_explicit_heat_source=true` 时热源温升在调用 PD-GCN 前加入，输出只承担面内输运；设为 `false` 时模型输入不预加热源，`q*`、物理 `ΔT_Q*` 和带源 residual 保持有效，输出承担完整增量。厚度方向传热由 FDM 模块处理。
