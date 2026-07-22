import torch
import torch.nn as nn
from torch_geometric.data import Data

from .config import PDGCNConfig
from .mlp import build_mlp


class Encoder(nn.Module):
    def __init__(self, config: PDGCNConfig):
        """初始化节点和边特征编码器。

        参数:
            config: ``PDGCNConfig``，提供输入维度、隐空间维度、归一化和 dropout 配置。

        返回:
            None。实例会创建 ``node_encoder`` 和 ``edge_encoder`` 两个 MLP。
        """

        super().__init__()
        self.config = config
        self.node_encoder = build_mlp(
            config.encoder_node_input_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.edge_encoder = build_mlp(
            config.edge_input_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )

    def forward(self, graph: Data) -> Data:
        """将原始图特征编码到统一隐空间。

        参数:
            graph: PyG ``Data`` 图对象；``x`` 形状为 ``[N, node_input_size]``，
                ``edge_attr`` 形状为 ``[E, edge_input_size]``；当
                ``include_global=True`` 时还需 ``global_attr``。

        返回:
            新的 PyG ``Data`` 图对象，``x`` 形状为 ``[N, hidden_size]``，
            ``edge_attr`` 形状为 ``[E, hidden_size]``，其他图属性被保留。
        """

        graph = _copy_data(graph)
        node_attr = graph.x
        if self.config.include_global:
            node_attr = _append_global_condition(node_attr, graph.global_attr)

        graph.x = self.node_encoder(node_attr)
        graph.edge_attr = self.edge_encoder(graph.edge_attr)
        return graph


def _append_global_condition(node_attr: torch.Tensor, global_attr: torch.Tensor) -> torch.Tensor:
    """将全局工艺条件广播并拼接到每个节点特征。

    参数:
        node_attr: 节点特征张量，形状 ``[N, F_node]``。
        global_attr: 全局特征张量，形状通常为 ``[G]`` 或 ``[1, G]``。

    返回:
        拼接后的节点特征张量，形状 ``[N, F_node + G]``。
    """

    if global_attr is None:
        raise ValueError("graph.global_attr is required when include_global=True.")

    global_flat = global_attr.reshape(1, -1).to(device=node_attr.device, dtype=node_attr.dtype)
    global_per_node = global_flat.expand(node_attr.shape[0], -1)
    return torch.cat([node_attr, global_per_node], dim=-1)


def _copy_data(graph: Data) -> Data:
    """浅复制 PyG ``Data`` 对象的已有字段。

    参数:
        graph: 输入 PyG ``Data`` 对象。

    返回:
        新的 ``Data`` 对象，包含与输入相同的键值；张量本身保持引用语义。
    """

    keys = graph.keys() if callable(graph.keys) else graph.keys
    return Data(**{key: graph[key] for key in keys})
