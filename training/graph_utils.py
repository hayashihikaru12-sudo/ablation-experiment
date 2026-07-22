from typing import Dict

import torch

from pde import compute_surface_source_delta_star


TEMPERATURE_SLICE = slice(6, 7)
Q_FEATURE_INDEX_ATTR = "q_feature_index"
DELTA_T_SOURCE_FEATURE_INDEX_ATTR = "delta_t_source_feature_index"
MISSING_FEATURE_INDEX = -1


def node_feature_indices_from_config(model_config):
    """根据模型配置返回可选节点特征列索引。"""

    next_index = 7
    q_index = MISSING_FEATURE_INDEX
    delta_t_source_index = MISSING_FEATURE_INDEX
    if bool(getattr(model_config, "include_q_in_features", False)):
        q_index = next_index
        next_index += 1
    if bool(getattr(model_config, "include_delta_t_source_in_features", False)):
        delta_t_source_index = next_index
    return q_index, delta_t_source_index


def set_graph_node_feature_layout(graph, model_config):
    """在图对象上记录可选节点特征的列布局。"""

    q_index, delta_t_source_index = node_feature_indices_from_config(model_config)
    graph.q_feature_index = int(q_index)
    graph.delta_t_source_feature_index = int(delta_t_source_index)
    graph.include_q_in_features = q_index >= 0
    graph.include_delta_t_source_in_features = delta_t_source_index >= 0
    return graph


def clone_graph_with_temperature(graph, temperature_star, *, delta_t_source_star=None):
    """复制图对象并替换节点温度和当前步显式源温升特征。

    参数:
        graph: PyG ``Data`` 图对象，``x`` 至少包含第 7 列温度特征。
        temperature_star: 新的无量纲温度张量，形状 ``[N, 1]``。
        delta_t_source_star: 可选当前步显式热源温升，形状 ``[N, 1]``。

    返回:
        新的图对象，``x[:, 6:7]`` 已替换为 ``temperature_star``，
        若图包含 ``delta_t_source`` 特征列，也会同步写入该列；
        其他字段继承自输入图。
    """

    cloned = graph.clone()
    cloned.x = graph.x.clone()
    temperature = temperature_star.to(device=cloned.x.device, dtype=cloned.x.dtype).reshape(graph.num_nodes, 1)
    cloned.x[:, TEMPERATURE_SLICE] = temperature
    delta_index = _feature_index(cloned, DELTA_T_SOURCE_FEATURE_INDEX_ATTR)
    if delta_index >= 0:
        if delta_t_source_star is None:
            delta_t_source = torch.zeros_like(temperature)
        else:
            delta_t_source = delta_t_source_star.to(device=cloned.x.device, dtype=cloned.x.dtype).reshape(
                graph.num_nodes,
                1,
            )
        cloned.x[:, delta_index : delta_index + 1] = delta_t_source
    return cloned


def graph_temperature(graph):
    """读取图中的节点无量纲温度特征。

    参数:
        graph: PyG ``Data`` 图对象，``x`` 形状 ``[N, >=7]``。

    返回:
        温度特征张量，形状 ``[N, 1]``，对应 ``x[:, 6:7]``。
    """

    return graph.x[:, TEMPERATURE_SLICE]


def graph_surface_heat_source(graph):
    """读取图中的无量纲表面热流。

    参数:
        graph: PyG ``Data`` 图对象，优先读取 ``q_surface_star`` 属性。

    返回:
        表面热流张量，形状 ``[N, 1]``。若图上没有热源字段，则返回零张量。
    """

    if hasattr(graph, "q_surface_star"):
        return graph.q_surface_star.reshape(graph.num_nodes, 1)
    if hasattr(graph, "q_star"):
        return graph.q_star.reshape(graph.num_nodes, 1)
    return torch.zeros_like(graph_temperature(graph))


def graph_explicit_source_delta(graph, model_config):
    """计算当前图的显式表面热源温升 ``delta_T_Q*``。"""

    if not bool(getattr(model_config, "use_explicit_heat_source", True)):
        return torch.zeros_like(graph_temperature(graph))

    source_coefficient = getattr(
        model_config,
        "source_coefficient",
        getattr(model_config, "pi_q", 0.0),
    )
    return compute_surface_source_delta_star(
        graph_surface_heat_source(graph),
        dt_star=getattr(model_config, "dt_star", 1.0),
        source_coefficient=source_coefficient,
        absorptivity=getattr(model_config, "heat_source_absorptivity", 1.0),
    ).to(device=graph.x.device, dtype=graph.x.dtype)


def _feature_index(graph, attr_name: str) -> int:
    index = int(getattr(graph, attr_name, MISSING_FEATURE_INDEX))
    if index < 0:
        return MISSING_FEATURE_INDEX
    if graph.x.ndim != 2 or graph.x.shape[1] <= index:
        return MISSING_FEATURE_INDEX
    return index


def graph_boundary_nodes(graph) -> Dict[str, torch.Tensor]:
    """从图对象中提取边界节点索引字典。

    参数:
        graph: PyG ``Data`` 图对象，需包含 ``upwind_nodes``、``side_nodes``、
            ``downwind_nodes`` 属性。

    返回:
        字典 ``{"upwind": ..., "side": ..., "downwind": ...}``，
        每个值为一维 ``torch.LongTensor`` 节点索引。
    """

    return {
        "upwind": graph.upwind_nodes,
        "side": graph.side_nodes,
        "downwind": graph.downwind_nodes,
    }


def graph_to_device(graph, device):
    """将图对象移动到指定设备。

    参数:
        graph: PyG ``Data`` 图对象。
        device: 目标设备；若为 ``None`` 则不移动。

    返回:
        位于目标设备的图对象；当 ``device=None`` 时返回原对象。
    """

    if device is None:
        return graph
    return graph.to(device)
