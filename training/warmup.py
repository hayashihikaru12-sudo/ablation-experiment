import torch

from pde import apply_dirichlet_boundary

from .graph_utils import (
    clone_graph_with_temperature,
    graph_boundary_nodes,
    graph_explicit_source_delta,
    graph_physical_source_delta,
    graph_temperature,
)


@torch.no_grad()
def pseudo_time_relax_initial_temperature(model, graph, warmup_steps: int):
    """使用当前模型权重自回归生成伪时间松弛初温。

    参数:
        model: PD-GCN 模型，输入单步图并输出 ``delta_T*``。
        graph: 冻结几何、热源、边特征和全局速度的 PyG 图对象。
        warmup_steps: 伪时间松弛步数；为 ``0`` 时返回图中已有温度。

    返回:
        无量纲初始温度张量，形状 ``[N, 1]``，已与 warmup 计算图断开。
    """

    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    if int(warmup_steps) == 0:
        return graph_temperature(graph).detach()

    current_temperature = torch.zeros_like(graph_temperature(graph))
    current_temperature = apply_dirichlet_boundary(
        current_temperature,
        graph_boundary_nodes(graph),
        value=getattr(model.config, "dirichlet_temperature_star", 0.0),
    )

    was_training = model.training
    model.eval()
    try:
        for _ in range(int(warmup_steps)):
            physical_source_delta = graph_physical_source_delta(graph, model.config)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + delta_t_source,
                graph_boundary_nodes(graph),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph_step = clone_graph_with_temperature(
                graph,
                source_temperature,
                delta_t_source_star=physical_source_delta,
            )
            delta_temperature = model(graph_step)
            current_temperature = apply_dirichlet_boundary(
                source_temperature + delta_temperature,
                graph_boundary_nodes(graph_step),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
    finally:
        if was_training:
            model.train()

    return current_temperature.detach()
