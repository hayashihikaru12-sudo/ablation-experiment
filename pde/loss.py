from typing import Dict, Iterable, Optional

import torch

from .residual import (
    _as_time_node,
    _broadcast_to_match,
    _restore_layout,
    compute_pde_residual,
)


def apply_dirichlet_boundary(T, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, value: float = 0.0):
    """将迎风和侧边界节点钳制为 Dirichlet 温度。

    参数:
        T: 无量纲温度张量，允许形状 ``[N]``、``[N, 1]``、``[K, N]``
            或 ``[K, N, 1]``。
        boundary_nodes: 边界节点字典；函数使用其中 ``upwind`` 和 ``side``。
        value: 无量纲 Dirichlet 温度值，默认 ``0.0`` 表示环境温度。

    返回:
        与 ``T`` 形状一致的新张量；指定边界节点已被替换为 ``value``。
    """

    nodes = _concat_boundary_nodes(boundary_nodes, ("upwind", "side"), device=torch.as_tensor(T).device)
    if nodes.numel() == 0:
        return T.clone()

    output = T.clone()
    fill_value = torch.as_tensor(value, device=output.device, dtype=output.dtype)
    if output.ndim == 1:
        output[nodes] = fill_value
    elif output.ndim == 2 and output.shape[1] == 1:
        output[nodes, :] = fill_value
    elif output.ndim == 2:
        output[:, nodes] = fill_value
    elif output.ndim == 3 and output.shape[2] == 1:
        output[:, nodes, :] = fill_value
    else:
        raise ValueError(f"T must have shape [N], [N, 1], [K, N], or [K, N, 1], got {tuple(output.shape)}.")
    return output


def compute_outflow_loss(T, edge_index, edge_attr, outflow_nodes, *, eps: float = 1e-12):
    """计算尾迹出流边界的无量纲 Neumann 软约束损失。

    参数:
        T: 无量纲温度张量，允许形状 ``[N]``、``[N, 1]``、``[K, N]``
            或 ``[K, N, 1]``。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``，使用 ``d`` 和 ``cos_theta``。
        outflow_nodes: 尾迹边界节点索引，一维张量或可转为张量的序列。
        eps: 分母下界，用于避免除零。

    返回:
        标量张量，表示出流边界法向温度梯度平方均值。
    """

    T_2d, _ = _as_time_node(T, name="T")
    device = T_2d.device
    dtype = T_2d.dtype
    outflow_nodes = torch.as_tensor(outflow_nodes, device=device, dtype=torch.long).reshape(-1)
    if outflow_nodes.numel() == 0:
        return T_2d.new_zeros(())

    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)
    sender = edge_index[0]
    receiver = edge_index[1]
    distance = edge_attr[:, 3].clamp_min(eps)
    weight = torch.relu(edge_attr[:, 4])

    edge_gradient = weight.reshape(1, -1) * (T_2d[:, receiver] - T_2d[:, sender]) / distance.reshape(1, -1)
    weighted_gradient_sum = torch.zeros_like(T_2d)
    weight_sum = torch.zeros_like(T_2d)
    weighted_gradient_sum.index_add_(1, receiver, edge_gradient)
    weight_sum.index_add_(1, receiver, weight.reshape(1, -1).expand(T_2d.shape[0], -1))

    normal_gradient = weighted_gradient_sum[:, outflow_nodes] / weight_sum[:, outflow_nodes].clamp_min(eps)
    return normal_gradient.square().mean()


def compute_graph_gradient_loss(T, edge_index, edge_attr, boundary_nodes=None, *, eps: float = 1e-12):
    """计算内部边上的一阶图梯度平滑损失。"""

    T_2d, _ = _as_time_node(T, name="T")
    device = T_2d.device
    dtype = T_2d.dtype
    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)
    if edge_index.numel() == 0:
        return T_2d.new_zeros(())

    sender = edge_index[0]
    receiver = edge_index[1]
    interior_nodes = _interior_nodes(T_2d.shape[1], boundary_nodes, device=device)
    if interior_nodes.numel() == 0:
        return T_2d.new_zeros(())

    interior_mask = torch.zeros(T_2d.shape[1], device=device, dtype=torch.bool)
    interior_mask[interior_nodes] = True
    edge_mask = interior_mask[sender] & interior_mask[receiver]
    if not bool(edge_mask.any().item()):
        return T_2d.new_zeros(())

    distance = edge_attr[edge_mask, 3].clamp_min(eps)
    gradient = (T_2d[:, receiver[edge_mask]] - T_2d[:, sender[edge_mask]]) / distance.reshape(1, -1)
    return gradient.square().mean()


def total_loss(
    *,
    T_next,
    T_current,
    v_scan_star,
    Q_star=None,
    q_surface_star=None,
    delta_t_source_star=None,
    dt_star,
    edge_index,
    edge_attr,
    boundary_nodes: Optional[Dict[str, torch.Tensor]] = None,
    lambda_pde: float = 1.0,
    inverse_pe: float = 1.0,
    pi_q: float = 1.0,
    k_ratio: float = 0.05,
    lambda_outflow: float = 1.0,
    gradient_regularization: float = 0.0,
    dirichlet_temperature_star: float = 0.0,
    thermal_loss_beta: float = 0.0,
    thermal_loss_base_temperature_star=0.0,
    residual_time_scheme: str = "explicit",
    convection_scale: float = 2.0,
    adaptive_pde_node_weight_enabled: bool = False,
    adaptive_pde_node_weight_scheme: str = "heat_flux",
    adaptive_pde_node_weight_min: float = 0.2,
    pde_node_weight_temperature_star=None,
    pde_node_weight_epoch: int = 0,
    temperature_pde_node_weight_beta: float = 0.5,
    temperature_pde_node_weight_max: float = 8.0,
    temperature_pde_node_weight_clamp_enabled: bool = True,
    temperature_pde_node_weight_threshold: float = 1.0,
    temperature_pde_node_weight_high: float = 4.0,
    adaptive_pde_node_weight_warmup_enabled: bool = False,
    adaptive_pde_node_weight_warmup_epochs: int = 50,
    return_components: bool = False,
    eps: float = 1e-12,
):
    """计算 PD-GCN 纯物理总损失。

    参数:
        T_next: 下一步无量纲温度预测，形状 ``[N]``、``[N, 1]``、
            ``[K, N]`` 或 ``[K, N, 1]``。
        T_current: 当前无量纲温度，形状同 ``T_next`` 或可广播到 ``T_next``。
        v_scan_star: 无量纲扫描速度，标量或长度为 ``K`` 的张量。
        Q_star: 兼容旧调用的保留参数；不直接使用。
        delta_t_source_star: 可选的当前步无量纲热源温升。取消显式热源推进时，
            用它约束网络输出的完整温度增量；显式推进模式保持为 ``None`` 或零。
        dt_star: 无量纲时间步长。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``。
        boundary_nodes: 边界节点字典，使用 ``upwind``、``side`` 和 ``downwind``。
        inverse_pe: 佩克莱特数倒数。
        pi_q: 兼容旧调用的保留参数；无源残差中不再使用。
        k_ratio: 横向/纵向导热系数比。
        lambda_pde: PDE 残差损失权重。
        lambda_outflow: 出流边界损失权重。
        gradient_regularization: 图梯度平滑损失权重，用于抑制预测温度的高频振荡。
        dirichlet_temperature_star: 硬 Dirichlet 边界的无量纲温度值。
        thermal_loss_beta: 兼容旧调用的保留参数；无源残差中不再使用。
        thermal_loss_base_temperature_star: 兼容旧调用的保留参数；无源残差中不再使用。
        residual_time_scheme: PDE 空间项和热耗散项的时间离散方式；
            ``explicit`` 使用当前温度，``backward`` 使用预测温度。
        return_components: 是否返回损失分量和中间张量。
        eps: 数值下界，用于防止除零。

    返回:
        若 ``return_components=False``，返回标量总损失张量；
        否则返回字典，包含 ``loss_total``、``loss_pde``、``loss_outflow``、
        ``loss_smooth``、``residual`` 和 ``T_next_bc``。
    """

    T_next_bc = apply_dirichlet_boundary(
        T_next,
        boundary_nodes,
        value=dirichlet_temperature_star,
    )
    residual = compute_pde_residual(
        T_next=T_next_bc,
        T_current=T_current,
        v_scan_star=v_scan_star,
        dt_star=dt_star,
        edge_index=edge_index,
        edge_attr=edge_attr,
        Q_star=Q_star,
        delta_t_source_star=delta_t_source_star,
        inverse_pe=inverse_pe,
        pi_q=pi_q,
        k_ratio=k_ratio,
        thermal_loss_beta=thermal_loss_beta,
        thermal_loss_base_temperature_star=thermal_loss_base_temperature_star,
        residual_time_scheme=residual_time_scheme,
        convection_scale=convection_scale,
        eps=eps,
    )

    residual_2d, _ = _as_time_node(residual, name="residual")
    T_next_bc_2d, t_next_layout = _as_time_node(T_next_bc, name="T_next_bc")
    thermal_loss_term_2d = torch.zeros_like(T_next_bc_2d.to(device=residual_2d.device, dtype=residual_2d.dtype))

    interior_nodes = _interior_nodes(residual_2d.shape[1], boundary_nodes, device=residual_2d.device)
    if interior_nodes.numel() == 0:
        loss_pde = residual_2d.square().mean()
        loss_beta = thermal_loss_term_2d.square().mean()
    else:
        loss_pde = _compute_pde_loss(
            residual_2d,
            interior_nodes,
            q_surface_star=q_surface_star,
            temperature_star=pde_node_weight_temperature_star,
            adaptive_enabled=adaptive_pde_node_weight_enabled,
            scheme=adaptive_pde_node_weight_scheme,
            min_weight=adaptive_pde_node_weight_min,
            temperature_beta=temperature_pde_node_weight_beta,
            temperature_max=temperature_pde_node_weight_max,
            temperature_clamp_enabled=temperature_pde_node_weight_clamp_enabled,
            temperature_threshold=temperature_pde_node_weight_threshold,
            temperature_high=temperature_pde_node_weight_high,
            warmup_enabled=adaptive_pde_node_weight_warmup_enabled,
            warmup_epochs=adaptive_pde_node_weight_warmup_epochs,
            epoch=pde_node_weight_epoch,
            eps=eps,
        )
        loss_beta = thermal_loss_term_2d[:, interior_nodes].square().mean()

    outflow_nodes = _concat_boundary_nodes(boundary_nodes, ("downwind",), device=residual_2d.device)
    loss_outflow = compute_outflow_loss(T_next_bc, edge_index, edge_attr, outflow_nodes, eps=eps)
    loss_smooth = compute_graph_gradient_loss(T_next_bc, edge_index, edge_attr, boundary_nodes, eps=eps)
    loss_total = float(lambda_pde) * loss_pde + float(lambda_outflow) * loss_outflow + float(gradient_regularization) * loss_smooth

    if not return_components:
        return loss_total

    return {
        "loss_total": loss_total,
        "loss_pde": loss_pde,
        "loss_transport": loss_pde,
        "loss_outflow": loss_outflow,
        "loss_beta": loss_beta,
        "loss_smooth": loss_smooth,
        "residual": residual,
        "T_next_bc": T_next_bc,
        "thermal_loss_term": _restore_layout(thermal_loss_term_2d, t_next_layout),
    }


def _concat_boundary_nodes(boundary_nodes: Optional[Dict[str, torch.Tensor]], names: Iterable[str], *, device):
    """从边界字典中合并指定类别节点。

    参数:
        boundary_nodes: 边界节点字典，值为一维索引张量；可为 ``None``。
        names: 要合并的边界名称序列。
        device: 返回索引张量所在设备。

    返回:
        一维 ``torch.LongTensor``，包含指定边界类别的唯一索引；
        若没有可用边界则返回空张量。
    """

    if not boundary_nodes:
        return torch.empty(0, device=device, dtype=torch.long)
    selected = [
        torch.as_tensor(boundary_nodes[name], device=device, dtype=torch.long).reshape(-1)
        for name in names
        if name in boundary_nodes and boundary_nodes[name] is not None
    ]
    if not selected:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.unique(torch.cat(selected, dim=0))


def _compute_pde_loss(
    residual_2d,
    interior_nodes,
    *,
    q_surface_star,
    temperature_star,
    adaptive_enabled: bool,
    scheme: str,
    min_weight: float,
    temperature_beta: float,
    temperature_max: float,
    temperature_clamp_enabled: bool,
    temperature_threshold: float,
    temperature_high: float,
    warmup_enabled: bool,
    warmup_epochs: int,
    epoch: int,
    eps: float,
):
    residual_interior = residual_2d[:, interior_nodes]
    scheme = str(scheme).strip().lower().replace("-", "_")
    if not adaptive_enabled or scheme == "none":
        return residual_interior.square().mean()

    if scheme == "heat_flux":
        weights = _heat_flux_pde_weights(
            q_surface_star,
            residual_2d,
            min_weight=float(min_weight),
            eps=float(eps),
        )
    elif scheme == "temperature_continuous_clamped":
        weights = _temperature_continuous_clamped_weights(
            temperature_star,
            residual_2d,
            beta=float(temperature_beta),
            max_weight=float(temperature_max),
            clamp_enabled=bool(temperature_clamp_enabled),
        )
        weights = _apply_temperature_weight_warmup(
            weights,
            warmup_enabled=bool(warmup_enabled),
            warmup_epochs=int(warmup_epochs),
            epoch=int(epoch),
        )
    elif scheme == "temperature_hard_threshold":
        weights = _temperature_hard_threshold_weights(
            temperature_star,
            residual_2d,
            threshold=float(temperature_threshold),
            high_weight=float(temperature_high),
        )
        weights = _apply_temperature_weight_warmup(
            weights,
            warmup_enabled=bool(warmup_enabled),
            warmup_epochs=int(warmup_epochs),
            epoch=int(epoch),
        )
    else:
        raise ValueError(
            "adaptive_pde_node_weight_scheme must be one of 'none', 'heat_flux', "
            "'temperature_continuous_clamped', or 'temperature_hard_threshold', "
            f"got {scheme!r}."
        )

    weights = weights[:, interior_nodes]
    normalized_weights = _normalize_pde_node_weights(weights, interior_nodes.numel(), eps=eps).detach()
    return (normalized_weights * residual_interior.square()).sum(dim=1).mean()


def _heat_flux_pde_weights(q_surface_star, residual_2d, *, min_weight: float, eps: float):
    if q_surface_star is None:
        return torch.ones_like(residual_2d)

    q_2d, _ = _as_time_node(q_surface_star, name="q_surface_star")
    q_2d = q_2d.to(device=residual_2d.device, dtype=residual_2d.dtype)
    q_2d = _broadcast_to_match(q_2d, residual_2d, name="q_surface_star")
    q_abs = q_2d.abs()
    max_abs = q_abs.max(dim=1, keepdim=True).values
    return float(min_weight) + (1.0 - float(min_weight)) * q_abs / max_abs.add(float(eps))


def _temperature_continuous_clamped_weights(
    temperature_star,
    residual_2d,
    *,
    beta: float,
    max_weight: float,
    clamp_enabled: bool,
):
    temperature_2d = _temperature_weight_signal(temperature_star, residual_2d)
    weights = 1.0 + float(beta) * torch.relu(temperature_2d)
    weights = weights.clamp_min(1.0)
    if bool(clamp_enabled):
        weights = weights.clamp(max=float(max_weight))
    return weights


def _temperature_hard_threshold_weights(temperature_star, residual_2d, *, threshold: float, high_weight: float):
    temperature_2d = _temperature_weight_signal(temperature_star, residual_2d)
    high = torch.full_like(temperature_2d, float(high_weight))
    low = torch.ones_like(temperature_2d)
    return torch.where(temperature_2d > float(threshold), high, low)


def _temperature_weight_signal(temperature_star, residual_2d):
    if temperature_star is None:
        raise ValueError(
            "pde_node_weight_temperature_star is required when adaptive_pde_node_weight_scheme "
            "uses a temperature-based scheme."
        )
    temperature_2d, _ = _as_time_node(temperature_star, name="pde_node_weight_temperature_star")
    temperature_2d = temperature_2d.to(device=residual_2d.device, dtype=residual_2d.dtype)
    return _broadcast_to_match(temperature_2d, residual_2d, name="pde_node_weight_temperature_star")


def _apply_temperature_weight_warmup(weights, *, warmup_enabled: bool, warmup_epochs: int, epoch: int):
    if not warmup_enabled:
        return weights
    if int(warmup_epochs) <= 0:
        raise ValueError(f"adaptive_pde_node_weight_warmup_epochs must be positive, got {warmup_epochs}.")
    if int(epoch) < int(warmup_epochs):
        return torch.ones_like(weights)
    if int(epoch) < 2 * int(warmup_epochs):
        factor = float(int(epoch) - int(warmup_epochs)) / float(warmup_epochs)
        return 1.0 + factor * (weights - 1.0)
    return weights


def _normalize_pde_node_weights(weights, num_nodes: int, *, eps: float):
    weight_sum = weights.sum(dim=1, keepdim=True)
    uniform_weights = torch.full_like(weights, 1.0 / float(num_nodes))
    normalized_weights = torch.where(
        weight_sum > float(eps),
        weights / weight_sum.clamp_min(float(eps)),
        uniform_weights,
    ).detach()
    return normalized_weights


def _interior_nodes(num_nodes: int, boundary_nodes: Optional[Dict[str, torch.Tensor]], *, device):
    """计算不属于任意边界的内部节点索引。

    参数:
        num_nodes: 图节点数量 ``N``。
        boundary_nodes: 边界节点字典；使用 ``upwind``、``side`` 和 ``downwind``。
        device: 返回索引张量所在设备。

    返回:
        一维 ``torch.LongTensor``，包含内部节点索引；若未提供边界则返回所有节点。
    """

    all_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    boundary = _concat_boundary_nodes(boundary_nodes, ("upwind", "side", "downwind"), device=device)
    if boundary.numel() == 0:
        return all_nodes
    mask = torch.ones(num_nodes, device=device, dtype=torch.bool)
    mask[boundary] = False
    return all_nodes[mask]
