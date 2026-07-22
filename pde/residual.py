from typing import Tuple

import torch


def compute_pde_residual(
    T_next,
    T_current,
    v_scan_star,
    dt_star,
    edge_index,
    edge_attr,
    *,
    Q_star=None,
    delta_t_source_star=None,
    inverse_pe: float = 1.0,
    pi_q: float = 1.0,
    k_ratio: float = 0.05,
    thermal_loss_beta: float = 0.0,
    thermal_loss_base_temperature_star=0.0,
    residual_time_scheme: str = "explicit",
    convection_scale: float = 2.0,
    eps: float = 1e-12,
):
    """计算每个节点的无源曲面内输运 PDE 残差。

    边特征布局为 [dx, dy, dz, d, cos_theta, cos_phi, cos_phi_sq]。
    对流项采用带符号边方向贡献：对每个接收节点按距离倒数归一化
    邻居权重，并累加速度在边方向上的带符号投影与边方向温度差分。
    输入可以是单步张量（[N]、[N, 1]），也可以是 TBPTT 窗口张量
    （[K, N]、[K, N, 1]）。返回残差的形状会与 ``T_next`` 保持一致。

    参数:
        T_next: 下一步无量纲温度，形状 ``[N]``、``[N, 1]``、``[K, N]``
            或 ``[K, N, 1]``。
        T_current: 当前无量纲温度，形状同 ``T_next``，也可用单步温度广播到窗口。
        v_scan_star: 无量纲扫描速度，标量张量、Python 标量或长度为 ``K`` 的张量。
        Q_star: 兼容旧调用的保留参数；不直接使用。
        delta_t_source_star: 可选的当前步无量纲热源温升，形状同温度或可广播。
            仅在取消显式热源推进、由网络预测完整温度增量时传入。
        dt_star: 无量纲时间步长，标量。
        edge_index: 图边索引，形状 ``[2, E]``。
        edge_attr: 原始边特征，形状 ``[E, >=7]``。
        inverse_pe: 佩克莱特数倒数 ``1 / Pe``。
        pi_q: 兼容旧调用的保留参数；无源残差中不再使用。
        k_ratio: 横向/纵向导热系数比 ``K_perp / K_parallel``。
        thermal_loss_beta: 兼容旧调用的保留参数；无源残差中不再使用。
        thermal_loss_base_temperature_star: 兼容旧调用的保留参数；无源残差中不再使用。
        residual_time_scheme: PDE 空间项和热耗散项的时间离散方式；
            ``explicit`` 使用 ``T_current``，``backward`` 使用 ``T_next``。
        eps: 数值下界，用于距离和时间步长防除零。

    返回:
        PDE 残差张量，形状与 ``T_next`` 保持一致。
    """

    T_next_2d, layout = _as_time_node(T_next, name="T_next")
    T_current_2d, _ = _as_time_node(T_current, name="T_current")
    T_current_2d = _broadcast_to_match(T_current_2d, T_next_2d, name="T_current")
    if delta_t_source_star is None:
        delta_t_source_2d = torch.zeros_like(T_current_2d)
    else:
        delta_t_source_2d, _ = _as_time_node(delta_t_source_star, name="delta_t_source_star")
        delta_t_source_2d = _broadcast_to_match(
            delta_t_source_2d,
            T_next_2d,
            name="delta_t_source_star",
        )
    _validate_graph(edge_index, edge_attr, T_next_2d.shape[1])

    device = T_next_2d.device
    dtype = T_next_2d.dtype
    T_current_2d = T_current_2d.to(device=device, dtype=dtype)
    delta_t_source_2d = delta_t_source_2d.to(device=device, dtype=dtype)
    edge_index = edge_index.to(device=device)
    edge_attr = edge_attr.to(device=device, dtype=dtype)
    T_eval_2d = _select_residual_temperature(
        residual_time_scheme,
        T_current_2d,
        T_next_2d,
    )

    sender = edge_index[0]
    receiver = edge_index[1]
    distance = edge_attr[:, 3].clamp_min(eps)
    cos_theta = edge_attr[:, 4]
    cos_phi_sq = edge_attr[:, 6]
    k_edge = cos_phi_sq + float(k_ratio) * (1.0 - cos_phi_sq)

    T_i = T_eval_2d[:, receiver]
    T_j = T_eval_2d[:, sender]

    v_scan = _as_time_scalar(v_scan_star, T_next_2d.shape[0], device=device, dtype=dtype)
    convection = _signed_directional_convection(
        T_eval_2d,
        v_scan,
        sender,
        receiver,
        distance,
        cos_theta,
        convection_scale=convection_scale,
        eps=eps,
    )
    diffusion_edge = k_edge.reshape(1, -1) * (T_j - T_i) / distance.square().reshape(1, -1)

    diffusion = torch.zeros_like(T_next_2d)
    diffusion.index_add_(1, receiver, diffusion_edge)

    transient = (T_next_2d - T_current_2d - delta_t_source_2d) / _as_scalar_tensor(
        dt_star,
        device=device,
        dtype=dtype,
    ).clamp_min(eps)
    residual = transient + convection - float(inverse_pe) * diffusion
    return _restore_layout(residual, layout)


def _signed_directional_convection(
    T_eval_2d,
    v_scan,
    sender,
    receiver,
    distance,
    cos_theta,
    *,
    convection_scale: float,
    eps: float,
):
    """按接收节点聚合带符号边方向对流贡献。

    当前边特征中的 ``cos_theta`` 使用接收节点切向速度方向与
    ``sender -> receiver`` 边方向的点积。对接收节点 ``i`` 和邻居
    ``j``，该方向为 ``x_i - x_j``，因此
    ``cos_theta * (T_i - T_j) / d`` 等价于文档中的
    ``(v_i · e_ij) * (T_j - T_i) / d``。
    """

    if sender.numel() == 0:
        return torch.zeros_like(T_eval_2d)

    inv_distance = distance.reciprocal()
    weight_sum = torch.zeros(T_eval_2d.shape[1], device=T_eval_2d.device, dtype=T_eval_2d.dtype)
    weight_sum.index_add_(0, receiver, inv_distance)
    alpha = inv_distance / weight_sum[receiver].clamp_min(float(eps))

    T_sender = T_eval_2d[:, sender]
    T_receiver = T_eval_2d[:, receiver]
    edge_contribution = (
        alpha.reshape(1, -1)
        * v_scan
        * cos_theta.reshape(1, -1)
        * (T_receiver - T_sender)
        / distance.reshape(1, -1)
    )

    convection = torch.zeros_like(T_eval_2d)
    convection.index_add_(1, receiver, edge_contribution)
    return convection * float(convection_scale)


def _as_time_node(value, *, name: str) -> Tuple[torch.Tensor, str]:
    """将温度/热源输入规范化为 ``[K, N]`` 形式。

    参数:
        value: 输入张量或数组，允许形状为 ``[N]``、``[N, 1]``、
            ``[K, N]`` 或 ``[K, N, 1]``。
        name: 参数名，用于错误信息。

    返回:
        ``(tensor_2d, layout)`` 二元组；``tensor_2d`` 形状为 ``[K, N]``，
        ``layout`` 记录原始布局以便恢复返回形状。
    """

    tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1), "node"
    if tensor.ndim == 2:
        if tensor.shape[1] == 1:
            return tensor.reshape(1, tensor.shape[0]), "node_col"
        return tensor, "time_node"
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        return tensor[:, :, 0], "time_node_col"
    raise ValueError(f"{name} must have shape [N], [N, 1], [K, N], or [K, N, 1], got {tuple(tensor.shape)}.")


def _restore_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    """根据记录的布局恢复张量形状。

    参数:
        value: 规范化计算后的二维张量，形状 ``[K, N]``。
        layout: ``_as_time_node`` 返回的布局标记。

    返回:
        恢复后的张量，形状为 ``[N]``、``[N, 1]``、``[K, N]``
        或 ``[K, N, 1]``。
    """

    if layout == "node":
        return value[0]
    if layout == "node_col":
        return value[0].reshape(-1, 1)
    if layout == "time_node":
        return value
    if layout == "time_node_col":
        return value.unsqueeze(-1)
    raise ValueError(f"Unsupported layout: {layout}.")


def _broadcast_to_match(value, target, *, name: str):
    """将单步节点张量广播到目标时间窗口形状。

    参数:
        value: 输入二维张量，形状 ``[1, N]`` 或 ``[K, N]``。
        target: 目标二维张量，形状 ``[K, N]``。
        name: 参数名，用于错误信息。

    返回:
        与 ``target`` 形状一致的张量；若 ``value`` 已匹配则原样返回。
    """

    if value.shape == target.shape:
        return value
    if value.shape[0] == 1 and value.shape[1] == target.shape[1]:
        return value.expand(target.shape[0], -1)
    raise ValueError(f"{name} shape {tuple(value.shape)} must match T_next shape {tuple(target.shape)}.")


def _as_base_temperature(value, target, *, device, dtype):
    """将基底温度转换为可与 ``target`` 广播的 ``[K, N]`` 张量。"""

    tensor = _as_scalar_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1)

    base_2d, _ = _as_time_node(tensor, name="thermal_loss_base_temperature_star")
    return _broadcast_to_match(
        base_2d.to(device=device, dtype=dtype),
        target,
        name="thermal_loss_base_temperature_star",
    )


def _select_residual_temperature(time_scheme: str, T_current, T_next):
    """按配置选择 PDE 空间项和热耗散项的评估温度。"""

    scheme = str(time_scheme).strip().lower().replace("-", "_")
    if scheme in ("explicit", "explicit_euler", "forward", "forward_euler"):
        return T_current
    if scheme in ("backward", "backward_euler", "implicit"):
        return T_next
    raise ValueError(
        "residual_time_scheme must be 'explicit' or 'backward', "
        f"got {time_scheme!r}."
    )


def _validate_graph(edge_index, edge_attr, num_nodes: int):
    """校验图拓扑和边特征是否可用于 PDE 残差计算。

    参数:
        edge_index: 图边索引张量，形状必须为 ``[2, E]``。
        edge_attr: 边特征张量，形状必须为 ``[E, >=7]``。
        num_nodes: 节点数量 ``N``，用于检查边索引是否越界。

    返回:
        None。若形状不合法或边索引越界则抛出 ``ValueError``。
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_attr.ndim != 2 or edge_attr.shape[1] < 7:
        raise ValueError(f"edge_attr must have shape [E, >=7], got {tuple(edge_attr.shape)}.")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr must have one row per edge_index column.")
    if edge_index.numel() > 0:
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= num_nodes:
            raise ValueError(f"edge_index values must be within [0, {num_nodes - 1}], got [{min_index}, {max_index}].")


def _as_time_scalar(value, num_steps: int, *, device, dtype):
    """将标量或时间序列速度转换为 ``[K, 1]`` 张量。

    参数:
        value: Python 标量或 ``torch.Tensor``；可为单个值或长度为 ``num_steps``。
        num_steps: 时间步数量 ``K``。
        device: 输出张量设备。
        dtype: 输出张量数据类型。

    返回:
        速度张量，形状为 ``[1, 1]`` 或 ``[K, 1]``。
    """

    tensor = _as_scalar_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1)
    if tensor.numel() == num_steps:
        return tensor.reshape(num_steps, 1)
    raise ValueError(f"v_scan_star must be scalar or have {num_steps} values, got {tensor.numel()}.")


def _as_scalar_tensor(value, *, device, dtype):
    """将标量值转换为指定设备和类型的张量。

    参数:
        value: Python 标量、数组或 ``torch.Tensor``。
        device: 输出张量设备。
        dtype: 输出张量数据类型。

    返回:
        ``torch.Tensor``，位于 ``device`` 且类型为 ``dtype``。
    """

    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)
