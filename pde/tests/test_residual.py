import torch

from pde.residual import compute_pde_residual


def _edge_attr(distance, cos_theta, cos_phi_sq):
    """构造用于残差测试的边特征张量。

    参数:
        distance: 边距离列表，长度为 ``E``。
        cos_theta: 边与扫描方向夹角余弦列表，长度为 ``E``。
        cos_phi_sq: 边与纤维方向夹角余弦平方列表，长度为 ``E``。

    返回:
        ``torch.FloatTensor``，形状 ``[E, 7]``，仅测试所需列被赋值。
    """

    rows = []
    for d, theta, phi_sq in zip(distance, cos_theta, cos_phi_sq):
        rows.append([0.0, 0.0, 0.0, d, theta, 0.0, phi_sq])
    return torch.tensor(rows, dtype=torch.float32)


def test_compute_pde_residual_matches_hand_calculation():
    """验证 PDE 残差计算与手工推导结果一致。

    参数:
        None。

    返回:
        None。断言失败时由测试框架报告错误。
    """

    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    edge_attr = _edge_attr(
        distance=[2.0, 1.0],
        cos_theta=[1.0, -1.0],
        cos_phi_sq=[1.0, 0.0],
    )
    T_next = torch.tensor([[1.0], [3.0], [5.0]])
    T_current = torch.tensor([[0.0], [1.0], [2.0]])

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=2.0,
        dt_star=2.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.5,
        k_ratio=0.1,
        convection_scale=1.0,
    )

    expected = torch.tensor([[0.5], [2.7416668], [1.5]])
    assert residual.shape == T_next.shape
    assert torch.allclose(residual, expected, atol=1e-6)


def test_signed_directional_convection_is_receiver_aggregated():
    """验证带符号对流项只按接收节点聚合，不强制全局守恒。"""

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0], cos_theta=[1.0], cos_phi_sq=[1.0])
    T_current = torch.tensor([[0.0], [2.0]])

    residual = compute_pde_residual(
        T_next=T_current,
        T_current=T_current,
        v_scan_star=1.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.1,
        convection_scale=1.0,
    )

    expected = torch.tensor([[0.0], [2.0]])
    assert torch.allclose(residual, expected, atol=1e-6)
    assert not torch.allclose(residual.sum(), torch.tensor(0.0), atol=1e-6)


def test_signed_directional_convection_defaults_to_scale_compensation():
    """验证默认对流缩放补偿因子会放大带符号边方向贡献。"""

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0], cos_theta=[1.0], cos_phi_sq=[1.0])
    T_current = torch.tensor([[0.0], [2.0]])

    residual = compute_pde_residual(
        T_next=T_current,
        T_current=T_current,
        v_scan_star=1.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.1,
    )

    expected = torch.tensor([[0.0], [4.0]])
    assert torch.allclose(residual, expected, atol=1e-6)


def test_signed_directional_convection_keeps_negative_projection():
    """验证反方向邻居保留带符号贡献而不是被截断。"""

    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    edge_attr = _edge_attr(
        distance=[1.0, 1.0],
        cos_theta=[1.0, -1.0],
        cos_phi_sq=[1.0, 1.0],
    )
    T_current = torch.tensor([[0.0], [1.0], [2.0]])

    residual = compute_pde_residual(
        T_next=T_current,
        T_current=T_current,
        v_scan_star=1.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.1,
        convection_scale=1.0,
    )

    expected = torch.tensor([[0.0], [1.0], [0.0]])
    assert torch.allclose(residual, expected, atol=1e-6)


def test_signed_directional_convection_uses_inverse_distance_weights():
    """验证邻居权重按距离倒数在接收节点内归一化。"""

    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    edge_attr = _edge_attr(
        distance=[1.0, 3.0],
        cos_theta=[1.0, 1.0],
        cos_phi_sq=[1.0, 1.0],
    )
    T_current = torch.tensor([[0.0], [3.0], [6.0]])

    residual = compute_pde_residual(
        T_next=T_current,
        T_current=T_current,
        v_scan_star=1.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.1,
        convection_scale=1.0,
    )

    expected = torch.tensor([[0.0], [0.0], [4.75]])
    assert torch.allclose(residual, expected, atol=1e-6)


def test_compute_pde_residual_supports_tbptt_window_shape():
    """验证 PDE 残差函数支持 TBPTT 时间窗口输入形状。

    参数:
        None。

    返回:
        None。断言输出形状和有限性。
    """

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_attr = _edge_attr(distance=[1.0], cos_theta=[1.0], cos_phi_sq=[1.0])
    T_next = torch.tensor([[[1.0], [2.0]], [[2.0], [4.0]]])
    T_current = torch.zeros_like(T_next)

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=torch.tensor([1.0, 2.0]),
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=1.0,
        k_ratio=0.05,
        convection_scale=1.0,
    )

    assert residual.shape == T_next.shape
    assert torch.isfinite(residual).all()


def test_compute_pde_residual_is_source_free_transport_only():
    """验证 PDE 残差不再包含热源项或单层等效热耗散项。"""

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 7), dtype=torch.float32)
    T_next = torch.tensor([[2.0], [4.0]])
    T_current = torch.tensor([[1.0], [1.0]])

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=0.0,
        dt_star=1.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        k_ratio=0.05,
        thermal_loss_beta=0.5,
        thermal_loss_base_temperature_star=0.5,
        convection_scale=1.0,
    )

    expected = torch.tensor([[1.0], [3.0]])
    assert torch.allclose(residual, expected, atol=1e-6)


def test_compute_pde_residual_accepts_source_for_full_delta_prediction():
    """取消显式推进时，网络预测完整源温升应满足带源 residual。"""

    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 7), dtype=torch.float32)
    T_current = torch.zeros(3, 1)
    delta_t_source = torch.tensor([[0.0], [0.25], [0.5]])

    residual = compute_pde_residual(
        T_next=T_current + delta_t_source,
        T_current=T_current,
        delta_t_source_star=delta_t_source,
        v_scan_star=0.0,
        dt_star=0.5,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.0,
        convection_scale=1.0,
    )

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-7)


def test_compute_pde_residual_can_use_backward_time_scheme():
    """验证后向残差会用预测温度计算空间项和热耗散项。"""

    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    edge_attr = _edge_attr(
        distance=[2.0, 1.0],
        cos_theta=[1.0, -1.0],
        cos_phi_sq=[1.0, 0.0],
    )
    T_next = torch.tensor([[1.0], [3.0], [5.0]])
    T_current = torch.tensor([[0.0], [1.0], [2.0]])

    residual = compute_pde_residual(
        T_next=T_next,
        T_current=T_current,
        v_scan_star=2.0,
        dt_star=2.0,
        edge_index=edge_index,
        edge_attr=edge_attr,
        inverse_pe=0.5,
        k_ratio=0.1,
        residual_time_scheme="backward",
        convection_scale=1.0,
    )

    expected = torch.tensor([[0.5], [4.483333], [1.5]])
    assert torch.allclose(residual, expected, atol=1e-6)
