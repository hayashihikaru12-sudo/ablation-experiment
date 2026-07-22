import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PDGCNConfig
from ..mlp import build_mlp


class EdgeBlock(nn.Module):
    def __init__(self, config: PDGCNConfig):
        """初始化边消息计算模块。

        参数:
            config: ``PDGCNConfig``，提供隐空间维度、迎风门控系数、
                各向异性门控开关、归一化和 dropout 配置。

        返回:
            None。实例会创建消息 MLP 和各向异性门控 MLP。
        """

        super().__init__()
        self.config = config
        self.message_mlp = build_mlp(
            3 * config.hidden_size,
            config.hidden_size,
            config.hidden_size,
            layer_norm=config.layer_norm,
            dropout=config.dropout,
        )
        self.aniso_mlp = build_mlp(
            2,
            config.hidden_size,
            config.hidden_size,
            layer_norm=False,
            dropout=config.dropout,
        )
        # 可学习的迎风门控系数，初始化为配置中的 gamma_upwind 值。
        # 设为 nn.Parameter 后梯度可直接优化，训练会自适应调整方向选择性强度。
        self.gamma_upwind = nn.Parameter(torch.tensor(float(config.gamma_upwind)))
        self.last_alpha = None
        self.last_aniso_gate = None
        self.last_gamma = None

    def forward(self, graph, raw_edge_attr):
        """计算带迎风和各向异性门控的边消息。

        参数:
            graph: PyG ``Data`` 图对象，``x`` 形状 ``[N, hidden_size]``，
                ``edge_index`` 形状 ``[2, E]``，``edge_attr`` 为编码后的边隐状态
                ``[E, hidden_size]``。
            raw_edge_attr: 原始边特征张量，形状 ``[E, 7]``，用于读取
                ``cos_theta``、``cos_phi`` 和 ``cos_phi_sq``。

        返回:
            更新后的 ``graph``，其中 ``edge_attr`` 形状 ``[E, hidden_size]``，
            表示门控后的边消息。
        """

        sender, receiver = graph.edge_index
        sender_attr = graph.x[sender]
        receiver_attr = graph.x[receiver]
        # 与设计稿略有差异：设计公式为 m_ij = MLP_msg(h_j, e_ij^(0))，
        # 只使用接收节点一侧的邻居特征 h_j；当前实现同时传入发送端和接收端特征。
        # 这是 PIGNN 风格的“双端消息”形式，表达能力更强。
        # 从物理含义上通常无害，但并不完全等同于设计文档中的公式。
        # 建议在论文或说明文档中明确记录实际实现，避免训练代码与文档描述不一致。
        message_input = torch.cat([sender_attr, receiver_attr, graph.edge_attr], dim=-1)
        message = self.message_mlp(message_input)

        alpha = self._upwind_gate(raw_edge_attr)
        aniso_gate = self._aniso_gate(raw_edge_attr, message)
        graph.edge_attr = alpha * aniso_gate * message

        self.last_alpha = alpha.detach()
        self.last_aniso_gate = aniso_gate.detach()
        return graph

    def _upwind_gate(self, raw_edge_attr):
        """根据边方向和扫描方向夹角计算宏观迎风权重。

        参数:
            raw_edge_attr: 原始边特征张量，形状 ``[E, 7]``，第 5 列为
                ``cos_theta``。

        返回:
            迎风权重张量，形状 ``[E, 1]``。本消融分支关闭迎风门控时
            恒为 1；否则为 ``ReLU(1 + gamma_upwind * cos_theta)``。
        """

        cos_theta = raw_edge_attr[:, 4:5]
        gamma = self.gamma_upwind
        self.last_gamma = gamma.detach()
        if not self.config.use_upwind_gate:
            return torch.ones_like(cos_theta)
        alpha = F.relu(1.0 + gamma * cos_theta)
        return alpha

    def _aniso_gate(self, raw_edge_attr, message):
        """根据纤维方向相关边特征计算微观各向异性门控。

        参数:
            raw_edge_attr: 原始边特征张量，形状 ``[E, 7]``，第 6-7 列为
                ``[cos_phi, cos_phi_sq]``。
            message: 待门控的边消息张量，形状 ``[E, hidden_size]``。

        返回:
            门控张量，形状 ``[E, hidden_size]``；若关闭各向异性门控则返回全 1。
        """

        if not self.config.use_aniso_gate:
            return torch.ones_like(message)
        gate_input = raw_edge_attr[:, 5:7].to(device=message.device, dtype=message.dtype)
        # Sigmoid 会把输出限制在 [0, 1] 范围内；这里保留该约束用于稳定门控幅值。

        # 若优先追求最大物理可解释性，可以移除该 MLP，并直接将
        # raw_edge_attr[:, 6:7]（即 $\cos^2\phi$）广播到 64 维后与消息相乘。
        # 若优先追求更低损失，保留当前 MLP + sigmoid 设计通常更合适。
        return torch.sigmoid(self.aniso_mlp(gate_input))
