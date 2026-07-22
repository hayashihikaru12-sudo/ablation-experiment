from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PDGCNConfig:
    node_input_size: Optional[int] = None
    edge_input_size: int = 7
    global_input_size: int = 1
    hidden_size: int = 64
    message_passing_num: int = 3
    output_size: int = 1

    gamma_upwind: float = 0.8
    # 消融分支默认关闭方向加权，使所有边的迎风门控恒为 1。
    use_upwind_gate: bool = False
    use_aniso_gate: bool = True
    include_global: bool = True
    include_q_in_features: bool = False
    include_delta_t_source_in_features: bool = False

    dropout: float = 0.0
    layer_norm: bool = True

    lambda_pde: float = 1.0
    lambda_outflow: float = 1.0
    inverse_pe: float = 1.0
    source_coefficient: float = 1.0
    heat_source_absorptivity: float = 1.0
    pi_q: float = 1.0
    k_ratio: float = 0.05
    dt_star: float = 1.0
    gradient_regularization: float = 0.0
    dirichlet_temperature_star: float = 0.0
    thermal_loss_beta: float = 0.0
    thermal_loss_base_temperature_star: float = 0.0
    residual_time_scheme: str = "explicit"
    convection_scale: float = 2.0
    adaptive_pde_node_weight_enabled: bool = True
    adaptive_pde_node_weight_scheme: str = "heat_flux"
    adaptive_pde_node_weight_min: float = 0.2
    temperature_pde_node_weight_beta: float = 0.5
    temperature_pde_node_weight_max: float = 8.0
    temperature_pde_node_weight_clamp_enabled: bool = True
    temperature_pde_node_weight_threshold: float = 1.0
    temperature_pde_node_weight_high: float = 4.0
    adaptive_pde_node_weight_warmup_enabled: bool = False
    adaptive_pde_node_weight_warmup_epochs: int = 50

    @property
    def encoder_node_input_size(self) -> int:
        """计算节点编码器实际输入维度。

        参数:
            self: ``PDGCNConfig`` 实例，包含节点特征维度、全局特征维度和开关。

        返回:
            Python ``int``。若 ``include_global=True``，返回
            ``node_input_size + global_input_size``；否则返回 ``node_input_size``。
        """

        if self.include_global:
            return self.node_input_size + self.global_input_size
        return self.node_input_size

    def __post_init__(self):
        """校验 PD-GCN 模型和物理损失配置。

        参数:
            self: ``PDGCNConfig`` 实例，包含维度、网络超参数和物理损失系数。

        返回:
            None。若维度或系数非法则抛出 ``ValueError``。
        """

        expected_node_input_size = (
            7
            + int(bool(self.include_q_in_features))
            + int(bool(self.include_delta_t_source_in_features))
        )
        if self.node_input_size is None:
            object.__setattr__(self, "node_input_size", expected_node_input_size)
        elif int(self.node_input_size) != expected_node_input_size:
            raise ValueError(
                "node_input_size must match enabled node feature channels: "
                f"expected {expected_node_input_size} for "
                f"include_q_in_features={self.include_q_in_features} and "
                "include_delta_t_source_in_features="
                f"{self.include_delta_t_source_in_features}, got {self.node_input_size}."
            )

        positive_ints = (
            "node_input_size",
            "edge_input_size",
            "global_input_size",
            "hidden_size",
            "message_passing_num",
            "output_size",
        )
        for field_name in positive_ints:
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if not isinstance(self.use_upwind_gate, bool):
            raise ValueError(f"use_upwind_gate must be a boolean, got {self.use_upwind_gate!r}.")
        if float(self.lambda_pde) < 0:
            raise ValueError(f"lambda_pde must be non-negative, got {self.lambda_pde}.")
        if float(self.lambda_outflow) < 0:
            raise ValueError(f"lambda_outflow must be non-negative, got {self.lambda_outflow}.")
        if float(self.inverse_pe) < 0:
            raise ValueError(f"inverse_pe must be non-negative, got {self.inverse_pe}.")
        if float(self.pi_q) < 0:
            raise ValueError(f"pi_q must be non-negative, got {self.pi_q}.")
        if float(self.source_coefficient) < 0:
            raise ValueError(f"source_coefficient must be non-negative, got {self.source_coefficient}.")
        if float(self.heat_source_absorptivity) < 0:
            raise ValueError(
                f"heat_source_absorptivity must be non-negative, got {self.heat_source_absorptivity}."
            )
        if float(self.k_ratio) < 0:
            raise ValueError(f"k_ratio must be non-negative, got {self.k_ratio}.")
        if float(self.dt_star) <= 0:
            raise ValueError(f"dt_star must be positive, got {self.dt_star}.")
        if float(self.gradient_regularization) < 0:
            raise ValueError(
                f"gradient_regularization must be non-negative, got {self.gradient_regularization}."
            )
        if float(self.convection_scale) < 0:
            raise ValueError(f"convection_scale must be non-negative, got {self.convection_scale}.")
        if float(self.thermal_loss_beta) < 0:
            raise ValueError(f"thermal_loss_beta must be non-negative, got {self.thermal_loss_beta}.")
        if not isinstance(self.adaptive_pde_node_weight_enabled, bool):
            raise ValueError(
                "adaptive_pde_node_weight_enabled must be a boolean, "
                f"got {self.adaptive_pde_node_weight_enabled!r}."
            )
        scheme = str(self.adaptive_pde_node_weight_scheme).strip().lower().replace("-", "_")
        valid_schemes = {
            "none",
            "heat_flux",
            "temperature_continuous_clamped",
            "temperature_hard_threshold",
        }
        if scheme not in valid_schemes:
            raise ValueError(
                "adaptive_pde_node_weight_scheme must be one of "
                "'none', 'heat_flux', 'temperature_continuous_clamped', or "
                f"'temperature_hard_threshold', got {self.adaptive_pde_node_weight_scheme!r}."
            )
        object.__setattr__(self, "adaptive_pde_node_weight_scheme", scheme)
        if not 0.0 <= float(self.adaptive_pde_node_weight_min) <= 1.0:
            raise ValueError(
                "adaptive_pde_node_weight_min must be in [0, 1], "
                f"got {self.adaptive_pde_node_weight_min}."
            )
        if float(self.temperature_pde_node_weight_beta) < 0:
            raise ValueError(
                "temperature_pde_node_weight_beta must be non-negative, "
                f"got {self.temperature_pde_node_weight_beta}."
            )
        if float(self.temperature_pde_node_weight_max) < 1.0:
            raise ValueError(
                "temperature_pde_node_weight_max must be at least 1, "
                f"got {self.temperature_pde_node_weight_max}."
            )
        if not isinstance(self.temperature_pde_node_weight_clamp_enabled, bool):
            raise ValueError(
                "temperature_pde_node_weight_clamp_enabled must be a boolean, "
                f"got {self.temperature_pde_node_weight_clamp_enabled!r}."
            )
        if float(self.temperature_pde_node_weight_high) < 1.0:
            raise ValueError(
                "temperature_pde_node_weight_high must be at least 1, "
                f"got {self.temperature_pde_node_weight_high}."
            )
        if float(self.temperature_pde_node_weight_threshold) <= 0:
            raise ValueError(
                "temperature_pde_node_weight_threshold must be positive, "
                f"got {self.temperature_pde_node_weight_threshold}."
            )
        if not isinstance(self.adaptive_pde_node_weight_warmup_enabled, bool):
            raise ValueError(
                "adaptive_pde_node_weight_warmup_enabled must be a boolean, "
                f"got {self.adaptive_pde_node_weight_warmup_enabled!r}."
            )
        if int(self.adaptive_pde_node_weight_warmup_epochs) <= 0:
            raise ValueError(
                "adaptive_pde_node_weight_warmup_epochs must be positive, "
                f"got {self.adaptive_pde_node_weight_warmup_epochs}."
            )
        if str(self.residual_time_scheme).strip().lower().replace("-", "_") not in (
            "explicit",
            "explicit_euler",
            "forward",
            "forward_euler",
            "backward",
            "backward_euler",
            "implicit",
        ):
            raise ValueError(
                "residual_time_scheme must be 'explicit' or 'backward', "
                f"got {self.residual_time_scheme!r}."
            )
