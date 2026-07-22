from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Optional, Sequence

import torch
from torch_geometric.data import Data

from data.dimensionless import ScaleParams, temperature_from_dimensionless, temperature_to_dimensionless
from data.velocity import tangent_velocity_direction
from data.static_cache import STATIC_FILE, HDF5FrameReader
from pde import apply_dirichlet_boundary, total_loss

from .config import TrainConfig
from .graph_utils import (
    clone_graph_with_temperature,
    graph_explicit_source_delta,
    graph_physical_source_delta,
    graph_residual_source_delta,
    graph_surface_heat_source,
    node_feature_indices_from_config,
)
from .lr_scheduler import build_lr_scheduler, optimizer_lr
from .warmup import pseudo_time_relax_initial_temperature


@dataclass
class StaticGraphState:
    """常驻设备的固定拓扑图状态。"""

    edge_index: torch.Tensor
    source: torch.Tensor
    receiver: torch.Tensor
    boundary_nodes: dict
    node_type: torch.Tensor
    num_nodes: int
    num_edges: int
    device: torch.device

    @classmethod
    def from_cache(cls, cache_dir, device=None):
        """从静态缓存加载固定拓扑并移动到目标设备。

        参数:
            cache_dir: ``build_static_cache`` 生成的缓存目录。
            device: 目标设备；若为 ``None``，优先使用 CUDA，否则使用 CPU。

        返回:
            ``StaticGraphState`` 实例，索引和边界节点已在目标设备上。
        """

        target_device = torch.device(device) if device is not None else _default_device()
        payload = torch.load(Path(cache_dir) / STATIC_FILE, map_location="cpu")
        edge_index = payload["edge_index"].to(device=target_device, dtype=torch.long, non_blocking=True)
        boundary_nodes = {
            name: value.to(device=target_device, dtype=torch.long, non_blocking=True)
            for name, value in payload["boundary_nodes"].items()
        }
        node_type = payload["node_type"].to(device=target_device, dtype=torch.long, non_blocking=True)
        return cls(
            edge_index=edge_index,
            source=edge_index[0],
            receiver=edge_index[1],
            boundary_nodes=boundary_nodes,
            node_type=node_type,
            num_nodes=int(payload["num_nodes"]),
            num_edges=int(payload["num_edges"]),
            device=target_device,
        )


class GpuFeatureBuilder:
    """在目标设备上由基础动态特征生成 PD-GCN 节点和边特征。"""

    def __init__(self, static_state: StaticGraphState, scale_params: ScaleParams, *, model_config=None, dtype=torch.float32):
        """初始化 GPU 特征构建器和复用缓冲区。

        参数:
            static_state: ``StaticGraphState``，提供固定拓扑和边界节点。
            scale_params: ``ScaleParams``，提供无量纲化标尺。
            dtype: 特征张量数据类型，默认 ``torch.float32``。

        返回:
            None。实例会预分配节点、边、基础特征和全局条件缓冲区。
        """

        self.static_state = static_state
        self.scale_params = scale_params
        self.device = static_state.device
        self.dtype = dtype
        self.model_config = model_config
        self.node_input_size = int(getattr(model_config, "node_input_size", 7))
        self.q_feature_index, self.delta_t_source_feature_index = node_feature_indices_from_config(model_config)
        self.node_base = torch.empty(static_state.num_nodes, 13, device=self.device, dtype=dtype)
        self.global_raw = torch.empty(1, device=self.device, dtype=dtype)
        self.x = torch.empty(static_state.num_nodes, self.node_input_size, device=self.device, dtype=dtype)
        self.edge_attr = torch.empty(static_state.num_edges, 7, device=self.device, dtype=dtype)
        self.q_surface_star = torch.empty(static_state.num_nodes, 1, device=self.device, dtype=dtype)
        self.graph = Data(
            x=self.x,
            edge_index=static_state.edge_index,
            edge_attr=self.edge_attr,
            node_type=static_state.node_type,
            global_attr=self.global_raw,
            pos=self.x[:, 0:3],
            q_surface_star=self.q_surface_star,
        )
        self.graph.num_nodes = static_state.num_nodes
        self.graph.upwind_nodes = static_state.boundary_nodes["upwind"]
        self.graph.downwind_nodes = static_state.boundary_nodes["downwind"]
        self.graph.side_nodes = static_state.boundary_nodes["side"]
        self.graph.q_feature_index = int(self.q_feature_index)
        self.graph.delta_t_source_feature_index = int(self.delta_t_source_feature_index)
        self.graph.include_q_in_features = self.q_feature_index >= 0
        self.graph.include_delta_t_source_in_features = self.delta_t_source_feature_index >= 0

    def build(self, node_base_cpu, global_cpu, temperature_star):
        """用当前帧基础特征更新复用图对象。

        参数:
            node_base_cpu: CPU 张量，形状 ``[N, 13]``，列为
                ``[x, y, z, fx, fy, fz, nx, ny, nz, vx, vy, vz, Q]``。
            global_cpu: CPU 张量，形状 ``[G]``，当前只使用第 1 个值作为真实扫描速度。
            temperature_star: 当前无量纲温度，形状 ``[N, 1]``。

        返回:
            复用的 PyG ``Data`` 图对象；其 ``x``、``edge_attr`` 和 ``global_attr``
            已指向当前帧特征缓冲区。
        """

        self.node_base = self.node_base.detach()
        self.global_raw = self.global_raw.detach()
        self.x = self.x.detach()
        self.edge_attr = self.edge_attr.detach()
        self.q_surface_star = self.q_surface_star.detach()

        self.node_base.copy_(node_base_cpu, non_blocking=self.device.type == "cuda")
        self.global_raw.copy_(global_cpu[:1], non_blocking=self.device.type == "cuda")

        coords_star = self.node_base[:, 0:3] / float(self.scale_params.L0)
        fibers_unit = _normalize_vectors(self.node_base[:, 3:6], eps=float(self.scale_params.eps))
        normals_unit = _normalize_vectors(self.node_base[:, 6:9], eps=float(self.scale_params.eps))
        velocity_direction = self.node_base[:, 9:12]
        q_surface_star = self.node_base[:, 12:13] / float(self.scale_params.Q0)
        temperature = temperature_star.to(device=self.device, dtype=self.dtype, non_blocking=True).reshape(-1, 1)

        self.x[:, 0:3] = coords_star
        self.x[:, 3:6] = fibers_unit
        self.x[:, 6:7] = temperature
        if self.q_feature_index >= 0:
            self.x[:, self.q_feature_index : self.q_feature_index + 1] = q_surface_star
        if self.delta_t_source_feature_index >= 0:
            self.x[:, self.delta_t_source_feature_index : self.delta_t_source_feature_index + 1].zero_()
        self.q_surface_star[:, 0:1] = q_surface_star

        source = self.static_state.source
        receiver = self.static_state.receiver
        delta = coords_star[receiver] - coords_star[source]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(float(self.scale_params.eps))
        direction = delta / distance
        tangent_velocity = tangent_velocity_direction(
            velocity_direction,
            normals_unit,
            eps=float(self.scale_params.eps),
        )
        fiber_mid = _normalize_vectors(fibers_unit[source] + fibers_unit[receiver], eps=float(self.scale_params.eps))
        cos_phi = torch.sum(fiber_mid * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        self.edge_attr[:, 0:3] = delta
        self.edge_attr[:, 3:4] = distance
        self.edge_attr[:, 4:5] = torch.sum(tangent_velocity[receiver] * direction, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        self.edge_attr[:, 5:6] = cos_phi
        self.edge_attr[:, 6:7] = cos_phi.square()

        self.global_raw[:] = self.global_raw / float(self.scale_params.v0)
        self.graph.x = self.x
        self.graph.edge_attr = self.edge_attr
        self.graph.global_attr = self.global_raw
        self.graph.pos = self.x[:, 0:3]
        self.graph.q_surface_star = self.q_surface_star
        self.graph.q_feature_index = int(self.q_feature_index)
        self.graph.delta_t_source_feature_index = int(self.delta_t_source_feature_index)
        self.graph.include_q_in_features = self.q_feature_index >= 0
        self.graph.include_delta_t_source_in_features = self.delta_t_source_feature_index >= 0
        return self.graph

    def initial_temperature(self):
        """创建当前静态图的冷态无量纲初温。

        参数:
            无。

        返回:
            形状 ``[N, 1]`` 的零张量，位于特征构建器所在设备。训练入口会在
            ``warmup_steps > 0`` 时用当前 PD-GCN 权重将该冷态松弛为初温。
        """

        return torch.zeros(self.static_state.num_nodes, 1, device=self.device, dtype=self.dtype)


def _snapshot_graph_for_tbptt(graph):
    """为单个 TBPTT 时间步创建保留梯度的图快照。

    ``GpuFeatureBuilder`` 会在下一帧原地复用特征缓冲区；窗口末才
    ``backward`` 时，真实 PD-GCN 会需要这些旧帧特征。这里仅克隆会被
    覆盖且可能被 autograd 保存的动态特征，静态拓扑索引继续共享。
    """

    snapshot = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr.clone(),
        node_type=graph.node_type,
        global_attr=graph.global_attr.clone(),
        pos=graph.pos.clone(),
    )
    if hasattr(graph, "q_surface_star"):
        snapshot.q_surface_star = graph.q_surface_star.clone()
    if hasattr(graph, "q_surface"):
        snapshot.q_surface = graph.q_surface.clone()
    snapshot.q_feature_index = int(getattr(graph, "q_feature_index", -1))
    snapshot.delta_t_source_feature_index = int(getattr(graph, "delta_t_source_feature_index", -1))
    snapshot.include_q_in_features = bool(getattr(graph, "include_q_in_features", False))
    snapshot.include_delta_t_source_in_features = bool(getattr(graph, "include_delta_t_source_in_features", False))
    snapshot.num_nodes = graph.num_nodes
    snapshot.upwind_nodes = graph.upwind_nodes
    snapshot.downwind_nodes = graph.downwind_nodes
    snapshot.side_nodes = graph.side_nodes
    return snapshot


def train_static_topology(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
    monitor_callback: Optional[Callable[[dict, dict], None]] = None,
    slice_callback: Optional[Callable[[dict], None]] = None,
    monitor_frame_index: Optional[int] = None,
    start_epoch: int = 0,
    supervision_config=None,
    peak_supervision_config=None,
    lr_scheduler_state: Optional[dict] = None,
    _lr_scheduler_state_out: Optional[list] = None,
):
    """使用固定拓扑流式数据管线训练 PD-GCN。

    参数:
        model: 待训练模型，输入当前帧图对象并输出 ``delta_T*``。
        frame_reader: ``HDF5FrameReader``，按帧提供 CPU 基础特征。
        static_state: ``StaticGraphState``，提供常驻设备的静态拓扑。
        feature_builder: ``GpuFeatureBuilder``，在设备上构建当前帧特征。
        config: ``TrainConfig``，提供训练轮数、窗口长度、warmup、学习率和设备。
        optimizer: 可选优化器；若为 ``None``，创建 Adam。
        epoch_callback: 可选回调；每个 epoch 结束后接收该 epoch 的历史记录。

    返回:
        训练历史列表；每个元素包含 ``epoch``、平均 ``loss`` 和窗口损失。
    """

    return train_static_topology_sequences(
        model,
        [frame_reader],
        static_state,
        feature_builder,
        config,
        optimizer=optimizer,
        epoch_callback=epoch_callback,
        monitor_callback=monitor_callback,
        slice_callback=slice_callback,
        monitor_frame_index=monitor_frame_index,
        start_epoch=start_epoch,
        supervision_config=supervision_config,
        peak_supervision_config=peak_supervision_config,
        lr_scheduler_state=lr_scheduler_state,
        _lr_scheduler_state_out=_lr_scheduler_state_out,
    )


def train_static_topology_sequences(
    model,
    frame_readers: Sequence[HDF5FrameReader],
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_callback: Optional[Callable[[dict], None]] = None,
    monitor_callback: Optional[Callable[[dict, dict], None]] = None,
    slice_callback: Optional[Callable[[dict], None]] = None,
    monitor_frame_index: Optional[int] = None,
    start_epoch: int = 0,
    supervision_config=None,
    peak_supervision_config=None,
    lr_scheduler_state: Optional[dict] = None,
    _lr_scheduler_state_out: Optional[list] = None,
):
    """按独立 HDF5 序列训练固定拓扑 PD-GCN。"""

    if not frame_readers:
        raise ValueError("frame_readers must contain at least one sequence.")

    device = static_state.device
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    supervision_enabled = _supervision_enabled(supervision_config)
    peak_supervision_enabled = _peak_supervision_enabled(peak_supervision_config)
    if supervision_enabled and peak_supervision_enabled:
        raise ValueError("supervision and peak_supervision cannot both be enabled.")
    if supervision_enabled or peak_supervision_enabled:
        _validate_supervised_readers(frame_readers)

    history = []
    start_epoch = int(start_epoch)
    lr_scheduler = build_lr_scheduler(optimizer, config)
    if lr_scheduler_state is not None:
        lr_scheduler.load_state_dict(lr_scheduler_state)
    _lr_state_out = _lr_scheduler_state_out if _lr_scheduler_state_out is not None else None
    training_start_time = time.perf_counter()
    for epoch in range(start_epoch, start_epoch + int(config.epochs)):
        _synchronize_if_cuda(device)
        epoch_start_time = time.perf_counter()
        lr_scheduler.begin_epoch(epoch)
        epoch_lr = optimizer_lr(optimizer)
        model.train()
        window_records = []
        file_window_counts = []
        last_snapshot = None
        for file_index, frame_reader in enumerate(frame_readers):
            sequence_records, sequence_snapshot = _train_one_static_sequence_epoch(
                model,
                frame_reader,
                static_state,
                feature_builder,
                config,
                optimizer,
                epoch=epoch,
                monitor_frame_index=monitor_frame_index,
                supervision_config=supervision_config,
                peak_supervision_config=peak_supervision_config,
            )
            window_records.extend(sequence_records)
            file_window_counts.append(len(sequence_records))
            if sequence_snapshot is not None:
                last_snapshot = sequence_snapshot
            if slice_callback is not None:
                slice_callback({"epoch": epoch, "slice_index": file_index, "frame_reader": frame_reader})

        window_losses = [record["loss_total"] for record in window_records]
        window_times = [float(record.get("window_time_seconds", 0.0)) for record in window_records]
        gamma_values = _collect_gamma_upwind(model)
        _synchronize_if_cuda(device)
        epoch_time_seconds = max(0.0, time.perf_counter() - epoch_start_time)
        elapsed_time_seconds = max(0.0, time.perf_counter() - training_start_time)
        epoch_record = {
            "epoch": epoch,
            "loss": sum(window_losses) / max(len(window_losses), 1),
            "lr": epoch_lr,
            "epoch_time_seconds": epoch_time_seconds,
            "elapsed_time_seconds": elapsed_time_seconds,
            "total_training_time_seconds": elapsed_time_seconds,
            "batch_time_seconds": window_times,
            "batch_time_mean_seconds": _mean_values(window_times),
            "window_time_seconds": window_times,
            "window_time_mean_seconds": _mean_values(window_times),
            "loss_total": _mean_records(window_records, "loss_total"),
            "loss_physics": _mean_records(window_records, "loss_physics"),
            "loss_supervised": _mean_records(window_records, "loss_supervised"),
            "loss_peak_temperature_rise": _mean_records(window_records, "loss_peak_temperature_rise"),
            "peak_temperature_rise_pred": _mean_records(window_records, "peak_temperature_rise_pred"),
            "peak_temperature_rise_fem": _mean_records(window_records, "peak_temperature_rise_fem"),
            "peak_temperature_rise_error": _mean_records(window_records, "peak_temperature_rise_error"),
            "peak_temperature_rise_abs_error": _mean_records(window_records, "peak_temperature_rise_abs_error"),
            "peak_temperature_rise_rmse": _mean_records(window_records, "peak_temperature_rise_rmse"),
            "lambda_peak_temperature_rise": _mean_records(window_records, "lambda_peak_temperature_rise"),
            "case_peak_temperature_pred": _mean_records(window_records, "case_peak_temperature_pred"),
            "case_peak_temperature_fem": _mean_records(window_records, "case_peak_temperature_fem"),
            "case_peak_temperature_error": _mean_records(window_records, "case_peak_temperature_error"),
            "case_peak_temperature_abs_error": _mean_records(window_records, "case_peak_temperature_abs_error"),
            "case_peak_temperature_rmse": _rmse_records(window_records, "case_peak_temperature_error"),
            "case_peak_temperature_rise_pred": _mean_records(window_records, "case_peak_temperature_rise_pred"),
            "case_peak_temperature_rise_fem": _mean_records(window_records, "case_peak_temperature_rise_fem"),
            "case_peak_temperature_rise_error": _mean_records(window_records, "case_peak_temperature_rise_error"),
            "case_peak_temperature_rise_abs_error": _mean_records(
                window_records,
                "case_peak_temperature_rise_abs_error",
            ),
            "case_peak_temperature_rise_rmse": _rmse_records(window_records, "case_peak_temperature_rise_error"),
            "case_peak_topk_temperature_rise_pred": _mean_records(
                window_records,
                "case_peak_topk_temperature_rise_pred",
            ),
            "case_peak_topk_temperature_rise_fem": _mean_records(
                window_records,
                "case_peak_topk_temperature_rise_fem",
            ),
            "case_peak_topk_temperature_rise_error": _mean_records(
                window_records,
                "case_peak_topk_temperature_rise_error",
            ),
            "case_peak_topk_temperature_rise_abs_error": _mean_records(
                window_records,
                "case_peak_topk_temperature_rise_abs_error",
            ),
            "case_peak_topk_temperature_rise_rmse": _rmse_records(
                window_records,
                "case_peak_topk_temperature_rise_error",
            ),
            "loss_temperature": _mean_records(window_records, "loss_temperature"),
            "loss_teacher_forcing_temperature": _mean_records(
                window_records, "loss_teacher_forcing_temperature"
            ),
            "loss_rollout_temperature": _mean_records(window_records, "loss_rollout_temperature"),
            "loss_pde": _mean_records(window_records, "loss_pde"),
            "loss_outflow": _mean_records(window_records, "loss_outflow"),
            "loss_beta": _mean_records(window_records, "loss_beta"),
            "loss_smooth": _mean_records(window_records, "loss_smooth"),
            "fem_temperature_rmse": _mean_records(window_records, "fem_temperature_rmse"),
            "fem_temperature_mae": _mean_records(window_records, "fem_temperature_mae"),
            "fem_temperature_max_error": _max_records(window_records, "fem_temperature_max_error"),
            "rollout_fem_temperature_rmse": _mean_records(
                window_records, "rollout_fem_temperature_rmse"
            ),
            "rollout_fem_temperature_mae": _mean_records(window_records, "rollout_fem_temperature_mae"),
            "rollout_fem_temperature_max_error": _max_records(
                window_records, "rollout_fem_temperature_max_error"
            ),
            "temperature_mean": _mean_records(window_records, "temperature_mean"),
            "temperature_max": _max_records(window_records, "temperature_max"),
            "temperature_min": _min_records(window_records, "temperature_min"),
            "temperature_var": _mean_records(window_records, "temperature_var"),
            "gamma_upwind": float(gamma_values.mean()) if len(gamma_values) > 0 else float(model.config.gamma_upwind),
            "gamma_upwind_std": float(gamma_values.std()) if len(gamma_values) > 1 else 0.0,
            "window_losses": window_losses,
            "file_window_counts": file_window_counts,
        }
        lr_scheduler.end_epoch(epoch_record["loss"])
        if _lr_state_out is not None:
            _lr_state_out[:] = [lr_scheduler.state_dict()]
        should_stop = config.loss_threshold is not None and epoch_record["loss"] < float(config.loss_threshold)
        if should_stop:
            epoch_record["stopped_early"] = True
            epoch_record["stop_reason"] = "loss_threshold"
        history.append(epoch_record)
        if monitor_callback is not None:
            monitor_callback(epoch_record, {"snapshot": last_snapshot} if last_snapshot is not None else {})
        if epoch_callback is not None:
            epoch_callback(epoch_record)
        if should_stop:
            break
    return history


def _train_one_static_sequence_epoch(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int = 0,
    monitor_frame_index: Optional[int] = None,
    supervision_config=None,
    peak_supervision_config=None,
):
    if _supervision_enabled(supervision_config):
        return _train_one_static_sequence_epoch_supervised(
            model,
            frame_reader,
            static_state,
            feature_builder,
            config,
            optimizer,
            supervision_config,
            epoch=epoch,
            monitor_frame_index=monitor_frame_index,
        )
    if _peak_supervision_enabled(peak_supervision_config):
        return _train_one_static_sequence_epoch_peak_supervised(
            model,
            frame_reader,
            static_state,
            feature_builder,
            config,
            optimizer,
            peak_supervision_config,
            epoch=epoch,
            monitor_frame_index=monitor_frame_index,
        )

    if int(config.warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, feature_builder.initial_temperature())
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(config.warmup_steps),
        )
    else:
        current_temperature = feature_builder.initial_temperature().detach()

    window_records = []
    selected_snapshot = None
    for start in range(0, frame_reader.num_frames, int(config.tbptt_window)):
        _synchronize_if_cuda(static_state.device)
        window_start_time = time.perf_counter()
        end = min(start + int(config.tbptt_window), frame_reader.num_frames)
        optimizer.zero_grad()
        loss_terms = []
        component_records = []
        window_temperature = current_temperature
        window_snapshot = None

        for frame_idx in range(start, end):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = _snapshot_graph_for_tbptt(
                feature_builder.build(node_base_cpu, global_cpu, window_temperature)
            )
            physical_source_delta = graph_physical_source_delta(graph, model.config)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                window_temperature + delta_t_source,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(
                graph,
                source_temperature,
                delta_t_source_star=physical_source_delta,
            )
            delta_temperature = model(graph)
            next_temperature = apply_dirichlet_boundary(
                source_temperature + delta_temperature,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            components = _compute_loss_components(
                model,
                next_temperature,
                source_temperature,
                graph,
                static_state,
                epoch=epoch,
            )
            loss_terms.append(components["loss_total"])
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx, monitor_frame_index, frame_reader.num_frames):
                window_snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx,
                )
            window_temperature = next_temperature

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        optimizer.step()
        _synchronize_if_cuda(static_state.device)
        window_time_seconds = max(0.0, time.perf_counter() - window_start_time)

        window_record = _aggregate_component_records(component_records)
        window_record.update(_temperature_stats(window_temperature, feature_builder.scale_params))
        window_record["loss_total"] = float(loss.detach().cpu())
        window_record["batch_time_seconds"] = window_time_seconds
        window_record["window_time_seconds"] = window_time_seconds
        window_records.append(window_record)
        if window_snapshot is not None:
            selected_snapshot = window_snapshot
        current_temperature = window_temperature.detach()
    return window_records, selected_snapshot


def _train_one_static_sequence_epoch_supervised(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    supervision_config,
    *,
    epoch: int = 0,
    monitor_frame_index: Optional[int] = None,
):
    if frame_reader.num_frames < 2:
        raise ValueError("Supervised training requires at least two FEM temperature frames.")

    mode = _supervision_mode(supervision_config)
    window_size = _supervision_window_size(config, supervision_config, mode)
    window_records = []
    selected_snapshot = None
    num_transitions = frame_reader.num_frames - 1
    for start in range(0, num_transitions, window_size):
        _synchronize_if_cuda(static_state.device)
        window_start_time = time.perf_counter()
        end = min(start + window_size, num_transitions)
        optimizer.zero_grad()
        loss_terms = []
        component_records = []
        window_temperature = None
        window_snapshot = None
        rollout_temperature = (
            _read_fem_temperature_star(frame_reader, start, feature_builder)
            if mode in {"rollout", "mixed"}
            else None
        )

        for frame_idx in range(start, end):
            fem_next = _read_fem_temperature_star(frame_reader, frame_idx + 1, feature_builder)
            fem_mask_next = _read_fem_valid_mask(frame_reader, frame_idx + 1, feature_builder)

            if mode == "teacher_forcing":
                fem_current = _read_fem_temperature_star(frame_reader, frame_idx, feature_builder)
                graph, _, next_temperature, components = _run_static_training_step(
                    model,
                    frame_reader,
                    static_state,
                    feature_builder,
                    frame_idx,
                    fem_current,
                    epoch=epoch,
                )
                supervision_components = _compute_supervision_components(
                    next_temperature,
                    fem_next,
                    fem_mask_next,
                    feature_builder.scale_params,
                    lambda_temperature=float(supervision_config.lambda_temperature),
                )
                _apply_teacher_forcing_supervision(components, supervision_components)
            else:
                graph, _, next_temperature, components = _run_static_training_step(
                    model,
                    frame_reader,
                    static_state,
                    feature_builder,
                    frame_idx,
                    rollout_temperature,
                    epoch=epoch,
                )
                rollout_components = _compute_supervision_components(
                    next_temperature,
                    fem_next,
                    fem_mask_next,
                    feature_builder.scale_params,
                    lambda_temperature=float(supervision_config.lambda_rollout_temperature),
                )
                _apply_rollout_supervision(components, rollout_components)
                if mode == "mixed":
                    fem_current = _read_fem_temperature_star(frame_reader, frame_idx, feature_builder)
                    _, _, teacher_temperature, _ = _run_static_training_step(
                        model,
                        frame_reader,
                        static_state,
                        feature_builder,
                        frame_idx,
                        fem_current,
                        epoch=epoch,
                        compute_components=False,
                    )
                    teacher_components = _compute_supervision_components(
                        teacher_temperature,
                        fem_next,
                        fem_mask_next,
                        feature_builder.scale_params,
                        lambda_temperature=float(supervision_config.lambda_temperature),
                    )
                    _apply_teacher_forcing_supervision(
                        components,
                        teacher_components,
                        include_general_metrics=False,
                    )
                rollout_temperature = next_temperature
            components["loss_total"] = components["loss_physics"] + components["loss_supervised"]
            loss_terms.append(components["loss_total"])
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx + 1, monitor_frame_index, frame_reader.num_frames):
                window_snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx + 1,
                    fem_temperature_star=fem_next,
                    pred_temperature_star=next_temperature,
                )
            window_temperature = next_temperature

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        optimizer.step()
        _synchronize_if_cuda(static_state.device)
        window_time_seconds = max(0.0, time.perf_counter() - window_start_time)

        window_record = _aggregate_component_records(component_records)
        if window_temperature is not None:
            window_record.update(_temperature_stats(window_temperature, feature_builder.scale_params))
        window_record["loss_total"] = float(loss.detach().cpu())
        window_record["batch_time_seconds"] = window_time_seconds
        window_record["window_time_seconds"] = window_time_seconds
        window_records.append(window_record)
        if window_snapshot is not None:
            selected_snapshot = window_snapshot
    return window_records, selected_snapshot


def _train_one_static_sequence_epoch_peak_supervised(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    peak_supervision_config,
    *,
    epoch: int = 0,
    monitor_frame_index: Optional[int] = None,
):
    if frame_reader.num_frames < 2:
        raise ValueError("Peak supervised training requires at least two FEM temperature frames.")

    window_size = _peak_supervision_window_size(config, peak_supervision_config)
    window_records = []
    selected_snapshot = None
    num_transitions = frame_reader.num_frames - 1
    rollout_temperature = _read_fem_temperature_star(frame_reader, 0, feature_builder).detach()
    case_peak_tracker = _init_case_peak_tracker()
    for start in range(0, num_transitions, window_size):
        _synchronize_if_cuda(static_state.device)
        window_start_time = time.perf_counter()
        end = min(start + window_size, num_transitions)
        optimizer.zero_grad()
        loss_terms = []
        component_records = []
        window_temperature = None
        window_snapshot = None
        for frame_idx in range(start, end):
            fem_next = _read_fem_temperature_star(frame_reader, frame_idx + 1, feature_builder)
            fem_mask_next = _read_fem_valid_mask(frame_reader, frame_idx + 1, feature_builder)
            graph, _, next_temperature, components = _run_static_training_step(
                model,
                frame_reader,
                static_state,
                feature_builder,
                frame_idx,
                rollout_temperature,
                epoch=epoch,
            )
            peak_components = _compute_peak_supervision_components(
                next_temperature,
                fem_next,
                fem_mask_next,
                feature_builder.scale_params,
                lambda_peak=float(peak_supervision_config.lambda_peak),
                topk=int(peak_supervision_config.topk),
                warmup_epochs=int(peak_supervision_config.warmup_epochs),
                epoch=epoch,
            )
            _apply_peak_supervision(components, peak_components)
            _update_case_peak_tracker(
                case_peak_tracker,
                next_temperature,
                fem_next,
                fem_mask_next,
                feature_builder.scale_params,
                topk=int(peak_supervision_config.topk),
            )
            components["loss_total"] = components["loss_physics"] + components["loss_supervised"]
            loss_terms.append(components["loss_total"])
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx + 1, monitor_frame_index, frame_reader.num_frames):
                window_snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx + 1,
                    fem_temperature_star=fem_next,
                    pred_temperature_star=next_temperature,
                )
            rollout_temperature = next_temperature
            window_temperature = next_temperature

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        if config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip_norm))
        optimizer.step()
        _synchronize_if_cuda(static_state.device)
        window_time_seconds = max(0.0, time.perf_counter() - window_start_time)

        window_record = _aggregate_component_records(component_records)
        if window_temperature is not None:
            window_record.update(_temperature_stats(window_temperature, feature_builder.scale_params))
        window_record["loss_total"] = float(loss.detach().cpu())
        window_record["batch_time_seconds"] = window_time_seconds
        window_record["window_time_seconds"] = window_time_seconds
        window_records.append(window_record)
        if window_snapshot is not None:
            selected_snapshot = window_snapshot
        if window_temperature is not None:
            rollout_temperature = window_temperature.detach()
    if window_records:
        window_records[-1].update(_finalize_case_peak_tracker(case_peak_tracker, feature_builder.scale_params))
    return window_records, selected_snapshot


def _run_static_training_step(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    frame_idx: int,
    current_temperature,
    *,
    epoch: int = 0,
    compute_components: bool = True,
):
    node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
    graph = _snapshot_graph_for_tbptt(
        feature_builder.build(node_base_cpu, global_cpu, current_temperature)
    )
    physical_source_delta = graph_physical_source_delta(graph, model.config)
    delta_t_source = graph_explicit_source_delta(graph, model.config)
    source_temperature = apply_dirichlet_boundary(
        current_temperature + delta_t_source,
        static_state.boundary_nodes,
        value=getattr(model.config, "dirichlet_temperature_star", 0.0),
    )
    graph = clone_graph_with_temperature(
        graph,
        source_temperature,
        delta_t_source_star=physical_source_delta,
    )
    delta_temperature = model(graph)
    next_temperature = apply_dirichlet_boundary(
        source_temperature + delta_temperature,
        static_state.boundary_nodes,
        value=getattr(model.config, "dirichlet_temperature_star", 0.0),
    )
    components = None
    if compute_components:
        components = _compute_loss_components(
            model,
            next_temperature,
            source_temperature,
            graph,
            static_state,
            epoch=epoch,
        )
    return graph, source_temperature, next_temperature, components


def _apply_teacher_forcing_supervision(components, supervision_components, *, include_general_metrics=True):
    components["loss_supervised"] = components["loss_supervised"] + supervision_components["loss_supervised"]
    components["loss_teacher_forcing_temperature"] = supervision_components["loss_temperature"]
    if include_general_metrics:
        components["loss_temperature"] = supervision_components["loss_temperature"]
        components["fem_temperature_rmse"] = supervision_components["fem_temperature_rmse"]
        components["fem_temperature_mae"] = supervision_components["fem_temperature_mae"]
        components["fem_temperature_max_error"] = supervision_components["fem_temperature_max_error"]
    else:
        components["loss_temperature"] = components["loss_temperature"] + supervision_components["loss_temperature"]


def _apply_rollout_supervision(components, supervision_components):
    components["loss_supervised"] = components["loss_supervised"] + supervision_components["loss_supervised"]
    components["loss_temperature"] = supervision_components["loss_temperature"]
    components["loss_rollout_temperature"] = supervision_components["loss_temperature"]
    components["fem_temperature_rmse"] = supervision_components["fem_temperature_rmse"]
    components["fem_temperature_mae"] = supervision_components["fem_temperature_mae"]
    components["fem_temperature_max_error"] = supervision_components["fem_temperature_max_error"]
    components["rollout_fem_temperature_rmse"] = supervision_components["fem_temperature_rmse"]
    components["rollout_fem_temperature_mae"] = supervision_components["fem_temperature_mae"]
    components["rollout_fem_temperature_max_error"] = supervision_components["fem_temperature_max_error"]


def _apply_peak_supervision(components, peak_components):
    components["loss_supervised"] = components["loss_supervised"] + peak_components["loss_supervised"]
    components["loss_peak_temperature_rise"] = peak_components["loss_peak_temperature_rise"]
    components["peak_temperature_rise_pred"] = peak_components["peak_temperature_rise_pred"]
    components["peak_temperature_rise_fem"] = peak_components["peak_temperature_rise_fem"]
    components["peak_temperature_rise_error"] = peak_components["peak_temperature_rise_error"]
    components["peak_temperature_rise_abs_error"] = peak_components["peak_temperature_rise_abs_error"]
    components["peak_temperature_rise_rmse"] = peak_components["peak_temperature_rise_rmse"]
    components["lambda_peak_temperature_rise"] = peak_components["lambda_peak_temperature_rise"]


def _supervision_mode(supervision_config) -> str:
    return str(getattr(supervision_config, "mode", "teacher_forcing")).strip().lower()


def _supervision_window_size(config: TrainConfig, supervision_config, mode: str) -> int:
    if mode in {"rollout", "mixed"} and getattr(supervision_config, "rollout_window", None) is not None:
        return int(supervision_config.rollout_window)
    return int(config.tbptt_window)


def _peak_supervision_window_size(config: TrainConfig, peak_supervision_config) -> int:
    if getattr(peak_supervision_config, "rollout_window", None) is not None:
        return int(peak_supervision_config.rollout_window)
    return int(config.tbptt_window)


@torch.no_grad()
def evaluate_static_topology_sequence(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    config: TrainConfig,
    *,
    epoch: int = 0,
    slice_index: int = 0,
    monitor_frame_index: Optional[int] = None,
):
    """无梯度评估一个 HDF5 切片并返回平均损失分量和监控快照。"""

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    current_temperature = feature_builder.initial_temperature()
    if int(config.warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, current_temperature)
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(config.warmup_steps),
        )

    component_records = []
    snapshot = None
    try:
        for frame_idx in range(frame_reader.num_frames):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)
            physical_source_delta = graph_physical_source_delta(graph, model.config)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + delta_t_source,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(
                graph,
                source_temperature,
                delta_t_source_star=physical_source_delta,
            )
            next_temperature = apply_dirichlet_boundary(
                source_temperature + model(graph),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            components = _compute_loss_components(
                model,
                next_temperature,
                source_temperature,
                graph,
                static_state,
                epoch=epoch,
            )
            component_records.append(_detach_loss_record(components))
            if _should_capture_frame(frame_idx, monitor_frame_index, frame_reader.num_frames):
                snapshot = _build_monitor_snapshot(
                    graph,
                    components["residual"],
                    next_temperature,
                    feature_builder.scale_params,
                    frame_idx=frame_idx,
                )
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    record = {
        "epoch": int(epoch),
        "slice_index": int(slice_index),
        **_aggregate_component_records(component_records),
        **_temperature_stats(current_temperature, feature_builder.scale_params),
    }
    record["loss"] = record["loss_total"]
    return record, {"snapshot": snapshot} if snapshot is not None else {}


def _compute_loss_components(
    model,
    next_temperature,
    current_temperature,
    graph,
    static_state: StaticGraphState,
    *,
    epoch: int = 0,
):
    components = total_loss(
        T_next=next_temperature,
        T_current=current_temperature,
        v_scan_star=graph.global_attr,
        q_surface_star=graph_surface_heat_source(graph),
        delta_t_source_star=graph_residual_source_delta(graph, model.config),
        dt_star=model.config.dt_star,
        edge_index=static_state.edge_index,
        edge_attr=graph.edge_attr,
        boundary_nodes=static_state.boundary_nodes,
        lambda_pde=model.config.lambda_pde,
        inverse_pe=model.config.inverse_pe,
        k_ratio=model.config.k_ratio,
        lambda_outflow=model.config.lambda_outflow,
        gradient_regularization=model.config.gradient_regularization,
        dirichlet_temperature_star=model.config.dirichlet_temperature_star,
        residual_time_scheme=model.config.residual_time_scheme,
        convection_scale=model.config.convection_scale,
        adaptive_pde_node_weight_enabled=model.config.adaptive_pde_node_weight_enabled,
        adaptive_pde_node_weight_scheme=model.config.adaptive_pde_node_weight_scheme,
        adaptive_pde_node_weight_min=model.config.adaptive_pde_node_weight_min,
        pde_node_weight_temperature_star=current_temperature,
        pde_node_weight_epoch=epoch,
        temperature_pde_node_weight_beta=model.config.temperature_pde_node_weight_beta,
        temperature_pde_node_weight_max=model.config.temperature_pde_node_weight_max,
        temperature_pde_node_weight_clamp_enabled=model.config.temperature_pde_node_weight_clamp_enabled,
        temperature_pde_node_weight_threshold=model.config.temperature_pde_node_weight_threshold,
        temperature_pde_node_weight_high=model.config.temperature_pde_node_weight_high,
        adaptive_pde_node_weight_warmup_enabled=model.config.adaptive_pde_node_weight_warmup_enabled,
        adaptive_pde_node_weight_warmup_epochs=model.config.adaptive_pde_node_weight_warmup_epochs,
        return_components=True,
    )
    components["loss_physics"] = components["loss_total"]
    components["loss_supervised"] = components["loss_total"].new_zeros(())
    components["loss_peak_temperature_rise"] = components["loss_total"].new_zeros(())
    components["peak_temperature_rise_pred"] = components["loss_total"].new_zeros(())
    components["peak_temperature_rise_fem"] = components["loss_total"].new_zeros(())
    components["peak_temperature_rise_error"] = components["loss_total"].new_zeros(())
    components["peak_temperature_rise_abs_error"] = components["loss_total"].new_zeros(())
    components["peak_temperature_rise_rmse"] = components["loss_total"].new_zeros(())
    components["lambda_peak_temperature_rise"] = components["loss_total"].new_zeros(())
    components["loss_temperature"] = components["loss_total"].new_zeros(())
    components["loss_teacher_forcing_temperature"] = components["loss_total"].new_zeros(())
    components["loss_rollout_temperature"] = components["loss_total"].new_zeros(())
    components["fem_temperature_rmse"] = components["loss_total"].new_zeros(())
    components["fem_temperature_mae"] = components["loss_total"].new_zeros(())
    components["fem_temperature_max_error"] = components["loss_total"].new_zeros(())
    components["rollout_fem_temperature_rmse"] = components["loss_total"].new_zeros(())
    components["rollout_fem_temperature_mae"] = components["loss_total"].new_zeros(())
    components["rollout_fem_temperature_max_error"] = components["loss_total"].new_zeros(())
    return components


def _compute_peak_supervision_components(
    predicted_temperature_star,
    fem_temperature_star,
    valid_mask,
    scale_params: ScaleParams,
    *,
    lambda_peak: float,
    topk: int,
    warmup_epochs: int,
    epoch: int,
):
    mask = valid_mask.to(device=predicted_temperature_star.device, dtype=torch.bool).reshape_as(
        predicted_temperature_star
    )
    target = fem_temperature_star.to(
        device=predicted_temperature_star.device,
        dtype=predicted_temperature_star.dtype,
    ).reshape_as(predicted_temperature_star)
    peak_pred_star = _masked_topk_mean(predicted_temperature_star, mask, topk=int(topk))
    peak_fem_star = _masked_topk_mean(target, mask, topk=int(topk))
    error_star = peak_pred_star - peak_fem_star
    loss_peak = error_star.square()
    lambda_effective = _peak_supervision_lambda(
        lambda_peak=float(lambda_peak),
        warmup_epochs=int(warmup_epochs),
        epoch=int(epoch),
        reference=loss_peak,
    )
    loss_supervised = lambda_effective * loss_peak
    error_temperature = error_star * float(scale_params.delta_T0)
    peak_pred_temperature_rise = peak_pred_star * float(scale_params.delta_T0)
    peak_fem_temperature_rise = peak_fem_star * float(scale_params.delta_T0)
    abs_error_temperature = error_temperature.abs()
    return {
        "loss_total": loss_supervised,
        "loss_supervised": loss_supervised,
        "loss_peak_temperature_rise": loss_peak,
        "peak_temperature_rise_pred": peak_pred_temperature_rise,
        "peak_temperature_rise_fem": peak_fem_temperature_rise,
        "peak_temperature_rise_error": error_temperature,
        "peak_temperature_rise_abs_error": abs_error_temperature,
        "peak_temperature_rise_rmse": abs_error_temperature,
        "lambda_peak_temperature_rise": lambda_effective,
    }


def _init_case_peak_tracker():
    return {
        "peak_temperature_pred": None,
        "peak_temperature_fem": None,
        "peak_topk_temperature_rise_pred": None,
        "peak_topk_temperature_rise_fem": None,
    }


def _update_case_peak_tracker(
    tracker,
    predicted_temperature_star,
    fem_temperature_star,
    valid_mask,
    scale_params: ScaleParams,
    *,
    topk: int,
):
    mask = valid_mask.to(device=predicted_temperature_star.device, dtype=torch.bool).reshape_as(
        predicted_temperature_star
    )
    if int(mask.sum().detach().cpu()) <= 0:
        return

    pred = predicted_temperature_star.detach()
    fem = fem_temperature_star.detach().to(device=pred.device, dtype=pred.dtype).reshape_as(pred)
    pred_peak_star = _masked_max(pred, mask)
    fem_peak_star = _masked_max(fem, mask)
    pred_topk_star = _masked_topk_mean(pred, mask, topk=int(topk))
    fem_topk_star = _masked_topk_mean(fem, mask, topk=int(topk))

    delta_t0 = float(scale_params.delta_T0)
    t_amb = float(scale_params.T_amb)
    _update_tracker_max(
        tracker,
        "peak_temperature_pred",
        float((pred_peak_star * delta_t0 + t_amb).detach().cpu()),
    )
    _update_tracker_max(
        tracker,
        "peak_temperature_fem",
        float((fem_peak_star * delta_t0 + t_amb).detach().cpu()),
    )
    _update_tracker_max(
        tracker,
        "peak_topk_temperature_rise_pred",
        float((pred_topk_star * delta_t0).detach().cpu()),
    )
    _update_tracker_max(
        tracker,
        "peak_topk_temperature_rise_fem",
        float((fem_topk_star * delta_t0).detach().cpu()),
    )


def _finalize_case_peak_tracker(tracker, scale_params: ScaleParams):
    pred_temperature = float(tracker["peak_temperature_pred"] or 0.0)
    fem_temperature = float(tracker["peak_temperature_fem"] or 0.0)
    temperature_error = pred_temperature - fem_temperature
    pred_temperature_rise = pred_temperature - float(scale_params.T_amb)
    fem_temperature_rise = fem_temperature - float(scale_params.T_amb)
    temperature_rise_error = pred_temperature_rise - fem_temperature_rise
    pred_topk_rise = float(tracker["peak_topk_temperature_rise_pred"] or 0.0)
    fem_topk_rise = float(tracker["peak_topk_temperature_rise_fem"] or 0.0)
    topk_rise_error = pred_topk_rise - fem_topk_rise
    return {
        "case_peak_temperature_pred": pred_temperature,
        "case_peak_temperature_fem": fem_temperature,
        "case_peak_temperature_error": temperature_error,
        "case_peak_temperature_abs_error": abs(temperature_error),
        "case_peak_temperature_rmse": abs(temperature_error),
        "case_peak_temperature_rise_pred": pred_temperature_rise,
        "case_peak_temperature_rise_fem": fem_temperature_rise,
        "case_peak_temperature_rise_error": temperature_rise_error,
        "case_peak_temperature_rise_abs_error": abs(temperature_rise_error),
        "case_peak_temperature_rise_rmse": abs(temperature_rise_error),
        "case_peak_topk_temperature_rise_pred": pred_topk_rise,
        "case_peak_topk_temperature_rise_fem": fem_topk_rise,
        "case_peak_topk_temperature_rise_error": topk_rise_error,
        "case_peak_topk_temperature_rise_abs_error": abs(topk_rise_error),
        "case_peak_topk_temperature_rise_rmse": abs(topk_rise_error),
    }


def _update_tracker_max(tracker, key: str, value: float):
    if tracker[key] is None or float(value) > float(tracker[key]):
        tracker[key] = float(value)


def _masked_max(temperature_star, mask):
    values = temperature_star.reshape(-1)
    valid = mask.reshape(-1)
    valid_values = values[valid]
    if valid_values.numel() == 0:
        return values.new_zeros(())
    return valid_values.max()


def _masked_topk_mean(temperature_star, mask, *, topk: int):
    values = temperature_star.reshape(-1)
    valid = mask.reshape(-1)
    valid_count = int(valid.sum().detach().cpu())
    if valid_count <= 0:
        return values.new_zeros(())
    k = min(int(topk), valid_count)
    valid_values = values[valid]
    return torch.topk(valid_values, k=k, largest=True).values.mean()


def _peak_supervision_lambda(*, lambda_peak: float, warmup_epochs: int, epoch: int, reference):
    if int(warmup_epochs) <= 0:
        scale = 1.0
    else:
        scale = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return reference.new_tensor(float(lambda_peak) * scale)


def _compute_supervision_components(
    predicted_temperature_star,
    fem_temperature_star,
    valid_mask,
    scale_params: ScaleParams,
    *,
    lambda_temperature: float,
):
    mask = valid_mask.to(device=predicted_temperature_star.device, dtype=predicted_temperature_star.dtype).reshape_as(
        predicted_temperature_star
    )
    target = fem_temperature_star.to(
        device=predicted_temperature_star.device,
        dtype=predicted_temperature_star.dtype,
    ).reshape_as(predicted_temperature_star)
    error_star = predicted_temperature_star - target
    denominator = mask.sum().clamp_min(float(scale_params.eps))
    loss_temperature = (mask * error_star.square()).sum() / denominator

    error_temperature = error_star * float(scale_params.delta_T0)
    abs_error_temperature = error_temperature.abs() * mask
    mse_temperature = (mask * error_temperature.square()).sum() / denominator
    mae_temperature = abs_error_temperature.sum() / denominator
    max_error_temperature = abs_error_temperature.max() if mask.sum() > 0 else loss_temperature.new_zeros(())
    loss_supervised = float(lambda_temperature) * loss_temperature
    return {
        "loss_total": loss_supervised,
        "loss_supervised": loss_supervised,
        "loss_temperature": loss_temperature,
        "fem_temperature_rmse": torch.sqrt(mse_temperature),
        "fem_temperature_mae": mae_temperature,
        "fem_temperature_max_error": max_error_temperature,
    }


def _detach_loss_record(components):
    return {
        "loss_total": float(components["loss_total"].detach().cpu()),
        "loss_physics": float(components["loss_physics"].detach().cpu()),
        "loss_supervised": float(components["loss_supervised"].detach().cpu()),
        "loss_peak_temperature_rise": float(components["loss_peak_temperature_rise"].detach().cpu()),
        "peak_temperature_rise_pred": float(components["peak_temperature_rise_pred"].detach().cpu()),
        "peak_temperature_rise_fem": float(components["peak_temperature_rise_fem"].detach().cpu()),
        "peak_temperature_rise_error": float(components["peak_temperature_rise_error"].detach().cpu()),
        "peak_temperature_rise_abs_error": float(components["peak_temperature_rise_abs_error"].detach().cpu()),
        "peak_temperature_rise_rmse": float(components["peak_temperature_rise_rmse"].detach().cpu()),
        "lambda_peak_temperature_rise": float(components["lambda_peak_temperature_rise"].detach().cpu()),
        "loss_temperature": float(components["loss_temperature"].detach().cpu()),
        "loss_teacher_forcing_temperature": float(
            components["loss_teacher_forcing_temperature"].detach().cpu()
        ),
        "loss_rollout_temperature": float(components["loss_rollout_temperature"].detach().cpu()),
        "loss_pde": float(components["loss_pde"].detach().cpu()),
        "loss_outflow": float(components["loss_outflow"].detach().cpu()),
        "loss_beta": float(components["loss_beta"].detach().cpu()),
        "loss_smooth": float(components["loss_smooth"].detach().cpu()),
        "fem_temperature_rmse": float(components["fem_temperature_rmse"].detach().cpu()),
        "fem_temperature_mae": float(components["fem_temperature_mae"].detach().cpu()),
        "fem_temperature_max_error": float(components["fem_temperature_max_error"].detach().cpu()),
        "rollout_fem_temperature_rmse": float(components["rollout_fem_temperature_rmse"].detach().cpu()),
        "rollout_fem_temperature_mae": float(components["rollout_fem_temperature_mae"].detach().cpu()),
        "rollout_fem_temperature_max_error": float(
            components["rollout_fem_temperature_max_error"].detach().cpu()
        ),
    }


def _aggregate_component_records(records):
    return {
        "loss_total": _mean_records(records, "loss_total"),
        "loss_physics": _mean_records(records, "loss_physics"),
        "loss_supervised": _mean_records(records, "loss_supervised"),
        "loss_peak_temperature_rise": _mean_records(records, "loss_peak_temperature_rise"),
        "peak_temperature_rise_pred": _mean_records(records, "peak_temperature_rise_pred"),
        "peak_temperature_rise_fem": _mean_records(records, "peak_temperature_rise_fem"),
        "peak_temperature_rise_error": _mean_records(records, "peak_temperature_rise_error"),
        "peak_temperature_rise_abs_error": _mean_records(records, "peak_temperature_rise_abs_error"),
        "peak_temperature_rise_rmse": _mean_records(records, "peak_temperature_rise_rmse"),
        "lambda_peak_temperature_rise": _mean_records(records, "lambda_peak_temperature_rise"),
        "loss_temperature": _mean_records(records, "loss_temperature"),
        "loss_teacher_forcing_temperature": _mean_records(records, "loss_teacher_forcing_temperature"),
        "loss_rollout_temperature": _mean_records(records, "loss_rollout_temperature"),
        "loss_pde": _mean_records(records, "loss_pde"),
        "loss_outflow": _mean_records(records, "loss_outflow"),
        "loss_beta": _mean_records(records, "loss_beta"),
        "loss_smooth": _mean_records(records, "loss_smooth"),
        "fem_temperature_rmse": _mean_records(records, "fem_temperature_rmse"),
        "fem_temperature_mae": _mean_records(records, "fem_temperature_mae"),
        "fem_temperature_max_error": _max_records(records, "fem_temperature_max_error"),
        "rollout_fem_temperature_rmse": _mean_records(records, "rollout_fem_temperature_rmse"),
        "rollout_fem_temperature_mae": _mean_records(records, "rollout_fem_temperature_mae"),
        "rollout_fem_temperature_max_error": _max_records(records, "rollout_fem_temperature_max_error"),
    }


def _temperature_stats(temperature_star, scale_params: ScaleParams):
    temperature = temperature_from_dimensionless(temperature_star.detach(), scale_params)
    return {
        "temperature_mean": float(temperature.mean().cpu()),
        "temperature_max": float(temperature.max().cpu()),
        "temperature_min": float(temperature.min().cpu()),
        "temperature_var": float(temperature.var(unbiased=False).cpu()),
    }


def _build_monitor_snapshot(
    graph,
    residual,
    temperature_star,
    scale_params: ScaleParams,
    *,
    frame_idx: int,
    fem_temperature_star=None,
    pred_temperature_star=None,
):
    snapshot = {
        "frame_index": int(frame_idx),
        "coords": graph.pos.detach().cpu().numpy(),
        "edge_index": graph.edge_index.detach().cpu().numpy(),
        "residual": residual.detach().reshape(-1).cpu().numpy(),
        "temperature": temperature_from_dimensionless(temperature_star.detach(), scale_params).reshape(-1).cpu().numpy(),
    }
    if fem_temperature_star is not None:
        fem_temperature = temperature_from_dimensionless(fem_temperature_star.detach(), scale_params)
        snapshot["fem_temperature"] = fem_temperature.reshape(-1).cpu().numpy()
    if pred_temperature_star is not None:
        pred_temperature = temperature_from_dimensionless(pred_temperature_star.detach(), scale_params)
        snapshot["pred_temperature"] = pred_temperature.reshape(-1).cpu().numpy()
    if fem_temperature_star is not None and pred_temperature_star is not None:
        fem_temperature = temperature_from_dimensionless(fem_temperature_star.detach(), scale_params)
        pred_temperature = temperature_from_dimensionless(pred_temperature_star.detach(), scale_params)
        snapshot["temperature_error"] = (pred_temperature - fem_temperature).reshape(-1).cpu().numpy()
    return snapshot


def _read_fem_temperature_star(frame_reader: HDF5FrameReader, frame_idx: int, feature_builder: GpuFeatureBuilder):
    temperature = frame_reader.read_fem_temperature(frame_idx)
    return temperature_to_dimensionless(temperature, feature_builder.scale_params).to(
        device=feature_builder.device,
        dtype=feature_builder.dtype,
        non_blocking=feature_builder.device.type == "cuda",
    )


def _read_fem_valid_mask(frame_reader: HDF5FrameReader, frame_idx: int, feature_builder: GpuFeatureBuilder):
    return frame_reader.read_fem_valid_mask(frame_idx).to(
        device=feature_builder.device,
        dtype=feature_builder.dtype,
        non_blocking=feature_builder.device.type == "cuda",
    )


def _supervision_enabled(supervision_config) -> bool:
    return bool(supervision_config is not None and getattr(supervision_config, "enabled", False))


def _peak_supervision_enabled(peak_supervision_config) -> bool:
    return bool(peak_supervision_config is not None and getattr(peak_supervision_config, "enabled", False))


def _validate_supervised_readers(frame_readers):
    for frame_reader in frame_readers:
        if not frame_reader.has_fem_temperature:
            raise KeyError(
                f"HDF5 file {frame_reader.h5_path} is missing required FEM temperature dataset."
            )


def _should_capture_frame(frame_idx: int, monitor_frame_index: Optional[int], num_frames: int) -> bool:
    target = int(num_frames) // 2 if monitor_frame_index is None else min(int(monitor_frame_index), int(num_frames) - 1)
    return int(frame_idx) == target


def _mean_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return sum(values) / max(len(values), 1)


def _mean_values(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def _rmse_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return 0.0
    return (sum(value * value for value in values) / len(values)) ** 0.5


def _max_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return max(values) if values else 0.0


def _min_records(records, key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return min(values) if values else 0.0


@torch.no_grad()
def rollout_static_topology(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    *,
    writer: Optional[Callable[[int, torch.Tensor], None]] = None,
    return_all: bool = False,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
):
    """使用固定拓扑数据管线进行流式推理。

    参数:
        model: 已训练模型。
        frame_reader: ``HDF5FrameReader``，提供动态基础特征。
        static_state: ``StaticGraphState``，提供固定拓扑。
        feature_builder: ``GpuFeatureBuilder``，构建当前帧图特征。
        steps: 推理步数。
        scale_params: ``ScaleParams``，用于把无量纲温度还原为真实温度。
        writer: 可选回调 ``writer(step, temperature)``；每步输出会先移动到 CPU。
        return_all: 是否把所有步输出堆叠返回；大图默认应保持 ``False``。
        return_dimensionless: ``writer`` 和返回值是否使用无量纲温度。
        warmup_steps: 可选 PD-GCN 伪时间松弛步数；默认 ``0`` 表示冷态初温。

    返回:
        若 ``return_all=False``，返回 ``None``；否则返回形状 ``[steps, N, 1]``
        的温度序列张量。
    """

    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if int(steps) > frame_reader.num_frames:
        raise ValueError(f"steps={steps} exceeds available frames {frame_reader.num_frames}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    current_temperature = feature_builder.initial_temperature()
    if int(warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = _snapshot_graph_for_tbptt(
            feature_builder.build(node_base_cpu, global_cpu, current_temperature)
        )
        current_temperature = pseudo_time_relax_initial_temperature(
            model,
            warmup_graph,
            int(warmup_steps),
        )
    outputs = []
    try:
        for frame_idx in range(int(steps)):
            node_base_cpu, global_cpu = frame_reader.read_frame(frame_idx)
            graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature)
            physical_source_delta = graph_physical_source_delta(graph, model.config)
            delta_t_source = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + delta_t_source,
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            graph = clone_graph_with_temperature(
                graph,
                source_temperature,
                delta_t_source_star=physical_source_delta,
            )
            next_temperature = apply_dirichlet_boundary(
                source_temperature + model(graph),
                static_state.boundary_nodes,
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            output = next_temperature if return_dimensionless else temperature_from_dimensionless(next_temperature, scale_params)
            if writer is not None:
                writer(frame_idx, output.detach().cpu())
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def _normalize_vectors(vectors, *, eps: float):
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm


def _collect_gamma_upwind(model):
    """从模型各 EdgeBlock 收集可学习的 gamma_upwind 值。

    遍历 Processor 中每层 GnBlock 的 EdgeBlock，提取
    ``gamma_upwind`` ``nn.Parameter`` 的当前值。
    若模型非 PDGCN（如测试 mock），返回空张量。

    参数:
        model: ``PDGCN`` 模型实例或测试 mock。

    返回:
        一维 ``torch.Tensor``，包含所有 EdgeBlock 的 gamma 值。
    """
    gamma_list = []
    processor = getattr(model, "processor", None)
    if processor is not None:
        blocks = getattr(processor, "blocks", [])
        for block in blocks:
            edge_block = getattr(block, "edge_block", None)
            if edge_block is not None and hasattr(edge_block, "gamma_upwind"):
                gamma_param = edge_block.gamma_upwind
                if isinstance(gamma_param, torch.nn.Parameter):
                    gamma_list.append(gamma_param.detach().cpu())
    if not gamma_list:
        return torch.tensor([], dtype=torch.float32)
    return torch.stack(gamma_list)


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
