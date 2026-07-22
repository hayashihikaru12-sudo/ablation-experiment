import inspect
import math
import time
from collections.abc import Mapping
from typing import Sequence, Union

import torch
from torch_geometric.data import Data

from data import build_edge_features
from data.dimensionless import temperature_from_dimensionless
from pde import apply_dirichlet_boundary
from training.graph_utils import (
    graph_boundary_nodes,
    graph_explicit_source_delta,
    graph_physical_source_delta,
    graph_to_device,
)
from training.warmup import pseudo_time_relax_initial_temperature

from .fdm import compute_layer_implicit_fdm_step


@torch.no_grad()
def rollout_multilayer_fdm(
    model,
    graph_init_or_seq: Union[object, Sequence],
    steps: int,
    scale_params,
    *,
    num_layers: int,
    layer_spacing: float,
    return_dimensionless: bool = False,
    return_all: bool = True,
    writer=None,
    warmup_steps: int = 0,
    initial_temperature_star=None,
    bottom_temperature_star: float = 0.0,
    allow_unstable_fdm: bool = False,
    layer_fiber_angles_deg=None,
    normal_offset_sign: int = -1,
    layer_batch_size=None,
    delta_smoothing_alpha: float = 0.2,
    delta_smoothing_steps: int = 1,
    use_pdgcn_inplane: bool = True,
    pdgcn_inplane_top_layer_only: bool = False,
    post_fdm_source_compensation_alpha: float = 0.0,
    post_fdm_output_layer_compensations=(),
    post_fdm_output_q_region_percent: float = 10.0,
    post_fdm_output_q_transition_percent: float = 5.0,
    timing_recorder=None,
):
    """Run multilayer PD-GCN inference coupled with implicit 1D FDM in thickness."""

    steps = int(steps)
    num_layers = int(num_layers)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if num_layers < 2:
        raise ValueError(f"num_layers must be at least 2, got {num_layers}.")
    if float(layer_spacing) <= 0:
        raise ValueError(f"layer_spacing must be positive, got {layer_spacing}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
    layer_fiber_angles_deg = _validate_layer_fiber_angles(layer_fiber_angles_deg, num_layers)
    if int(normal_offset_sign) not in (-1, 1):
        raise ValueError(f"normal_offset_sign must be -1 or 1, got {normal_offset_sign}.")
    if not 0.0 <= float(delta_smoothing_alpha) <= 1.0:
        raise ValueError(f"delta_smoothing_alpha must be in [0, 1], got {delta_smoothing_alpha}.")
    if not 0.0 <= float(post_fdm_source_compensation_alpha) <= 1.0:
        raise ValueError(
            "post_fdm_source_compensation_alpha must be in [0, 1], "
            f"got {post_fdm_source_compensation_alpha}."
        )
    output_layer_compensations = _normalize_output_layer_compensations(
        post_fdm_output_layer_compensations,
        num_layers=num_layers,
    )
    q_region_percent = float(post_fdm_output_q_region_percent)
    if not 0.0 < q_region_percent <= 100.0:
        raise ValueError(
            "post_fdm_output_q_region_percent must be in (0, 100], "
            f"got {post_fdm_output_q_region_percent}."
        )
    q_transition_percent = float(post_fdm_output_q_transition_percent)
    if q_transition_percent < 0.0 or q_region_percent + q_transition_percent > 100.0:
        raise ValueError(
            "post_fdm_output_q_transition_percent must be non-negative and "
            "q_region_percent + q_transition_percent must not exceed 100, "
            f"got {post_fdm_output_q_transition_percent}."
        )
    delta_smoothing_steps = _as_non_negative_integer(
        delta_smoothing_steps,
        "delta_smoothing_steps",
    )

    initial_setup_start = time.perf_counter()
    model_device = next(model.parameters()).device
    graph0 = _graph_for_step(graph_init_or_seq, 0, steps, model_device)
    current_temperature = _initial_multilayer_temperature(
        model,
        graph0,
        num_layers,
        initial_temperature_star=initial_temperature_star,
        warmup_steps=int(warmup_steps),
        bottom_temperature_star=bottom_temperature_star,
    )
    initial_setup_seconds = time.perf_counter() - initial_setup_start
    effective_layer_batch_size = _resolve_layer_batch_size(layer_batch_size, num_layers, model_device)

    layer_spacing_star = float(layer_spacing) / float(scale_params.L0)
    outputs = []
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            step_start = time.perf_counter()
            graph = graph0 if step == 0 else _graph_for_step(graph_init_or_seq, step, steps, model_device)
            physical_source_delta = torch.zeros_like(current_temperature)
            physical_source_delta[0] = graph_physical_source_delta(graph, model.config)
            source_delta = torch.zeros_like(current_temperature)
            source_delta[0] = graph_explicit_source_delta(graph, model.config)
            source_temperature = apply_dirichlet_boundary(
                current_temperature + source_delta,
                graph_boundary_nodes(graph),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            if bool(use_pdgcn_inplane):
                if bool(pdgcn_inplane_top_layer_only):
                    delta_net = torch.zeros_like(source_temperature)
                    delta_top = _forward_model_by_layer_batches(
                        model,
                        graph,
                        source_temperature[0:1],
                        physical_source_delta[0:1],
                        effective_layer_batch_size=1,
                        layer_spacing_star=layer_spacing_star,
                        layer_fiber_angles_deg=layer_fiber_angles_deg,
                        normal_offset_sign=int(normal_offset_sign),
                    )
                    delta_net[0:1] = _smooth_delta_by_graph(
                        delta_top,
                        graph.edge_index,
                        graph_boundary_nodes(graph),
                        alpha=float(delta_smoothing_alpha),
                        steps=delta_smoothing_steps,
                    )
                else:
                    delta_net = _forward_model_by_layer_batches(
                        model,
                        graph,
                        source_temperature,
                        physical_source_delta,
                        effective_layer_batch_size=effective_layer_batch_size,
                        layer_spacing_star=layer_spacing_star,
                        layer_fiber_angles_deg=layer_fiber_angles_deg,
                        normal_offset_sign=int(normal_offset_sign),
                    )
                    delta_net = _smooth_delta_by_graph(
                        delta_net,
                        graph.edge_index,
                        graph_boundary_nodes(graph),
                        alpha=float(delta_smoothing_alpha),
                        steps=delta_smoothing_steps,
                    )
            else:
                delta_net = torch.zeros_like(source_temperature)
            inplane_temperature = apply_dirichlet_boundary(
                source_temperature + delta_net,
                graph_boundary_nodes(graph),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )

            next_temperature = compute_layer_implicit_fdm_step(
                inplane_temperature,
                dt_star=getattr(model.config, "dt_star", 1.0),
                inverse_pe=getattr(model.config, "inverse_pe", 1.0),
                k_ratio=getattr(model.config, "k_ratio", 0.0),
                layer_spacing_star=layer_spacing_star,
                bottom_temperature_star=bottom_temperature_star,
            )
            next_temperature[0] = (
                next_temperature[0]
                + float(post_fdm_source_compensation_alpha) * source_delta[0]
            )
            next_temperature = apply_dirichlet_boundary(
                next_temperature,
                graph_boundary_nodes(graph),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            next_temperature[-1, :, :] = torch.as_tensor(
                bottom_temperature_star,
                device=next_temperature.device,
                dtype=next_temperature.dtype,
            )

            output_temperature = _apply_nonrollout_output_compensation(
                next_temperature,
                graph,
                layer_compensations=output_layer_compensations,
                delta_t0=float(scale_params.delta_T0),
                q_region_percent=q_region_percent,
                q_transition_percent=q_transition_percent,
                boundary_value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            output = (
                output_temperature
                if return_dimensionless
                else temperature_from_dimensionless(output_temperature, scale_params)
            )
            if writer is not None:
                render_seconds = _call_writer(writer, step, output.detach().cpu(), None)
            else:
                render_seconds = 0.0
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
            if timing_recorder is not None:
                inference_seconds = time.perf_counter() - step_start - float(render_seconds or 0.0)
                if step == 0:
                    inference_seconds += initial_setup_seconds
                timing_recorder(max(0.0, inference_seconds))
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def _apply_nonrollout_output_compensation(
    rollout_temperature,
    graph,
    *,
    layer_compensations,
    delta_t0: float,
    q_region_percent: float,
    q_transition_percent: float,
    boundary_value: float,
):
    """Add per-layer output-only offsets without feeding them back into rollout state."""

    if not layer_compensations:
        return rollout_temperature
    q_surface_star = getattr(graph, "q_surface_star", None)
    if q_surface_star is None:
        raise ValueError("graph.q_surface_star is required to define the high-Q compensation region.")
    q_surface_star = q_surface_star.to(device=rollout_temperature.device).reshape(-1)
    num_nodes = int(q_surface_star.numel())
    if num_nodes != int(rollout_temperature.shape[1]):
        raise ValueError(
            "graph.q_surface_star node count must match rollout temperature node count, "
            f"got {num_nodes} and {rollout_temperature.shape[1]}."
        )
    if not bool(torch.any(q_surface_star > 0).item()):
        return rollout_temperature
    compensation_weight = _smooth_high_q_compensation_weight(
        q_surface_star,
        core_percent=float(q_region_percent),
        transition_percent=float(q_transition_percent),
    ).reshape(num_nodes, 1)
    compensated = rollout_temperature.clone()
    for entry in layer_compensations:
        layer_index = int(entry["layer"]) - 1
        compensation_temperature_star = float(entry["temperature"]) / float(delta_t0)
        compensated[layer_index] = compensated[layer_index] + (
            compensation_weight.to(dtype=compensated.dtype) * compensation_temperature_star
        )
    compensated = apply_dirichlet_boundary(
        compensated,
        graph_boundary_nodes(graph),
        value=boundary_value,
    )
    return compensated


def _smooth_high_q_compensation_weight(q_surface_star, *, core_percent: float, transition_percent: float):
    """Build a shared high-Q weight with a full-strength core and smoothstep fade band."""

    q_surface_star = q_surface_star.reshape(-1)
    num_nodes = int(q_surface_star.numel())
    core_count = max(1, min(num_nodes, int(math.ceil(num_nodes * float(core_percent) / 100.0))))
    outer_percent = min(100.0, float(core_percent) + float(transition_percent))
    outer_count = max(core_count, min(num_nodes, int(math.ceil(num_nodes * outer_percent / 100.0))))
    sorted_q = torch.topk(q_surface_star, k=outer_count, largest=True).values
    q_core = sorted_q[core_count - 1]
    if outer_count == core_count or float(transition_percent) <= 0.0:
        return (q_surface_star >= q_core).to(dtype=q_surface_star.dtype)

    q_outer = sorted_q[outer_count - 1]
    scale = q_core - q_outer
    eps = torch.finfo(q_surface_star.dtype).eps
    if bool(torch.abs(scale).le(eps).item()):
        return (q_surface_star >= q_core).to(dtype=q_surface_star.dtype)
    normalized = ((q_surface_star - q_outer) / scale).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)


def _normalize_output_layer_compensations(
    value,
    *,
    num_layers: int,
):
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError("post_fdm_output_layer_compensations must be a sequence of objects.")
    try:
        entries = tuple(value)
    except TypeError as error:
        raise ValueError("post_fdm_output_layer_compensations must be a sequence of objects.") from error

    normalized = []
    layers = []
    valid_keys = {"layer", "temperature"}
    for index, entry in enumerate(entries):
        name = f"post_fdm_output_layer_compensations[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{name} must be an object.")
        unknown = sorted(set(entry) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown keys in {name}: {unknown}")
        missing = sorted(valid_keys - set(entry))
        if missing:
            raise ValueError(f"Missing required keys in {name}: {missing}")
        layer = entry["layer"]
        if isinstance(layer, bool) or int(layer) != layer or int(layer) <= 0:
            raise ValueError(f"{name}.layer must be a positive one-based integer, got {layer}.")
        layer = int(layer)
        if layer >= int(num_layers):
            raise ValueError(
                f"{name}.layer must refer to a non-bottom layer 1..{int(num_layers) - 1}, got {layer}."
            )
        temperature = float(entry["temperature"])
        if temperature < 0.0:
            raise ValueError(f"{name}.temperature must be non-negative, got {temperature}.")
        layers.append(layer)
        if temperature > 0.0:
            normalized.append(
                {
                    "layer": layer,
                    "temperature": temperature,
                }
            )
    if len(set(layers)) != len(layers):
        raise ValueError("post_fdm_output_layer_compensations must not contain duplicate layer indices.")
    return tuple(normalized)


def _resolve_layer_batch_size(layer_batch_size, num_layers: int, device):
    if layer_batch_size is not None:
        batch_size = int(layer_batch_size)
        if batch_size <= 0:
            raise ValueError(f"layer_batch_size must be positive when set, got {layer_batch_size}.")
        return min(batch_size, int(num_layers))
    if getattr(device, "type", None) == "cuda":
        return min(int(num_layers), 4)
    return int(num_layers)


def _forward_model_by_layer_batches(
    model,
    graph,
    current_temperature,
    source_delta,
    *,
    effective_layer_batch_size: int,
    layer_spacing_star: float,
    layer_fiber_angles_deg,
    normal_offset_sign: int,
):
    num_layers, num_nodes, _ = current_temperature.shape
    delta_net = None
    layer_start = 0
    batch_size = int(effective_layer_batch_size)
    while layer_start < num_layers:
        layer_end = min(num_layers, layer_start + batch_size)
        try:
            layer_indices = torch.arange(layer_start, layer_end, device=current_temperature.device, dtype=torch.long)
            graph_step = _build_multilayer_graph(
                graph,
                current_temperature[layer_start:layer_end],
                source_delta[layer_start:layer_end],
                layer_spacing_star=layer_spacing_star,
                layer_fiber_angles_deg=layer_fiber_angles_deg,
                normal_offset_sign=int(normal_offset_sign),
                layer_indices=layer_indices,
            )
            delta_chunk = model(graph_step).reshape(layer_end - layer_start, num_nodes, -1)
            if delta_chunk.shape[-1] != 1:
                raise ValueError(f"model output_size must be 1 for thermal rollout, got {delta_chunk.shape[-1]}.")
            if delta_net is None:
                delta_net = torch.empty(
                    (num_layers, num_nodes, 1),
                    device=delta_chunk.device,
                    dtype=delta_chunk.dtype,
                )
            delta_net[layer_start:layer_end] = delta_chunk
            layer_start = layer_end
        except torch.cuda.OutOfMemoryError as error:
            if getattr(current_temperature.device, "type", None) != "cuda" or batch_size <= 1:
                raise RuntimeError(
                    "CUDA out of memory during multilayer inference even with layer_batch_size=1. "
                    "Reduce graph size, run fewer layers, or use a GPU with more memory."
                ) from error
            if "graph_step" in locals():
                del graph_step
            if "delta_chunk" in locals():
                del delta_chunk
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
    return delta_net


def _smooth_delta_by_graph(delta, edge_index, boundary_nodes, *, alpha: float, steps: int):
    alpha = float(alpha)
    steps = _as_non_negative_integer(steps, "steps")
    if steps <= 0 or alpha <= 0.0:
        return delta

    if delta.ndim != 3 or delta.shape[2] != 1:
        raise ValueError(f"delta must have shape [L, N, 1], got {tuple(delta.shape)}.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
    if edge_index.numel() == 0:
        return delta

    num_layers, num_nodes, _ = delta.shape
    device = delta.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    source = edge_index[0]
    receiver = edge_index[1]
    if int(source.min().item()) < 0 or int(receiver.min().item()) < 0:
        raise ValueError("edge_index must contain non-negative node indices.")
    if int(source.max().item()) >= num_nodes or int(receiver.max().item()) >= num_nodes:
        raise ValueError(f"edge_index values must be within [0, {num_nodes - 1}].")

    update_mask = _internal_node_mask(num_nodes, boundary_nodes, device=device)
    incoming_count = torch.zeros(num_nodes, device=device, dtype=delta.dtype)
    incoming_count.index_add_(0, receiver, torch.ones_like(receiver, dtype=delta.dtype))
    update_mask = update_mask & (incoming_count > 0)
    if not bool(update_mask.any().item()):
        return delta

    current = delta[:, :, 0]
    for _ in range(steps):
        neighbor_sum = torch.zeros((num_layers, num_nodes), device=device, dtype=current.dtype)
        neighbor_sum.index_add_(1, receiver, current[:, source])
        neighbor_mean = neighbor_sum / incoming_count.clamp_min(1.0).reshape(1, num_nodes)
        smoothed = current.clone()
        smoothed[:, update_mask] = (1.0 - alpha) * current[:, update_mask] + alpha * neighbor_mean[:, update_mask]
        current = smoothed
    return current.unsqueeze(-1)


def _as_non_negative_integer(value, name: str):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    try:
        int_value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.") from error
    if int_value < 0 or int_value != value:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")
    return int_value


def _internal_node_mask(num_nodes: int, boundary_nodes, *, device):
    mask = torch.ones(int(num_nodes), device=device, dtype=torch.bool)
    if not boundary_nodes:
        return mask
    selected = [
        torch.as_tensor(boundary_nodes[name], device=device, dtype=torch.long).reshape(-1)
        for name in ("upwind", "side", "downwind")
        if name in boundary_nodes and boundary_nodes[name] is not None
    ]
    if not selected:
        return mask
    boundary = torch.unique(torch.cat(selected, dim=0))
    if boundary.numel() > 0:
        mask[boundary] = False
    return mask


def _call_writer(writer, step, output, graph_step):
    if _writer_accepts_graph(writer):
        result = writer(step, output, graph_step)
    else:
        result = writer(step, output)
    return 0.0 if result is None else float(result)


def _writer_accepts_graph(writer):
    try:
        signature = inspect.signature(writer)
    except (TypeError, ValueError):
        return True
    positional_count = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional_count += 1
    return positional_count >= 3


def _graph_for_step(graph_init_or_seq, step: int, steps: int, device):
    if callable(graph_init_or_seq) and not isinstance(graph_init_or_seq, Data):
        return graph_to_device(graph_init_or_seq(int(step)), device)
    if isinstance(graph_init_or_seq, (list, tuple)):
        if len(graph_init_or_seq) < steps:
            raise ValueError(f"graph sequence length {len(graph_init_or_seq)} is shorter than steps={steps}.")
        return graph_to_device(graph_init_or_seq[int(step)], device)
    return graph_to_device(graph_init_or_seq, device)


def _initial_multilayer_temperature(
    model,
    graph,
    num_layers: int,
    *,
    initial_temperature_star,
    warmup_steps: int,
    bottom_temperature_star: float,
):
    device = graph.x.device
    dtype = graph.x.dtype
    if initial_temperature_star is not None:
        initial = torch.as_tensor(initial_temperature_star, device=device, dtype=dtype)
        if initial.shape != (num_layers, graph.num_nodes, 1):
            raise ValueError(
                "initial_temperature_star must have shape "
                f"({num_layers}, {graph.num_nodes}, 1), got {tuple(initial.shape)}."
            )
        output = initial.clone()
    else:
        if warmup_steps > 0:
            top_temperature = pseudo_time_relax_initial_temperature(model, graph, int(warmup_steps))
        else:
            top_temperature = graph.x[:, 6:7]
        output = torch.full(
            (num_layers, graph.num_nodes, 1),
            float(bottom_temperature_star),
            device=device,
            dtype=dtype,
        )
        output[0] = top_temperature.to(device=device, dtype=dtype)
    output[-1, :, :] = torch.as_tensor(bottom_temperature_star, device=device, dtype=dtype)
    return output


def _build_multilayer_graph(
    graph,
    temperature_star,
    source_delta_star=None,
    *,
    layer_spacing_star: float = 0.0,
    layer_fiber_angles_deg=None,
    normal_offset_sign: int = -1,
    layer_indices=None,
):
    return _build_multilayer_graph_impl(
        graph,
        temperature_star,
        source_delta_star,
        layer_spacing_star=layer_spacing_star,
        layer_fiber_angles_deg=layer_fiber_angles_deg,
        normal_offset_sign=normal_offset_sign,
        layer_indices=layer_indices,
    )


def _build_multilayer_graph_impl(
    graph,
    temperature_star,
    source_delta_star=None,
    *,
    layer_spacing_star: float,
    layer_fiber_angles_deg,
    normal_offset_sign: int,
    layer_indices,
):
    num_layers, num_nodes, _ = temperature_star.shape
    edge_index = graph.edge_index
    edge_count = edge_index.shape[1]
    device = graph.x.device

    multilayer_geometry = _build_multilayer_geometry(
        graph,
        num_layers,
        layer_spacing_star=float(layer_spacing_star),
        layer_fiber_angles_deg=layer_fiber_angles_deg,
        normal_offset_sign=int(normal_offset_sign),
        layer_indices=layer_indices,
    )

    x = graph.x.repeat(num_layers, 1)
    x[:, 0:3] = multilayer_geometry["pos"].reshape(num_layers * num_nodes, 3).to(device=device, dtype=x.dtype)
    x[:, 3:6] = multilayer_geometry["fiber"].reshape(num_layers * num_nodes, 3).to(device=device, dtype=x.dtype)
    x[:, 6:7] = temperature_star.reshape(num_layers * num_nodes, 1).to(device=device, dtype=x.dtype)
    _apply_multilayer_source_features(x, graph, layer_indices, num_nodes, source_delta_star)

    offsets = (torch.arange(num_layers, device=device, dtype=edge_index.dtype) * num_nodes).reshape(num_layers, 1, 1)
    edge_index_batched = (edge_index.reshape(1, 2, edge_count) + offsets).permute(1, 0, 2).reshape(
        2,
        num_layers * edge_count,
    )
    if multilayer_geometry["normal"] is not None and getattr(graph, "velocity_direction", None) is not None:
        edge_attr = build_edge_features(
            x[:, 0:3],
            edge_index_batched,
            x[:, 3:6],
            multilayer_geometry["normal"].reshape(num_layers * num_nodes, 3).to(device=device, dtype=x.dtype),
            graph.velocity_direction,
        )
    else:
        edge_attr = graph.edge_attr.repeat(num_layers, 1)

    data = Data(
        x=x,
        edge_index=edge_index_batched,
        edge_attr=edge_attr,
        global_attr=graph.global_attr,
    )
    data.num_nodes = num_layers * num_nodes
    data.layer_count = num_layers
    data.nodes_per_layer = num_nodes
    if hasattr(graph, "node_type"):
        data.node_type = graph.node_type.repeat(num_layers)
    data.pos = x[:, 0:3]
    if multilayer_geometry["normal"] is not None:
        data.normal = multilayer_geometry["normal"].reshape(num_layers * num_nodes, 3)
    if getattr(graph, "velocity_direction", None) is not None:
        data.velocity_direction = graph.velocity_direction
    data.q_feature_index = int(getattr(graph, "q_feature_index", -1))
    data.delta_t_source_feature_index = int(getattr(graph, "delta_t_source_feature_index", -1))
    data.include_q_in_features = bool(getattr(graph, "include_q_in_features", False))
    data.include_delta_t_source_in_features = bool(
        getattr(graph, "include_delta_t_source_in_features", False)
    )
    return data


def _apply_multilayer_source_features(x, graph, layer_indices, num_nodes: int, source_delta_star):
    inferred_layers = x.shape[0] // int(num_nodes)
    layer_indices = _as_layer_indices(layer_indices, inferred_layers, x.device, dtype=torch.long)
    num_layers = int(layer_indices.numel())

    q_index = int(getattr(graph, "q_feature_index", -1))
    if q_index >= 0 and x.shape[1] > q_index:
        x[:, q_index : q_index + 1] = 0.0
        top_mask = layer_indices == 0
        if bool(top_mask.any().item()):
            q_source = _graph_surface_heat_source_for_features(graph).to(device=x.device, dtype=x.dtype)
            q_view = x[:, q_index : q_index + 1].reshape(num_layers, num_nodes, 1)
            q_view[top_mask] = q_source.reshape(1, num_nodes, 1)

    delta_index = int(getattr(graph, "delta_t_source_feature_index", -1))
    if delta_index >= 0 and x.shape[1] > delta_index:
        x[:, delta_index : delta_index + 1] = 0.0
        if source_delta_star is not None:
            delta = source_delta_star.to(device=x.device, dtype=x.dtype).reshape(num_layers, num_nodes, 1)
            x[:, delta_index : delta_index + 1] = delta.reshape(num_layers * num_nodes, 1)


def _graph_surface_heat_source_for_features(graph):
    if hasattr(graph, "q_surface_star"):
        return graph.q_surface_star.reshape(graph.num_nodes, 1)
    return graph.x.new_zeros(graph.num_nodes, 1)


def _validate_layer_fiber_angles(layer_fiber_angles_deg, num_layers: int):
    if layer_fiber_angles_deg is None:
        return [0.0] * int(num_layers)
    if len(layer_fiber_angles_deg) != int(num_layers):
        raise ValueError(
            "layer_fiber_angles_deg length must match num_layers, "
            f"got {len(layer_fiber_angles_deg)} for num_layers={num_layers}."
        )
    angles = [float(angle) for angle in layer_fiber_angles_deg]
    if abs(angles[0]) > 1e-12:
        raise ValueError("layer_fiber_angles_deg[0] must be 0.0 for the base layer.")
    return angles


def _build_multilayer_geometry(
    graph,
    num_layers: int,
    *,
    layer_spacing_star: float,
    layer_fiber_angles_deg,
    normal_offset_sign: int,
    layer_indices=None,
):
    if layer_fiber_angles_deg is None:
        layer_fiber_angles_deg = [0.0] * int(num_layers)
    pos0 = getattr(graph, "pos", None)
    if pos0 is None:
        pos0 = graph.x[:, 0:3]
    fiber0 = graph.x[:, 3:6]
    normal0 = getattr(graph, "normal", None)

    absolute_layer_indices = _as_layer_indices(layer_indices, num_layers, pos0.device, dtype=torch.long)

    if normal0 is None:
        return {
            "pos": pos0.reshape(1, graph.num_nodes, 3).repeat(num_layers, 1, 1),
            "fiber": fiber0.reshape(1, graph.num_nodes, 3).repeat(num_layers, 1, 1),
            "normal": None,
        }

    normal_unit = _normalize_vectors(normal0.to(device=pos0.device, dtype=pos0.dtype))
    layer_index = absolute_layer_indices.to(device=pos0.device, dtype=pos0.dtype).reshape(num_layers, 1, 1)
    pos_layers = pos0.reshape(1, graph.num_nodes, 3) + (
        float(normal_offset_sign) * layer_index * float(layer_spacing_star) * normal_unit.reshape(1, graph.num_nodes, 3)
    )

    angle_values = torch.as_tensor(layer_fiber_angles_deg, device=pos0.device, dtype=pos0.dtype)
    if int(absolute_layer_indices.max().item()) >= int(angle_values.numel()):
        raise ValueError(
            "layer_fiber_angles_deg length must cover all requested absolute layer indices, "
            f"got {angle_values.numel()} angles for max layer index {int(absolute_layer_indices.max().item())}."
        )
    angles = angle_values[absolute_layer_indices].reshape(num_layers, 1, 1)
    angles = torch.deg2rad(angles)
    fiber_layers = _rotate_vectors_about_axis(
        fiber0.reshape(1, graph.num_nodes, 3).expand(num_layers, -1, -1),
        normal_unit.reshape(1, graph.num_nodes, 3).expand(num_layers, -1, -1),
        angles,
    )
    normal_layers = normal_unit.reshape(1, graph.num_nodes, 3).expand(num_layers, -1, -1)
    return {"pos": pos_layers, "fiber": fiber_layers, "normal": normal_layers}


def _as_layer_indices(layer_indices, num_layers: int, device, dtype):
    if layer_indices is None:
        return torch.arange(int(num_layers), device=device, dtype=dtype)
    indices = torch.as_tensor(layer_indices, device=device, dtype=dtype).reshape(-1)
    if indices.numel() != int(num_layers):
        raise ValueError(
            f"layer_indices length must match temperature layer count {num_layers}, got {indices.numel()}."
        )
    if indices.numel() == 0 or int(indices.min().item()) < 0:
        raise ValueError("layer_indices must contain non-negative layer indices.")
    return indices


def _rotate_vectors_about_axis(vectors, axes, angles):
    cos_angle = torch.cos(angles)
    sin_angle = torch.sin(angles)
    rotated = (
        vectors * cos_angle
        + torch.cross(axes, vectors, dim=-1) * sin_angle
        + axes * (axes * vectors).sum(dim=-1, keepdim=True) * (1.0 - cos_angle)
    )
    return _normalize_vectors(rotated)


def _normalize_vectors(vectors, eps: float = 1e-12):
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
