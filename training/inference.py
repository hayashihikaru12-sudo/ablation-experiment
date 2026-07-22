from typing import Sequence, Union

import torch

from data.dimensionless import temperature_from_dimensionless
from pde import apply_dirichlet_boundary

from .graph_utils import (
    clone_graph_with_temperature,
    graph_boundary_nodes,
    graph_explicit_source_delta,
    graph_physical_source_delta,
    graph_to_device,
)
from .warmup import pseudo_time_relax_initial_temperature


@torch.no_grad()
def rollout(
    model,
    graph_init_or_seq: Union[object, Sequence],
    steps: int,
    scale_params,
    *,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
):
    """Run single-layer autoregressive PD-GCN inference."""

    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")

    model_device = next(model.parameters()).device
    graphs = _as_graph_sequence(graph_init_or_seq, int(steps), model_device)
    current_temperature = pseudo_time_relax_initial_temperature(model, graphs[0], int(warmup_steps))
    predictions_star = []

    was_training = model.training
    model.eval()
    try:
        for step in range(int(steps)):
            graph = graphs[step]
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
            next_temperature = source_temperature + delta_temperature
            next_temperature = apply_dirichlet_boundary(
                next_temperature,
                graph_boundary_nodes(graph_step),
                value=getattr(model.config, "dirichlet_temperature_star", 0.0),
            )
            predictions_star.append(next_temperature)
            current_temperature = next_temperature
    finally:
        if was_training:
            model.train()

    temperature_star = torch.stack(predictions_star, dim=0)
    temperature = temperature_from_dimensionless(temperature_star, scale_params)
    if return_dimensionless:
        return {"temperature": temperature, "temperature_star": temperature_star}
    return temperature


def _as_graph_sequence(graph_init_or_seq, steps: int, device):
    """Normalize one graph or a graph sequence to a fixed-length list."""

    if isinstance(graph_init_or_seq, (list, tuple)):
        if len(graph_init_or_seq) < steps:
            raise ValueError(f"graph sequence length {len(graph_init_or_seq)} is shorter than steps={steps}.")
        return [graph_to_device(graph, device) for graph in graph_init_or_seq[:steps]]
    graph = graph_to_device(graph_init_or_seq, device)
    return [graph] * steps
