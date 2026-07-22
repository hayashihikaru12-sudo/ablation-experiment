import json
import shutil
import time
from dataclasses import MISSING, asdict, fields
from pathlib import Path

import h5py
import numpy as np
import torch

from data import HDF5FrameReader, HDF5Loader, build_graph, build_static_cache
from data.dimensionless import ScaleParams, temperature_from_dimensionless
from data.static_cache import META_FILE, STATIC_FILE
from pde import apply_dirichlet_boundary
from training.graph_utils import (
    clone_graph_with_temperature,
    graph_explicit_source_delta,
    graph_physical_source_delta,
)
from training.run_config import load_run_config, pdgcn_config_from_scale
from training.static_topology import GpuFeatureBuilder, StaticGraphState
from training.train_entry import derive_timing_from_hdf5, discover_hdf5_files
from training.warmup import pseudo_time_relax_initial_temperature
from visualization import write_surface_vtu

from .config import SingleLayerInferenceRunConfig
from .io import (
    _resolve_path,
    _should_write_cloud_step,
    load_model_from_checkpoint,
    read_hdf5_temperature_shape,
)


DEFAULT_PREDICTION_GROUP_PATH = "prediction/pdgcn_single_layer"


def run_single_layer_inference_from_config(
    config_path,
    *,
    checkpoint=None,
    h5_path=None,
    batch=None,
    h5_dir=None,
    output_path=None,
    output_dir=None,
    output_prefix=None,
    vtu_output_dir=None,
    vtu_interval=None,
    mode=None,
):
    """Run single-layer PD-GCN inference from a split inference JSON config."""

    config_path = Path(config_path)
    (
        run_config,
        inference_config,
        training_base_dir,
        inference_base_dir,
        training_config_path,
    ) = load_single_layer_inference_run_context(config_path)
    overrides = asdict(inference_config)
    if mode is not None:
        overrides["mode"] = str(mode)
    if vtu_interval is not None:
        overrides["vtu_interval"] = int(vtu_interval)
    if output_prefix is not None:
        overrides["output_prefix"] = str(output_prefix)
    inference_config = SingleLayerInferenceRunConfig(**overrides)

    if int(inference_config.dataset_index) >= len(run_config.datasets):
        raise IndexError(
            f"single_layer_inference.dataset_index={inference_config.dataset_index} exceeds "
            f"datasets length {len(run_config.datasets)}."
        )

    dataset = run_config.datasets[int(inference_config.dataset_index)]
    scale_params = dataset.scale.to_scale_params()
    selected_checkpoint = (
        _resolve_path(inference_base_dir, checkpoint)
        if checkpoint
        else _resolve_path(
            training_base_dir,
            run_config.outputs.checkpoint_path if run_config.outputs is not None else run_config.data.checkpoint_path,
        )
    )
    device = torch.device(run_config.training.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dir = _resolve_path(training_base_dir, dataset.cache_dir)
    batch_mode = bool(inference_config.batch_mode) if batch is None else bool(batch)
    mode_value = str(inference_config.mode).strip().lower()
    warmup_steps = _resolve_single_layer_warmup_steps(inference_config, run_config)
    require_fem = False
    prediction_group_path = str(inference_config.prediction_group_path).strip("/")

    if batch_mode:
        if h5_path or inference_config.h5_path:
            raise ValueError("single_layer batch mode uses h5_dir; do not set h5_path or --h5.")
        if output_path is not None:
            raise ValueError("single_layer batch mode uses output_dir; do not set output_path or --output.")
        selected_h5_dir = (
            _resolve_path(inference_base_dir, h5_dir or inference_config.h5_dir)
            if h5_dir or inference_config.h5_dir
            else _resolve_path(training_base_dir, dataset.h5_dir)
        )
        selected_h5_paths = discover_hdf5_files(selected_h5_dir)
        selected_output_dir_value = output_dir or inference_config.output_dir
        if not selected_output_dir_value:
            raise ValueError("single_layer batch mode requires output_dir or --output-dir.")
        selected_output_dir = _resolve_path(inference_base_dir, selected_output_dir_value)
        selected_output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        skipped_paths = set()
        runtime = None
        for candidate_h5 in selected_h5_paths:
            candidate_output = selected_output_dir / f"{inference_config.output_prefix}{candidate_h5.name}"
            try:
                runtime = _prepare_single_layer_runtime(
                    candidate_h5,
                    selected_checkpoint=selected_checkpoint,
                    cache_dir=cache_dir,
                    scale_params=scale_params,
                    scan_velocity=dataset.scan_velocity,
                    model_overrides=run_config.model,
                    device=device,
                )
                break
            except Exception as error:  # noqa: BLE001 - batch mode must summarize per-file failures.
                skipped_paths.add(candidate_h5)
                results.append(
                    {
                        "status": "failed",
                        "h5_path": str(candidate_h5),
                        "output_path": str(candidate_output),
                        "error": str(error),
                    }
                )
        if runtime is None:
            failed = [item for item in results if item["status"] == "failed"]
            return {
                "batch_mode": True,
                "checkpoint_path": str(selected_checkpoint),
                "h5_dir": str(selected_h5_dir),
                "output_dir": str(selected_output_dir),
                "output_prefix": str(inference_config.output_prefix),
                "prediction_group_path": prediction_group_path,
                "processed_count": len(results),
                "succeeded_count": 0,
                "failed_count": len(failed),
                "results": results,
            }
        model, checkpoint_payload, static_state = runtime
        feature_builder = GpuFeatureBuilder(static_state, scale_params, model_config=model.config)
        for selected_h5 in selected_h5_paths:
            if selected_h5 in skipped_paths:
                continue
            selected_output = selected_output_dir / f"{inference_config.output_prefix}{selected_h5.name}"
            selected_vtu_dir = selected_output_dir / f"{inference_config.output_prefix}{selected_h5.stem}_vtu"
            try:
                item = _run_single_layer_inference_for_h5(
                    config_path=config_path,
                    training_config_path=training_config_path,
                    selected_h5=selected_h5,
                    selected_output=selected_output,
                    selected_vtu_dir=selected_vtu_dir,
                    selected_checkpoint=selected_checkpoint,
                    checkpoint_payload=checkpoint_payload,
                    model=model,
                    static_state=static_state,
                    feature_builder=feature_builder,
                    scale_params=scale_params,
                    scan_velocity=dataset.scan_velocity,
                    inference_config=inference_config,
                    warmup_steps=warmup_steps,
                    mode_value=mode_value,
                    require_fem=require_fem,
                    prediction_group_path=prediction_group_path,
                    write_vtu=bool(inference_config.write_vtu),
                    write_fem_vtu=bool(inference_config.write_fem_vtu),
                )
                item["status"] = "succeeded"
                results.append(item)
            except Exception as error:  # noqa: BLE001 - batch mode must summarize per-file failures.
                results.append(
                    {
                        "status": "failed",
                        "h5_path": str(selected_h5),
                        "output_path": str(selected_output),
                        "error": str(error),
                    }
                )
        succeeded = [item for item in results if item["status"] == "succeeded"]
        failed = [item for item in results if item["status"] == "failed"]
        return {
            "batch_mode": True,
            "checkpoint_path": str(selected_checkpoint),
            "h5_dir": str(selected_h5_dir),
            "output_dir": str(selected_output_dir),
            "output_prefix": str(inference_config.output_prefix),
            "prediction_group_path": prediction_group_path,
            "processed_count": len(results),
            "succeeded_count": len(succeeded),
            "failed_count": len(failed),
            "results": results,
        }

    selected_h5 = (
        _resolve_path(inference_base_dir, h5_path or inference_config.h5_path)
        if h5_path or inference_config.h5_path
        else discover_hdf5_files(_resolve_path(training_base_dir, dataset.h5_dir))[0]
    )
    selected_output = _resolve_path(inference_base_dir, output_path or inference_config.output_path)
    selected_vtu_dir = (
        _resolve_path(inference_base_dir, vtu_output_dir or inference_config.vtu_output_dir)
        if vtu_output_dir or inference_config.vtu_output_dir
        else selected_output.with_name(f"{selected_output.stem}_vtu")
    )
    model, checkpoint_payload, static_state = _prepare_single_layer_runtime(
        selected_h5,
        selected_checkpoint=selected_checkpoint,
        cache_dir=cache_dir,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
        model_overrides=run_config.model,
        device=device,
    )
    feature_builder = GpuFeatureBuilder(static_state, scale_params, model_config=model.config)
    return _run_single_layer_inference_for_h5(
        config_path=config_path,
        training_config_path=training_config_path,
        selected_h5=selected_h5,
        selected_output=selected_output,
        selected_vtu_dir=selected_vtu_dir,
        selected_checkpoint=selected_checkpoint,
        checkpoint_payload=checkpoint_payload,
        model=model,
        static_state=static_state,
        feature_builder=feature_builder,
        scale_params=scale_params,
        scan_velocity=dataset.scan_velocity,
        inference_config=inference_config,
        warmup_steps=warmup_steps,
        mode_value=mode_value,
        require_fem=require_fem,
        prediction_group_path=prediction_group_path,
        write_vtu=bool(inference_config.write_vtu),
        write_fem_vtu=bool(inference_config.write_fem_vtu),
    )


def _resolve_single_layer_warmup_steps(inference_config, run_config):
    return (
        int(inference_config.warmup_steps)
        if inference_config.warmup_steps is not None
        else int(run_config.training.warmup_steps)
    )


def _prepare_single_layer_runtime(
    selected_h5,
    *,
    selected_checkpoint,
    cache_dir,
    scale_params,
    scan_velocity,
    model_overrides,
    device,
):
    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=scan_velocity)
    fallback_model_config = pdgcn_config_from_scale(
        scale_params,
        dt=timing["dt"],
        model_overrides=model_overrides,
    )
    model, checkpoint_payload = load_model_from_checkpoint(selected_checkpoint, fallback_model_config, device)
    _ensure_static_cache(selected_h5, cache_dir, scale_params, scan_velocity=scan_velocity)
    static_state = StaticGraphState.from_cache(cache_dir, device=device)
    return model, checkpoint_payload, static_state


def _run_single_layer_inference_for_h5(
    *,
    config_path,
    training_config_path,
    selected_h5,
    selected_output,
    selected_vtu_dir,
    selected_checkpoint,
    checkpoint_payload,
    model,
    static_state,
    feature_builder,
    scale_params,
    scan_velocity,
    inference_config,
    warmup_steps: int,
    mode_value: str,
    require_fem: bool,
    prediction_group_path: str,
    write_vtu: bool,
    write_fem_vtu: bool,
):
    timing = derive_timing_from_hdf5(selected_h5, scale_params, scan_velocity=scan_velocity)
    num_frames, _ = read_hdf5_temperature_shape(selected_h5)
    steps = int(inference_config.steps) if inference_config.steps is not None else int(num_frames)
    if steps > num_frames:
        raise ValueError(f"single_layer_inference.steps={steps} exceeds available frames {num_frames}.")

    selected_output = Path(selected_output)
    selected_vtu_dir = Path(selected_vtu_dir)
    temp_output = _copy_hdf5_to_prediction_output(selected_h5, selected_output)

    try:
        with HDF5FrameReader(
            selected_h5,
            expected_num_nodes=static_state.num_nodes,
            scale_params=scale_params,
            scan_velocity=scan_velocity,
            require_fem_temperature=require_fem,
            fem_temperature_dataset=inference_config.fem_temperature_dataset,
            fem_valid_mask_dataset=inference_config.fem_valid_mask_dataset,
        ) as frame_reader:
            timing_summary = write_single_layer_hdf5(
                temp_output,
                model=model,
                frame_reader=frame_reader,
                static_state=static_state,
                feature_builder=feature_builder,
                steps=steps,
                scale_params=scale_params,
                warmup_steps=warmup_steps,
                prediction_group_path=prediction_group_path,
            )

        render_summary = {
            "render_seconds": 0.0,
            "rendered_steps": [],
            "fem_rendered_steps": [],
            "fem_vtu_written": False,
            "vtu_output_dir": str(selected_vtu_dir),
        }
        if write_vtu:
            render_summary = render_single_layer_surfaces_from_hdf5(
                temp_output,
                vtu_interval=int(inference_config.vtu_interval),
                vtu_output_dir=selected_vtu_dir,
                prediction_group_path=prediction_group_path,
                source_h5=temp_output,
                scale_params=scale_params,
                scan_velocity=timing.get("velocity_speed", scale_params.v0),
                write_fem_vtu=bool(write_fem_vtu),
            )
        total_seconds = float(timing_summary["inference_seconds"]) + float(render_summary["render_seconds"])
        temp_output.replace(selected_output)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    return {
        "output_path": str(selected_output),
        "checkpoint_path": str(selected_checkpoint),
        "h5_path": str(selected_h5),
        "steps": steps,
        "mode": mode_value,
        "num_layers": 1,
        "vtu_interval": int(inference_config.vtu_interval),
        **timing_summary,
        **render_summary,
        "total_seconds": total_seconds,
        "prediction_group_path": prediction_group_path,
    }


def _copy_hdf5_to_prediction_output(source_h5, output_h5):
    source_h5 = Path(source_h5).resolve()
    output_h5 = Path(output_h5).resolve()
    if source_h5 == output_h5:
        raise ValueError("single-layer inference output_path must differ from the source HDF5 path.")
    if not source_h5.exists():
        raise FileNotFoundError(f"HDF5 file not found: {source_h5}")
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_h5.with_name(f".{output_h5.name}.tmp")
    if temp_output.exists():
        temp_output.unlink()
    shutil.copy2(source_h5, temp_output)
    return temp_output


def load_single_layer_inference_run_context(config_path):
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("single-layer inference config JSON must contain an object at the top level.")
    unknown = sorted(set(payload) - {"training_config", "single_layer_inference"})
    if unknown:
        raise ValueError(f"Unknown keys in single-layer inference config: {unknown}")
    training_config_value = payload.get("training_config")
    if not isinstance(training_config_value, str) or not training_config_value:
        raise ValueError("'training_config' must be a non-empty string path.")

    inference_base_dir = config_path.resolve().parent
    training_config_path = _resolve_path(inference_base_dir, training_config_value)
    run_config = load_run_config(training_config_path)
    inference_config = _build_single_layer_inference_run_config(payload.get("single_layer_inference"))
    return (
        run_config,
        inference_config,
        training_config_path.resolve().parent,
        inference_base_dir,
        training_config_path,
    )


def _build_single_layer_inference_run_config(value) -> SingleLayerInferenceRunConfig:
    if value is None:
        raise ValueError("Missing required 'single_layer_inference' section in inference config.")
    if not isinstance(value, dict):
        raise ValueError("'single_layer_inference' section must be an object.")

    field_defs = fields(SingleLayerInferenceRunConfig)
    valid = {field.name for field in field_defs}
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ValueError(f"Unknown keys in 'single_layer_inference' section: {unknown}")
    missing = [
        field.name
        for field in field_defs
        if field.default is MISSING and field.default_factory is MISSING and field.name not in value
    ]
    if missing:
        raise ValueError(f"Missing required keys in 'single_layer_inference' section: {missing}")
    return SingleLayerInferenceRunConfig(**dict(value))


def write_single_layer_hdf5(
    output_path,
    *,
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    warmup_steps: int,
    prediction_group_path: str = DEFAULT_PREDICTION_GROUP_PATH,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_group_path = str(prediction_group_path).strip("/")
    step_inference_seconds = []
    timing_summary = {
        "inference_seconds": 0.0,
        "render_seconds": 0.0,
        "total_seconds": 0.0,
        "average_inference_seconds": 0.0,
        "max_inference_seconds": 0.0,
        "min_inference_seconds": 0.0,
        "rendered_steps": [],
    }

    with h5py.File(output_path, "r+") as output_file:
        if prediction_group_path in output_file:
            del output_file[prediction_group_path]
        output_group = output_file.create_group(prediction_group_path)
        temperature_dataset = output_group.create_dataset(
            "temperature",
            shape=(int(steps), int(static_state.num_nodes), 1),
            dtype="float32",
        )

        start_total = time.perf_counter()

        def writer(step, temperature_star):
            temperature = temperature_from_dimensionless(temperature_star, scale_params)
            temperature_dataset[int(step)] = temperature.numpy()

        rollout_single_layer_static(
            model,
            frame_reader,
            static_state,
            feature_builder,
            int(steps),
            scale_params,
            writer=writer,
            return_all=False,
            return_dimensionless=True,
            warmup_steps=int(warmup_steps),
            timing_recorder=step_inference_seconds.append,
        )

        timing_summary["total_seconds"] = time.perf_counter() - start_total
        timing_summary["inference_seconds"] = timing_summary["total_seconds"]
        if step_inference_seconds:
            timing_summary["average_inference_seconds"] = float(np.mean(step_inference_seconds))
            timing_summary["max_inference_seconds"] = float(np.max(step_inference_seconds))
            timing_summary["min_inference_seconds"] = float(np.min(step_inference_seconds))

    return timing_summary


@torch.no_grad()
def rollout_single_layer_static(
    model,
    frame_reader: HDF5FrameReader,
    static_state: StaticGraphState,
    feature_builder: GpuFeatureBuilder,
    steps: int,
    scale_params: ScaleParams,
    *,
    writer=None,
    return_all: bool = False,
    return_dimensionless: bool = False,
    warmup_steps: int = 0,
    timing_recorder=None,
):
    if int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if int(steps) > frame_reader.num_frames:
        raise ValueError(f"steps={steps} exceeds available frames {frame_reader.num_frames}.")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    model.to(static_state.device)
    was_training = model.training
    model.eval()
    setup_start = time.perf_counter()
    current_temperature = feature_builder.initial_temperature()
    if int(warmup_steps) > 0:
        node_base_cpu, global_cpu = frame_reader.read_frame(0)
        warmup_graph = feature_builder.build(node_base_cpu, global_cpu, current_temperature).clone()
        current_temperature = pseudo_time_relax_initial_temperature(model, warmup_graph, int(warmup_steps))
    setup_seconds = time.perf_counter() - setup_start
    outputs = []
    try:
        for frame_idx in range(int(steps)):
            step_start = time.perf_counter()
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
            output = (
                next_temperature
                if return_dimensionless
                else temperature_from_dimensionless(next_temperature, scale_params)
            )
            if writer is not None:
                writer(frame_idx, output.detach().cpu())
            if return_all:
                outputs.append(output.detach().cpu())
            current_temperature = next_temperature
            if timing_recorder is not None:
                elapsed = time.perf_counter() - step_start
                if frame_idx == 0:
                    elapsed += setup_seconds
                timing_recorder(max(0.0, elapsed))
    finally:
        if was_training:
            model.train()

    if return_all:
        return torch.stack(outputs, dim=0)
    return None


def render_single_layer_surfaces_from_hdf5(
    prediction_h5,
    *,
    vtu_interval=None,
    vtu_output_dir=None,
    prediction_group_path=DEFAULT_PREDICTION_GROUP_PATH,
    source_h5=None,
    scale_params=None,
    scan_velocity=None,
    write_fem_vtu: bool = True,
):
    prediction_h5 = Path(prediction_h5)
    metadata = None
    if source_h5 is None or scale_params is None:
        try:
            metadata = _read_single_layer_prediction_metadata(
                prediction_h5,
                prediction_group_path=prediction_group_path,
            )
        except KeyError as exc:
            raise KeyError(
                "Minimal single-layer prediction HDF5 files do not store rendering metadata; "
                "pass source_h5 and scale_params when rendering from prediction/pdgcn_single_layer."
            ) from exc
    if scale_params is None:
        scale_params = ScaleParams(**metadata["scale_params"])
    elif isinstance(scale_params, dict):
        scale_params = ScaleParams(**scale_params)
    if source_h5 is None:
        source_h5 = metadata["source_h5"]
    source_h5 = Path(source_h5)
    if not source_h5.is_absolute():
        source_h5 = (prediction_h5.parent / source_h5).resolve()

    default_vtu_interval = metadata.get("vtu_interval", 20) if metadata is not None else 20
    vtu_interval = int(vtu_interval if vtu_interval is not None else default_vtu_interval)
    if vtu_interval <= 0:
        raise ValueError(f"vtu_interval must be positive, got {vtu_interval}.")
    if vtu_output_dir is not None:
        vtu_output_dir = Path(vtu_output_dir).resolve()
    else:
        default_vtu_output_dir = (
            metadata.get("vtu_output_dir") if metadata is not None else None
        ) or prediction_h5.with_name(f"{prediction_h5.stem}_vtu")
        vtu_output_dir = Path(default_vtu_output_dir)
        if not vtu_output_dir.is_absolute():
            vtu_output_dir = (prediction_h5.parent / vtu_output_dir).resolve()

    hdf5_timing = metadata.get("hdf5_timing", {}) if metadata is not None else {}
    if scan_velocity is None:
        scan_velocity = hdf5_timing.get("velocity_speed", scale_params.v0)
    loader = HDF5Loader(source_h5, scale_params=scale_params)
    rendered_steps = []
    fem_rendered_steps = []
    start_render = time.perf_counter()
    with h5py.File(prediction_h5, "r") as output_file:
        output_group = _resolve_single_layer_prediction_group(
            output_file,
            prediction_group_path=prediction_group_path,
        )
        # 输出 HDF5 是源文件副本，根级 attrs 保留 heat_source_qmax 与 velocity_speed，
        # 用它们构造文件名 Q/V 标记，便于一眼区分不同工况的可视化结果。
        qv_tag = _build_qv_tag(output_file)
        # FEM 原始温度场保存在输出 HDF5 副本的根级（fem/temperature），
        # 与推理预测组同源，可直接复用同一曲面网格生成对比 vtu。
        fem_temperature_dataset = output_file["fem/temperature"] if "fem/temperature" in output_file else None
        fem_valid_mask_dataset = (
            output_file["fem/valid_mask"]
            if fem_temperature_dataset is not None and "fem/valid_mask" in output_file
            else None
        )
        fem_available = write_fem_vtu and fem_temperature_dataset is not None

        def _maybe_write_fem(step, coords, edge_index, num_nodes):
            if not fem_available:
                return
            if int(step) >= int(fem_temperature_dataset.shape[0]):
                return
            _write_single_layer_fem_step_vtu(
                coords,
                edge_index,
                num_nodes,
                fem_temperature_dataset,
                fem_valid_mask_dataset,
                vtu_output_dir=vtu_output_dir,
                step=int(step),
                qv_tag=qv_tag,
            )
            fem_rendered_steps.append(int(step))

        if "temperature" in output_group:
            steps = int(output_group["temperature"].shape[0])
            for step in range(steps):
                if not _should_write_cloud_step(step, vtu_interval):
                    continue
                coords, edge_index, num_nodes = _build_single_layer_step_surface(
                    loader, scale_params, scan_velocity, step
                )
                _write_single_layer_step_vtu(
                    output_group,
                    coords,
                    edge_index,
                    num_nodes,
                    vtu_output_dir=vtu_output_dir,
                    step=int(step),
                    teacher_only=False,
                    qv_tag=qv_tag,
                )
                _maybe_write_fem(step, coords, edge_index, num_nodes)
                rendered_steps.append(int(step))
        elif "teacher_forcing" in output_group:
            frame_indices = np.asarray(output_group["teacher_forcing/frame_index"], dtype=np.int64)
            for teacher_index, step in enumerate(frame_indices):
                if not _should_write_cloud_step(int(step), vtu_interval):
                    continue
                coords, edge_index, num_nodes = _build_single_layer_step_surface(
                    loader, scale_params, scan_velocity, step
                )
                _write_single_layer_step_vtu(
                    output_group,
                    coords,
                    edge_index,
                    num_nodes,
                    vtu_output_dir=vtu_output_dir,
                    step=int(step),
                    teacher_only=True,
                    teacher_index=int(teacher_index),
                    qv_tag=qv_tag,
                )
                _maybe_write_fem(step, coords, edge_index, num_nodes)
                rendered_steps.append(int(step))

    return {
        "render_seconds": time.perf_counter() - start_render,
        "rendered_steps": rendered_steps,
        "fem_rendered_steps": fem_rendered_steps,
        "fem_vtu_written": bool(fem_rendered_steps),
        "vtu_output_dir": str(vtu_output_dir),
    }


def _resolve_single_layer_prediction_group(output_file, *, prediction_group_path=DEFAULT_PREDICTION_GROUP_PATH):
    prediction_group_path = str(prediction_group_path or DEFAULT_PREDICTION_GROUP_PATH).strip("/")
    if prediction_group_path and prediction_group_path in output_file:
        return output_file[prediction_group_path]
    if "temperature" in output_file or "teacher_forcing" in output_file:
        return output_file
    raise KeyError(
        f"Prediction group '{prediction_group_path}' not found and no legacy single-layer prediction root datasets exist."
    )


def _read_single_layer_prediction_metadata(prediction_h5, *, prediction_group_path=DEFAULT_PREDICTION_GROUP_PATH):
    with h5py.File(prediction_h5, "r") as output_file:
        group = _resolve_single_layer_prediction_group(
            output_file,
            prediction_group_path=prediction_group_path,
        )
        return _read_single_layer_metadata_from_group(group)


def _read_single_layer_metadata_from_group(group):
    if "metadata" in group.attrs:
        metadata_json = group.attrs["metadata"]
    else:
        metadata_json = group["metadata"][()]
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8")
    return json.loads(str(metadata_json))


def _format_filename_scalar(value, *, max_decimals: int = 6) -> str:
    """Format a physical scalar for a filename, case-style (``.`` -> ``p``).

    Integer-valued quantities render without a decimal point (``25`` -> ``25``);
    fractional values keep significant digits with ``p`` in place of ``.``
    (``0.6666667`` -> ``0p666667``), matching the source ``case_*.h5`` naming.
    """
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _build_single_layer_vtu_name(prefix: str, step: int, qv_tag: str) -> str:
    """Compose a single-layer VTU filename with optional Q/V tag.

    Without a tag: ``INF_temperature_step_000000.vtu``
    With a tag:    ``INF_temperature_step_Q0p666667_V25_000000.vtu``
    """
    if qv_tag:
        return f"{prefix}_temperature_step_{qv_tag}_{int(step):06d}.vtu"
    return f"{prefix}_temperature_step_{int(step):06d}.vtu"


def _build_qv_tag(h5_file) -> str:
    """Build a ``Q<v>_V<v>`` filename tag from HDF5 root attributes.

    ``Q`` is read from ``heat_source_qmax`` (native W/mm^2) and ``V`` from
    ``velocity_speed`` (native mm/s). When either attribute is missing the
    corresponding token is omitted, so the tag degrades gracefully rather than
    failing rendering.
    """
    tokens = []
    if "heat_source_qmax" in h5_file.attrs:
        tokens.append(f"Q{_format_filename_scalar(h5_file.attrs['heat_source_qmax'])}")
    if "velocity_speed" in h5_file.attrs:
        tokens.append(f"V{_format_filename_scalar(h5_file.attrs['velocity_speed'])}")
    return "_".join(tokens)


def _build_single_layer_step_surface(loader, scale_params, scan_velocity, step):
    """Construct the single-layer surface geometry (coords + edges) for one frame.

    Shared by the INF and FEM VTU writers so both files use an identical mesh
    and are directly comparable in ParaView.
    """
    raw = loader.load_graph_data(int(step), device=torch.device("cpu"))
    graph = build_graph(
        raw,
        scale_params,
        scan_velocity=scan_velocity,
        initial_temperature=torch.full(
            (raw.xyz.shape[0], 1),
            float(scale_params.T_amb),
            device=raw.xyz.device,
            dtype=raw.xyz.dtype,
        ),
    )
    coords = graph.pos.detach().cpu().numpy() * float(scale_params.L0)
    edge_index = graph.edge_index.detach().cpu().numpy()
    return coords, edge_index, int(graph.num_nodes)


def _write_single_layer_step_vtu(
    output_file,
    coords,
    edge_index,
    num_nodes,
    *,
    vtu_output_dir,
    step: int,
    teacher_only: bool,
    teacher_index=None,
    qv_tag: str = "",
):
    """Write the PDGCN inference temperature field as an INF_ prefixed VTU."""

    if teacher_only:
        group = output_file["teacher_forcing"]
        index = int(teacher_index)
        temperature = np.asarray(group["temperature"][index], dtype=np.float64).reshape(-1)
        temperature_star = (
            np.asarray(group["temperature_star"][index], dtype=np.float64).reshape(-1)
            if "temperature_star" in group
            else None
        )
    else:
        temperature = np.asarray(output_file["temperature"][step], dtype=np.float64).reshape(-1)
        temperature_star = (
            np.asarray(output_file["temperature_star"][step], dtype=np.float64).reshape(-1)
            if "temperature_star" in output_file
            else None
        )

    point_values = {
        "temperature": temperature,
        "time_step": np.full((num_nodes,), float(step), dtype=np.float32),
    }
    if temperature_star is not None:
        point_values["temperature_star"] = temperature_star

    write_surface_vtu(
        Path(vtu_output_dir) / _build_single_layer_vtu_name("INF", step, qv_tag),
        coords,
        point_data=point_values,
        edge_index=edge_index,
        title=f"PDGCN step {step} single layer",
    )


def _write_single_layer_fem_step_vtu(
    coords,
    edge_index,
    num_nodes,
    fem_temperature_dataset,
    fem_valid_mask_dataset,
    *,
    vtu_output_dir,
    step: int,
    qv_tag: str = "",
):
    """Write the original FEM temperature field as a FEM_ prefixed VTU.

    The scalar field is named ``temperature`` to match the INF VTU, so the two
    files can share one ParaView color map for direct side-by-side comparison.
    """
    temperature = np.asarray(fem_temperature_dataset[int(step)], dtype=np.float64).reshape(-1)
    point_values = {
        "temperature": temperature,
        "time_step": np.full((num_nodes,), float(step), dtype=np.float32),
    }
    if fem_valid_mask_dataset is not None and int(step) < int(fem_valid_mask_dataset.shape[0]):
        point_values["fem_valid_mask"] = np.asarray(
            fem_valid_mask_dataset[int(step)], dtype=np.float64
        ).reshape(-1)

    write_surface_vtu(
        Path(vtu_output_dir) / _build_single_layer_vtu_name("FEM", step, qv_tag),
        coords,
        point_data=point_values,
        edge_index=edge_index,
        title=f"FEM step {step} single layer",
    )


def _ensure_static_cache(h5_path, cache_dir, scale_params, *, scan_velocity):
    cache_dir = Path(cache_dir)
    required = [cache_dir / name for name in (STATIC_FILE, META_FILE)]
    if all(path.exists() for path in required):
        return cache_dir
    return build_static_cache(
        h5_path,
        cache_dir,
        scale_params,
        scan_velocity=scan_velocity,
    )
