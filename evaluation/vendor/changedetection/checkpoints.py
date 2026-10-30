import os
import re
from collections.abc import Mapping

import torch
import torch.distributed as dist

from changedetection.logging_utils import format_log_block

_FULL_MODEL_CHECKPOINT_PREFIXES = (
    "encoder.",
    "encoder_1.",
    "encoder_2.",
    "decoder.",
    "decoder_",
    "main_clf",
    "main_clf_",
    "aux_clf",
)


def _read_checkpoint(path):
    if not os.path.isfile(path):
        raise RuntimeError(f"No checkpoint found at '{path}'")
    return torch.load(path, map_location="cpu")


def _unwrap_model(model):
    return getattr(model, "module", model)


def _model_state_dict_for_checkpoint(model):
    model = _unwrap_model(model)
    if hasattr(model, "state_dict_for_checkpoint"):
        return model.state_dict_for_checkpoint()
    return model.state_dict()


def _model_checkpoint_metadata(model):
    model = _unwrap_model(model)
    if hasattr(model, "checkpoint_metadata"):
        return model.checkpoint_metadata()
    return {}


def _strip_prefix_if_present(state_dict, prefix):
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _normalize_state_dict(state_dict):
    normalized = dict(state_dict)
    for prefix in ("module.", "model."):
        normalized = _strip_prefix_if_present(normalized, prefix)
    return normalized


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _normalize_state_dict(value)
    if isinstance(checkpoint, Mapping):
        return _normalize_state_dict(checkpoint)
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")


def _indexed_module_count(model_state, prefix, cache):
    cached = cache.get(prefix)
    if cached is not None:
        return cached

    indices = set()
    prefix_len = len(prefix)
    for key in model_state:
        if not key.startswith(prefix):
            continue
        index_token = key[prefix_len:].split(".", 1)[0]
        if index_token.isdigit():
            indices.add(int(index_token))

    count = max(indices) + 1 if indices else 0
    cache[prefix] = count
    return count


def _resolve_change_decoder_legacy_key(key, model_state, count_cache):
    submodule_match = re.match(r"(?P<prefix>.+?\.)st_block_(?P<stage>\d+)(?P<sub>\d)\.(?P<rest>.*)$", key)
    if submodule_match:
        prefix = submodule_match.group("prefix")
        stage_count = _indexed_module_count(model_state, f"{prefix}stage_blocks.", count_cache)
        stage = int(submodule_match.group("stage"))
        new_stage_idx = stage_count - stage
        submodule_name = {"1": "cat", "2": "interleave", "3": "split"}.get(submodule_match.group("sub"))
        if submodule_name is not None and 0 <= new_stage_idx < stage_count:
            candidate = f"{prefix}stage_blocks.{new_stage_idx}.{submodule_name}.{submodule_match.group('rest')}"
            if candidate in model_state:
                return candidate

    fuse_match = re.match(r"(?P<prefix>.+?\.)fuse_layer_(?P<stage>\d+)\.(?P<rest>.*)$", key)
    if fuse_match:
        prefix = fuse_match.group("prefix")
        fuse_count = _indexed_module_count(model_state, f"{prefix}fuse_layers.", count_cache)
        stage = int(fuse_match.group("stage"))
        new_stage_idx = fuse_count - stage
        if 0 <= new_stage_idx < fuse_count:
            candidate = f"{prefix}fuse_layers.{new_stage_idx}.{fuse_match.group('rest')}"
            if candidate in model_state:
                return candidate

    smooth_match = re.match(r"(?P<prefix>.+?\.)smooth_layer_(?P<stage>\d+)\.(?P<rest>.*)$", key)
    if smooth_match:
        prefix = smooth_match.group("prefix")
        smooth_count = _indexed_module_count(model_state, f"{prefix}smooth_layers.", count_cache)
        stage = int(smooth_match.group("stage"))
        new_stage_idx = smooth_count - stage
        if 0 <= new_stage_idx < smooth_count:
            candidate = f"{prefix}smooth_layers.{new_stage_idx}.{smooth_match.group('rest')}"
            if candidate in model_state:
                return candidate

    return key


def _resolve_semantic_decoder_legacy_key(key, model_state, count_cache):
    stage_match = re.match(r"(?P<prefix>.+?\.)st_block_(?P<stage>\d+)_semantic\.(?P<rest>.*)$", key)
    if stage_match:
        prefix = stage_match.group("prefix")
        stage_count = _indexed_module_count(model_state, f"{prefix}stage_blocks.", count_cache)
        stage = int(stage_match.group("stage"))
        new_stage_idx = stage_count - stage
        if 0 <= new_stage_idx < stage_count:
            candidate = f"{prefix}stage_blocks.{new_stage_idx}.{stage_match.group('rest')}"
            if candidate in model_state:
                return candidate

    transition_match = re.match(r"(?P<prefix>.+?\.)trans_layer_(?P<stage>\d+)\.(?P<rest>.*)$", key)
    if transition_match:
        prefix = transition_match.group("prefix")
        transition_count = _indexed_module_count(model_state, f"{prefix}transition_layers.", count_cache)
        stage = int(transition_match.group("stage"))
        new_stage_idx = transition_count - stage
        if 0 <= new_stage_idx < transition_count:
            candidate = f"{prefix}transition_layers.{new_stage_idx}.{transition_match.group('rest')}"
            if candidate in model_state:
                return candidate

    smooth_match = re.match(r"(?P<prefix>.+?\.)smooth_layer_(?P<stage>\d+)_semantic\.(?P<rest>.*)$", key)
    if smooth_match:
        prefix = smooth_match.group("prefix")
        smooth_count = _indexed_module_count(model_state, f"{prefix}smooth_layers.", count_cache)
        stage = int(smooth_match.group("stage"))
        new_stage_idx = smooth_count - 1 - stage
        if 0 <= new_stage_idx < smooth_count:
            candidate = f"{prefix}smooth_layers.{new_stage_idx}.{smooth_match.group('rest')}"
            if candidate in model_state:
                return candidate

    return key


def _resolve_legacy_checkpoint_key(key, model_state, count_cache):
    remapped_key = _resolve_semantic_decoder_legacy_key(key, model_state, count_cache)
    if remapped_key != key:
        return remapped_key
    return _resolve_change_decoder_legacy_key(key, model_state, count_cache)


def _match_state_dict(model, checkpoint_state_dict):
    model = _unwrap_model(model)
    model_state = model.state_dict()
    matched = {}
    unexpected = []
    mismatched = []
    remapped_legacy_keys = []
    count_cache = {}

    for key, value in checkpoint_state_dict.items():
        resolved_key = key
        if key not in model_state:
            resolved_key = _resolve_legacy_checkpoint_key(key, model_state, count_cache)
            if resolved_key != key:
                remapped_legacy_keys.append({"source": key, "target": resolved_key})

        if resolved_key not in model_state:
            unexpected.append(key)
            continue
        if getattr(model_state[resolved_key], "shape", None) != getattr(value, "shape", None):
            mismatched.append(
                {
                    "key": resolved_key,
                    "source_key": key,
                    "model_shape": tuple(model_state[resolved_key].shape),
                    "checkpoint_shape": tuple(value.shape),
                }
            )
            continue
        matched[resolved_key] = value

    merged_state = dict(model_state)
    merged_state.update(matched)
    model.load_state_dict(merged_state)
    return {
        "loaded_keys": len(matched),
        "missing_keys": sorted(set(model_state) - set(matched)),
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "remapped_legacy_keys": remapped_legacy_keys,
    }


def load_model_weights(model, path):
    checkpoint = _read_checkpoint(path)
    load_info = _match_state_dict(model, extract_model_state_dict(checkpoint))
    load_info["path"] = path
    return load_info


def _looks_like_full_model_state_dict(state_dict):
    return any(key.startswith(_FULL_MODEL_CHECKPOINT_PREFIXES) for key in state_dict)


def load_encoder_pretrained_weights(model, path):
    checkpoint = _read_checkpoint(path)
    state_dict = extract_model_state_dict(checkpoint)
    if _looks_like_full_model_state_dict(state_dict):
        raise RuntimeError(
            "The provided checkpoint looks like a full ChangeMamba model checkpoint, not encoder-only pretrained "
            "weights. Use --model_checkpoint_path for full-model loading or --resume_training_path for training "
            "resume."
        )

    load_info = _match_state_dict(model, state_dict)
    load_info["path"] = path
    return load_info


def _preview_sequence(items, max_items=8):
    items = list(items)
    if not items:
        return "[]"
    preview_items = items[:max_items]
    if len(items) > max_items:
        preview_items.append(f"... (+{len(items) - max_items} more)")
    return preview_items


def _preview_mismatched_keys(items, max_items=6):
    if not items:
        return "[]"
    preview_items = []
    for item in items[:max_items]:
        preview_items.append(
            f"{item['key']} (ckpt={item['checkpoint_shape']}, model={item['model_shape']})"
        )
    if len(items) > max_items:
        preview_items.append(f"... (+{len(items) - max_items} more)")
    return preview_items


def _preview_remapped_keys(items, max_items=6):
    if not items:
        return "[]"
    preview_items = []
    for item in items[:max_items]:
        preview_items.append(f"{item['source']} -> {item['target']}")
    if len(items) > max_items:
        preview_items.append(f"... (+{len(items) - max_items} more)")
    return preview_items


def format_checkpoint_load_report(load_info, title="CHECKPOINT Load"):
    values = {
        "matched": load_info["loaded_keys"],
        "missing": len(load_info["missing_keys"]),
        "unexpected": len(load_info["unexpected_keys"]),
        "mismatched": len(load_info["mismatched_keys"]),
    }
    if load_info.get("remapped_legacy_keys"):
        values["legacy_remapped"] = len(load_info["remapped_legacy_keys"])
    if load_info["missing_keys"]:
        values["missing_preview"] = _preview_sequence(load_info["missing_keys"])
    if load_info["unexpected_keys"]:
        values["unexpected_preview"] = _preview_sequence(load_info["unexpected_keys"])
    if load_info["mismatched_keys"]:
        values["mismatched_preview"] = _preview_mismatched_keys(load_info["mismatched_keys"])
    if load_info.get("remapped_legacy_keys"):
        values["legacy_remap_preview"] = _preview_remapped_keys(load_info["remapped_legacy_keys"])
    meta = {"path": load_info["path"]} if load_info.get("path") else None
    return format_log_block(
        title,
        values,
        meta=meta,
    )


def _resume_error(path, detail):
    return RuntimeError(
        f"Cannot perform full training resume from '{path}': {detail} "
        "Full resume requires compatible model, optimizer, scheduler configuration, and iteration state. "
        "Use --model_checkpoint_path instead when the checkpoint should initialize model weights only."
    )


def resume_training_state(path, *, model, optimizer=None, scheduler=None):
    checkpoint = _read_checkpoint(path)
    if not isinstance(checkpoint, Mapping):
        raise _resume_error(path, f"expected a mapping checkpoint, got {type(checkpoint)!r}.")

    iteration_key = next((key for key in ("iteration", "iter") if key in checkpoint), None)
    if iteration_key is None:
        raise _resume_error(path, "iteration state is missing.")
    try:
        iteration = int(checkpoint[iteration_key])
    except (TypeError, ValueError) as exc:
        raise _resume_error(path, f"iteration state is invalid: {checkpoint[iteration_key]!r}.") from exc
    if iteration < 0:
        raise _resume_error(path, f"iteration state must be non-negative, got {iteration}.")

    optimizer_state = checkpoint.get("optimizer")
    if optimizer is None:
        raise _resume_error(path, "the current training run has no optimizer to restore.")
    if optimizer_state is None:
        raise _resume_error(path, "optimizer state is missing.")
    try:
        optimizer.load_state_dict(optimizer_state)
    except Exception as exc:
        raise _resume_error(path, f"optimizer state is incompatible: {exc}") from exc

    scheduler_state = checkpoint.get("scheduler")
    if scheduler is None and scheduler_state is not None:
        raise _resume_error(
            path,
            "the checkpoint contains scheduler state, but the current run has no scheduler; use the original "
            "scheduler configuration.",
        )
    if scheduler is not None and scheduler_state is None:
        raise _resume_error(path, "scheduler state is missing for the configured scheduler.")
    if scheduler is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception as exc:
            raise _resume_error(path, f"scheduler state is incompatible: {exc}") from exc

    load_info = _match_state_dict(model, extract_model_state_dict(checkpoint))
    scheduler_loaded = scheduler is not None

    return {
        "path": path,
        "iteration": iteration,
        "best_score": checkpoint.get("best_score"),
        "best_record": checkpoint.get("best_record"),
        "task_name": checkpoint.get("task_name"),
        "config_dump": checkpoint.get("config_dump"),
        "args_snapshot": checkpoint.get("args_snapshot"),
        "extra_state": checkpoint.get("extra_state", {}),
        "optimizer_loaded": True,
        "optimizer_error": None,
        "scheduler_loaded": scheduler_loaded,
        "scheduler_error": None,
        "has_optimizer_state": True,
        "has_scheduler_state": scheduler_state is not None,
        "has_iteration_state": True,
        "load_info": load_info,
    }


def save_training_checkpoint(
    path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    iteration=0,
    best_score=None,
    best_record=None,
    task_name=None,
    config=None,
    args=None,
    extra_state=None,
):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": _model_state_dict_for_checkpoint(model),
        "model_checkpoint_meta": _model_checkpoint_metadata(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "iteration": int(iteration),
        "best_score": best_score,
        "best_record": best_record,
        "task_name": task_name,
        "config_dump": config.dump() if config is not None and hasattr(config, "dump") else None,
        "args_snapshot": vars(args) if args is not None else None,
        "extra_state": extra_state or {},
    }
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)
