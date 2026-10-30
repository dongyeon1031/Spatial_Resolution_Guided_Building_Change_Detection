import numpy as np
import torch
import torch.nn.functional as F

def _sliding_window_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts

def _predict_sliding_single(model, pre_img, post_img, window_size, stride, window_batch_size):
    if window_size <= 0:
        raise ValueError(f"Sliding window size must be positive, got {window_size}.")
    if stride <= 0 or stride > window_size:
        raise ValueError(f"Sliding stride must be in [1, {window_size}], got {stride}.")

    _, _, height, width = pre_img.shape
    pad_h = max(window_size - height, 0)
    pad_w = max(window_size - width, 0)
    if pad_h > 0 or pad_w > 0:
        pre_img = F.pad(pre_img, (0, pad_w, 0, pad_h), value=0.0)
        post_img = F.pad(post_img, (0, pad_w, 0, pad_h), value=0.0)

    padded_h, padded_w = pre_img.shape[-2:]
    h_starts = _sliding_window_starts(padded_h, window_size, stride)
    w_starts = _sliding_window_starts(padded_w, window_size, stride)
    accum = None
    counts = None
    windows = []

    def flush_windows():
        nonlocal accum, counts, windows
        if not windows:
            return
        pre_windows = torch.cat([item[0] for item in windows], dim=0)
        post_windows = torch.cat([item[1] for item in windows], dim=0)
        logits = model(pre_windows, post_windows)
        if accum is None:
            num_classes = logits.shape[1]
            accum = logits.new_zeros((1, num_classes, padded_h, padded_w))
            counts = logits.new_zeros((1, 1, padded_h, padded_w))
        for logit, (_, _, h_start, w_start) in zip(logits, windows):
            h_end = h_start + window_size
            w_end = w_start + window_size
            accum[:, :, h_start:h_end, w_start:w_end] += logit.unsqueeze(0)
            counts[:, :, h_start:h_end, w_start:w_end] += 1.0
        windows = []

    for h_start in h_starts:
        for w_start in w_starts:
            h_end = h_start + window_size
            w_end = w_start + window_size
            windows.append((
                pre_img[:, :, h_start:h_end, w_start:w_end],
                post_img[:, :, h_start:h_end, w_start:w_end],
                h_start,
                w_start,
            ))
            if len(windows) >= window_batch_size:
                flush_windows()
    flush_windows()
    if accum is None or counts is None:
        raise RuntimeError("Sliding inference produced no windows.")
    uncovered = counts == 0
    if bool(uncovered.any().item()):
        uncovered_pixels = int(uncovered.sum().item())
        raise RuntimeError(
            f"Sliding inference left {uncovered_pixels} pixels uncovered; "
            f"window_size={window_size}, stride={stride}."
        )
    return (accum / counts)[:, :, :height, :width]

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

    return normalized_img

def to_channel_first(img):
    return np.transpose(img, (2, 0, 1))

def normalize_binary_mask(mask):
    """Convert common binary-mask encodings to float32 values in {0, 1}."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = np.max(mask[..., :3], axis=-1)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}.")
    if mask.size == 0:
        return mask.astype(np.float32)

    mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.maximum(mask, 0.0)
    max_value = float(mask.max())
    if max_value <= 1.0:
        threshold = 0.5
    elif max_value <= 255.0:
        threshold = 127.5
    else:
        threshold = max_value * 0.5
    return (mask > threshold).astype(np.float32)
