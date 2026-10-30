"""Evaluate one manuscript entry using separately supplied inference-only weights."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "vendor"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_names(path):
    names = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate split entries")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe relative image name")
    return names


def checked_checkpoint(path, row):
    weight = Path(path).expanduser().resolve(strict=True)
    if not weight.is_file():
        raise ValueError("--checkpoint must name an inference-only weight file")
    if sha256(weight) != row["weight_sha256"]:
        raise ValueError("Weight checksum mismatch: the external checkpoint is not the recorded export")
    return weight


def run(args, row, profile):
    weight = checked_checkpoint(args.checkpoint, row)
    import numpy as np
    from PIL import Image
    import torch
    try:
        import selective_scan_cuda_oflex  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install evaluation/vendor/kernels/selective_scan first; see README.md") from exc
    from changedetection.models.ChangeMambaBCD import ChangeMambaBCD
    from changedetection.evaluation.binary import BinaryChangeEvaluator
    from frozen_helpers import normalize_img, to_channel_first, normalize_binary_mask, _predict_sliding_single

    if not torch.cuda.is_available():
        raise RuntimeError("The reported BF16 evaluation requires a CUDA GPU")
    output = Path(args.output).expanduser().resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("Write evaluation output outside the immutable release directory")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(args.data_root).expanduser().resolve()
    names = safe_names(ROOT / row["test_list"])
    if sha256(ROOT / row["test_list"]) != Path(row["test_list"]).stem:
        raise ValueError("Test split checksum mismatch")
    if len(names) != int(row["test_pairs"]):
        raise ValueError("Test split count differs from manifest")
    if args.limit:
        names = names[:args.limit]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    kwargs = json.loads((HERE / "model.json").read_text())
    model = ChangeMambaBCD(pretrained=None, **kwargs)
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    if set(payload) != {"model"} or not all(torch.is_tensor(v) for v in payload["model"].values()):
        raise ValueError("Expected inference-only model tensors")
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model = model.cuda().eval()
    evaluator = BinaryChangeEvaluator()
    per_image = []
    folders = profile["layout"]
    if args.save_predictions:
        (output / "predictions").mkdir()
    with torch.inference_mode():
        for index, name in enumerate(names, 1):
            arrays = []
            for folder in folders:
                path = root / folder / name
                with Image.open(path) as image:
                    arrays.append(np.array(image, dtype=np.float32))
            before, after, gt = arrays
            if before.ndim != 3 or before.shape[-1] != 3 or after.shape != before.shape:
                raise ValueError(f"Expected aligned RGB pairs: {name}")
            if profile["label_policy"] == "divide255":
                gt = (gt / 255 > 0.5).astype(np.int64)
            else:
                gt = normalize_binary_mask(gt).astype(np.int64)
            if gt.ndim != 2 or gt.shape != before.shape[:2]:
                raise ValueError(f"Image/GT shape mismatch: {name}")
            before = torch.from_numpy(np.ascontiguousarray(to_channel_first(normalize_img(before)))).unsqueeze(0).cuda().float()
            after = torch.from_numpy(np.ascontiguousarray(to_channel_first(normalize_img(after)))).unsqueeze(0).cuda().float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = _predict_sliding_single(model, before, after, profile["tile_size"], profile["stride"], profile["tile_batch_size"])
            prediction = logits.argmax(1).cpu().numpy()[0]
            old = evaluator.confusion_matrix.copy()
            evaluator.add_batch(gt, prediction)
            matrix = evaluator.confusion_matrix - old
            per_image.append(dict(image_id=name, tn=int(matrix[0, 0]), fp=int(matrix[0, 1]),
                fn=int(matrix[1, 0]), tp=int(matrix[1, 1])))
            if args.save_predictions:
                target = output / "predictions" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction * 255).astype(np.uint8)).save(target)
            if index % 100 == 0 or index == len(names):
                print(f"{row['entry_id']} {index}/{len(names)}", flush=True)
    metrics = asdict(evaluator.compute())
    full_test = not args.limit or args.limit >= int(row["test_pairs"])
    matches = full_test and all(f"{metrics[k] * 100:.2f}" == row[k + "_percent"] for k in ("f1", "iou"))
    result = dict(entry_id=row["entry_id"], image_pairs=len(names), full_test=full_test,
        metrics=metrics, expected_percent={k:row[k+"_percent"] for k in ("f1","iou")},
        matches_manuscript_at_two_decimals=matches, checkpoint_sha256=row["weight_sha256"],
        inference=profile, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        confusion=evaluator.confusion_matrix.tolist())
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    print(json.dumps(result, indent=2), flush=True)
    if args.require_match and not matches:
        raise SystemExit("Evaluation did not reproduce the manuscript; output is retained for inspection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe", metavar="ENTRY", help="Show required input layout and test split without CUDA")
    parser.add_argument("--entry")
    parser.add_argument("--checkpoint", help="External inference-only .pth export matching the manifest SHA-256; weights are not included")
    parser.add_argument("--data-root", help="Prepared test split containing the folders named by the profile")
    parser.add_argument("--output", help="New output directory outside the release")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--limit", type=int, help="Smoke-test only; never certifies a paper score")
    args = parser.parse_args()
    with (ROOT / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if args.list:
        for row in rows:
            print(row["entry_id"], row["status"], row["f1_percent"], row["iou_percent"])
        return
    if args.describe:
        matches = [r for r in rows if r['entry_id'] == args.describe]
        if len(matches) != 1:
            parser.error('Unknown entry; use --list')
        profiles = json.loads((HERE / 'profiles.json').read_text())
        print(json.dumps(dict(entry=matches[0], profile=profiles.get(args.describe)), indent=2))
        return
    if not args.entry or not args.data_root or not args.output or not args.checkpoint:
        parser.error("--entry, --checkpoint, --data-root and --output are required; weights are not included")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    matches = [r for r in rows if r["entry_id"] == args.entry]
    if len(matches) != 1:
        parser.error("Unknown entry; use --list")
    row = matches[0]
    if row["status"] != "external_weights_required":
        parser.error("This entry is withheld: " + row["note"])
    profiles = json.loads((HERE / "profiles.json").read_text())
    run(args, row, profiles[row["entry_id"]])


if __name__ == "__main__":
    main()
