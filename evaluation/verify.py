"""Verify the code-only repository without CUDA, model weights or datasets."""

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".zip", ".gz", ".tar",
    ".so", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".docx", ".xlsx",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_name(name):
    path = Path(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts, f"Unsafe path: {name}")
    return path


def split_names(path):
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    require(bool(names) and len(names) == len(set(names)), f"Empty or duplicate split: {path.name}")
    for name in names:
        relative_name(name)
    return names


def main():
    files = {}
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
        for name in dirs + names:
            path = Path(directory) / name
            require(not path.is_symlink(), f"Symlink is not portable: {path.relative_to(ROOT)}")
        for name in names:
            path = Path(directory) / name
            require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"Forbidden publication file: {name}")
            require(path.stat().st_size < 50 * 1024 * 1024, f"Unexpected large file: {name}")
            files[path.relative_to(ROOT).as_posix()] = path
    require(not (ROOT / "weights").exists(), "Model weights must remain external")
    entries = {}
    for line in (ROOT / "evaluation/checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative_name(name)
        require(name not in entries, f"Duplicate checksum: {name}")
        entries[name] = expected
    require(set(files) == set(entries) | {"evaluation/checksums.sha256"}, "File inventory differs from checksums")
    for name, expected in entries.items():
        require(digest(files[name]) == expected, f"Checksum mismatch: {name}")

    with (ROOT / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require(len({row["entry_id"] for row in rows}) == len(rows), "Duplicate manifest entries")
    profiles = json.loads((ROOT / "evaluation/profiles.json").read_text())
    weights = json.loads((ROOT / "evaluation/weights.json").read_text())
    catalog = {value["file"]: value for value in weights.values()}
    require(len(catalog) == len(weights), "Duplicate weight filenames")
    bound = {row["entry_id"] for row in rows if row["status"] == "external_weights_required"}
    require(set(profiles) == bound, "Profile/entry mismatch")
    referenced = set()
    for row in rows:
        if row["status"] == "withheld":
            require(not row["weight_file"] and bool(row["note"]), "Invalid unresolved entry")
            continue
        require(row["status"] == "external_weights_required", "Unexpected distribution status")
        reference = relative_name(row["weight_file"])
        require(reference.parts[0] == "weights", "Invalid logical weight reference")
        require(not (ROOT / reference).exists(), "A model weight was bundled")
        item = catalog[row["weight_file"]]
        require(item["distribution"] == "not_included", "Weight distribution status mismatch")
        require(item["sha256"] == row["weight_sha256"], "Weight catalog/hash mismatch")
        require(len(item["sha256"]) == 64 and all(c in "0123456789abcdef" for c in item["sha256"]), "Invalid weight hash")
        referenced.add(row["weight_file"])
        split = relative_name(row["test_list"])
        require(entries[row["test_list"]] == split.stem, "Split ID/hash mismatch")
        require(len(split_names(ROOT / split)) == int(row["test_pairs"]), "Test count mismatch")
        profile = profiles[row["entry_id"]]
        require(profile["tile_size"] == profile["stride"] == 256, "Unexpected tiling")
        require(profile["tile_batch_size"] > 0 and profile["precision"] == "bfloat16", "Unexpected batching/precision")
        require(profile["decision"] == "argmax" and not profile["resizing"] and not profile["tta"], "Unexpected inference policy")
    require(referenced == set(catalog), "Unreferenced checkpoint metadata")
    release = json.loads((ROOT / "evaluation/release.json").read_text())
    require(release["weights_included"] is False, "Invalid release scope")
    require(release["referenced_unique_weights"] == len(catalog), "Weight metadata count mismatch")
    require(release["externally_bound_entries"] == len(bound), "Mapped entry count mismatch")
    require(set(release["withheld_entries"]) == {row["entry_id"] for row in rows if row["status"] == "withheld"}, "Unresolved entry mismatch")
    with (ROOT / "splits/index.csv").open() as stream:
        for row in csv.DictReader(stream):
            relative_name(row["file"])
            require(entries[row["file"]] == row["sha256"], "Split index hash mismatch")
            require(len(split_names(ROOT / row["file"])) == int(row["pairs"]), "Split count mismatch")
    print("Code-only integrity and manifest: PASS")
    print("Bundled model weights: 0")
    print("External weight references:", len(catalog))
    print("Mapped / unresolved entries:", len(bound), "/", len(rows) - len(bound))
    print("Full manuscript numerical reproduction: not certified")


if __name__ == "__main__":
    main()
