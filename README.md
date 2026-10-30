# Spatial-Resolution-Guided Building Change Detection

Evaluation code and split lists accompanying **Performance Analysis of
Spatial-Resolution-Guided Integrated Pretraining and Domain-Specific Fine-Tuning
for Building Change Detection Using Korean and International Datasets**.

## Scope

This repository provides MambaBCD evaluation code, inference profiles, recorded
training settings, prepared-data split lists and manuscript-linked metadata.
**Model weights, original imagery, ground-truth masks, training states and bulk
predictions are not included.** There is no bundled or automatic weight download.
Running inference requires separately supplied, hash-matching inference-only
weights and the corresponding prepared data.

| Path | Contents |
| --- | --- |
| `manifest.csv` | Table entries, reported F1/change IoU, checkpoint hashes and test lists |
| `evaluation/` | Evaluator, model sources, profiles, settings, dependency and license records |
| `splits/` | Train/selection/test lists and prepared-image identities |
| `CITATION.cff` | Citation metadata for GitHub and a future Zenodo deposit |

The manifest preserves the dataset-level numbers from the manuscript revision
identified by `evaluation/release.json`. It contains 21 entries with recorded
checkpoint bindings and three unresolved historical Single entries. The status
`external_weights_required` means a binding is recorded, **not** that its weights
are supplied or its manuscript value is independently certified. `weight_file`
and the `file` field in `evaluation/weights.json` are logical export names, not
download locations. Unresolved entries remain `withheld`; no replacement model
is presented as the original. Complete reproduction of all tables is not certified.

## Setup

Reference environment: Python 3.12, NVIDIA PyTorch 25.08, CUDA and a BF16-capable
NVIDIA GPU. See `evaluation/environment.json` and `evaluation/THIRD_PARTY.txt`.

```bash
python -m pip install -r evaluation/requirements.txt
python -m pip install --no-build-isolation evaluation/vendor/kernels/selective_scan
```

Build the supplied CUDA extension for your Python/PyTorch/CUDA/GPU combination.
Compiled binaries are not distributed. Local runtime checks used an already
installed compatible extension; compilation in a clean container has not been
certified.

## Data And Evaluation

Official download/access links are listed in `evaluation/datasets.csv`. Obtain
the datasets under their providers' terms and keep images and masks outside this
repository. `--data-root` names a prepared **test split** containing `A/B/label`
or `T1/T2/GT` folders, as specified by the selected profile.

The following commands do not require a GPU, weights or source imagery:

```bash
python -B evaluation/verify.py
python -B evaluation/evaluate.py --list
python -B evaluation/evaluate.py --describe single70k_aihub
```

With an external checkpoint whose SHA-256 matches the manifest:

```bash
CUDA_VISIBLE_DEVICES=0 python -B evaluation/evaluate.py \
  --entry single70k_aihub \
  --checkpoint /path/to/external/single70k_aihub.pth \
  --data-root /path/to/prepared/test \
  --output ../evaluation_runs/single70k_aihub --require-match
```

The checkpoint must be the recorded inference-only export: a dictionary with
one `model` key containing model tensors. A training checkpoint or another export
does not pass the hash check. Outputs must go to a new directory outside this
repository. `metrics.json` and `per_image.csv` are always written;
`--save-predictions` adds binary PNG masks. `--limit N` runs a smoke test only.
`--require-match` fails if a full evaluation differs from the printed F1/IoU at
two decimal places; it does not adjust the expected numbers.

Metrics pool pixels across the complete split. IoU denotes **change-class IoU,
not mIoU**. Inference uses native-resolution inputs, 256-pixel tiles/stride,
edge-aligned last windows, uniform logit averaging, BF16 and argmax, without
resizing or test-time augmentation. Tile batch sizes are recorded per profile.

`splits/source_images.csv` describes prepared inputs, not necessarily original
provider IDs. AI Hub/WHU provider-to-prepared mapping is incomplete and marked
accordingly. Automatic preparation from fresh downloads is outside the verified
scope. No historical online-crop coordinates have been invented: training crops
were sampled dynamically, whereas test windows are deterministic.

## Selection And Checks

Table 5 records selected best checkpoints after training through 70,000 steps,
including the inherited best when it was not surpassed. These are not necessarily
the final-step checkpoints. See `evaluation/continuations.json`. Historical 50k
references and newly evaluated 70k results are not automatically a paired estimate.
LEVIR-CD+ test-as-validation usage is recorded in `evaluation/training.json`;
this is not independent validation.

`evaluation/verify.py` checks code/split integrity and metadata consistency, not
full numerical reproducibility. With NumPy and PyTorch installed, run focused
tests with `python -B -m unittest discover -s tests -v`.

## Licensing And Citation

New evaluation/test code is licensed under **Apache-2.0** (root `LICENSE`).
Split lists and descriptive metadata are licensed under **CC BY 4.0**.
Third-party source retains its existing notices and terms. See
`evaluation/LICENSE_SCOPE.txt` for the file-level scope and bundled license texts.
No license to excluded imagery, masks or weights is granted by this repository.

For Zenodo compatibility, `CITATION.cff` identifies the primary software license
with the single value `Apache-2.0`. This does not change the CC BY 4.0 license of
the split lists and descriptive metadata. After import, also add CC BY 4.0 to
the Zenodo record and retain this file-level scope in its description.

Use `CITATION.cff` to cite this companion software. No Zenodo DOI has been assigned
yet. Connect this repository to Zenodo before publishing a GitHub release, then
review the deposit metadata and both applicable license scopes. A Git push alone
is not a Zenodo publication.
