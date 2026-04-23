# CoDA: A Cognitive-Inspired Approach for Domain Adaptation

Reference implementation for the paper

> **CoDA: A Cognitive-Inspired Approach for Domain Adaptation**
> Cavide Balki Gemirter, Emin Erkan Korkmaz, Dionysis Goularas
> Yeditepe University, Department of Computer Engineering
> Submitted to *Applied Sciences* (MDPI), April 2026.
> Full manuscript: [`CoDA.pdf`](./CoDA.pdf)

> [!IMPORTANT]
> ### 📣 Citation is mandatory
>
> **If you use this code, the Fruits-3D dataset, the 3D assets, or any
> derived artefact — in academic work, a technical report, a product, a
> blog post, a tutorial, or a fork — you must cite the paper.**
>
> Copy the BibTeX entry from the [Citation](#citation) section at the bottom
> of this README, or use GitHub's "Cite this repository" button (backed by
> [`CITATION.cff`](./CITATION.cff)). Redistributions of the 3D models and
> the Fruits-3D renders are conditional on this attribution.

CoDA is a synthetic-to-real Unsupervised Domain Adaptation (UDA) framework
that takes inspiration from infant cognitive development. It generates a
shape-biased synthetic source dataset from textured 3D assets, trains a
classifier from scratch with a Network Stability Scheduler (NSS), and adapts
the model to unlabeled real-world targets via Dynamic Top-K Pseudo-Labeling.
On VegFru, Fruits-262, and Open Images v7 (5-class subsets), the framework
matches or exceeds an ImageNet-pretrained baseline while training on only
12,000 synthetic images.

The codebase is organised as two independent modules behind a shared
launcher:

- **`image_gen_3d_to_2d/`** — Phase I: Blender-based renderer that turns
  textured `.usdz` assets into 2D training views across four rendering modes
  (HDR Nature, HDR Studio, Solid Color, Sculpture).
- **`train_eval/`** — Phases II and III: source-only training, joint
  optimisation with pseudo-labels, and multi-target evaluation.

---

## Repository layout

```
CoDA/
├── README.md                  This file.
├── CoDA.pdf                   The paper (read alongside this README).
├── run.py                     Unified Python launcher (subcommands).
├── run.sh                     Background launcher (nohup + log + auto-shutdown).
├── requirements.txt           Shared Python dependencies.
├── .env.template              Template for local secrets (copy to .env).
├── .gitignore
│
├── scripts/
│   └── setup.sh               One-shot bootstrap: pyenv, .venv, CUDA, deps.
│
├── train_eval/                Phases II + III: train and evaluate.
│   ├── main.py                Module entrypoint (called by run.py).
│   ├── yamls/                 4 paper-aligned configs.
│   │   ├── train_source.yaml         Phase II: source-only training.
│   │   ├── train_target.yaml         Phase III: pseudo-label adaptation.
│   │   ├── train_upper_target.yaml   Supervised target upper-bound.
│   │   └── eval.yaml                 Multi-target evaluation.
│   ├── configs/               Typed config schema (YAML → dataclasses).
│   ├── data/                  Datasets, stages, types, stats, losses, lambdas.
│   ├── managers/              Train, evaluation, checkpoint, tensorboard.
│   ├── model/                 Feature extractor, classifier, model wrapper.
│   └── util/                  Logging, file I/O, device detection.
│
└── image_gen_3d_to_2d/        Phase I: synthesise 2D views from 3D assets.
    ├── main.py                Module entrypoint (called by run.py).
    ├── config_loader.py       Reads .cfg files into the typed config tree.
    ├── render_pipeline.py     Walks .usdz files, renders, logs success/failure.
    ├── renderer.py            Per-model Blender renderer.
    ├── render_logger.py       Resumable success/failure CSV writer.
    ├── usdz_tools.py          .usdz unpacking helpers.
    ├── utils.py               Colour, image, and geometry helpers.
    ├── neutral_colors.csv     Neutral background palette (Sculpture Mode).
    ├── solid_colors.csv       Desaturated background palette (Solid Color Mode).
    ├── cfgs/                  4 paper-aligned rendering configs.
    │   ├── hdr_nature.cfg     HDR maps of natural landscapes.
    │   ├── hdr_studio.cfg     HDR maps of indoor studio configurations.
    │   ├── solid_color.cfg    Desaturated solid-colour backgrounds.
    │   └── sculpture.cfg      Texture stripped; geometry-only.
    ├── configs/               Typed config schema (.cfg sections → dataclasses).
    ├── blender/               Blender ops: camera, lighting, material, pose, etc.
    ├── data/                  Backgrounds, bounds, XY/XYZ samplers.
    └── values/                Sampling distributions (camera, light, materials).
```

---

## Setup

```bash
git clone <your-fork-url> CoDA
cd CoDA
cp .env.template .env          # then edit .env with your own values
./scripts/setup.sh
```

`scripts/setup.sh` is idempotent and works on macOS (Homebrew + pyenv) and
Ubuntu (apt + pyenv + NVIDIA driver R550 + CUDA 12.4). It pins Python
**3.10.14**, creates `.venv/`, installs everything in `requirements.txt`, and
bootstraps `.env` from `.env.template` if you haven't already.

Manual install (if you skip the script):

```bash
pyenv install 3.10.14 && pyenv shell 3.10.14
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Phase I — Synthetic source generation (`gen_2d`)

Renders 2D views from a folder of `.usdz` 3D assets per the paper's four
rendering modes (Section 3.1.2, Figure 4). The pipeline samples poses,
lights, materials, and backgrounds per the active config; resumable via
`success.csv` MD5 lookups.

### Quick start

```bash
./.venv/bin/python run.py gen_2d --cfg hdr_nature.cfg
```

`--cfg` accepts either a bare filename (resolved against
`image_gen_3d_to_2d/cfgs/`) or an absolute path.

### Background

```bash
./run.sh gen_2d hdr_nature.cfg
```

### The four rendering modes

| File              | Paper mode      | Background           | Per-class quota | Notes                                       |
| ----------------- | --------------- | -------------------- | --------------- | ------------------------------------------- |
| `hdr_nature.cfg`  | HDR Nature      | HDR (Nature maps)    | 100 / class     | High-frequency foliage, non-uniform light.  |
| `hdr_studio.cfg`  | HDR Studio      | HDR (Studio maps)    | 50 / class      | Soft, controlled product-photo lighting.    |
| `solid_color.cfg` | Solid Color     | Desaturated solid    | 50 / class      | Foreground-isolation curriculum.            |
| `sculpture.cfg`   | Sculpture       | Neutral solid        | 40 / class      | Texture stripped → forces shape learning.   |

The total of 240 images per 3D object yields the **Fruits-3D** dataset
(12,000 images for 50 SketchFab assets across 5 classes — paper §3.1.3).

### Phase I → Phase II handoff

Each `gen_2d` run writes to its own `destination_folder` (e.g.
`./data/SketchFab_HDRn/`). `train_eval` expects a **single** source folder
with the four mode outputs consolidated side-by-side, named to encode the
per-class quota. With the default configs, the expected folder name matches
the `source_folder` set in `train_eval/yamls/*.yaml`:

```
./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40/
├── apple/       ← images from all 4 gen_2d destinations, merged per class
├── banana/
├── pineapple/
├── pomegranate/
└── pumpkin/
```

Run all four modes and merge:

```bash
./run.sh gen_2d hdr_nature.cfg
./run.sh gen_2d hdr_studio.cfg
./run.sh gen_2d solid_color.cfg
./run.sh gen_2d sculpture.cfg

mkdir -p ./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40
rsync -a ./data/SketchFab_HDRn/     ./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40/
rsync -a ./data/SketchFab_HDRs/     ./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40/
rsync -a ./data/SketchFab_SC/       ./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40/
rsync -a ./data/SketchFab_Sculpture/ ./data/Fruits-3D/hdr_nature_n100_hdr_studio_n50_solid_color_n50_sculpture_n40/
```

If you rename the folder, update `storage.source_folder` in all four YAMLs
to match.

### `.cfg` schema (top-level sections)

Defined in `image_gen_3d_to_2d/config_loader.py`:

| Section            | Class                  | What it controls                                              |
| ------------------ | ---------------------- | ------------------------------------------------------------- |
| `default`          | `DefaultConfig`        | `cleanup`, `debug`, `seed`, source / destination / log paths. |
| `blender`          | `BlenderConfig`        | World colour, tiny-object epsilon, island epsilon.            |
| `render`           | `RenderConfig`         | Resolution, device (`mps`/`cuda`/`cpu`), samples, exposure, art toggles. |
| `post_process`     | `PostProcessConfig`    | Min foreground ratio.                                         |
| `lighting`         | `LightingConfig`       | Number of lights, energy ranges, HDR dir, temperature.        |
| `camera`           | `CameraConfig`         | Azimuth / elevation / radius, intrinsics, depth-of-field.     |
| `pose`             | `PoseConfig`           | Per-axis rotation, location, scale ranges.                    |
| `material`         | `MaterialConfig`       | Hue/saturation/value, specular, roughness, metallic, UV.      |
| `texture_jitter`   | `TextureJitterConfig`  | Per-texture HSV deltas.                                       |
| `line_art`         | `LineArtConfig`        | Compositor line-art parameters (when enabled).                |

---

## Phases II + III — Train and evaluate (`train_eval`)

### Quick start

```bash
# Phase II — source-only training (paper §3.2)
./.venv/bin/python run.py train_eval --cfg_file train_source.yaml

# Phase III — pseudo-label adaptation (paper §3.3)
./.venv/bin/python run.py train_eval --cfg_file train_target.yaml

# Supervised upper bound on the labelled target
./.venv/bin/python run.py train_eval --cfg_file train_upper_target.yaml

# Multi-target evaluation across VegFru / Fruits-262 / OI-v7
./.venv/bin/python run.py train_eval --cfg_file eval.yaml
```

### Background

```bash
./run.sh train_eval train_source.yaml
```

The launcher writes `logs/run_<timestamp>_<cfg>.log`, refreshes
`logs/latest.log`, and arms a watcher that triggers `sudo -n shutdown now`
once the run ends and no other independent `run.py` is alive. Override the
grace period with `SHUTDOWN_GRACE_SECS=600 ./run.sh ...`.

### The four configs

| File                       | Operation    | Stage                | Purpose                                            |
| -------------------------- | ------------ | -------------------- | -------------------------------------------------- |
| `train_source.yaml`        | `Training`   | `train_source`       | Phase II: train backbone + classifier on Fruits-3D from scratch. |
| `train_target.yaml`        | `Training`   | `train_target`       | Phase III: joint optimisation with Dynamic Top-K pseudo-labels (Eq. 6, λ_S=1.0, λ_T=0.7). |
| `train_upper_target.yaml`  | `Training`   | `train_upper_target` | Supervised upper-bound on the labelled target (paper Table 5 reference). |
| `eval.yaml`                | `Evaluation` | —                    | Macro-F1 across VegFru / Fruits-262 / OI-v7 test splits. |

### YAML schema (top-level sections)

Defined in `train_eval/configs/base/configs.py`. Every YAML must provide:

| Section            | Class                    | What it controls                                              |
| ------------------ | ------------------------ | ------------------------------------------------------------- |
| `general`          | `GeneralConfig`          | `tag`, `operation`, `stage`, `seed`.                          |
| `storage`          | `StorageConfig`          | Local / remote / source / target / test / backup folders.     |
| `pretrained`       | `PretrainedConfig`       | Which checkpoint to start from per CV fold.                   |
| `dataset`          | `DatasetConfig`          | Dataset type, CV folds, augmentations, num_workers.           |
| `training`         | `TrainingConfig`         | Batch sizes, GPU id, force_cpu, λ_S / λ_T weights.            |
| `feature_extractor`| `FeatureExtractorConfig` | timm backbone, optimiser, scheduler, LR, drop rates.          |
| `classifier`       | `ClassifierConfig`       | MLP head dims, optimiser, dropout.                            |
| `pseudo_label`     | `PseudoLabelConfig`      | Dynamic Top-K thresholds (paper §3.3.1).                      |
| `focal_loss`       | `FocalLossConfig`        | β, γ (γ=2.0 baseline → γ=0.0 tuning, paper §3.2).             |
| `mixup_criterion`  | `MixupCriterionConfig`   | Mixup/CutMix per-stage params and scheduler.                  |
| `stage_scheduler`  | `StageSchedulerConfig`   | NSS — patience, warmup, early-stop tolerances (paper §3.2.2). |
| `evaluation`       | `EvaluationConfig`       | Source/target/report toggles.                                 |

### Operation modes

`general.operation` (`train_eval/data/types/operation_type.py`):

- `Training`
- `Evaluation`
- `Hyperparameter Tuning`

### Stages

`general.stage` (`train_eval/data/stages/stage_types.py`):

- `train_source` — Phase II baseline (paper §3.2, focal γ=2.0, drop-rate 0.3).
- `finetune_source_dropout` — NSS-driven drop-rate ramp during the tuning step.
- `finetune_source_mixup` — NSS-driven mixup/cutmix ramp.
- `eval_models` — model-selection sweep across saved checkpoints.
- `train_target` — Phase III pseudo-label UDA.
- `train_upper_target` — supervised upper bound.

### Cross-validation

`main.py` (lines 97–121) iterates `dataset.source_cross_val_k` /
`target_cross_val_k`. Paper Table 1 uses 5-fold CV on Fruits-3D and on each
target's training pool.

---

## Datasets and reproducing the paper

| Domain    | Dataset                  | Split                            | Image count |
| --------- | ------------------------ | -------------------------------- | ----------- |
| Synthetic | **Fruits-3D** (ours)     | Train / Validation (5-fold)      | 12,000      |
| Real      | VegFru (5 categories)    | Train / Validation / Test        | 500 / 250 / 4,405 |
| Real      | Fruits-262 (5 categories)| Train / Validation (5-fold) / Test | 3,914 / 976 |
| Real      | OI-v7 (5 categories)     | Train / Validation / Test        | 1,980 / 60 / 184 |

All domains share five categories: **apple, banana, pineapple, pomegranate,
pumpkin** — chosen for taxonomic consistency, geometric diversity (spherical /
elongated / irregular), and representational sufficiency (paper §4.1).

Headline result (paper Table 5, RepVGG-B3 from scratch, average Macro-F1
across the three target test sets): **CoDA 77.12 % vs. ImageNet-pretrained
baseline 78.23 %.**

### Data access — 3D models

The 50 textured `.usdz` assets used to render Fruits-3D are hosted on
Google Drive (access-gated):

<https://drive.google.com/drive/folders/1TVn_gcFz-BqqygZnVE9uqhYaQqqHG11a?usp=drive_link>

The folder is private — click **Request access** on Drive with a short note
about intended use (research / reproduction of the paper). Once approved,
download the folder and extract it so the layout matches the
`source_folder` set in each `image_gen_3d_to_2d/cfgs/*.cfg`:

```
./data/SketchFab_3D/
├── apple/         ← *.usdz files per category
├── banana/
├── pineapple/
├── pomegranate/
└── pumpkin/
```

These 3D models are the input to **Phase I only** (`gen_2d`). Phase II /
III consume the rendered 2D outputs described in the *Phase I → Phase II
handoff* section above, not the raw `.usdz` files.

The three real-world target datasets (VegFru, Fruits-262, OI-v7) are
public and available at their respective project pages; see the paper's
Section 4.1 for the exact subset definitions.

---

## TensorBoard

Every training run writes scalars, images, and classification reports to
`<destination_folder>/Models/<model_id>/tensorboard/`. To browse them:

```bash
# Edit train_eval/managers/tensorboard/viewer.yaml once (set log_dir), then:
./.venv/bin/python run.py tb

# Or override on the CLI:
./.venv/bin/python run.py tb --logdir ./runs/train_source/1a2b.../tensorboard --port 6008
```

The `viewer.yaml` template is tracked; edits you make to `log_dir` stay
local because the file is a template you overwrite per machine. If the port
is busy the viewer auto-increments until a free port is found.

---

## Logs and process management

- `run.sh` writes to `logs/run_<timestamp>_<cfg>.log` and updates
  `logs/latest.log`.
- `logs/run.pid` holds the most recent `run.py` PID.
- The auto-shutdown watcher only fires `sudo -n shutdown now`; if `sudo`
  requires a password the shutdown is silently skipped.
- To stop a background run: `kill "$(cat logs/run.pid)"`.

---

## Data privacy

This repository is intentionally free of personal data, secrets, and
machine-specific paths so it can be published alongside the paper:

- `.env` and `credentials.json` are **gitignored**. Use `.env.template` as
  the starting point and never commit a populated `.env`.
- All `.cfg` and `.yaml` files use **placeholder paths** (e.g.
  `./data/Fruits-3D`). Edit them in place to point at your local data; the
  paths you set will not leave your machine because the configs themselves
  are tracked but they reference a local layout you control.
- No GitHub PATs, GCP keys, or hardcoded user names live in any tracked
  script. `scripts/setup.sh` reads from `.env`; populate it locally.

> **If you fork from an earlier revision of this repo**, note that
> `credentials.json` and a GitHub PAT were committed historically and remain
> reachable via `git log`. Rotate any GCP service-account key and any
> GitHub PAT that match the historical values *before* publishing or
> sharing the repo.

---

## Requirements

See `requirements.txt`. Key versions:

- Python 3.10.14
- torch 2.9.1 / torchvision 0.24.1
- timm 1.0.22 (RepVGG-B3, DeiT)
- tensorboard 2.20.0
- bpy 4.4.0 (Blender as a Python module — required for `gen_2d`)
- mathutils 3.3.0, scipy 1.16.1, openpyxl 3.1.5
- numpy 1.26.4, pandas 2.3.3, seaborn 0.13.2, matplotlib 3.10.3
- pillow 11.2.1, tqdm 4.67.1
- gcsfs 2025.10.0 (only needed for GCS-backed datasets)

---

## Citation

**Citation is a required condition for any use** of this repository — source
code, the Fruits-3D dataset, the 3D `.usdz` assets shared via Drive, and any
rendered artefacts derived from them. This applies to research papers,
technical reports, student projects, blog posts, tutorials, product
integrations, forks, and any public derivative work. If the paper is not
cited, permission to use these artefacts is not granted.

BibTeX:

```bibtex

@Article{app16094115,
AUTHOR = {Gemirter, Cavide Balkı and Korkmaz, Emin Erkan and Goularas, Dionysis},
TITLE = {CoDA: A Cognitive-Inspired Approach for Domain Adaptation},
JOURNAL = {Applied Sciences},
VOLUME = {16},
YEAR = {2026},
NUMBER = {9},
ARTICLE-NUMBER = {4115},
URL = {https://www.mdpi.com/2076-3417/16/9/4115},
ISSN = {2076-3417},
DOI = {10.3390/app16094115}
}
```

Plain-text:

> Gemirter, C. B., Korkmaz, E. E., & Goularas, D. (2026). *CoDA: A
> Cognitive-Inspired Approach for Domain Adaptation*. Applied Sciences
> (submitted). https://github.com/cbalkig/CoDA

GitHub users: click **“Cite this repository”** in the sidebar — the button is
powered by [`CITATION.cff`](./CITATION.cff) and produces both BibTeX and
APA formats.
