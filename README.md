# HITL-HAR: Human-in-the-Loop Incremental Human Activity Recognition

A plug-and-play framework for incrementally teaching a wearable sensor system new human activities with minimal user effort. Built on a frozen SimCLR self-supervised backbone with per-activity binary classifiers.

**Paper**: *Human-in-the-loop Incremental Learning for Human Activity
Recognition with Multimodal Wearable Sensors* — submitted to IMWUT (ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)

---

## Key ideas

- **No catastrophic forgetting**: each activity gets its own binary head; adding a new one never touches existing heads.
- **Human-in-the-loop**: when adding activity B, the system detects that B co-occurs with already-known activity A, surfaces this to the user for confirmation, and automatically retrains A's head if needed.
- **Sensor gating**: each head applies a fixed, activity-specific attention mask over sensor streams (e.g. ankle sensors matter more for walking, wrist sensors matter more for writing).
- **Mutual exclusion**: the user can mark two activities as mutually exclusive; the system suppresses lower-confidence simultaneous predictions.

---

## Repository structure

```
HITL-class-incremental-learning/
├── configs/
│   ├── paaws_lab.json          # PAAWS Lab dataset config (worked example)
│   ├── hparams.json            # All hyperparameters and simulation settings
│   ├── paths.example.json      # Template — copy to paths.json and fill in
│   └── paths.json              # Your local paths (gitignored)
├── scripts/
│   ├── config_loader.py        # Loads all JSON configs into a single Config object
│   ├── encoder.py              # Model-agnostic encoder loading and feature extraction
│   ├── helpers.py              # Dataset loading utilities
│   ├── helpers_hitl.py         # GatedHead, training, evaluation, ME registry
│   ├── hitl_simulation.py      # HITL interaction simulation layer
│   ├── logger.py               # Structured JSON run logger
│   ├── simclr_models_pt.py     # SimCLR model architecture (PyTorch)
│   ├── validate_dataset.py     # Pre-flight checks before running experiments
│   ├── E1_train_seeds.py       # Step 1: train binary heads for seed activities
│   ├── E2_add_activity.py      # Step 2: add one new activity (full HITL pipeline)
│   └── E3_ablation.py          # Step 3: ablation study over full incremental sequence
├── models/                     # Place your pretrained SimCLR .pt file here
├── output/                     # Generated state files, logs, and weights
├── figs/                       # Generated figures
├── environment.yml
└── README.md
```

---

## Quickstart (PAAWS Lab)

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate hitl_har
```

### 2. Configure paths

```bash
cp configs/paths.example.json configs/paths.json
```

Edit `configs/paths.json` and fill in your paths:

```json
{
  "data_dir":       "/absolute/path/to/your/dataset/",
  "working_dir":    "./output/",
  "encoder_paths":  "./models/your_simclr.pt",
  "participants":   ["DS_11"],
  "dataset_config": "paaws_lab.json",
  "hparams_config": "hparams.json"
}
```

`data_dir` should be an absolute path. `working_dir`, `encoder_paths`, and `figs_dir` can be relative — they resolve relative to the repo root regardless of where you invoke the script from.

### 3. Validate your setup

```bash
python scripts/validate_dataset.py
```

This checks that your data directory, file shapes, seed activities, sensor hints, and encoder weights are all correct before anything starts training.

### 4. Run the experiment pipeline

```bash
# E1: Train seed activity heads
python scripts/E1_train_seeds.py

# E2: Add one new activity (run once per activity, chaining state files)
python scripts/E2_add_activity.py --activity Treadmill_2mph_Lab
python scripts/E2_add_activity.py --activity Sit_Writing_Lab --state output/<timestamp>_state_after_Treadmill_2mph_Lab.pkl

# E3: Full ablation study (all conditions, all activities in one run)
python scripts/E3_ablation.py
```

Each step saves a `.pkl` state file and a `.json` run log to `working_dir`. The log contains a full config snapshot so every result is self-describing and reproducible.

---

## Data format

Data is stored as `.npy` files, one file per activity per participant:

```
data_dir/
  DS_11/
    Walking.npy            # shape: (N, T, DIM, C)
    Sitting_Still.npy
    Treadmill_2mph_Lab.npy
    ...
  DS_12/
    ...
```

| Dimension | Meaning | PAAWS Lab value |
|-----------|---------|-----------------|
| `N` | Number of windows for this participant × activity | varies |
| `T` | Window length in samples | 100 (5 s at 20 Hz) |
| `DIM` | Number of sensor streams | 8 |
| `C` | Channels per stream | 3 (x, y, z accelerometer) |

The sample shape `(T, DIM, C)` is inferred automatically from the first valid file loaded — you do not need to set it manually.

---

## Adapting to your own dataset

### 1. Preprocess your raw sensor data

Window your raw sensor recordings into the `.npy` format above. The `preprocessing` section in your dataset JSON documents the expected `source_fs`, `target_fs`, `window_size_samples`, and `window_overlap` for reference when writing your preprocessing script. The framework expects data to already be windowed.

### 2. Create a dataset config

Copy `configs/paaws_lab.json` and fill in each section:

| Section | What to fill in |
|---------|-----------------|
| `dataset` | `name`, `dim`, `stream_names`, `sampling_rate_hz` |
| `seed_activities` | 3–5 well-separated activities to start from |
| `training_exclusions` | Activities to exclude from the negative training set per head — semantically related ones that would contaminate negatives |
| `cooccurrence_graph` | Ground-truth co-occurrence relationships used to *evaluate* the simulation — which heads should fire when each new activity is introduced |
| `mutual_exclusions` | Activities that cannot physically co-occur |
| `sensor_hints` | Per-activity sensor relevance weights, one float per stream (0.0 = irrelevant, 1.0 = highly relevant) |

`training_exclusions` and `cooccurrence_graph` look similar but serve different purposes: the former is used at training time to filter negatives; the latter is used at simulation time to score co-occurrence detection quality.

### 3. Train or bring a SimCLR encoder

Train an encoder on your data using `SimCLR.py`, or bring your own `.pt` weights. The encoder must accept `(B, T, C)` input and output `(B, D)` embeddings. Set `simclr_path` in `configs/paths.json` to point to your weights — this is always the authoritative source for the model path, regardless of what the dataset config says.

To use a different architecture, subclass `StreamEncoder` in `scripts/encoder.py` and register it:

```python
class MyEncoder(StreamEncoder):
    def encode(self, X: np.ndarray, batch_size: int = 200) -> np.ndarray:
        ...  # X: (N, T, C) → return (N, D)

    @property
    def embed_dim(self) -> int:
        return 128

EncoderRegistry.register("my_encoder", MyEncoder)
```

Then set `"encoder_type": "my_encoder"` in `configs/paths.json`.

### 4. Point paths.json at your new config

```json
"dataset_config": "my_dataset.json"
```

### 5. Validate

```bash
python scripts/validate_dataset.py
```

---

## Configuration reference

### `configs/paths.json` (gitignored)

| Key | Description | Notes |
|-----|-------------|-------|
| `data_dir` | Dataset root directory | Absolute path recommended |
| `working_dir` | Output directory for state files, logs, weights | Relative to repo root |
| `simclr_path` | Pretrained encoder `.pt` file | Relative to repo root |
| `figs_dir` | Output directory for figures | Defaults to `working_dir/figs` |
| `participants` | Participant IDs to include | `null` = all subfolders |
| `dataset_config` | Dataset JSON filename | Relative to `configs/` |
| `hparams_config` | Hyperparameter JSON filename | Relative to `configs/` |
| `fusion` | Fusion strategy | `"early"` (default) or `"late"` |
| `encoder_type` | Encoder architecture | `"simclr_pt"` (default) |

### `configs/hparams.json`

All training hyperparameters, threshold bounds, co-occurrence firing threshold, and HITL simulation settings in one place. See the inline `_comment` fields in the file for documentation of each parameter. Key sections:

**`training`** — epochs, learning rates, batch size, early stopping patience, focal loss gamma, class weight cap.

**`threshold`** — min/max bounds for F1-optimal threshold search, fallback value, seed head nudge factor.

**`hitl_simulation`** — controls how user interactions are simulated:
- `"confirmation_source": "ground_truth"` — perfect oracle using dataset labels (used in all published experiments)
- `"confirmation_source": "noisy"` — adds configurable label noise via `label_noise_rate` (reserved for robustness experiments)

**`ablation.conditions`** — list of ablation conditions run by E3, each with `name`, `sensor_gating`, `me`, and `retraining` flags. Add, remove, or rename conditions here without touching `E3_ablation.py`.

---

## Extending the framework

### Different binary classifier

Replace `GatedHead` in `helpers_hitl.py`. The pipeline only calls `train_head_fast`, `evaluate_head_fast`, `find_optimal_threshold_fast`, and `retrain_head_fast` — swap these for any implementation that accepts precomputed `(N, DIM, D)` features and returns the same metric dicts. If your classifier does not support sensor gating, set all activities to `uniform_hint` in your dataset config.

### Different HITL simulation strategy

Edit `hitl_simulation.py`. The functions `simulate_cooccurrence_confirmation` and `simulate_mutual_exclusion_marking` return the same types that real user input would return — replacing simulation with real UI responses only requires modifying these two functions.

### Multiple encoders for different sensor streams

Configure `encoders` and `stream_to_encoder` in your dataset JSON to assign different model weights to different streams:

```json
"encoders": {
  "body":  {"path": "placeholder", "input_shape": [100, 3], "embed_dim": 96},
  "wrist": {"path": "placeholder", "input_shape": [100, 3], "embed_dim": 96}
},
"stream_to_encoder": {
  "LeftAnkle":  "body",
  "LeftWrist":  "wrist",
  "RightWrist": "wrist"
}
```

Set the actual model paths in `paths.json` (or use absolute paths in the dataset config). Any stream not listed in `stream_to_encoder` falls back to the first encoder defined.

---

## Output files

| File | Contents |
|------|----------|
| `<ts>_e1_state_<fusion>.pkl` | Full system state after E1: heads, thresholds, replay buffer |
| `<ts>_e1_state_<fusion>_log.json` | Run log: config snapshot, per-seed metrics, all events |
| `<ts>_seed_<activity>_<fusion>_weights.pt` | Trained head weights (one per seed activity) |
| `<ts>_state_after_<activity>.pkl` | System state after adding one activity (E2) |
| `<ts>_e3_ablation_histories_<fusion>.pkl` | Full per-condition histories from E3 |
| `<ts>_abl_f1_trend_<fusion>.pdf` | Weighted F1 over incremental steps, per ablation condition |
| `<ts>_abl_final_metrics_<fusion>.pdf` | Final AUC/F1/precision/recall bar chart, seed vs all heads |
| `<ts>_abl_forgetting_<fusion>.pdf` | Seed head AUC delta vs E1 baseline across steps |
| `<ts>_abl_heatmap_<fusion>.pdf` | Per-activity F1 heatmap across all ablation conditions |

Every `.pkl` has a sibling `_log.json` containing the complete config snapshot used to produce it, so any result can be reproduced exactly.

---

## Citation

```bibtex
@article{le2026hitlhar,
  title     = {Human-in-the-loop Incremental Learning for Human Activity Recognition with Multimodal Wearable Sensors},
  author    = {Le, Ha and Choube, Akshat and Kanel, Pranjal and Mishra, Varun and Intille, Stephen S.},
  journal   = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume    = {},
  number    = {},
  pages     = {},
  year      = {2026},
  publisher = {ACM},
  doi       = {}
}
```

---

## License

MIT License