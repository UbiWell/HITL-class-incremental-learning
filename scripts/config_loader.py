"""
config_loader.py
================
Replaces the old config.py. Loads all settings from three JSON files:
  - paths.json          (gitignored, user-specific)
  - configs/<dataset>.json  (dataset structure, activities, gating, ME)
  - configs/hparams.json    (training hyperparameters and simulation params)

Usage
-----
  from config_loader import cfg

  cfg.DATA_DIR
  cfg.SEED_ACTIVITIES
  cfg.get_sensor_hint("Walking")
  cfg.get_training_exclusions("Treadmill_2mph_Lab")
  cfg.get_cooccurrences_for("Treadmill_2mph_Lab", trained_activities)
  cfg.get_mutual_exclusions_for("Walking", trained_activities)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# JSON loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"  If this is paths.json, copy paths.example.json and fill in your paths."
        )
    with open(path) as f:
        raw = json.load(f)
    # Strip _comment keys (used as inline documentation in the JSON)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _strip_comments(d: Any) -> Any:
    """Recursively strip _comment* keys from nested dicts."""
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items()
                if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_comments(i) for i in d]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Config class
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Single config object that merges paths, dataset config, and hparams.
    Construct via Config.from_paths_file(path) or Config.from_dicts(...).
    """

    def __init__(self, paths: dict, dataset: dict, hparams: dict,
                 repo_root: Path | None = None):
        self._paths     = paths
        self._dataset   = _strip_comments(dataset)
        self._hparams   = _strip_comments(hparams)
        self._repo_root = repo_root   # used to resolve relative paths in paths.json
        self._validate()

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_paths_file(cls, paths_file: str | Path = "paths.json") -> "Config":
        """Load config from paths.json which points to dataset and hparams configs.

        Relative paths in paths.json (e.g. 'configs/paaws_lab.json') are resolved
        relative to the repo root — defined as the parent of paths.json itself.
        This means paths.json at repo root, dataset configs at repo_root/configs/,
        and scripts anywhere under repo root all work correctly.
        """
        paths_file = Path(paths_file).resolve()
        p          = _load_json(paths_file)
        repo_root  = paths_file.parent.parent    # configs/paths.json -> repo root

        # dataset_config and hparams_config are resolved relative to repo root
        # so "./configs/paaws_lab.json" and "paaws_lab.json" both work
        dataset_path = Path(p["dataset_config"])
        if not dataset_path.is_absolute():
            dataset_path = repo_root / dataset_path

        hparams_path = Path(p["hparams_config"])
        if not hparams_path.is_absolute():
            hparams_path = repo_root / hparams_path

        return cls(
            paths     = p,
            dataset   = _load_json(dataset_path),
            hparams   = _load_json(hparams_path),
            repo_root = repo_root,
        )

    @classmethod
    def from_dicts(cls, paths: dict, dataset: dict, hparams: dict) -> "Config":
        """Construct directly from dicts (useful for testing)."""
        return cls(paths, dataset, hparams)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        required_path_keys = ["data_dir", "working_dir", "encoder_paths"]
        for k in required_path_keys:
            if k not in self._paths:
                raise KeyError(f"paths.json is missing required key: '{k}'")

        ds = self._dataset
        for k in ["dataset", "seed_activities", "sensor_hints",
                  "cooccurrence_graph", "training_exclusions", "mutual_exclusions"]:
            if k not in ds:
                raise KeyError(f"Dataset config is missing required section: '{k}'")

        n_streams = len(ds["dataset"]["stream_names"])
        hints = ds["sensor_hints"].get("hints", {})
        uniform = ds["sensor_hints"].get("uniform_hint", [])
        if len(uniform) != n_streams:
            raise ValueError(
                f"uniform_hint length ({len(uniform)}) != number of streams ({n_streams})"
            )
        for act, hint in hints.items():
            if len(hint) != n_streams:
                raise ValueError(
                    f"sensor_hint for '{act}' has length {len(hint)}, expected {n_streams}"
                )

    # ── Paths ─────────────────────────────────────────────────────────────────

    def _resolve(self, p: str) -> str:
        """
        Resolve a path string from paths.json.
        - Absolute paths are returned as-is.
        - Relative paths are resolved against repo_root (parent of paths.json)
          so they work regardless of the working directory the script is run from.
        """
        path = Path(p)
        if path.is_absolute():
            return str(path)
        if self._repo_root is not None:
            return str((self._repo_root / path).resolve())
        return str(path.resolve())

    @property
    def DATA_DIR(self) -> str:
        return self._resolve(self._paths["data_dir"])

    @property
    def WORKING_DIR(self) -> str:
        d = self._resolve(self._paths["working_dir"])
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def ENCODER_PATHS(self) -> dict[str, str]:
        """
        Dict mapping encoder key -> resolved absolute path.
        Defined in paths.json as encoder_paths: {"default": "./models/..."}
        Keys must match the 'encoders' section in the dataset config.
        """
        raw = self._paths["encoder_paths"]
        return {k: self._resolve(v) for k, v in raw.items()}

    @property
    def FIGS_DIR(self) -> str:
        raw = self._paths.get("figs_dir")
        d   = self._resolve(raw) if raw else os.path.join(self.WORKING_DIR, "figs")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def PARTICIPANTS(self) -> list[str] | None:
        return self._paths.get("participants", None)

    # ── Dataset ───────────────────────────────────────────────────────────────

    @property
    def DIM(self) -> int:
        return self._dataset["dataset"]["dim"]

    @property
    def STREAM_NAMES(self) -> list[str]:
        return self._dataset["dataset"]["stream_names"]

    @property
    def DATASET_NAME(self) -> str:
        return self._dataset["dataset"]["name"]

    @property
    def SEED_ACTIVITIES(self) -> list[str]:
        return self._dataset["seed_activities"]

    @property
    def ENCODERS(self) -> dict:
        return self._dataset.get("encoders", {})

    @property
    def STREAM_TO_ENCODER(self) -> dict:
        return self._dataset.get("stream_to_encoder", {})

    # ── Activity relationship lookups ─────────────────────────────────────────

    def get_training_exclusions(self, activity: str) -> list[str]:
        """
        Activities to exclude from the negative training set when training a head
        for `activity`. These are semantically related activities that would
        contaminate the negatives.
        """
        return self._dataset["training_exclusions"].get(activity, [])

    def get_cooccurrences_for(self, activity: str,
                               trained_activities: list[str]) -> list[str]:
        """
        Ground-truth co-occurrences for `activity`, filtered to only those
        activities already trained. Used in HITL simulation to evaluate
        whether the system correctly detected co-occurrences.
        """
        all_cooc = self._dataset["cooccurrence_graph"].get(activity, [])
        trained_set = set(trained_activities)
        return [a for a in all_cooc if a in trained_set]

    def get_mutual_exclusions_for(self, activity: str,
                                   trained_activities: list[str]) -> list[str]:
        """
        Activities that are mutually exclusive with `activity`, filtered to
        only those already trained. Used during ME registration.
        """
        all_me = self._dataset["mutual_exclusions"].get(activity, [])
        trained_set = set(trained_activities)
        return [a for a in all_me if a in trained_set]

    def get_sensor_hint(self, activity: str) -> list[float]:
        """
        Sensor gating hint for `activity`. Falls back to uniform_hint if not specified.
        """
        hints   = self._dataset["sensor_hints"]["hints"]
        uniform = self._dataset["sensor_hints"]["uniform_hint"]
        return hints.get(activity, uniform)

    @property
    def UNIFORM_HINT(self) -> list[float]:
        return self._dataset["sensor_hints"]["uniform_hint"]

    @property
    def MUTUAL_EXCLUSIONS(self) -> list[tuple[str, str]]:
        """Flat list of (a, b) pairs from the mutual_exclusions dict."""
        pairs = []
        seen  = set()
        me    = self._dataset["mutual_exclusions"]
        for a, others in me.items():
            for b in others:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    pairs.append((a, b))
                    seen.add(key)
        return pairs

    @property
    def AMBIGUOUS_COOCCURRENCE(self) -> list[str]:
        return self._dataset.get("ambiguous_cooccurrence", {}).get("activities", [])

    @property
    def ALL_ACTIVITIES(self) -> list[str]:
        """All activities defined in the cooccurrence_graph (used as the incremental order pool)."""
        return list(self._dataset["cooccurrence_graph"].keys())

    # ── Hyperparameters ───────────────────────────────────────────────────────

    @property
    def SEED(self) -> int:
        return self._paths.get("seed", 42)

    @property
    def FUSION(self) -> str:
        fusion = self._paths.get("fusion", "early")
        if fusion not in ("early", "late"):
            raise ValueError(
                f"Invalid fusion value '{fusion}' in paths.json. Must be 'early' or 'late'."
            )
        return fusion

    @property
    def ENCODER_TYPE(self) -> str:
        """Default encoder type. Can be overridden per-encoder in dataset config."""
        return self._paths.get("encoder_type", "simclr_pt")

    @property
    def INTERMEDIATE_LAYER(self) -> int:
        return self._paths.get("intermediate_layer", 8)

    @property
    def BATCH_SIZE(self) -> int:
        return self._hparams["training"]["batch_size"]

    @property
    def SEED_HEAD_EPOCHS(self) -> int:
        return self._hparams["training"]["seed_head_epochs"]

    @property
    def INCREMENTAL_HEAD_EPOCHS(self) -> int:
        return self._hparams["training"]["incremental_head_epochs"]

    @property
    def RETRAIN_EPOCHS(self) -> int:
        return self._hparams["training"]["retrain_epochs"]

    @property
    def LEARNING_RATE(self) -> float:
        return self._hparams["training"]["learning_rate"]

    @property
    def RETRAIN_LR(self) -> float:
        return self._hparams["training"]["retrain_learning_rate"]

    @property
    def EARLY_STOPPING_PATIENCE(self) -> int:
        return self._hparams["training"]["early_stopping_patience"]

    @property
    def FOCAL_GAMMA(self) -> float:
        return self._hparams["training"]["focal_gamma"]

    @property
    def MAX_CLASS_WEIGHT(self) -> float:
        return self._hparams["training"]["max_class_weight"]

    @property
    def THRESHOLD_MIN(self) -> float:
        return self._hparams["threshold"]["min"]

    @property
    def THRESHOLD_MAX(self) -> float:
        return self._hparams["threshold"]["max"]

    @property
    def THRESHOLD_FALLBACK(self) -> float:
        return self._hparams["threshold"]["fallback"]

    @property
    def SEED_NUDGE_FACTOR(self) -> float:
        return self._hparams["threshold"]["seed_nudge_factor"]

    @property
    def COOC_FIRE_THRESHOLD(self) -> float:
        return self._hparams["cooccurrence"]["fire_pct_threshold"]

    @property
    def GATE_SCALE(self) -> float:
        return self._hparams["sensor_gating"]["gate_scale"]

    @property
    def HITL_SIMULATION(self) -> dict:
        return self._hparams.get("hitl_simulation", {})

    @property
    def ABLATION_CONDITIONS(self) -> list[dict]:
        return self._hparams.get("ablation", {}).get("conditions", [])

    # ── Convenience ───────────────────────────────────────────────────────────

    def summary(self):
        print(f"Dataset       : {self.DATASET_NAME}")
        print(f"Streams       : {self.DIM}  {self.STREAM_NAMES}")
        print(f"Seed activities ({len(self.SEED_ACTIVITIES)}): {self.SEED_ACTIVITIES}")
        print(f"All activities  ({len(self.ALL_ACTIVITIES)})")
        print(f"Data dir      : {self.DATA_DIR}")
        print(f"Working dir   : {self.WORKING_DIR}")
        print(f"Encoder paths : {self.ENCODER_PATHS}")
        print(f"Fusion        : {self.FUSION}")
        print(f"Batch size    : {self.BATCH_SIZE}")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — scripts do: from config_loader import cfg
# ─────────────────────────────────────────────────────────────────────────────

# Resolve repo root robustly — .resolve() handles .pyc cache, symlinks, and
# any working directory the script is invoked from.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPTS_DIR.parent
_PATHS_FILE  = _REPO_ROOT / "configs" / "paths.json"

try:
    cfg = Config.from_paths_file(_PATHS_FILE)
except FileNotFoundError:
    # Allow import to succeed so the scripts can show a clear error at runtime.
    import warnings
    warnings.warn(
        "configs/paths.json not found. "
        "Copy configs/paths.example.json to configs/paths.json and fill in your paths. "
        "cfg is None until then.",
        stacklevel=2,
    )
    cfg = None  # type: ignore