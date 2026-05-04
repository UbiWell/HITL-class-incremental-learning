"""
helpers_hitl.py
===============
Core HITL-HAR building blocks — PyTorch version.

Key design rule: this file imports NOTHING from config_loader.
All tuneable values (DIM, BATCH_SIZE, thresholds, epochs, etc.) are passed
as function/method arguments, with sensible defaults matching hparams.json.
This makes the module reusable across datasets without modification.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, precision_recall_curve,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_limb_inputs(X, dim):
    """Split (N, T, DIM, C) into list of DIM arrays of shape (N, T, C)."""
    return [X[:, :, i, :] for i in range(dim)]


def make_binary_labels(X, y_int, class_name, label_dict,
                       excluded_classes=None, extra_positive_classes=None,
                       negative_classes=None):
    positive_names = [class_name] + (extra_positive_classes or [])
    pos_idxs       = {label_dict[c] for c in positive_names if c in label_dict}
    excl_idxs      = {label_dict[c] for c in (excluded_classes or []) if c in label_dict}

    if negative_classes is not None:
        neg_idxs  = {label_dict[c] for c in negative_classes if c in label_dict}
        keep_idxs = pos_idxs | neg_idxs
        keep_mask = np.array([i in keep_idxs for i in y_int])
    else:
        keep_mask = np.array([i not in excl_idxs for i in y_int])

    X_out    = X[keep_mask]
    y_out    = y_int[keep_mask]
    y_binary = np.array([1 if i in pos_idxs else 0 for i in y_out], dtype=np.int32)
    return X_out, y_binary


def make_multilabel_binary(activity, y_int_all, label_dict, cooccurrence_graph=None):
    """
    Build binary labels for `activity` treating co-occurring activities as positives.

    Parameters
    ----------
    cooccurrence_graph : dict | None
        {activity: [co-occurring activities]}. If None, only the exact activity
        class is treated as positive (no co-occurrence expansion).
        Pass cfg._dataset["cooccurrence_graph"] to match original behaviour.
    """
    class_idx     = label_dict.get(activity, -1)
    positive_idxs = set()
    if class_idx >= 0:
        positive_idxs.add(class_idx)

    if cooccurrence_graph is not None:
        for cls_name, cooc_list in cooccurrence_graph.items():
            if activity in cooc_list and cls_name in label_dict:
                positive_idxs.add(label_dict[cls_name])

    return np.array([1 if i in positive_idxs else 0 for i in y_int_all], dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION + CACHING
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(pretrained_model, X, intermediate_layer=8, batch_size=200):
    """
    Run the frozen SimCLR encoder on X once and return precomputed features.

    Parameters
    ----------
    pretrained_model : SimCLRModel (PyTorch)
    X : np.ndarray  shape (N, T, DIM, C)
    intermediate_layer : int  (unused — kept for API compatibility; encoder is always used)
    batch_size : int

    Returns
    -------
    Z : np.ndarray  shape (N, DIM, D)
    """
    pretrained_model.eval()
    encoder = pretrained_model.encoder   # BaseEncoder -> (B, D)

    N, T, DIM, C = X.shape
    X_limbs = X.transpose(0, 2, 1, 3).reshape(N * DIM, T, C).astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_limbs), batch_size):
            xb = torch.from_numpy(X_limbs[i:i + batch_size]).to(DEVICE)
            preds.append(encoder(xb).cpu().numpy())

    Z_flat = np.concatenate(preds, axis=0)   # (N*DIM, D)
    D      = Z_flat.shape[-1]
    return Z_flat.reshape(N, DIM, D).astype(np.float32)


class FeatureCache:
    """Stores precomputed (N, DIM, D) encoder features for train/val/test."""
    def __init__(self, Z_train, Z_val, Z_test):
        self.train = Z_train
        self.val   = Z_val
        self.test  = Z_test

    def subset(self, split, mask):
        return getattr(self, split)[mask]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — GatedHead
# ─────────────────────────────────────────────────────────────────────────────

class GatedHead(nn.Module):
    """
    Lightweight binary head that operates on precomputed (N, DIM, D) features.

    The gate is a fixed (non-trainable) buffer — sigmoid of scaled sensor hints
    broadcast over the embedding dimension.

    Early fusion:
        gate → flatten → Dense(256, ReLU) → Dropout(0.3) → Dense(1)  [logit]

    Late fusion:
        per-limb: Dense(128, ReLU) → Dense(1) [logit]
        gate per-limb logits → Dense(DIM, ReLU) → Dense(1) [logit]

    Output: raw logit (apply sigmoid at inference).
    """

    def __init__(self, hint: list, feature_dim: int,
                 fusion: str = "early", gate_scale: float = 4.0):
        super().__init__()
        self.fusion      = fusion
        self.feature_dim = feature_dim
        dim              = len(hint)

        # Fixed gate: (1, DIM, 1) — non-trainable
        hint_arr = np.array(hint, dtype=np.float32)
        gates    = 1.0 / (1.0 + np.exp(-gate_scale * (hint_arr - 0.5)))
        self.register_buffer(
            "gate",
            torch.from_numpy(gates).reshape(1, dim, 1)
        )

        if fusion == "early":
            self.head = nn.Sequential(
                nn.Linear(dim * feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
        else:
            self.limb_projs = nn.ModuleList([
                nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 1))
                for _ in range(dim)
            ])
            self.fuser = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Z : (N, DIM, D) → (N, 1) raw logit"""
        Z = Z * self.gate

        if self.fusion == "early":
            z = Z.reshape(Z.shape[0], -1)
            return self.head(z)
        else:
            logits = torch.cat(
                [proj(Z[:, i, :]) for i, proj in enumerate(self.limb_projs)],
                dim=1
            )
            logits = logits * self.gate.squeeze(-1)
            return self.fuser(logits)


def build_gated_head_from_features(hint, feature_dim,
                                   fusion="early", gate_scale=4.0):
    """Construct and move a GatedHead to DEVICE."""
    return GatedHead(hint, feature_dim, fusion=fusion, gate_scale=gate_scale).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss. gamma=0 → standard BCE. Expects raw logits."""
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pw      = torch.tensor([self.pos_weight], device=logits.device)
        logits  = logits.squeeze(-1)
        bce     = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        probs   = torch.sigmoid(logits)
        p_t     = targets * probs + (1 - targets) * (1 - probs)
        return (bce * (1 - p_t) ** self.gamma).mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(Z, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(Z.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def train_head_fast(model, Z_tr, y_tr, Z_vl, y_vl, save_path,
                    epochs=20, lr=1e-3, batch_size=200,
                    focal_gamma=2.0, max_class_weight=10.0,
                    early_stopping_patience=10):
    """
    Train a GatedHead on precomputed (N, DIM, D) features.
    Saves best-val-AUC weights to save_path. Returns model with best weights loaded.
    """
    pos_weight = float(min(
        (y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        max_class_weight
    ))

    criterion  = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    tr_loader  = _make_loader(Z_tr, y_tr, batch_size, shuffle=True)

    best_auc   = -1.0
    no_improve = 0

    # Ensure save_path ends with .pt
    weights_path = save_path if save_path.endswith(".pt") else save_path + ".pt"

    for epoch in range(epochs):
        model.train()
        for Zb, yb in tr_loader:
            Zb, yb = Zb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(Zb), yb).backward()
            optimizer.step()

        probs = _predict_probs(model, Z_vl, batch_size=batch_size)
        auc   = roc_auc_score(y_vl, probs) if len(np.unique(y_vl)) > 1 else 0.0

        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break

    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    except Exception:
        pass

    return model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _predict_probs(model, Z, batch_size=512):
    """Run model on numpy Z (N, DIM, D); return (N,) sigmoid probabilities."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Z), batch_size):
            Zb    = torch.from_numpy(Z[i:i + batch_size].astype(np.float32)).to(DEVICE)
            logit = model(Zb).squeeze(-1)
            preds.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(preds)


def evaluate_head_fast(model, Z, y, threshold=0.5):
    """Evaluate a GatedHead. Returns dict of metrics including raw probs."""
    probs = _predict_probs(model, Z)
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
    return {
        "auc":       float(auc),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "probs":     probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold_fast(model, Z_val, y_val,
                                 t_min=0.20, t_max=0.80, fallback=0.50):
    """
    Find threshold maximising F1 on the validation PR curve,
    constrained to [t_min, t_max]. Returns fallback if no valid threshold found.
    """
    probs                        = _predict_probs(model, Z_val)
    precision, recall, pr_thresholds = precision_recall_curve(y_val, probs)
    f1s   = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    valid = (pr_thresholds >= t_min) & (pr_thresholds <= t_max)
    if valid.any():
        best_t = float(pr_thresholds[valid][np.argmax(f1s[:-1][valid])])
    else:
        best_t = fallback
    return float(np.clip(best_t, t_min, t_max))


# ─────────────────────────────────────────────────────────────────────────────
# CO-OCCURRENCE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_cooccurrence(new_activity, Z_new, heads, thresholds,
                       fusion="early", fire_threshold=0.50):
    """
    Run new activity features through all existing heads.

    A head 'fires' if the fraction of windows predicted positive
    exceeds fire_threshold.
    """
    results = {}
    for (activity, f), model in heads.items():
        if f != fusion:
            continue
        probs  = _predict_probs(model, Z_new)
        thresh = thresholds.get((activity, fusion), 0.5)
        preds  = (probs >= thresh).astype(int)
        results[activity] = {
            "mean_prob":    float(probs.mean()),
            "pct_positive": float(preds.mean()),
            "fires":        bool(preds.mean() > fire_threshold),
            "threshold":    thresh,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OOD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_heads_fast(heads, Z_all, y_int_all, label_dict,
                             thresholds, fusion="early",
                             cooccurrence_graph=None):
    """
    Evaluate all trained heads on precomputed features.

    cooccurrence_graph is passed to make_multilabel_binary so co-occurring
    activities are treated as positives when scoring a head.
    Pass cfg._dataset["cooccurrence_graph"] or None for exact-class-only scoring.
    """
    results = {}
    for (activity, f), model in heads.items():
        if f != fusion or activity not in label_dict:
            continue
        y_binary = make_multilabel_binary(
            activity, y_int_all, label_dict,
            cooccurrence_graph=cooccurrence_graph
        )
        if len(np.unique(y_binary)) < 2:
            continue
        probs  = _predict_probs(model, Z_all)
        thresh = thresholds.get((activity, fusion), 0.5)
        preds  = (probs >= thresh).astype(int)
        results[activity] = {
            "auc":       float(roc_auc_score(y_binary, probs)),
            "f1":        float(f1_score(y_binary, preds, zero_division=0)),
            "accuracy":  float(accuracy_score(y_binary, preds)),
            "precision": float(precision_score(y_binary, preds, zero_division=0)),
            "recall":    float(recall_score(y_binary, preds, zero_division=0)),
            "n_pos":     int(y_binary.sum()),
            "n_neg":     int((y_binary == 0).sum()),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def retrain_head_fast(activity, model, replay_buffer, Z_new, y_new_label,
                      Z_val, y_val, Z_train_all, y_train_int, label_dict,
                      trained_activities=None,
                      epochs=20, lr=1e-4, batch_size=200,
                      focal_gamma=2.0, max_class_weight=10.0,
                      early_stopping_patience=10,
                      timestamp="", working_dir="."):
    """Fine-tune an existing GatedHead with new samples."""
    metrics_before = evaluate_head_fast(model, Z_val, y_val)
    print(f"  [{activity}] Before — "
          f"AUC:{metrics_before['auc']:.4f} F1:{metrics_before['f1']:.4f}")

    class_idx = label_dict.get(activity, -1)

    # ── Positive set ──────────────────────────────────────────────────────────
    Z_pos_replay = replay_buffer.get_positives(activity)
    if y_new_label == 1:
        Z_pos = np.concatenate([Z_pos_replay, Z_new], axis=0) \
                if Z_pos_replay is not None else Z_new.copy()
        replay_buffer.store_positives(activity, Z_new)
    else:
        Z_pos = Z_pos_replay if Z_pos_replay is not None \
                else np.zeros((0, *Z_new.shape[1:]), dtype=np.float32)

    # ── Negative set ──────────────────────────────────────────────────────────
    if trained_activities is not None:
        trained_idxs = {label_dict[a] for a in trained_activities
                        if a in label_dict and a != activity}
        neg_mask = np.array([i in trained_idxs for i in y_train_int])
    else:
        neg_mask = (y_train_int != class_idx)

    Z_neg = Z_train_all[neg_mask]
    if y_new_label == 0:
        Z_neg = np.concatenate([Z_neg, Z_new], axis=0)

    if Z_pos.shape[0] == 0:
        print(f"  Warning: no positives for '{activity}', skipping retrain.")
        return model, metrics_before, metrics_before

    y_pos      = np.ones(Z_pos.shape[0],  dtype=np.int32)
    y_neg      = np.zeros(Z_neg.shape[0], dtype=np.int32)
    Z_combined = np.concatenate([Z_pos, Z_neg], axis=0)
    y_combined = np.concatenate([y_pos, y_neg], axis=0)

    print(f"  Combined: {Z_combined.shape[0]} samples "
          f"({Z_pos.shape[0]} pos, {Z_neg.shape[0]} neg)")

    safe_name  = activity.replace(" ", "_").replace(":", "_")
    save_path  = os.path.join(working_dir, f"{timestamp}_retrain_{safe_name}.pt")

    model = train_head_fast(
        model, Z_combined, y_combined, Z_val, y_val,
        save_path,
        epochs=epochs, lr=lr, batch_size=batch_size,
        focal_gamma=focal_gamma, max_class_weight=max_class_weight,
        early_stopping_patience=early_stopping_patience,
    )

    metrics_after = evaluate_head_fast(model, Z_val, y_val)
    print(f"  [{activity}] After  — "
          f"AUC:{metrics_after['auc']:.4f} F1:{metrics_after['f1']:.4f} "
          f"(ΔAUC:{metrics_after['auc']-metrics_before['auc']:+.4f})")

    return model, metrics_before, metrics_after


# ─────────────────────────────────────────────────────────────────────────────
# STATEFUL CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Stores positive feature vectors per activity for replay during retraining."""

    def __init__(self, max_per_activity=None):
        """
        max_per_activity : int | None
            Cap on stored positives per activity. None = unlimited.
            Set via hparams.json replay_buffer.max_positives_per_activity.
        """
        self.positives        = {}
        self.max_per_activity = max_per_activity

    def store_positives(self, activity, Z_pos):
        if activity in self.positives:
            combined = np.concatenate([self.positives[activity], Z_pos], axis=0)
        else:
            combined = Z_pos.copy()
        if self.max_per_activity is not None:
            combined = combined[-self.max_per_activity:]
        self.positives[activity] = combined

    def get_positives(self, activity):
        return self.positives.get(activity, None)

    def has(self, activity):
        return activity in self.positives

    def store(self, activity, Z, y):
        Z_pos = Z[y == 1]
        if Z_pos.shape[0] > 0:
            self.store_positives(activity, Z_pos)

    def get(self, activity):
        Z_pos = self.get_positives(activity)
        if Z_pos is None:
            return None, None
        return Z_pos, np.ones(Z_pos.shape[0], dtype=np.int32)


class MutualExclusionRegistry:
    """Tracks mutual exclusion rules and suppresses conflicting head predictions."""

    def __init__(self):
        self.rules = {}

    def add(self, a, b):
        self.rules.setdefault(a, set()).add(b)
        self.rules.setdefault(b, set()).add(a)

    def add_for_activity(self, activity, trained_activities, cfg=None):
        """
        Register ME rules for a newly added activity.

        Pass cfg (Config object) to read from the dataset config.
        If cfg is None, no rules are added (silent no-op for compatibility).
        """
        if cfg is None:
            return []
        exclusive = cfg.get_mutual_exclusions_for(activity, trained_activities)
        for e in exclusive:
            self.add(activity, e)
        if exclusive:
            print(f"  ME rules for '{activity}': {exclusive}")
        return exclusive

    def get_conflicts(self, fired):
        fired  = set(fired)
        seen   = set()
        result = []
        for a in fired:
            for b in self.rules.get(a, set()):
                if b in fired and (b, a) not in seen:
                    result.append((a, b))
                    seen.add((a, b))
        return result

    def apply_suppression(self, predictions):
        import copy
        preds      = copy.deepcopy(predictions)
        fired      = [a for a, r in preds.items() if r["fires"]]
        conflicts  = self.get_conflicts(fired)
        suppressed = []
        for a, b in conflicts:
            loser  = a if preds[a]["mean_prob"] <= preds[b]["mean_prob"] else b
            winner = b if loser == a else a
            preds[loser]["fires"]      = False
            preds[loser]["suppressed"] = True
            suppressed.append(loser)
            print(f"  Suppressed '{loser}' "
                  f"(conflict with '{winner}', "
                  f"prob: {preds[loser]['mean_prob']:.3f})")
        return preds, suppressed

    def summary(self):
        printed = set()
        for a, bs in sorted(self.rules.items()):
            for b in sorted(bs):
                if (b, a) not in printed:
                    print(f"  {a}  ✗↔  {b}")
                    printed.add((a, b))


class NegativeBuffer:
    """Accumulates negative feature vectors for later use in retraining."""

    def __init__(self):
        self.buffer = {}

    def add(self, activity, Z):
        self.buffer.setdefault(activity, []).append(Z)

    def get(self, activity):
        if activity not in self.buffer:
            return None
        return np.concatenate(self.buffer[activity], axis=0)

    def drain(self):
        result = {a: self.get(a) for a in self.buffer}
        self.buffer.clear()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# STATE SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_heads_from_state(state, fusion, cfg=None):
    """
    Rebuild GatedHead objects from a saved state dict.

    Parameters
    ----------
    state  : dict  loaded from .pkl (must have weights_paths, feature_dim)
    fusion : str
    cfg    : Config | None
        If provided, sensor hints are read from cfg.get_sensor_hint().
        If None, uniform hints (all 0.5) are used as fallback.
    """
    D     = state["feature_dim"]
    heads = {}

    for (activity, f), wpath in state["weights_paths"].items():
        if f != fusion:
            continue
        if not os.path.exists(wpath):
            print(f"  Warning: weights not found for '{activity}' at {wpath}, skipping.")
            continue

        if cfg is not None:
            hint = cfg.get_sensor_hint(activity)
        else:
            # Infer DIM from the first available weight file shape
            hint = [0.5] * state.get("dim", D)

        model = GatedHead(hint, D, fusion=fusion).to(DEVICE)
        model.load_state_dict(torch.load(wpath, map_location=DEVICE))
        model.eval()
        heads[(activity, f)] = model
        print(f"  Loaded '{activity}' ({f})")

    return heads


def save_head_weights(activity, fusion, model, working_dir, timestamp):
    """Save a single head's weights and return the path."""
    safe  = activity.replace(" ", "_")
    wpath = os.path.join(working_dir, f"{timestamp}_head_{safe}_{fusion}_weights.pt")
    torch.save(model.state_dict(), wpath)
    return wpath
