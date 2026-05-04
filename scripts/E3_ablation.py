"""
E3_ablation.py
==============
Ablation study over the full incremental HITL simulation.

Ablation conditions are defined in configs/hparams.json under "ablation.conditions".
Each condition is a dict with: name, sensor_gating, me, retraining flags.

Reads all settings from paths.json / dataset config / hparams.json.
"""
import sys
from pathlib import Path
# Ensure repo root (parent of scripts/) is on sys.path so config_loader and
# helpers resolve correctly regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/

import os
import time
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from config_loader import cfg
from logger import RunLogger
from hitl_simulation import (
    simulate_cooccurrence_confirmation,
    should_retrain_for_fn,
)
from helpers import create_dataset_file_split
from helpers_hitl import (
    build_gated_head_from_features,
    train_head_fast, evaluate_head_fast,
    find_optimal_threshold_fast,
    FeatureCache,
    check_cooccurrence, retrain_head_fast,
    MutualExclusionRegistry,
    evaluate_all_heads_fast, make_multilabel_binary,
    load_heads_from_state,
)
from E2_add_activity import load_state

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SEED_SET  = set(cfg.SEED_ACTIVITIES)

# Incremental order: all activities not in seed set, shuffled deterministically
rng            = np.random.default_rng(cfg.SEED)
ALL_ACTIVITIES = cfg.ALL_ACTIVITIES
NEW_ACTIVITIES = [a for a in rng.permutation(ALL_ACTIVITIES) if a not in SEED_SET]

# Ablation conditions from hparams.json
ABLATION_CONDITIONS = cfg.ABLATION_CONDITIONS


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def weighted_f1(snap):
    if not snap:
        return 0.0
    f1s     = np.array([v["f1"]    for v in snap.values()])
    weights = np.array([v["n_pos"] for v in snap.values()], dtype=float)
    if weights.sum() == 0:
        return 0.0
    return float(np.dot(f1s, weights / weights.sum()))


def weighted_metric(snap, metric):
    if not snap:
        return 0.0
    vals    = np.array([v.get(metric, 0.0) for v in snap.values()])
    weights = np.array([v["n_pos"]          for v in snap.values()], dtype=float)
    if weights.sum() == 0:
        return 0.0
    return float(np.dot(vals, weights / weights.sum()))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE CONDITION RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_condition(
    condition: dict,
    e1_state_path: str,
    feat_cache, np_train_raw, np_val_raw, np_test_raw, label_dict,
    logger: RunLogger,
):
    condition_name  = condition["name"]
    use_sensor_gating = condition["sensor_gating"]
    use_me            = condition["me"]
    use_retraining    = condition["retraining"]

    print(f"\n{'#'*65}")
    print(f"ABLATION: {condition_name}")
    print(f"  sensor_gating={use_sensor_gating}  ME={use_me}  retraining={use_retraining}")
    print(f"{'#'*65}")
    logger.event("INFO",
        f"Ablation condition: '{condition_name}' "
        f"(gating={use_sensor_gating} me={use_me} retrain={use_retraining})"
    )

    state         = load_state(e1_state_path)
    fusion        = state["fusion"]
    heads         = state["heads"]
    thresholds    = state["thresholds"]
    replay_buffer = state["replay_buffer"]
    trained       = list(state["trained_activities"])
    D             = state["feature_dim"]

    baseline_metrics = {}
    for activity in cfg.SEED_ACTIVITIES:
        if (activity, fusion) not in heads:
            continue
        y_te = make_multilabel_binary(activity, np_test_raw[1], label_dict, cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
        if len(np.unique(y_te)) < 2:
            continue
        baseline_metrics[activity] = evaluate_head_fast(
            heads[(activity, fusion)], feat_cache.test, y_te,
            threshold=thresholds.get((activity, fusion), 0.5)
        )

    me_registry = MutualExclusionRegistry()
    if use_me:
        for a, b in cfg.MUTUAL_EXCLUSIONS:
            me_registry.add(a, b)

    history = {
        "ood":               {"baseline": evaluate_all_heads_fast(
                                  heads, feat_cache.val, np_val_raw[1],
                                  label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])},
        "test_ood":          {"baseline": evaluate_all_heads_fast(
                                  heads, feat_cache.test, np_test_raw[1],
                                  label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])},
        "cooccurrence":      [],
        "forgetting":        [],
        "retraining_events": 0,
        "head_birth_step":   {},
        "head_retrain_steps": {},
        "test_metrics":      None,
        "flags": {
            "sensor_gating": use_sensor_gating,
            "me":            use_me,
            "retraining":    use_retraining,
        },
    }

    for step, new_activity in enumerate(NEW_ACTIVITIES):
        if new_activity not in label_dict:
            continue

        class_idx = label_dict[new_activity]
        mask_val  = (np_val_raw[1]   == class_idx)
        mask_tr   = (np_train_raw[1] == class_idx)
        Z_new_val = feat_cache.val[mask_val]
        Z_new_tr  = feat_cache.train[mask_tr]

        if Z_new_val.shape[0] == 0 or Z_new_tr.shape[0] == 0:
            print(f"  Skipping '{new_activity}' — no samples")
            continue

        print(f"\n[{step+1:02d}/{len(NEW_ACTIVITIES)}] '{new_activity}' "
              f"(val:{Z_new_val.shape[0]} tr:{Z_new_tr.shape[0]})")

        retrained_this_step = set()
        pre_test_ood = evaluate_all_heads_fast(
            heads, feat_cache.test, np_test_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])

    # ── Co-occurrence check ──────────────────────────────────────────────
        cooc_results = check_cooccurrence(
            new_activity, Z_new_val, heads, thresholds, fusion,
            fire_threshold=cfg.COOC_FIRE_THRESHOLD,
        )

        if use_me:
            me_registry.add_for_activity(new_activity, trained, cfg=cfg)
            me_registry.apply_suppression(cooc_results)

        confirmed, missed, false_pos = simulate_cooccurrence_confirmation(
            new_activity, cooc_results, trained, cfg)

        # Compute co-occurrence detection metrics
        gt_cooc   = set(cfg.get_cooccurrences_for(new_activity, trained))
        fired     = {a for a, r in cooc_results.items() if r["fires"]}
        tp        = fired & gt_cooc
        fp        = fired - gt_cooc
        fn        = gt_cooc - fired if not (fired & gt_cooc) else set()
        precision = len(tp) / max(len(fired), 1)
        recall    = 1.0 if (gt_cooc and tp) else (0.0 if gt_cooc else 1.0)
        f1_cooc   = 2 * precision * recall / max(precision + recall, 1e-8)
        history["cooccurrence"].append({
            "step": step, "activity": new_activity,
            "precision": precision, "recall": recall, "f1": f1_cooc,
            "TP": sorted(tp), "FP": sorted(fp), "FN": sorted(fn),
        })

        # ── Retraining ───────────────────────────────────────────────────────
        if use_retraining:
            for activity in missed:
                if not should_retrain_for_fn(activity, cfg):
                    continue
                if activity not in label_dict:
                    continue
                y_vl_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
                if y_vl_bin.sum() == 0:
                    continue
                heads[(activity, fusion)], _, _ = retrain_head_fast(
                    activity=activity,
                    model=heads[(activity, fusion)],
                    replay_buffer=replay_buffer,
                    Z_new=Z_new_val, y_new_label=1,
                    Z_val=feat_cache.val, y_val=y_vl_bin,
                    Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                    label_dict=label_dict, trained_activities=trained,
                    epochs=cfg.RETRAIN_EPOCHS, lr=cfg.RETRAIN_LR,
                    timestamp=TIMESTAMP, working_dir=cfg.WORKING_DIR,
                )
                history["retraining_events"] += 1
                retrained_this_step.add(activity)

            for activity in false_pos:
                if (activity, fusion) not in heads or activity not in label_dict:
                    continue
                y_vl_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
                if y_vl_bin.sum() == 0:
                    continue
                heads[(activity, fusion)], _, _ = retrain_head_fast(
                    activity=activity,
                    model=heads[(activity, fusion)],
                    replay_buffer=replay_buffer,
                    Z_new=Z_new_val, y_new_label=0,
                    Z_val=feat_cache.val, y_val=y_vl_bin,
                    Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                    label_dict=label_dict, trained_activities=trained,
                    epochs=cfg.RETRAIN_EPOCHS, lr=cfg.RETRAIN_LR,
                    timestamp=TIMESTAMP, working_dir=cfg.WORKING_DIR,
                )
                history["retraining_events"] += 1
                retrained_this_step.add(activity)

        # ── Train new activity head ───────────────────────────────────────────
        y_tr_new = (np_train_raw[1] == class_idx).astype(np.int32)
        y_vl_new = (np_val_raw[1]   == class_idx).astype(np.int32)
        hint     = cfg.get_sensor_hint(new_activity) \
                   if use_sensor_gating else cfg.UNIFORM_HINT
        model    = build_gated_head_from_features(hint, D, fusion=fusion,
                                                  gate_scale=cfg.GATE_SCALE)

        safe  = new_activity.replace(" ", "_")
        cname = condition_name.replace(" ", "_").replace("+", "").replace("/", "")
        spath = os.path.join(
            cfg.WORKING_DIR,
            f"{TIMESTAMP}_e3_abl_{safe}_{fusion}_{cname}.pt"
        )
        model = train_head_fast(
            model, feat_cache.train, y_tr_new,
            feat_cache.val, y_vl_new,
            spath,
            epochs=cfg.INCREMENTAL_HEAD_EPOCHS,
            lr=cfg.LEARNING_RATE,
            focal_gamma=cfg.FOCAL_GAMMA,
            max_class_weight=cfg.MAX_CLASS_WEIGHT,
        )
        thresh = find_optimal_threshold_fast(
            model, feat_cache.val, y_vl_new,
            t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
            fallback=cfg.THRESHOLD_FALLBACK,
        )
        heads[(new_activity, fusion)]      = model
        thresholds[(new_activity, fusion)] = thresh
        Z_pos_new = feat_cache.train[np_train_raw[1] == class_idx]
        replay_buffer.store_positives(new_activity, Z_pos_new)
        trained.append(new_activity)
        history["head_birth_step"][new_activity] = step

        for act in retrained_this_step:
            history["head_retrain_steps"].setdefault(act, []).append(step)
        history["head_retrain_steps"].setdefault(new_activity, []).append(step)

        if retrained_this_step:
            trained_idxs = {label_dict[a] for a in trained if a in label_dict}
            mask_seen    = np.array([i in trained_idxs for i in np_val_raw[1]])
            Z_val_seen   = feat_cache.val[mask_seen]
            y_val_seen   = np_val_raw[1][mask_seen]
            for activity in retrained_this_step:
                if (activity, fusion) not in heads or activity not in label_dict:
                    continue
                y_bin = make_multilabel_binary(activity, y_val_seen, label_dict, cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
                if y_bin.sum() == 0:
                    continue
                thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                    heads[(activity, fusion)], Z_val_seen, y_bin,
                    t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                    fallback=cfg.THRESHOLD_FALLBACK,
                )

        # ── Snapshot metrics ─────────────────────────────────────────────────
        pre_val_ood   = evaluate_all_heads_fast(
            heads, feat_cache.val, np_val_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
        changed_heads = retrained_this_step | {new_activity}

        def reeval_changed(pre_snap, Z, y):
            snap = dict(pre_snap)
            for act in changed_heads:
                if act not in label_dict:
                    continue
                y_bin  = make_multilabel_binary(act, y, label_dict, cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
                if len(np.unique(y_bin)) < 2:
                    continue
                probs  = evaluate_head_fast(heads[(act, fusion)], Z, y_bin)["probs"]
                thresh = thresholds.get((act, fusion), 0.5)
                preds  = (probs >= thresh).astype(int)
                snap[act] = {
                    "auc":       float(roc_auc_score(y_bin, probs)),
                    "f1":        float(f1_score(y_bin, preds, zero_division=0)),
                    "accuracy":  float(accuracy_score(y_bin, preds)),
                    "precision": float(precision_score(y_bin, preds, zero_division=0)),
                    "recall":    float(recall_score(y_bin, preds, zero_division=0)),
                    "n_pos":     int(y_bin.sum()),
                    "n_neg":     int((y_bin == 0).sum()),
                }
            return snap

        ood      = reeval_changed(pre_val_ood,  feat_cache.val,  np_val_raw[1])
        test_ood = reeval_changed(pre_test_ood, feat_cache.test, np_test_raw[1])
        history["ood"][new_activity]      = ood
        history["test_ood"][new_activity] = test_ood

        forgetting = {}
        for activity in cfg.SEED_ACTIVITIES:
            if activity not in baseline_metrics:
                continue
            y_te_s = make_multilabel_binary(activity, np_test_raw[1], label_dict, cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
            if len(np.unique(y_te_s)) < 2:
                continue
            cur = evaluate_head_fast(
                heads[(activity, fusion)], feat_cache.test, y_te_s,
                threshold=thresholds.get((activity, fusion), 0.5))
            forgetting[activity] = {
                "auc_baseline": baseline_metrics[activity]["auc"],
                "auc_current":  cur["auc"],
                "delta":        cur["auc"] - baseline_metrics[activity]["auc"],
            }
        history["forgetting"].append({
            "step": step, "activity": new_activity, "forgetting": forgetting})

        seed_metrics = {a: ood[a] for a in cfg.SEED_ACTIVITIES if a in ood}
        act_metrics  = {a: ood[a] for a in ood if a not in SEED_SET}
        print(f"  Seeds F1:{np.mean([m['f1'] for m in seed_metrics.values()]):.4f} | "
              f"Activities({len(act_metrics)}) F1:"
              f"{np.mean([m['f1'] for m in act_metrics.values()]):.4f} | "
              f"Co-occ F1:{history['cooccurrence'][-1]['f1']:.3f} | "
              f"Retrains:{history['retraining_events']}")

    last_activity = next((a for a in reversed(NEW_ACTIVITIES)
                          if a in history["test_ood"]), None)
    history["test_metrics"] = history["test_ood"].get(
        last_activity,
        evaluate_all_heads_fast(heads, feat_cache.test, np_test_raw[1],
                                label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
    )

    seed_test = {a: v for a, v in history["test_metrics"].items() if a in SEED_SET}
    act_test  = {a: v for a, v in history["test_metrics"].items() if a not in SEED_SET}
    all_test  = {**seed_test, **act_test}
    print(f"\n[Final] TEST — "
          f"All Weighted F1:{weighted_f1(all_test):.4f} | "
          f"Seeds Weighted F1:{weighted_f1(seed_test):.4f} | "
          f"Activities({len(act_test)}) Weighted F1:{weighted_f1(act_test):.4f}")

    logger.log_ablation_condition(
        condition_name,
        flags={"sensor_gating": use_sensor_gating, "me": use_me, "retraining": use_retraining},
        final_metrics=history["test_metrics"],
        retraining_events=history["retraining_events"],
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

COND_COLORS = ["#9E9E9E", "#2E7D32", "#FF8A65", "#64B5F6", "#AB47BC"]
COND_STYLES = [":",       "-",        "--",      "-.",      (0,(3,1,1,1))]


def plot_ablation(all_histories):
    cond_names  = [c["name"] for c in ABLATION_CONDITIONS]
    cond_labels = cond_names

    # ── 1. Weighted F1 trend ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, title, filter_fn in zip(
        axes,
        [f"All {len(ALL_ACTIVITIES)} Activities", "Seed Heads Only"],
        [lambda a: True, lambda a: a in SEED_SET],
    ):
        for (cname, color, style) in zip(cond_names, COND_COLORS, COND_STYLES):
            history = all_histories[cname]
            steps   = list(history["test_ood"].keys())
            means   = []
            for s in steps:
                snap = {a: v for a, v in history["test_ood"][s].items()
                        if filter_fn(a) and "f1" in v and "n_pos" in v}
                if not snap:
                    means.append(np.nan); continue
                f1s     = np.array([v["f1"]   for v in snap.values()])
                weights = np.array([v["n_pos"] for v in snap.values()], dtype=float)
                weights /= weights.sum()
                means.append(float(np.dot(weights, f1s)))
            ax.plot(range(len(steps)), means, label=cname,
                    color=color, linestyle=style, linewidth=2)
        ax.set_xlabel("Step (activities added)")
        ax.set_ylabel("Weighted F1")
        ax.set_title(f"Ablation — Weighted F1 Trend\n{title}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p = os.path.join(cfg.FIGS_DIR, f"{TIMESTAMP}_abl_f1_trend_{cfg.FUSION}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── 2. Final metrics bar chart ────────────────────────────────────────────
    metrics_list = ["auc", "f1", "precision", "recall"]
    x     = np.arange(len(metrics_list))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, title, filter_fn in zip(
        axes,
        ["Seed Heads", "All Activity Heads"],
        [lambda a: a in SEED_SET, lambda a: a not in SEED_SET],
    ):
        for i, (cname, color) in enumerate(zip(cond_names, COND_COLORS)):
            snap = {a: v for a, v in all_histories[cname]["test_metrics"].items()
                    if filter_fn(a) and "n_pos" in v}
            if not snap: continue
            wts   = np.array([v["n_pos"] for v in snap.values()], dtype=float)
            wts  /= wts.sum()
            means = [float(np.dot([v.get(m, 0.0) for v in snap.values()], wts))
                     for m in metrics_list]
            offset = (i - len(cond_names) / 2 + 0.5) * width
            bars   = ax.bar(x + offset, means, width, label=cname, color=color, alpha=0.88)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics_list])
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Ablation — Final Metrics (Test)\n{title}", fontsize=11)
        ax.set_ylabel("Weighted Score"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(cfg.FIGS_DIR, f"{TIMESTAMP}_abl_final_metrics_{cfg.FUSION}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── 3. Seed forgetting ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(cond_names),
                              figsize=(6 * len(cond_names), 5), sharey=True)
    cmap_seed = cm.get_cmap("tab10", len(cfg.SEED_ACTIVITIES))
    for ax, cname, color in zip(axes, cond_names, COND_COLORS):
        history = all_histories[cname]
        steps   = history["forgetting"]
        for j, activity in enumerate(cfg.SEED_ACTIVITIES):
            deltas = [s["forgetting"].get(activity, {}).get("delta", np.nan) for s in steps]
            ax.plot(range(len(steps)), deltas, label=activity,
                    color=cmap_seed(j), linewidth=1.5, marker=".", markersize=3)
        ax.axhline(0,     color="black", linewidth=0.8, linestyle="--")
        ax.axhline(-0.01, color="red",   linewidth=0.8, linestyle=":", label="−0.01")
        ax.set_title(f"{cname}", fontsize=9)
        ax.set_xlabel("Activities added"); ax.set_ylabel("AUC delta vs E1")
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    fig.suptitle("Seed Head Forgetting across Ablation Conditions", fontsize=12)
    plt.tight_layout()
    p = os.path.join(cfg.FIGS_DIR, f"{TIMESTAMP}_abl_forgetting_{cfg.FUSION}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── 4. Per-activity heatmap ───────────────────────────────────────────────
    all_acts = cfg.SEED_ACTIVITIES + sorted({
        a for cname in cond_names
        for a in all_histories[cname]["test_metrics"]
        if a not in SEED_SET
    })
    matrix = np.full((len(cond_names), len(all_acts)), np.nan)
    for i, cname in enumerate(cond_names):
        tm = all_histories[cname]["test_metrics"]
        for j, act in enumerate(all_acts):
            if act in tm:
                matrix[i, j] = tm[act].get("f1", np.nan)

    fig, ax = plt.subplots(figsize=(max(20, len(all_acts) * 0.5), 4))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(len(all_acts)))
    ax.set_xticklabels([a.replace("_", " ") for a in all_acts],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(cond_names))); ax.set_yticklabels(cond_labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="black")
    seed_idxs = [j for j, a in enumerate(all_acts) if a in SEED_SET]
    for j in seed_idxs:
        ax.add_patch(plt.Rectangle((j-0.5,-0.5), 1, len(cond_names),
                                    linewidth=1.5, edgecolor="gold",
                                    facecolor="none", zorder=3))
    plt.colorbar(im, ax=ax, shrink=0.6, label="F1")
    ax.set_title("Per-Activity Final F1 Heatmap (Test) — Ablation\n(gold = seed activities)",
                 fontsize=11)
    plt.tight_layout()
    p = os.path.join(cfg.FIGS_DIR, f"{TIMESTAMP}_abl_heatmap_{cfg.FUSION}.pdf")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    logger = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP, script="E3_ablation")
    logger.log_run_start(cfg)

    print("Loading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )
    print(f"  Activities to add: {len(NEW_ACTIVITIES)}")

    # ── Load encoder(s) ──────────────────────────────────────────────────────
    logger.event("INFO", "Loading encoder(s)...")
    from encoder import load_encoders_from_cfg, extract_all_features
    encoders = load_encoders_from_cfg(cfg)

    # ── Precompute encoder features ───────────────────────────────────────────
    logger.event("INFO", "Precomputing encoder features...")
    Z_train = extract_all_features(np_train_raw[0], encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE)
    Z_val   = extract_all_features(np_val_raw[0],   encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE)
    Z_test  = extract_all_features(np_test_raw[0],  encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE)
    feat_cache = FeatureCache(Z_train, Z_val, Z_test)
    D = Z_train.shape[-1]
    logger.event("INFO",
        f"Train: {Z_train.shape}  Val: {Z_val.shape}  Test: {Z_test.shape}"
    )

    print(f"  Train:{Z_train.shape}  Val:{Z_val.shape}  Test:{Z_test.shape}")

    # Auto-select E1 state
    matches = sorted(glob.glob(
        os.path.join(cfg.WORKING_DIR, f"*_e1_state_{cfg.FUSION}.pkl")
    ))
    if not matches:
        raise FileNotFoundError(
            f"No E1 state file found in {cfg.WORKING_DIR}. Run E1_train_seeds.py first."
        )
    e1_state_path = matches[-1]
    print(f"\nUsing E1 state: {e1_state_path}")
    logger.event("INFO", f"E1 state: {e1_state_path}")

    all_histories = {}
    for condition in ABLATION_CONDITIONS:
        cname = condition["name"]
        all_histories[cname] = run_ablation_condition(
            condition=condition,
            e1_state_path=e1_state_path,
            feat_cache=feat_cache,
            np_train_raw=np_train_raw,
            np_val_raw=np_val_raw,
            np_test_raw=np_test_raw,
            label_dict=label_dict,
            logger=logger,
        )

    print(f"\n{'='*65}")
    print("ABLATION FINAL SUMMARY (Test Set)")
    for condition in ABLATION_CONDITIONS:
        cname     = condition["name"]
        history   = all_histories[cname]
        seed_snap = {a: v for a, v in history["test_metrics"].items() if a in SEED_SET}
        act_snap  = {a: v for a, v in history["test_metrics"].items() if a not in SEED_SET}
        all_snap  = {**seed_snap, **act_snap}
        cooc      = history["cooccurrence"]
        mean_cooc = np.mean([r["f1"] for r in cooc]) if cooc else 0.0
        print(f"\n  [{cname}]")
        print(f"    flags: gating={condition['sensor_gating']} ME={condition['me']} "
              f"retrain={condition['retraining']}")
        print(f"    All heads      ({len(all_snap):2d}) — "
              f"Weighted F1:{weighted_f1(all_snap):.4f}")
        print(f"    Seed heads     ({len(seed_snap):2d}) — "
              f"Weighted F1:{weighted_f1(seed_snap):.4f}")
        print(f"    Activity heads ({len(act_snap):2d}) — "
              f"Weighted F1:{weighted_f1(act_snap):.4f}")
        print(f"    Co-occ mean F1 : {mean_cooc:.4f}")
        print(f"    Retraining events: {history['retraining_events']}")

    # Save histories
    pkl_path = os.path.join(
        cfg.WORKING_DIR,
        f"{TIMESTAMP}_e3_ablation_histories_{cfg.FUSION}.pkl"
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(all_histories, f)
    print(f"\nHistories saved to {pkl_path}")

    log_path = logger.save_alongside(pkl_path)

    print("\nGenerating plots...")
    plot_ablation(all_histories)
    print(f"\nDone.")
    print(f"  Histories : {pkl_path}")
    print(f"  Log       : {log_path}")
    print(f"  Figures   : {cfg.FIGS_DIR}")