"""
E2_add_activity.py
==================
Add one new activity to an existing HITL-HAR system.

Reads all settings from paths.json / dataset config / hparams.json via config_loader.
HITL interactions (co-occurrence confirmation, ME marking) are handled by
hitl_simulation.py — swap that module to change simulation strategy.

Usage
-----
  python E2_add_activity.py --activity Treadmill_2mph_Lab --state output/DS_11_e1_state_early.pkl
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
import argparse
import random
import numpy as np
import torch
from datetime import datetime

from config_loader import cfg
from logger import RunLogger
from hitl_simulation import (
    simulate_cooccurrence_confirmation,
    simulate_mutual_exclusion_marking,
    should_retrain_for_fn,
)
from helpers import create_dataset_file_split
from helpers_hitl import (
    make_binary_labels, build_gated_head_from_features,
    train_head_fast, evaluate_head_fast,
    find_optimal_threshold_fast,
    FeatureCache,
    check_cooccurrence, retrain_head_fast,
    MutualExclusionRegistry, NegativeBuffer, ReplayBuffer,
    evaluate_all_heads_fast, make_multilabel_binary,
    load_heads_from_state, save_head_weights,
)

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_path: str) -> dict:
    print(f"Loading state from {state_path}...")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    fusion = state["fusion"]
    D      = state.get("feature_dim")
    if D is None:
        raise ValueError("State missing 'feature_dim' — re-run E1.")
    state["heads"] = load_heads_from_state(state, fusion, cfg=cfg)
    print(f"  Loaded {len(state['heads'])} heads  D={D}")
    return state


def save_state(state: dict, state_path: str):
    """Save updated weights for any new/retrained heads, then pickle metadata."""
    for (activity, f), model in state["heads"].items():
        if state.get("_updated", {}).get((activity, f), False):
            wpath = save_head_weights(
                activity, f, model, cfg.WORKING_DIR,
                state.get("timestamp", TIMESTAMP)
            )
            state["weights_paths"][(activity, f)] = wpath

    state_to_save = {k: v for k, v in state.items()
                     if k not in ("heads", "_updated")}
    with open(state_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  State saved to {state_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HITL STEP
# ─────────────────────────────────────────────────────────────────────────────

def run_add_activity(new_activity, state,
                     feat_cache, np_train_raw, np_val_raw, np_test_raw,
                     label_dict, me_registry, logger: RunLogger):
    fusion             = state["fusion"]
    heads              = state["heads"]
    thresholds         = state["thresholds"]
    replay_buffer      = state["replay_buffer"]
    trained_activities = state["trained_activities"]
    D                  = state["feature_dim"]

    if new_activity not in label_dict:
        logger.warn(f"'{new_activity}' not in label_dict — skipping.")
        return state, None, None

    class_idx = label_dict[new_activity]
    mask_val  = (np_val_raw[1]   == class_idx)
    mask_tr   = (np_train_raw[1] == class_idx)
    Z_new_val = feat_cache.val[mask_val]

    print(f"\n{'='*60}")
    print(f"Adding: '{new_activity}'")
    logger.event("INFO", f"Adding activity: '{new_activity}'")

    if Z_new_val.shape[0] == 0:
        logger.warn(f"No val samples for '{new_activity}' — skipping.")
        return state, None, None

    step_info = {"activity": new_activity}

    # ── Step 1: Co-occurrence check ───────────────────────────────────────────
    print(f"\n[Step 1] Co-occurrence check")
    cooc_results = check_cooccurrence(
        new_activity, Z_new_val, heads, thresholds,
        fusion=fusion,
        fire_threshold=cfg.COOC_FIRE_THRESHOLD,
    )
    print(f"  {'Activity':<40} {'Mean P':>7} {'%Pos':>6} {'Fires':>6} {'Thresh':>7}")
    print(f"  {'-'*65}")
    for act, r in sorted(cooc_results.items()):
        print(f"  {act:<40} {r['mean_prob']:>7.3f} "
              f"{r['pct_positive']:>6.2%} "
              f"{'✓' if r['fires'] else '✗':>6} "
              f"{r['threshold']:>7.2f}")

    # ── Step 2: User confirms co-occurrences (simulated) ─────────────────────
    print(f"\n[Step 2] User confirms co-occurrences")
    confirmed, missed, false_pos = simulate_cooccurrence_confirmation(
        new_activity, cooc_results, trained_activities, cfg
    )
    step_info.update({"confirmed": confirmed, "missed": missed, "false_pos": false_pos})

    # ── Step 3: User marks mutual exclusions (simulated) ─────────────────────
    print(f"\n[Step 3] User marks mutual exclusions")
    simulate_mutual_exclusion_marking(new_activity, trained_activities, cfg)
    me_registry.add_for_activity(new_activity, trained_activities, cfg=cfg)

    # ── Step 4: Post-hoc ME suppression ──────────────────────────────────────
    print(f"\n[Step 4] Post-hoc ME suppression")
    _, suppressed = me_registry.apply_suppression(cooc_results)
    if not suppressed:
        print("  No conflicts — nothing suppressed.")
    step_info["suppressed"] = suppressed

    # ── Step 5: Pre-training OOD snapshot ─────────────────────────────────────
    print(f"\n[Step 5] Pre-training OOD evaluation")
    mask_test_new = (np_test_raw[1] == class_idx)
    Z_new_test    = feat_cache.test[mask_test_new]

    pre_Z_val  = np.concatenate([feat_cache.val, Z_new_val], axis=0)
    pre_y_val  = np.concatenate([np_val_raw[1],
                                  np.full(Z_new_val.shape[0], -1)], axis=0)
    pre_Z_test = (np.concatenate([feat_cache.test, Z_new_test], axis=0)
                  if Z_new_test.shape[0] > 0 else feat_cache.test)
    pre_y_test = (np.concatenate([np_test_raw[1],
                                   np.full(Z_new_test.shape[0], -1)], axis=0)
                  if Z_new_test.shape[0] > 0 else np_test_raw[1])

    pre_val_ood  = evaluate_all_heads_fast(
        heads, pre_Z_val,  pre_y_val,  label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
    pre_test_ood = evaluate_all_heads_fast(
        heads, pre_Z_test, pre_y_test, label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])

    # ── Step 6: Retrain for FNs ───────────────────────────────────────────────
    if missed:
        print(f"\n[Step 6] Retraining for FNs: {missed}")
        for activity in missed:
            if not should_retrain_for_fn(activity, cfg):
                continue
            if (activity, fusion) not in heads:
                continue
            y_val_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
            heads[(activity, fusion)], _, _ = retrain_head_fast(
                activity=activity,
                model=heads[(activity, fusion)],
                replay_buffer=replay_buffer,
                Z_new=Z_new_val, y_new_label=1,
                Z_val=feat_cache.val, y_val=y_val_bin,
                Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                label_dict=label_dict,
                trained_activities=trained_activities,
                epochs=cfg.RETRAIN_EPOCHS,
                lr=cfg.RETRAIN_LR,
                timestamp=TIMESTAMP,
                working_dir=cfg.WORKING_DIR,
            )
            thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                heads[(activity, fusion)], feat_cache.val, y_val_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )
            state.setdefault("_updated", {})[(activity, fusion)] = True
    else:
        print(f"\n[Step 6] No FNs — no retraining needed.")

    # ── Step 7: Retrain for FPs ───────────────────────────────────────────────
    if false_pos:
        print(f"\n[Step 7] Retraining for FPs: {false_pos}")
        for activity in false_pos:
            if (activity, fusion) not in heads:
                continue
            y_val_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
            heads[(activity, fusion)], _, _ = retrain_head_fast(
                activity=activity,
                model=heads[(activity, fusion)],
                replay_buffer=replay_buffer,
                Z_new=Z_new_val, y_new_label=0,
                Z_val=feat_cache.val, y_val=y_val_bin,
                Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                label_dict=label_dict,
                trained_activities=trained_activities,
                epochs=cfg.RETRAIN_EPOCHS,
                lr=cfg.RETRAIN_LR,
                timestamp=TIMESTAMP,
                working_dir=cfg.WORKING_DIR,
            )
            thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                heads[(activity, fusion)], feat_cache.val, y_val_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )
            state.setdefault("_updated", {})[(activity, fusion)] = True
    else:
        print(f"\n[Step 7] No FPs — no retraining needed.")

    # ── Step 8: Train new activity head ───────────────────────────────────────
    print(f"\n[Step 8] Training head for '{new_activity}'")
    excl_idxs = {label_dict[a]
                 for a in cfg.get_training_exclusions(new_activity)
                 if a in label_dict}
    tr_mask   = np.array([i == class_idx or i not in excl_idxs
                          for i in np_train_raw[1]])
    vl_mask_t = np.array([i == class_idx or i not in excl_idxs
                          for i in np_val_raw[1]])
    y_tr_new  = (np_train_raw[1][tr_mask]  == class_idx).astype(np.int32)
    y_vl_new  = (np_val_raw[1][vl_mask_t]  == class_idx).astype(np.int32)
    Z_tr_new  = feat_cache.train[tr_mask]
    Z_vl_new  = feat_cache.val[vl_mask_t]

    hint      = cfg.get_sensor_hint(new_activity)
    model     = build_gated_head_from_features(hint, D, fusion=fusion,
                                               gate_scale=cfg.GATE_SCALE)
    safe_name = new_activity.replace(" ", "_")
    save_path = os.path.join(cfg.WORKING_DIR,
                             f"{TIMESTAMP}_head_{safe_name}_{fusion}.pt")

    t0    = time.time()
    model = train_head_fast(
        model, feat_cache.train, y_tr_new,
        feat_cache.val, y_vl_new,
        save_path,
        epochs=cfg.INCREMENTAL_HEAD_EPOCHS,
        lr=cfg.LEARNING_RATE,
        focal_gamma=cfg.FOCAL_GAMMA,
        max_class_weight=cfg.MAX_CLASS_WEIGHT,
    )
    print(f"  Trained in {time.time()-t0:.1f}s")

    thresh = find_optimal_threshold_fast(
        model, feat_cache.val, y_vl_new,
        t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
        fallback=cfg.THRESHOLD_FALLBACK,
    )
    new_metrics = evaluate_head_fast(model, feat_cache.test,
                                     (np_test_raw[1] == class_idx).astype(np.int32),
                                     threshold=thresh)
    print(f"  Threshold: {thresh:.2f}  "
          f"AUC:{new_metrics['auc']:.4f} F1:{new_metrics['f1']:.4f}")
    step_info["new_head_metrics"] = new_metrics
    step_info["threshold"]        = thresh

    heads[(new_activity, fusion)]      = model
    thresholds[(new_activity, fusion)] = thresh
    wpath = save_head_weights(new_activity, fusion, model,
                              cfg.WORKING_DIR, TIMESTAMP)
    state["weights_paths"][(new_activity, fusion)] = wpath
    Z_pos_new = feat_cache.train[np_train_raw[1] == class_idx]
    replay_buffer.store_positives(new_activity, Z_pos_new)
    trained_activities.append(new_activity)
    state.setdefault("_updated", {})[(new_activity, fusion)] = True

    # ── Step 9: Recalibrate thresholds for retrained heads ────────────────────
    retrained = set(missed) | set(false_pos)
    if retrained:
        trained_idxs = {label_dict[a] for a in trained_activities if a in label_dict}
        mask_seen    = np.array([i in trained_idxs for i in np_val_raw[1]])
        y_val_seen   = np_val_raw[1][mask_seen]
        Z_val_seen   = feat_cache.val[mask_seen]
        for activity in retrained:
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

    # ── Step 10/11: OOD evaluation ─────────────────────────────────────────────
    print(f"\n[Step 10] Val OOD evaluation")
    val_ood = evaluate_all_heads_fast(
        heads, feat_cache.val, np_val_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
    print(f"\n[Step 11] Test OOD evaluation")
    test_ood = evaluate_all_heads_fast(
        heads, feat_cache.test, np_test_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])

    step_info["pre_val_ood"]   = pre_val_ood
    step_info["post_val_ood"]  = val_ood
    step_info["pre_test_ood"]  = pre_test_ood
    step_info["post_test_ood"] = test_ood
    step_info["retrained"]     = sorted(retrained)

    # ── Before/After table ────────────────────────────────────────────────────
    all_acts = sorted(set(pre_val_ood) | set(val_ood))
    print(f"\n  {'─'*112}")
    print(f"  {'Activity':<45} {'':^4} {'':^6} "
          f"{'Val AUC':>8} {'Val F1':>7} {'Tst AUC':>8} {'Tst F1':>7}")
    print(f"  {'─'*112}")
    for act in all_acts:
        seed_tag = " *" if act in set(cfg.SEED_ACTIVITIES) else ""
        role     = "[NEW]" if act == new_activity else \
                   ("[RTR]" if act in retrained else "")
        pre_v, pre_t = pre_val_ood.get(act, {}), pre_test_ood.get(act, {})
        pst_v, pst_t = val_ood.get(act, {}),     test_ood.get(act, {})
        def fmt(d, k): return f"{d[k]:>7.4f}" if k in d else f"{'—':>7}"
        print(f"  {act+seed_tag:<45} {'BEF':>4} {role:>6} "
              f"{fmt(pre_v,'auc'):>8} {fmt(pre_v,'f1'):>7} "
              f"{fmt(pre_t,'auc'):>8} {fmt(pre_t,'f1'):>7}")
        print(f"  {'':<45} {'AFT':>4} {role:>6} "
              f"{fmt(pst_v,'auc'):>8} {fmt(pst_v,'f1'):>7} "
              f"{fmt(pst_t,'auc'):>8} {fmt(pst_t,'f1'):>7}")
        if act in retrained or act == new_activity:
            print(f"  {'':<45} {'Δ':>4} {role:>6} "
                  f"{pst_v.get('auc',0)-pre_v.get('auc',0):>+8.4f} "
                  f"{pst_v.get('f1', 0)-pre_v.get('f1', 0):>+7.4f} "
                  f"{pst_t.get('auc',0)-pre_t.get('auc',0):>+8.4f} "
                  f"{pst_t.get('f1', 0)-pre_t.get('f1', 0):>+7.4f}")
        print(f"  {'·'*112}")

    # ── Update state ──────────────────────────────────────────────────────────
    state["heads"]              = heads
    state["thresholds"]         = thresholds
    state["replay_buffer"]      = replay_buffer
    state["trained_activities"] = trained_activities
    state.setdefault("ood_history",          {})[new_activity] = val_ood
    state.setdefault("test_ood_history",     {})[new_activity] = test_ood
    state.setdefault("pre_ood_history",      {})[new_activity] = pre_val_ood
    state.setdefault("pre_test_ood_history", {})[new_activity] = pre_test_ood
    state.setdefault("head_birth_step",      {})[new_activity] = \
        len(trained_activities) - 1

    logger.log_activity_step(new_activity, step_info)
    return state, val_ood, test_ood


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", type=str, required=True,
                        help="Name of the activity to add (must exist in label_dict).")
    parser.add_argument("--state",    type=str,
                        default=None,
                        help="Path to .pkl state file from E1 or a previous E2 run.")
    args = parser.parse_args()

    logger = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP, script="E2_add_activity")
    logger.log_run_start(cfg)

    print("Loading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )

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

    state_path = args.state
    if state_path is None:
        import glob
        matches = sorted(glob.glob(
            os.path.join(cfg.WORKING_DIR, f"*_e1_state_{cfg.FUSION}.pkl")
        ))
        if not matches:
            raise FileNotFoundError(
                f"No E1 state file found in {cfg.WORKING_DIR}. "
                "Run E1_train_seeds.py first, or pass --state explicitly."
            )
        state_path = matches[-1]
        print(f"Auto-selected state: {state_path}")

    state       = load_state(state_path)
    me_registry = MutualExclusionRegistry()
    for a, b in cfg.MUTUAL_EXCLUSIONS:
        me_registry.add(a, b)

    state, val_ood, test_ood = run_add_activity(
        new_activity=args.activity,
        state=state,
        feat_cache=feat_cache,
        np_train_raw=np_train_raw,
        np_val_raw=np_val_raw,
        np_test_raw=np_test_raw,
        label_dict=label_dict,
        me_registry=me_registry,
        logger=logger,
    )

    safe_act  = args.activity.replace(" ", "_")
    out_path  = os.path.join(
        cfg.WORKING_DIR,
        f"{TIMESTAMP}_state_after_{safe_act}.pkl"
    )
    save_state(state, out_path)
    log_path = logger.save_alongside(out_path)

    print(f"\nDone.")
    print(f"  State : {out_path}")
    print(f"  Log   : {log_path}")
    print(f"\nNext run: python E2_add_activity.py --activity <NEXT_ACTIVITY> --state {out_path}")