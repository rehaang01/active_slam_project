#!/usr/bin/env python3
"""
eval_rl.py — Evaluate a trained RecurrentPPO Active SLAM agent
==============================================================
Loads the saved model from training, runs it deterministically on the
live Gazebo environment, and saves a CSV log in exactly the same schema
as all the baseline scripts so plot_comparison.py can read it directly.

CSV saved to:  logs/rl_eval_TIMESTAMP.csv

Usage
-----
  # Use the best model found during training (recommended):
  python3 eval_rl.py

  # Use a specific checkpoint:
  python3 eval_rl.py --model logs/checkpoints/active_slam_25000_steps.zip

  # Use the final model:
  python3 eval_rl.py --model logs/final_model.zip

  # Override number of steps:
  python3 eval_rl.py --max-steps 1000

What it does
------------
  1. Loads the saved RecurrentPPO model (best_model/ by default).
  2. Loads VecNormalize stats so observations are scaled the same as during training.
  3. Resets the environment and runs the policy deterministically (no exploration noise).
  4. At every step, reads the info dict returned by ActiveSLAMEnv.step() and logs it.
  5. Saves logs/rl_eval_TIMESTAMP.csv when done.

The CSV schema matches baseline_frontier.py exactly:
  step, timestamp, coverage, known_cells, total_cells, cov_trace,
  pos_x, pos_y, altitude, frontier_count, nearest_frontier_dist,
  loop_closures, tracking_lost_events, new_cells, note
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

LOG_DIR         = os.path.join(PROJECT_ROOT, "logs")
BEST_MODEL_DIR  = os.path.join(LOG_DIR, "best_model")
VEC_NORM_PATH   = os.path.join(LOG_DIR, "vec_normalize.pkl")

# Default model search order
DEFAULT_MODEL_PATHS = [
    os.path.join(BEST_MODEL_DIR, "best_model.zip"),   # EvalCallback saves this
    os.path.join(LOG_DIR, "final_model.zip"),
    os.path.join(LOG_DIR, "final_model"),              # SB3 saves without .zip too
]


def find_default_model() -> str | None:
    for p in DEFAULT_MODEL_PATHS:
        if os.path.isfile(p):
            return p
        # SB3 also saves without extension
        if os.path.isfile(p + ".zip"):
            return p + ".zip"
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained RecurrentPPO agent")
    p.add_argument("--model",      default=None,
                   help="Path to model .zip (default: auto-detect from logs/)")
    p.add_argument("--vec-norm",   default=VEC_NORM_PATH,
                   help=f"Path to VecNormalize .pkl (default: {VEC_NORM_PATH})")
    p.add_argument("--max-steps",  type=int, default=None,
                   help="Override MAX_STEPS from env (default: use env's MAX_STEPS)")
    p.add_argument("--output",     default=None,
                   help="Output CSV path (default: logs/rl_eval_TIMESTAMP.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve model path ────────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        model_path = find_default_model()
    if model_path is None:
        print(
            "[Eval] ERROR: No trained model found.\n"
            "  Looked in:\n" +
            "\n".join(f"    {p}" for p in DEFAULT_MODEL_PATHS) +
            "\n  Train first with:  python3 train.py"
        )
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"[Eval] ERROR: Model file not found: {model_path}")
        sys.exit(1)
    print(f"[Eval] Model:    {model_path}")

    # ── Imports (after path check so we fail fast on missing model) ───────────
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as e:
        print(f"[Eval] ERROR: Missing dependency: {e}")
        print("  Install with:  pip install sb3-contrib stable-baselines3")
        sys.exit(1)

    from envs.active_slam_env import ActiveSLAMEnv

    # ── Build environment ─────────────────────────────────────────────────────
    print("[Eval] Creating environment …")
    env = DummyVecEnv([lambda: ActiveSLAMEnv()])

    # Load VecNormalize stats if available (critical — obs must be scaled the same)
    if os.path.isfile(args.vec_norm):
        print(f"[Eval] Loading VecNormalize stats: {args.vec_norm}")
        env = VecNormalize.load(args.vec_norm, env)
        env.training = False       # Freeze running stats during eval
        env.norm_reward = False    # Don't normalise rewards during eval
    else:
        print(
            f"[Eval] WARNING: VecNormalize stats not found at {args.vec_norm}\n"
            "  Continuing without normalisation — performance may differ from training.\n"
            "  If training used VecNormalize (it does by default), results may be degraded."
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print("[Eval] Loading model …")
    model = RecurrentPPO.load(model_path, env=env)
    print("[Eval] Model loaded.")

    # ── Determine max steps ───────────────────────────────────────────────────
    # Pull MAX_STEPS from the env constants if not overridden
    try:
        from envs.active_slam_env import MAX_STEPS as ENV_MAX_STEPS
        max_steps = args.max_steps if args.max_steps is not None else ENV_MAX_STEPS
    except ImportError:
        max_steps = args.max_steps if args.max_steps is not None else 2000
    print(f"[Eval] Max steps: {max_steps}")

    # ── Output path ───────────────────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join(LOG_DIR, f"rl_eval_{timestamp}.csv")
    print(f"[Eval] Output:   {output_path}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    CSV_FIELDS = [
        "step", "timestamp", "coverage", "known_cells", "total_cells",
        "cov_trace", "pos_x", "pos_y", "altitude", "frontier_count",
        "nearest_frontier_dist", "loop_closures", "tracking_lost_events",
        "new_cells", "note",
    ]
    metrics: list[dict] = []

    # ── Episode-level tracking ────────────────────────────────────────────────
    total_loop_closures    = 0
    total_tracking_lost    = 0
    prev_lc_id             = None   # Will be set after first step
    prev_known_cells       = 0

    # ── Reset ─────────────────────────────────────────────────────────────────
    print("\n[Eval] Resetting environment …")
    obs = env.reset()

    # RecurrentPPO needs the LSTM hidden state carried across steps
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    print("[Eval] Starting evaluation …\n")
    print("=" * 64)
    start_time = time.time()

    step = 0
    done_flag = False

    try:
        for step in range(1, max_steps + 1):

            # ── Policy action ─────────────────────────────────────────────────
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,   # No exploration noise during eval
            )
            episode_starts = np.zeros((1,), dtype=bool)

            # ── Environment step ──────────────────────────────────────────────
            obs, reward, done, info_list = env.step(action)
            info = info_list[0]   # DummyVecEnv wraps info in a list

            # ── Extract metrics from info dict ────────────────────────────────
            # These keys are set by ActiveSLAMEnv._compute_reward() and step()
            coverage   = float(info.get("coverage", 0.0))
            cov_trace  = float(info.get("cov_trace", 0.0))
            new_cells  = int(info.get("new_cells", 0))
            pos_x      = float(info.get("pos_x", 0.0))
            pos_y      = float(info.get("pos_y", 0.0))
            altitude   = float(info.get("altitude", 0.0))
            trk_lost   = bool(info.get("tracking_lost", False))

            # known_cells and total_cells — derive from coverage if not direct
            # (env exposes them via the ROS node; we can back-calculate)
            # If the env exposes them directly in info, use those.
            known_cells = int(info.get("known_cells", 0))
            total_cells = int(info.get("total_cells", 0))
            if total_cells == 0 and coverage > 0:
                # Fallback: approximate from coverage ratio
                total_cells = int(known_cells / coverage) if coverage > 0 else 0

            # Frontier info
            frontier_count        = int(info.get("frontier_count", 0))
            nearest_frontier_dist = float(info.get("nearest_frontier_dist", -1))

            # Loop closures — increment counter when lc_id changes
            lc_id = info.get("loop_closure_id", None)
            if lc_id is not None:
                if prev_lc_id is not None and lc_id != 0 and lc_id != prev_lc_id:
                    total_loop_closures += 1
                prev_lc_id = lc_id
            # Also accept direct count if env exposes it
            if "loop_closures" in info:
                total_loop_closures = int(info["loop_closures"])

            # Tracking lost events
            if trk_lost:
                total_tracking_lost += 1

            # Build note string
            note_parts = []
            if info.get("panic_mode"):
                note_parts.append("panic")
            if trk_lost:
                note_parts.append("trk_lost")
            if info.get("r_coverage_bonus", 0) > 0:
                note_parts.append("cov_bonus!")
            note = ",".join(note_parts) if note_parts else ""

            # ── Log row ───────────────────────────────────────────────────────
            row = {
                "step":                  step,
                "timestamp":             time.time(),
                "coverage":              round(coverage, 4),
                "known_cells":           known_cells,
                "total_cells":           total_cells,
                "cov_trace":             round(cov_trace, 6),
                "pos_x":                 round(pos_x, 3),
                "pos_y":                 round(pos_y, 3),
                "altitude":              round(altitude, 3),
                "frontier_count":        frontier_count,
                "nearest_frontier_dist": round(nearest_frontier_dist, 3),
                "loop_closures":         total_loop_closures,
                "tracking_lost_events":  total_tracking_lost,
                "new_cells":             new_cells,
                "note":                  note,
            }
            metrics.append(row)

            # ── Console progress ───────────────────────────────────────────────
            if step % 50 == 0 or step == 1:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(
                    f"[Step {step:4d}/{max_steps}] "
                    f"Cov:{coverage:.1%}  "
                    f"CovTr:{cov_trace:.4f}  "
                    f"NewCells:{new_cells:3d}  "
                    f"LC:{total_loop_closures}  "
                    f"TrkLost:{total_tracking_lost}  "
                    f"Reward:{float(reward[0]):.2f}  "
                    f"Pos:({pos_x:.1f},{pos_y:.1f})  "
                    f"{elapsed}"
                )

            # ── Episode end ───────────────────────────────────────────────────
            if done[0]:
                done_flag = True
                episode_starts = np.ones((1,), dtype=bool)
                lstm_states = None   # Reset LSTM state at episode boundary
                print(f"\n[Eval] Episode ended at step {step}. "
                      f"Final coverage: {coverage:.1%}")
                # Don't break — keep going if max_steps not reached
                # (env resets itself; we continue logging the next episode)

    except KeyboardInterrupt:
        print("\n[Eval] Interrupted by user.")

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    final = metrics[-1] if metrics else {}

    print("\n" + "=" * 64)
    print("  RL EVAL RESULTS")
    print("=" * 64)
    print(f"  Steps:    {len(metrics)}")
    print(f"  Coverage: {final.get('coverage', 0):.1%}")
    print(f"  CovTrace: {final.get('cov_trace', 0):.4f}")
    print(f"  LoopClos: {final.get('loop_closures', 0)}")
    print(f"  TrkLost:  {final.get('tracking_lost_events', 0)}")
    print(f"  Time:     {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}")
    print("=" * 64)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if metrics:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(metrics)
        print(f"\n[Eval] CSV saved → {output_path}")
        print(f"[Eval] Rows: {len(metrics)}")
        print(f"\n[Eval] To plot against baselines, run:")
        print(f"         python3 plot_comparison.py")
        print(f"       (it will auto-detect the CSV in logs/)")
    else:
        print("[Eval] WARNING: No steps logged — CSV not saved.")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()