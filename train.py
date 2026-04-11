#!/usr/bin/env python3
"""
Training script for Active SLAM RL agent.

Architecture:
  RecurrentPPO (sb3-contrib) with MultiInputLstmPolicy
  - Adds LSTM/GRU memory automatically after the feature extractor
  - Handles Dict observation space with recurrent state
  - SLAMFeatureExtractor (CNN + scalar branches) feeds into LSTM

Key changes from old train.py (mapped to audit bugs):
  - RecurrentPPO instead of vanilla PPO (Bug 8: no temporal memory)
  - features_dim=512 (Bug 11: was 128 — extreme bottleneck)
  - net_arch=[256, 256] (Bug 11: was [64, 64] — too small)
  - n_steps=2048, batch_size=256, n_epochs=10 (Bug 12: all were too small)
  - ent_coef=0.05 (Bug 12: was 0.01 — premature convergence)
  - total_timesteps=1_000_000 (Bug: was 10_000 — smoke test only)
  - VecNormalize for reward scaling (audit: unbounded reward variance)
  - EvalCallback for convergence measurement (audit: no eval)
  - device='cuda' explicit (audit: silent CPU fallback)
  - Episode-level logging, not per-step (Bug 22, 23)
  - Relative paths (Bug: hardcoded /root/)
  - seed=42 for reproducibility

System: Ubuntu 25.04 | Xeon w7-3465X (56 threads) | RTX A4000 16GB | 128GB RAM
"""

import os
import sys
import time
import numpy as np

# ============================================
# Path setup (relative, not hardcoded)
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
TB_DIR = os.path.join(LOG_DIR, "tb")
BEST_MODEL_DIR = os.path.join(LOG_DIR, "best_model")

# Create directories
for d in [LOG_DIR, CHECKPOINT_DIR, TB_DIR, BEST_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================
# Imports (after path setup)
# ============================================
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback, EvalCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.active_slam_env import ActiveSLAMEnv
from agents.feature_extractor import SLAMFeatureExtractor


# ============================================
# Training Hyperparameters
# ============================================
TOTAL_TIMESTEPS = 1_000_000   # Minimum for meaningful learning
N_STEPS = 2048                # Rollout buffer size (large for stable updates)
BATCH_SIZE = 256              # Mini-batch size for GPU utilization
N_EPOCHS = 10                 # PPO update epochs per rollout
LEARNING_RATE = 3e-4          # Standard for PPO
ENT_COEF = 0.05              # Higher than default for exploration
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE parameter
CLIP_RANGE = 0.2              # PPO clipping range
FEATURES_DIM = 512            # Feature extractor output dimension
SEED = 42                     # Reproducibility
EVAL_FREQ = 10_000            # Evaluate every N training steps
EVAL_EPISODES = 3             # Episodes per evaluation
CHECKPOINT_FREQ = 25_000      # Save checkpoint every N steps
VEC_NORM_CLIP_REWARD = 10.0   # VecNormalize reward clip


# ============================================
# Episode-Level Logging Callback
# ============================================
class ExplorationLogCallback(BaseCallback):
    """Logs exploration metrics at episode boundaries (not per-step).

    Fixes from audit:
    - episode_rewards is actually populated (Bug 22)
    - Coverage logged only at episode end (Bug 23)
    - Tracks all key metrics for Tensorboard
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.episode_lengths = []
        self.episode_cov_traces = []
        self.episode_count = 0
        self.training_start_time = None

    def _on_training_start(self):
        self.training_start_time = time.time()

    def _on_step(self):
        # Check for episode completion via SB3's built-in episode tracking
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self.episode_count += 1

                # Extract episode metrics
                coverage = info.get("coverage", 0.0)
                cov_trace = info.get("cov_trace", 0.0)
                step_count = info.get("step", 0)

                # Get cumulative episode reward from SB3's monitor wrapper
                # "episode" key is added by SB3 when an episode ends
                ep_info = info.get("episode")
                if ep_info is not None:
                    ep_reward = ep_info.get("r", 0.0)
                    ep_length = ep_info.get("l", step_count)
                else:
                    ep_reward = 0.0
                    ep_length = step_count

                self.episode_coverages.append(coverage)
                self.episode_cov_traces.append(cov_trace)
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Log to Tensorboard
                if self.logger is not None:
                    self.logger.record("episode/coverage", coverage)
                    self.logger.record("episode/cov_trace", cov_trace)
                    self.logger.record("episode/reward", ep_reward)
                    self.logger.record("episode/length", ep_length)
                    self.logger.record("episode/count", self.episode_count)

                    # Running averages (last 10 episodes)
                    if len(self.episode_coverages) >= 2:
                        recent = min(10, len(self.episode_coverages))
                        self.logger.record(
                            "episode/avg_coverage_10",
                            np.mean(self.episode_coverages[-recent:])
                        )
                        self.logger.record(
                            "episode/avg_reward_10",
                            np.mean(self.episode_rewards[-recent:])
                        )

                # Console output
                elapsed = time.time() - self.training_start_time if self.training_start_time else 0
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

                print(f"\n[Episode {self.episode_count}] "
                      f"Steps: {ep_length} | "
                      f"Coverage: {coverage:.1%} | "
                      f"Reward: {ep_reward:.2f} | "
                      f"CovTrace: {cov_trace:.4f} | "
                      f"Timestep: {self.num_timesteps} | "
                      f"Time: {elapsed_str}")

                if len(self.episode_coverages) >= 10:
                    avg_cov = np.mean(self.episode_coverages[-10:])
                    avg_rew = np.mean(self.episode_rewards[-10:])
                    print(f"  [Avg last 10] Coverage: {avg_cov:.1%} | Reward: {avg_rew:.2f}")

        return True

# ============================================
# Main Training Function
# ============================================
def main():
    print("=" * 64)
    print("  Active SLAM — RL Training")
    print("  RecurrentPPO + SLAMFeatureExtractor")
    print("=" * 64)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Tensorboard:  tensorboard --logdir {TB_DIR}")
    print(f"  Checkpoints:  {CHECKPOINT_DIR}")
    print(f"  Total steps:  {TOTAL_TIMESTEPS:,}")
    print("=" * 64)

    # ----------------------------------------------------------
    # Verify CUDA
    # ----------------------------------------------------------
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nWARNING: CUDA not available. Training will be VERY slow on CPU.")
            print("  Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            device = "cuda"
    except ImportError:
        print("\nWARNING: PyTorch not found. Install it first.")
        device = "auto"

    # ----------------------------------------------------------
    # Create training environment wrapped in VecNormalize
    # ----------------------------------------------------------
    print("\n[Train] Creating training environment...")
    train_env = DummyVecEnv([lambda: ActiveSLAMEnv()])
    train_env = VecNormalize(
        train_env,
        norm_obs=False,         # Don't normalize observations (already [0,1])
        norm_reward=True,       # Normalize rewards (critical for stable training)
        clip_reward=VEC_NORM_CLIP_REWARD,
        gamma=GAMMA,
    )

    # ----------------------------------------------------------
    # Policy configuration
    # ----------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=SLAMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        # LSTM config (RecurrentPPO adds this after the feature extractor)
        lstm_hidden_size=128,
        n_lstm_layers=1,
    )

    # ----------------------------------------------------------
    # Create RecurrentPPO agent
    # ----------------------------------------------------------
    print("[Train] Creating RecurrentPPO agent...")
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=SEED,
        tensorboard_log=TB_DIR,
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------
    # 1. Checkpoint saving
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="active_slam",
        verbose=1,
    )

    # 2. Episode-level logging
    log_cb = ExplorationLogCallback()

    # 3. Evaluation (deterministic policy, separate from training noise)
    # Note: We reuse the same env for eval since we only have one sim.
    # For true separation, you'd need a second Gazebo instance.
    eval_cb = EvalCallback(
        train_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        best_model_save_path=BEST_MODEL_DIR,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, log_cb, eval_cb])

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    print(f"\n[Train] Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"[Train] Estimated time: ~{TOTAL_TIMESTEPS * 1.0 / 3600:.0f} hours "
          f"(at ~1s/step)")
    print(f"[Train] Monitor with: tensorboard --logdir {TB_DIR}\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[Train] Training interrupted by user.")
    except Exception as e:
        print(f"\n[Train] Training error: {e}")
        import traceback
        traceback.print_exc()

    # ----------------------------------------------------------
    # Save final model + VecNormalize stats
    # ----------------------------------------------------------
    final_model_path = os.path.join(LOG_DIR, "final_model")
    model.save(final_model_path)
    print(f"[Train] Model saved: {final_model_path}")

    # Save VecNormalize stats (needed to load the model later)
    vec_norm_path = os.path.join(LOG_DIR, "vec_normalize.pkl")
    train_env.save(vec_norm_path)
    print(f"[Train] VecNormalize stats saved: {vec_norm_path}")

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    train_env.env_method("close")
    print("[Train] Training complete.")


if __name__ == "__main__":
    main()