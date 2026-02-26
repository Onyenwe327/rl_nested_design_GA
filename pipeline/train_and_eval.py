"""
pipeline/train_and_eval.py
--------------------------

Entry point for a single training + evaluation run.

Workflow:
    params (dict)
      ↓
    build_xml_from_params()  → generates MuJoCo XML
      ↓
    make_env_from_xml()      → creates Gymnasium-compatible env
      ↓
    SB3 (PPO) training + evaluation
      ↓
    Returns a scalar score (higher = better)

Note:
    This version is designed for joint-position tracking tasks
    where the environment reward is the negative squared error:
        r = -||q - u||^2 - c * ||dq||^2
    The algorithm maximizes expected reward, so it automatically
    minimizes tracking error. No changes to the reward are needed.
"""

from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
import tempfile
import json
import time
import os
import traceback
from dataclasses import asdict, is_dataclass

# --- Pipeline imports ---
from pipeline.xml_builder import build_xml_from_params
from pipeline.env_factory import make_env_from_xml

# --- RL imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


# ------------------------------------------------------------------------
# Optional rendering callback (for debugging or local visualization)
# ------------------------------------------------------------------------
class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.training_env.render()
        return True
    
class TerminateOnThreshold(BaseCallback):
    """
    Stop training when:
      - total timesteps >= threshold_timestep
      - AND eval reward < reward_threshold
    """
    def __init__(self, eval_callback, threshold_timestep=500_000, reward_threshold=2000, verbose=1):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.threshold_timestep = threshold_timestep
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        # Only act after an evaluation has happened
        last_mean_reward = self.eval_callback.last_mean_reward

        # If no evaluation yet, do nothing
        if last_mean_reward is None:
            return True

        # Check conditions
        if (
            self.num_timesteps >= self.threshold_timestep
            and last_mean_reward < self.reward_threshold
        ):
            if self.verbose > 0:
                print(
                    f"⛔ Terminating: At {self.num_timesteps} steps,"
                    f" eval reward {last_mean_reward:.2f} < {self.reward_threshold}"
                )
            return False  # Stop training

        return True
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------
def _jsonable_params(params: Dict) -> Dict:
    """Convert nested dataclasses or objects to JSON-safe dicts."""
    out = {}
    for k, v in params.items():
        if is_dataclass(v):
            out[k] = asdict(v)
        elif hasattr(v, "name") and hasattr(v, "__dict__") and k == "motor":
            out[k] = {"name": getattr(v, "name", str(v))}
        else:
            out[k] = v
    return out


# def _make_run_tag(params: Dict) -> str:
#     """Generate a concise run ID based on morphology parameters."""
#     motor_name = getattr(params.get("motor", None), "name", "UnknownMotor")
#     motor_name = str(motor_name).replace("-", "").replace(" ", "")
#     dof = params.get("dof_per_leg", "x")
#     L = float(params.get("link_length", 0.0))
#     return f"dof{dof}_L{L:.3f}_M{motor_name}"
def _make_run_tag(params: Dict) -> str:
    """Generate a concise run ID based on morphology parameters."""
    motor_val = params.get("motor_type", params.get("motor", "Unknown"))
    if hasattr(motor_val, "name"):
        motor_name = motor_val.name
    else:
        motor_name = str(motor_val)
    
    motor_name = motor_name.replace("-", "").replace(" ", "")

    if "link_lengths" in params and isinstance(params["link_lengths"], list):
        L_str = "_".join([f"{x:.2f}" for x in params["link_lengths"]])
    else:
        L = float(params.get("link_length", 0.0))
        L_str = f"{L:.3f}"

    dof = params.get("dof_per_leg", "x")
    gear = params.get("gear_ratio", "")
    
    tag = f"dof{dof}_L{L_str}_M{motor_name}"
    if gear:
        tag += f"_G{gear}"
        
    return tag


def make_single_env(params, xml_path, rank, seed):
    """Factory to create independent envs for vectorized training."""
    def _init():
        xml_tmp = Path(tempfile.mkdtemp()) / f"model_{rank}.xml"
        build_xml_from_params(params, str(xml_tmp))
        env = make_env_from_xml(str(xml_tmp), seed=seed + rank, params=params)
        return env
    return _init


# ------------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------------
def train_and_eval(
    params: Dict,
    *,
    algo: str = "PPO",
    total_timesteps: int = 1_000_00,
    eval_episodes: int = 5,
    seed: Optional[int] = None,
    log_dir: str = "runs",
    log_file: str = "ga_eval.log",
    num_envs: int = 4,
    device: str = "cpu",
    return_info: bool = False, 
) -> float:
    """
    Run a full training + evaluation cycle and return a single scalar score.

    For tracking-error rewards (negative values):
        - The RL agent will try to make reward less negative (closer to zero).
        - The "score" returned to outer loops is simply the mean reward.
        - Higher (less negative) score = better tracking performance.
    """
    t0 = time.time()

    # --- Random seed setup ---
    if seed is None:
        seed = int(time.time() * 1000) % 2_147_483_647

    # --- Prepare run directories ---
    run_tag = _make_run_tag(params)
    base_log_dir = Path(log_dir)
    run_dir = base_log_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Save parameter snapshot
    with open(run_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable_params(params), f, indent=2)

    score: float = float("nan")
    status: str = "ok"
    fail_reason: str = ""

    vec_env = None
    model = None
    env = None

    with tempfile.TemporaryDirectory() as td:
        xml_path = str(Path(td) / "model.xml")

        try:
            # 1) Build MuJoCo XML from current parameters
            build_xml_from_params(params, xml_path)

            # Keep a copy of the XML in the log folder
            try:
                Path(run_dir / "model.xml").write_text(Path(xml_path).read_text())
            except Exception:
                pass

            set_random_seed(seed)

            # 2) Train RL policy (PPO by default)
            if algo.upper() != "PPO":
                raise NotImplementedError("Only PPO is implemented in this entrypoint.")

            render_cb = RenderCallback()

            if num_envs > 1:
                # Parallel training with multiple envs
                env_fns = [
                    make_single_env(params, xml_path, rank=i, seed=seed)
                    for i in range(num_envs)
                ]
                # eval_env_fns = [
                #     make_single_env(params, xml_path, rank=i, seed=seed+1000)
                #     for i in range(num_envs)
                # ]
                vec_env = SubprocVecEnv(env_fns, start_method="spawn")
                vec_env = VecMonitor(vec_env)
                env = vec_env

                # eval_vec_env = SubprocVecEnv(eval_env_fns, start_method="spawn")
                # eval_vec_env = VecMonitor(eval_vec_env)
                # eval_env = eval_vec_env
            else:
                # Single-env mode (useful for debugging)
                env = make_env_from_xml(xml_path, seed=seed)
                
            eval_env = make_env_from_xml(xml_path, seed=seed + 1000, params=params)

            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=3,
                min_evals=5,
                verbose=1
            )
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),     # saves best_model.zip here
                log_path=str(run_dir / "eval_logs"),
                eval_freq=5000,                        # tune this based on total_timesteps
                deterministic=True,
                render=False,
                callback_after_eval=stop_cb,
                verbose=1,
            )
            terminate_cb = TerminateOnThreshold(
                eval_callback=eval_cb,
                threshold_timestep=1_000_000,   # 1M steps
                reward_threshold=1500,          # your reward cutoff
                verbose=1
            )


            # Create PPO model
            model = PPO(
                "MlpPolicy",
                env,
                seed=seed,
                verbose=0,
                device=device,
                tensorboard_log=str(base_log_dir),
                learning_rate=linear_schedule(0.001),
            )

            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=False,
                callback=[eval_cb, terminate_cb] if num_envs > 1 else [render_cb, eval_cb, terminate_cb],
                tb_log_name=run_tag,
            )

            # 3) Evaluate policy on deterministic rollouts
            mean_reward, _ = evaluate_policy(
                model, env, n_eval_episodes=eval_episodes, deterministic=True
            )

            # Note: for tracking tasks, rewards are typically negative;
            # "higher" (less negative) is still better.
            print(f"Evaluation results: mean_reward={mean_reward:.3f}")
            score = float(mean_reward)

            # Save final model
            try:
                model.save(str(run_dir / "final_model.zip"))
                model.save(str(run_dir / "best_model.zip"))
            except Exception:
                pass

        except Exception as e:
            status = "fail"
            fail_reason = f"{type(e).__name__}: {e}"
            print(f"Training failed: {fail_reason}")
            score = -1e9
            fail_reason += " | " + "; ".join(traceback.format_exc().splitlines()[-3:])

        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    # --- Structured logging ---
    elapsed = time.time() - t0
    log_line = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(elapsed, 3),
        "algo": algo,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_episodes": eval_episodes,
        "params": _jsonable_params(params),
        "run_tag": run_tag,
        "status": status,
        "score": float(score),
        "fail_reason": fail_reason,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line) + "\n")

    elapsed_sec = time.time() - t0

    if return_info:
        info = {
            "artifact_dir": str(run_dir),   # <<<<<<<<<< 关键行
            "elapsed_sec": elapsed_sec,
            "status": status,
            "fail_reason": fail_reason,
        }
        return score, info
    else:
        return score
# ------------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.param_schema import make_params

    test_params = make_params()
    print("▶ Running train_and_eval() self-test with params:", test_params)

    result = train_and_eval(
        test_params,
        algo="PPO",
        total_timesteps=100_000,  # smaller for quick check
        eval_episodes=3,
        seed=0,
        log_dir="runs",
        log_file="ga_eval.log",
        num_envs=1,
    )
    print(f"✅ train_and_eval() returned score: {result:.3f}")
