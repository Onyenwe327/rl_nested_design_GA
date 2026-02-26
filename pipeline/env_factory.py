"""
pipeline/env_factory.py

Creates a Gymnasium-compatible environment from a MuJoCo XML path.
Priority:
  1) Use your custom SingleLegEnv from leg_env.py
  2) If SingleLegEnv doesn't accept xml_path, load model+data and set on the env
  3) Fallback: tiny placeholder env (for wiring tests only)
"""

from __future__ import annotations
from typing import Optional
import argparse
import os
import sys
import random
from pipeline.param_schema import GEAR_RATIOS, MOTOR_TYPES
from typing import Dict, Any, List

SingleLegEnv = None  # type: ignore

try:
    from leg_env import SingleLegEnv as _SingleLegEnv  # noqa: E402

    SingleLegEnv = _SingleLegEnv
except Exception:
    print("Warning: Could not import 'SingleLegEnv' from 'leg_env.py'.")  # noqa: T201
    pass


def make_env_from_xml(
    xml_path: str, seed: Optional[int] = None, params: Optional[dict] = None
):
    """
    Return a Gymnasium-compatible env loading the given XML.
    Must call env.reset(seed=seed) before returning.
    Ensures render_mode='rgb_array' so that video recording always works.
    """
    # A) Constructor supports xml_path
    link_lengths = [float(x) for x in params.get("link_lengths", [])] if params else []
    link_lengths_sum = sum(link_lengths) if link_lengths else None
    # ✅ Force rgb_array for video capture
    env = SingleLegEnv(xml_path=xml_path, render_mode="rgb_array", link_length_sum=link_lengths_sum)  # type: ignore
    env.reset(seed=seed)
    env._debug_xml_path = str(xml_path)
    return env

if __name__ == "__main__":

    def _safe_reset(env):
        res = env.reset()
        return res[0] if isinstance(res, tuple) and len(res) >= 1 else res

    # Configure these values here (no CLI parsing)
    xml_path = "./single_leg_robstride02.xml"  # <-- set your XML path
    seed = 0  # <-- set an int seed or None
    steps = 5  # <-- number of steps to run

    params: Dict[str, Any] = {
        "dof_per_leg": 3,  # fixed for now
        "link_lengths": [0.2, 0.2, 0.2],
        "motor_type": random.choice(MOTOR_TYPES),
        "gear_ratio": random.choice(GEAR_RATIOS),
    }

    if not xml_path:
        print("Usage: set xml_path at the top of this script.")
        sys.exit(2)
    if not os.path.exists(xml_path):
        print(f"XML not found: {xml_path}")
        sys.exit(2)

    try:
        env = make_env_from_xml(xml_path, seed=seed, params=params)
    except Exception as exc:
        print("Failed to create environment:", exc)
        raise

    print("Environment created:", type(env))
    try:
        frame = None
        try:
            frame = env.render()
        except Exception:
            # Some envs require an explicit call to reset before render
            try:
                _safe_reset(env)
                frame = env.render()
            except Exception as e:
                print("Rendering not available:", e)
                frame = None

        if frame is None:
            print("No frame returned by render().")
        else:
            shape = getattr(frame, "shape", None)
            dtype = getattr(frame, "dtype", None)
            print(f"Rendered frame shape={shape}, dtype={dtype}")
    except Exception as e:
        print("Render check failed:", e)

    obs = None
    try:
        obs = _safe_reset(env)
    except Exception:
        pass

    for i in range(steps):
        try:
            action = env.action_space.sample()
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                terminated = bool(done)
                truncated = False
            else:
                obs, reward, terminated, truncated, info = result
            print(
                f"step {i}: reward={reward}, terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                obs = _safe_reset(env)
        except Exception as e:
            print("Step failed:", e)
            break

    try:
        env.close()
    except Exception:
        pass
