# pipeline/post_training.py
"""
Post-training automation:
- Select Top-2 from HOF + 3 random others
- Evaluate rollouts to collect x-velocity & actuator-torque time-series
- Plot comparison figures
- Record videos for ALL 5 models (Top-2 + Random-3)
- Try to overlay live x-velocity on frames (best-effort; auto-disable if PIL missing)
- Save per-model "design combo" (genome + resolved motor details)
"""

from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

# Try to import Pillow to draw velocity text on frames; if not available, skip overlay gracefully.
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# Video IO
import imageio

# SB3 imports (loader kept for compatibility; direct policy use is optional)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm

try:
    from pipeline.motor_db import MOTOR_DB
except Exception:
    MOTOR_DB = {}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModelArtifact:
    """Pointers to a trained model and metadata for a single individual."""
    name: str
    genome: Sequence
    artifact_dir: Path
    model_path: Path
    extra: Dict = None


@dataclass
class RolloutSeries:
    """Time-series captured from a single evaluation rollout."""
    t: np.ndarray
    x: np.ndarray
    xvel: np.ndarray
    torque: np.ndarray
    fps: float


# ---------------------------------------------------------------------------
# Helpers: environment + SB3 model loading
# ---------------------------------------------------------------------------

def _default_model_loader(model_path: Path):
    """Default SB3 model loader; look for best_model.zip or final_model.zip."""
    p = Path(model_path)
    if p.is_dir():
        for fname in ["best_model.zip", "final_model.zip"]:
            f = p / fname
            if f.exists():
                return PPO.load(str(f))
        raise FileNotFoundError(f"No model zip under {p}")
    elif p.suffix == ".zip":
        return PPO.load(str(p))
    else:
        raise ValueError(f"Invalid model_path: {model_path}")


def _make_vec_env(env_maker: Callable):
    """Return a single-env VecEnv for SB3 inference (headless-safe)."""
    # IMPORTANT: env_maker should create env with render_mode="rgb_array"
    return DummyVecEnv([env_maker])


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
def _try_render_frame(env) -> Optional[np.ndarray]:
    """
    Try hard to fetch an RGB frame (H,W,3, uint8) from a wide range of Gym/MuJoCo envs.
    Returns None if all attempts fail.
    """
    frame = None

    # 1) Plain env.render() (Gymnasium requires render_mode="rgb_array" at construction)
    try:
        f = env.render()
        if isinstance(f, (np.ndarray,)):
            frame = f
    except Exception:
        pass

    # 2) Old-style mode argument
    if frame is None:
        try:
            f = env.render(mode="rgb_array")
            if isinstance(f, (np.ndarray,)):
                frame = f
        except Exception:
            pass

    # 3) MuJoCo renderer helper (Gymnasium MujocoEnv exposes .mujoco_renderer)
    if frame is None:
        try:
            mr = getattr(env, "mujoco_renderer", None) or getattr(getattr(env, "unwrapped", env), "mujoco_renderer", None)
            if mr is not None and hasattr(mr, "render"):
                f = mr.render()
                if isinstance(f, (np.ndarray,)):
                    frame = f
        except Exception:
            pass

    # 4) Direct sim.render fallback (rarely needed; try best-effort)
    if frame is None:
        try:
            sim = getattr(env, "sim", None) or getattr(getattr(env, "unwrapped", env), "sim", None)
            if sim is not None and hasattr(sim, "render"):
                # (width, height, camera_id, depth): pick a safe default size
                f = sim.render(640, 480, 0, False)
                if isinstance(f, (np.ndarray,)):
                    frame = f
        except Exception:
            pass

    # Normalize to (H, W, 3) uint8
    if isinstance(frame, np.ndarray):
        # Some renderers return float in [0,1]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        # Drop alpha if present
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        # Ensure 3 channels
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return frame

    return None
def _synth_frame(width: int, height: int, text: str = "") -> np.ndarray:
    """
    Generate a simple synthetic RGB frame (H,W,3) uint8.
    Used as a fallback when env.render() returns None.
    """
    import numpy as np
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # simple gray grid background
    img[:, ::8, :] = 30
    img[::8, :, :] = 30

    # optional overlay text
    if text:
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(img)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            draw.rectangle([10, 10, 10 + len(text) * 6, 28], fill=(0, 0, 0))
            draw.text((12, 12), text, font=font, fill=(255, 255, 255))
            img = np.asarray(im)
        except Exception:
            pass  # if PIL unavailable, skip text
    return img

def _draw_velocity_overlay(frame: np.ndarray, vx: float) -> np.ndarray:
    """Overlay current x-velocity on the frame (top-left). If PIL missing, return frame unchanged."""
    if not _PIL_OK:
        return frame
    try:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text = f"vx = {vx:.3f} m/s"
        # Draw a small black rectangle behind to improve readability
        bbox = draw.textbbox((6, 6), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((6, 6), text, fill=(255, 255, 255), font=font)
        return np.asarray(img)
    except Exception:
        # If anything goes wrong, silently skip overlay
        return frame


def _open_video_writer(path: Path, fps: int):
    """Robust writer: try MP4 via ffmpeg, fallback to bundled ffmpeg, then GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try MP4 (requires ffmpeg)
    try:
        return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    except Exception:
        pass

    # Try locating a bundled ffmpeg
    try:
        import imageio_ffmpeg  # noqa: F401
        os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
        return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    except Exception:
        pass

    # Fallback to GIF
    try:
        gif_path = path.with_suffix(".gif")
        print(f"[POST][WARN] MP4 backend unavailable; falling back to GIF: {gif_path.name}")
        return imageio.get_writer(str(gif_path), mode="I", duration=1.0 / max(fps, 1))
    except Exception as e:
        raise RuntimeError(
            "No video backend available. Run `pip install -U imageio imageio-ffmpeg`."
        ) from e

# ---- Tracking camera for MuJoCo --------------------------------------
def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = np.clip(frame * (255.0 if frame.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame

class _TrackingCam:
    """A tiny wrapper around mujoco.Renderer + mjvCamera in TRACKING mode."""
    def __init__(self, model, track_body_id: int,
                 width: int = 1280, height: int = 720,
                 distance: float = 2.0, azimuth: float = 90.0, elevation: float = -20.0):
        import mujoco
        self.mj = mujoco
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = int(track_body_id)
        self.cam.distance = float(distance)
        self.cam.azimuth = float(azimuth)
        self.cam.elevation = float(elevation)

    def render(self, model, data) -> np.ndarray:
        # Important: always update scene with our tracking camera
        self.renderer.update_scene(data, camera=self.cam)
        frame = self.renderer.render()
        return _to_uint8(frame)

# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------

def _find_ref_handles(model):
    """Try to find a 'base' or first free-joint body; fall back to body 1."""
    import mujoco
    # 1) look for a body named like base/root/torso
    candidates = ["base", "root", "torso", "pelvis"]
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            return dict(body_id=bid, site_id=-1)
    # 2) first body after world
    return dict(body_id=1 if model.nbody > 1 else 0, site_id=-1)


def _get_body_xpos(data, body_id) -> float:
    """World-frame x position of a body's frame origin."""
    # data.xpos is (nbody, 3)
    return float(data.xpos[body_id][0])


def _get_com_xvel(model, data) -> float:
    """Approximate COM x-velocity; if a free joint exists at root, use qvel[0]."""
    try:
        if model.njnt > 0 and model.nv >= 6:
            # Joint type 0 == free joint in MuJoCo; typically the first joint for floating base
            if hasattr(model, "jnt_type") and int(model.jnt_type[0]) == 0:
                return float(data.qvel[0])
    except Exception:
        pass
    return 0.0


def _aggregate_torque(model, data) -> float:
    """Aggregate actuator effort; prefer actuator_force, fallback to ctrl."""
    try:
        if hasattr(data, "actuator_force") and data.actuator_force is not None:
            return float(np.sum(np.abs(data.actuator_force)))
        if hasattr(data, "qfrc_actuator") and data.qfrc_actuator is not None:
            return float(np.sum(np.abs(data.qfrc_actuator)))
        if hasattr(data, "ctrl") and data.ctrl is not None:
            return float(np.sum(np.abs(data.ctrl)))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Series normalization / validation
# ---------------------------------------------------------------------------

def _normalize_series(s: Dict | SimpleNamespace,
                      default_fps: float,
                      dt_hint: Optional[float] = None) -> RolloutSeries:
    """
    Normalize raw series dict/namespace into a RolloutSeries with keys:
    - t (seconds), x (meters), xvel (m/s), torque (scalar per step), fps (float)
    If x is missing but xvel is available, integrate xvel over time to reconstruct x.
    """
    # Access as namespace
    if isinstance(s, dict):
        s = SimpleNamespace(**s)

    # fps
    fps = getattr(s, "fps", None)
    if fps is None or fps <= 0:
        fps = float(default_fps)

    # dt priority: explicit dt_hint > derived from t > 1/fps
    t = getattr(s, "t", None)
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("t must be 1D")
        # dt from t if uniformly spaced; else fallback to 1/fps
        if len(t) >= 2:
            dt = float(np.mean(np.diff(t)))
        else:
            dt = (dt_hint if dt_hint is not None else 1.0 / fps)
    else:
        dt = (dt_hint if dt_hint is not None else 1.0 / fps)

    # xvel
    xvel = getattr(s, "xvel", None)
    if xvel is None:
        # Accept common aliases: vx, vel, x_velocity
        for alias in ("vx", "vel", "x_velocity"):
            xvel = getattr(s, alias, None)
            if xvel is not None:
                break
    if xvel is None:
        raise ValueError("series missing xvel/vx")
    xvel = np.asarray(xvel, dtype=float)

    # torque (allow missing -> zeros)
    torque = getattr(s, "torque", None)
    if torque is None:
        torque = np.zeros_like(xvel, dtype=float)
    else:
        torque = np.asarray(torque, dtype=float)
        if torque.ndim > 1:
            # Reduce vector torque per step to scalar aggregate (L1 norm)
            torque = np.sum(np.abs(torque), axis=-1)

    # Build/validate t
    T = len(xvel)
    if t is None:
        t = np.arange(T, dtype=float) * dt
    if len(t) != T:
        raise ValueError(f"length mismatch: len(t)={len(t)} vs len(xvel)={T}")

    # x: prefer provided; else integrate xvel
    x = getattr(s, "x", None)
    if x is None:
        # Simple cumulative integration: x[k] = sum_{i<k} xvel[i] * dt
        x = np.cumsum(xvel) * dt
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != T:
            raise ValueError(f"length mismatch: len(x)={len(x)} vs len(xvel)={T}")

    return RolloutSeries(t=t, x=x, xvel=xvel, torque=torque, fps=fps)


# ---------------------------------------------------------------------------
# Rollout + (optional) video recording with overlay
# ---------------------------------------------------------------------------

def _rollout_collect_series(model: Optional[BaseAlgorithm],
                            env_maker: Callable[[], object],
                            max_steps: int,
                            fps: int,
                            video_out_path: Path,
                            overlay_velocity: bool = True) -> Dict:
    """
    Run an evaluation rollout and capture: time, xvel, (optional) x, torque, fps.
    Always tries to produce a video:
      - First frame is forced (synthetic fallback if real render is unavailable)
      - Each step writes a frame (real or synthetic), so the writer always flushes.
    Also prints debug info about whether a policy is used and frames written.
    """
    env = env_maker()
    try:
        xml_dbg = getattr(env, "_debug_xml_path", None)
        if xml_dbg is not None:
            print(f"[POST][DBG] Using XML: {xml_dbg}")
        # Access MuJoCo model/data if available
        mj_model = getattr(env, "model", None)
        mj_data = getattr(env, "data", None)
        if mj_model is None or mj_data is None:
            u = getattr(env, "unwrapped", env)
            mj_model = getattr(u, "model", None)
            mj_data = getattr(u, "data", None)

        handles = None
        if mj_model is not None:
            handles = _find_ref_handles(mj_model)
        tracking_cam = None
        if mj_model is not None and handles is not None:
            try:
                tracking_cam = _TrackingCam(
                    mj_model,
                    track_body_id=handles["body_id"],
                    width=1280, height=720,
                    distance=2.2, azimuth=90.0, elevation=-18.0
                )
            except Exception as e:
                print(f"[POST][WARN] tracking camera disabled: {e}")
                tracking_cam = None
        # Video writer
        writer = _open_video_writer(Path(video_out_path), fps=fps)

        # Reset (Gymnasium API: obs, info)
        obs, info = env.reset()

        # Timebase
        dt = getattr(env, "dt", None)
        if dt is None or dt <= 0:
            dt = 1.0 / float(fps)
        t_series: List[float] = []
        vx_series: List[float] = []
        tq_series: List[float] = []
        x_series: List[float] = []

        # Force-write a first frame (real render if possible, otherwise synthetic)
        title = f"{Path(video_out_path).stem}"
        frame = None
        if tracking_cam is not None:
            try:
                frame = tracking_cam.render(mj_model, mj_data)
            except Exception:
                frame = None
        if frame is None:
            frame = _try_render_frame(env)
        if frame is None:
            frame = _synth_frame(640, 480, text=f"{title} (synthetic)")
        if overlay_velocity and frame is not None:
            frame = _draw_velocity_overlay(frame, 0.0)
        writer.append_data(frame)
        frames_written = 1
        # State for finite-difference velocity and integrated position
        prev_x_for_v = None
        running_x = 0.0
        t = 0.0

        # If a policy is provided, use it; otherwise zero action
        use_policy = isinstance(model, BaseAlgorithm)
        print(f"[POST][DBG] use_policy={use_policy}, model_cls={getattr(model, '__class__', type(model)).__name__}")
        printed_first_action = False

        for _ in range(max_steps):
            if use_policy:
                action, _ = model.predict(obs, deterministic=True)
                if not printed_first_action and isinstance(action, np.ndarray):
                    print(f"[POST][DBG] first action shape={action.shape}, dtype={action.dtype}")
                    printed_first_action = True
            else:
                try:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                except Exception:
                    action = None

            obs, reward, terminated, truncated, info = env.step(action)

            # Compute x-velocity
            vx = 0.0
            x_probe = None
            if mj_model is not None and mj_data is not None and handles is not None:
                try:
                    x_probe = _get_body_xpos(mj_data, handles["body_id"])
                    if prev_x_for_v is not None:
                        vx = (x_probe - prev_x_for_v) / dt
                    else:
                        # First step: try COM velocity fallback
                        vx = _get_com_xvel(mj_model, mj_data)
                    prev_x_for_v = x_probe
                except Exception:
                    vx = _get_com_xvel(mj_model, mj_data)
            else:
                # Fallback from info dict (if env provides one)
                vx = float(info.get("vx", 0.0))

            # Aggregate torque
            tq = _aggregate_torque(mj_model, mj_data) if (mj_model is not None and mj_data is not None) else float(info.get("torque", 0.0))

            # Integrate position from velocity for a consistent x series
            running_x += vx * dt

            # Append series
            t_series.append(t)
            vx_series.append(vx)
            tq_series.append(tq)
            x_series.append(running_x)

            # Render frame (real or synthetic), then (optionally) overlay velocity and write
            frame = None
            if tracking_cam is not None:
                try:
                    frame = tracking_cam.render(mj_model, mj_data)
                except Exception:
                    frame = None
            if frame is None:
                frame = _try_render_frame(env)
            if frame is None:
                frame = _synth_frame(640, 480, text=f"vx={vx:.3f} m/s (synthetic)")
            if overlay_velocity and frame is not None:
                frame = _draw_velocity_overlay(frame, vx)
            writer.append_data(frame)
            frames_written += 1

            t += dt
            if terminated or truncated:
                break

        writer.close()
        print(f"[POST] {Path(video_out_path).name}: frames_written={frames_written}")

        if len(t_series) == 0:
            print(f"[POST][WARN] No steps collected for {video_out_path}. Check env.step/termination.")
        if frames_written == 0:
            print(f"[POST][WARN] No frames captured. Ensure render_mode='rgb_array' or EGL is configured.")

        # Return a raw dict; will be normalized upstream
        return dict(
            t=np.asarray(t_series, dtype=float),
            x=np.asarray(x_series, dtype=float),
            xvel=np.asarray(vx_series, dtype=float),
            torque=np.asarray(tq_series, dtype=float),
            fps=float(fps),
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth(y: np.ndarray, k: int = 100) -> np.ndarray:
    """Simple moving average to smooth curves."""
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def _plot_velocity(series_map: Dict[str, RolloutSeries], out_path: Path) -> None:
    plt.figure(figsize=(8.5, 4.8))
    for name, s in series_map.items():
        smoothed = _smooth(s.xvel, k=120)
        # distinguish Top-2 vs Random-3
        linestyle = "--" if name.startswith("RND") else "-"
        plt.plot(
            s.t, smoothed,
            label=name,
            linewidth=1.6 if linestyle == "-" else 1.2,
            alpha=0.9,
            linestyle=linestyle,
        )

    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("X Velocity (m/s)", fontsize=11)
    plt.title("Velocity Comparison (Top-2 vs Random-3)", fontsize=12, pad=10)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_torque(series_map: Dict[str, RolloutSeries], out_path: Path) -> None:
    plt.figure(figsize=(8.5, 4.8))
    for name, s in series_map.items():
        smoothed = _smooth(s.torque, k=120)
        linestyle = "--" if name.startswith("RND") else "-"
        plt.plot(
            s.t, smoothed,
            label=name,
            linewidth=1.6 if linestyle == "-" else 1.2,
            alpha=0.9,
            linestyle=linestyle,
        )

    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Actuator Torque (N·m, aggregate)", fontsize=11)
    plt.title("Torque Comparison (Top-2 vs Random-3)", fontsize=12, pad=10)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Design combo saver
# ---------------------------------------------------------------------------

def _save_design_combo(artifact: ModelArtifact, out_path: Path) -> None:
    """Write a compact JSON for the model's design combo (genome + motor info)."""
    genome = list(artifact.genome)
    combo = {"name": artifact.name, "genome": genome}

    motor_info = None
    if len(genome) >= 3:
        motor_id = genome[2]
        if motor_id in MOTOR_DB:
            motor_info = MOTOR_DB[motor_id]
    if motor_info is not None:
        combo["motor"] = dict(motor_info)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combo, f, indent=2)


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _select_top2_and_random3(hof: Sequence[Sequence],
                             evaluated_pool: List[Tuple[Sequence, Path]],
                             rng: random.Random) -> Tuple[List[Tuple[Sequence, Path]],
                                                          List[Tuple[Sequence, Path]]]:
    """Return (top2, random3) where each item is (genome, artifact_dir)."""
    hof_list = list(hof)[:2]
    top2: List[Tuple[Sequence, Path]] = []

    def _match(g):
        for (gn, art) in evaluated_pool:
            if tuple(gn) == tuple(g):
                return (gn, art)
        return None

    for g in hof_list:
        matched = _match(g)
        if matched is None:
            raise RuntimeError("HOF genome not found in evaluated_pool.")
        top2.append(matched)

    excluded = set([tuple(g) for (g, _) in top2])
    candidates = [(g, d) for (g, d) in evaluated_pool if tuple(g) not in excluded]
    k = min(3, len(candidates))
    random3 = rng.sample(candidates, k=k)
    return top2, random3


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _find_model_path(artifact_dir: Path) -> Path:
    """Search for a model zip under artifact_dir."""
    for fname in ("best_model.zip", "final_model.zip"):
        p = artifact_dir / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"No model zip under {artifact_dir}")

from functools import partial

def run_post_training_bundle(
    hof,
    evaluated_pool,
    env_maker_for: Callable[[Sequence, Path], object],  # Build env per (genome, artifact_dir)
    out_dir: Path,
    *,
    rng_seed: int = 2025,
    max_eval_steps: int = 1500,
    eval_fps: int = 30,
    model_loader=None,
    name_fn=None,
    overlay_velocity_on_video: bool = True,
):
    """Automate plots + videos after training finishes.

    NEW:
    - Record videos for ALL 5 models (Top-2 + Random-3)
    - Try to overlay live x-velocity on the frames (optional; defaults True)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)
    loader = model_loader or _default_model_loader

    # 1) Select Top-2 and Random-3
    top2, random3 = _select_top2_and_random3(hof, evaluated_pool, rng)
    picked = top2 + random3

    # 2) Build ModelArtifact objects
    artifacts: List[ModelArtifact] = []
    for idx, (genome, artifact_dir) in enumerate(picked):
        artifact_dir = Path(artifact_dir)
        # ✅ 改：支持 best_model.zip / final_model.zip
        model_path = _find_model_path(artifact_dir)

        if name_fn is not None:
            name = name_fn(genome)
        else:
            name = f"HOF#{idx+1}" if idx < 2 else f"RND#{idx-1}"

        artifacts.append(
            ModelArtifact(
                name=name,
                genome=genome,
                artifact_dir=artifact_dir,
                model_path=model_path,
                extra={},
            )
        )

    # 3) Evaluate all 5 models; record videos for all
    series_map: Dict[str, RolloutSeries] = {}
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    for art in artifacts:
        model = loader(art.model_path)

        # Save design combo for traceability
        _save_design_combo(art, art.artifact_dir / f"{art.name}_design_combo.json")

        # Every model gets its own video
        video_path = videos_dir / f"{art.name}_eval.mp4"

        # ✅ 改：用 partial，更直观
        env_maker = partial(env_maker_for, art.genome, art.artifact_dir)

        raw = _rollout_collect_series(
            model=model,
            env_maker=env_maker,
            max_steps=max_eval_steps,
            fps=eval_fps,
            video_out_path=video_path,
            overlay_velocity=overlay_velocity_on_video,
        )

        # Normalize -> RolloutSeries
        series = _normalize_series(raw, default_fps=eval_fps)
        series_map[art.name] = series

        # Also persist the raw series (compressed NPZ)
        npz_path = art.artifact_dir / f"{art.name}_eval_timeseries.npz"
        np.savez_compressed(
            npz_path,
            t=series.t,
            x=series.x,
            xvel=series.xvel,
            torque=series.torque,
            fps=series.fps,
        )

    # 4) Make the comparison plots
    vel_plot = out_dir / "velocity_comparison.png"
    tq_plot = out_dir / "torque_comparison.png"
    _plot_velocity(series_map, vel_plot)
    _plot_torque(series_map, tq_plot)

    return {
        "velocity_plot": vel_plot,
        "torque_plot": tq_plot,
        "videos_dir": videos_dir,
    }
