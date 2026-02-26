# train_brax.py
import os
import functools
from datetime import datetime
import matplotlib.pyplot as plt
import jax
import brax
from brax.training.agents import ppo
from leg_env_mjx import SingleLegMJX  # <-- your MJX env
from brax.training.agents.ppo.train import train as ppo_train
from brax.training import checkpoint
from brax.io import html, mjcf, model
import mediapy as media
import mujoco
from mujoco import mjx
import numpy as np
# --- Setup environment ---
xml_path = os.path.join(os.path.dirname(__file__), "single_leg_robstride02.xml")
env = SingleLegMJX(xml_path, link_length_sum=0.6)

# --- Plotting setup ---
x_data, y_data, ydataerr = [], [], []
vel_rew, vel_rew_std = [], []
alive_rew, alive_rew_std = [], []
times = [datetime.now()]
max_y, min_y = 1500, 0

# Create a persistent figure with 4 subplots
plt.ion()  # interactive mode ON (non-blocking)
fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
plt.show(block=False)

def linear_lr_schedule(progress):
    """Linearly decay the learning rate from initial_lr to 0."""
    initial_lr = 1e-3
    return initial_lr * progress

# --- Define PPO train function ---
train_fn = functools.partial(
    ppo_train,
    num_timesteps=5_000_000,
    num_evals=5,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=1.0e-3,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=256,
    seed=0,
)

# --- Plotting setup ---
x_data, y_data, ydataerr, vel_rew, vel_rew_std, alive_rew, alive_rew_std, act_penalty, act_penalty_std, qvel_penalty, qvel_penalty_std = [], [], [], [], [], [], [], [], [], [], []
times = [datetime.now()]
max_y, min_y = 1500, 0

def progress(num_steps, metrics):
    # print(f"\n[Step {num_steps}] Available metric keys:")
    # for k in sorted(metrics.keys()):
    #     print("   ", k)
    times.append(datetime.now())
    x_data.append(num_steps)

    # Safely extract metrics
    y_data.append(metrics.get('eval/episode_reward', 0.0))
    ydataerr.append(metrics.get('eval/episode_reward_std', 0.0))
    vel_rew.append(metrics.get('eval/episode_vel_reward', 0.0))
    vel_rew_std.append(metrics.get('eval/episode_vel_reward_std', 0.0))
    alive_rew.append(metrics.get('eval/episode_reward_alive', 0.0))
    alive_rew_std.append(metrics.get('eval/episode_reward_alive_std', 0.0))
    act_penalty.append(metrics.get('eval/episode_action_rate_penalty', 0.0))
    act_penalty_std.append(metrics.get('eval/episode_action_rate_penalty_std', 0.0))
    qvel_penalty.append(metrics.get('eval/episode_joint_vel_penalty', 0.0))
    qvel_penalty_std.append(metrics.get('eval/episode_joint_vel_penalty_std', 0.0))

    # --- Update each subplot (reusing persistent axes) ---
    axes[0].cla()
    axes[0].errorbar(x_data, y_data, yerr=ydataerr, fmt='-o', ecolor='gray', capsize=3, label='Total Reward')
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].cla()
    axes[1].errorbar(x_data, vel_rew, yerr=vel_rew_std, fmt='-o', ecolor='lightcoral', capsize=3, color='r', label='Velocity Reward')
    axes[1].set_ylabel("Velocity Reward")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].cla()
    axes[2].errorbar(x_data, alive_rew, yerr=alive_rew_std, fmt='-o', ecolor='lightgreen', capsize=3, color='g', label='Alive Reward')
    axes[2].set_xlabel("# Environment Steps")
    axes[2].set_ylabel("Alive Reward")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].cla()
    axes[3].errorbar(x_data, act_penalty, yerr=act_penalty_std, fmt='-o', ecolor='lightblue', capsize=3, color='b', label='Action Rate Penalty')
    axes[3].set_xlabel("# Environment Steps")
    axes[3].set_ylabel("Action Rate Penalty")
    axes[3].legend()
    axes[3].grid(True)

    axes[4].cla()
    axes[4].errorbar(x_data, qvel_penalty, yerr=qvel_penalty_std, fmt='-o', ecolor='orange', capsize=3, color='orange', label='Joint Vel Penalty')
    axes[4].set_xlabel("# Environment Steps")
    axes[4].set_ylabel("Joint Vel Penalty")
    axes[4].legend()
    axes[4].grid(True)

    fig.suptitle(f"Step {num_steps} | "
                 f"Total: {y_data[-1]:.2f}, "
                 f"Vel: {vel_rew[-1]:.2f}, "
                 f"Alive: {alive_rew[-1]:.2f}, "
                 f"ActPenalty: {act_penalty[-1]:.2f}, "
                 f"QVelPenalty: {qvel_penalty[-1]:.2f}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    fig.canvas.flush_events()


# --- Run PPO training ---
print("🚀 Starting Brax PPO training...")
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
# --- Print timing info ---
print(f"\n⏱️  Time to JIT compile: {times[1] - times[0]}")
print(f"⏱️  Time to train: {times[-1] - times[1]}")

model_path = '/tmp/mjx_brax_policy'
model.save_params(model_path, params)

# Visualize Policy
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
qvel_history = []

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)
    qvel_history.append(np.array(state.pipeline_state.qvel))

    if state.done:
        break

# Save qvel history to npy file
qvel_array = np.array(qvel_history)
np.save("qvel_trajectory.npy", qvel_array)
print(f"Saved qvel trajectory to qvel_trajectory.npy with shape {qvel_array.shape}")

def render_with_camera(env, rollout, camera="side_track_cam", width=640, height=480):
    """Render MJX rollout frames using a MuJoCo camera."""
    model = env.sys.mj_model
    renderer = mujoco.Renderer(model, width, height)
    frames = []

    # Loop through MJX states
    for s in rollout:
        # Convert MJX state → MuJoCo data
        d = mujoco.MjData(model)
        d.qpos[:] = np.array(s.qpos)
        d.qvel[:] = np.array(s.qvel)
        mujoco.mj_forward(model, d)

        # Render from desired camera
        renderer.update_scene(d, camera=camera)
        frame = renderer.render()
        frames.append(frame)

    renderer.close()
    return np.array(frames)

print("Rendering rollout...")
frames = render_with_camera(env, rollout, camera="side_track_cam", width=720, height=1280)
media.write_video("evaluation.mp4", frames, fps=1.0/env.dt)
print("Video saved as evaluation.mp4")
