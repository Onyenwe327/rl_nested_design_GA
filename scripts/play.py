import gymnasium as gym
import leg_env
from stable_baselines3 import PPO
import glob
import os
import numpy as np

baseline_xml_path = os.path.join(
    os.path.dirname(__file__), "single_leg_robstride02.xml"
)
link_length_sum = 0.6


def get_latest_checkpoint(
    checkpoint_dir="../logs/20251019_115257", prefix="ppo_single_leg"
):
    """Return path to the latest checkpoint file."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip"))
    if not checkpoints:
        return None
    # sort by step count extracted from filename
    checkpoints.sort(
        key=lambda f: int(f.split("_")[-2])
    )  # e.g., .../ppo_quad_50000_steps.zip
    return checkpoints[-1]


def main():
    env = gym.make(
        "SingleLeg-v0",
        render_mode="human",
        xml_path=baseline_xml_path,
        link_length_sum=link_length_sum,
    )
    xvels = []
    torques = []
    qvels = []
    data_path = os.path.join(os.path.dirname(__file__), "data.npz")

    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt is None:
        print("No checkpoint found in ../logs/checkpoints")
    else:
        print(f"Loading checkpoint: {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=env)

    obs, _ = env.reset()
    for _ in range(1000):
        if latest_ckpt is None:
            action = env.action_space.sample()  # random action
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        xvels.append(obs[1])
        torques.append(env.unwrapped.data.actuator_force.copy())
        qvels.append(env.unwrapped.data.qvel.copy())
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

    try:
        np.savez_compressed(
            data_path,
            xvels=np.array(xvels),
            torques=np.array(torques),
            qvels=np.array(qvels),
        )
        print(f"✅ Saved data to {data_path}")
        print(f"   xvels shape  = {np.array(xvels).shape}")
        print(f"   torques shape = {np.array(torques).shape}")
        print(f"   qvels shape   = {np.array(qvels).shape}")
    except Exception as e:
        print(f"❌ Failed to save data: {e}")


if __name__ == "__main__":
    main()
