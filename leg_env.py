import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import os

class SingleLegEnv(gym.Env):
    # ✅ allow rgb_array so env.render() returns frames
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path, render_mode="human", link_length_sum=1.0,
                 width: int = 640, height: int = 480):
        super().__init__()

        self.xml = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml)
        self.data = mujoco.MjData(self.model)

        FOOT_RADIUS_OFFSET = 0.16
        SAFETY_MARGIN = 0.1

        self.reset_height = link_length_sum + FOOT_RADIUS_OFFSET + SAFETY_MARGIN
        self.reset_speed = 0.5  # m/s

        # render config
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None            # ✅ offscreen renderer
        self._width = int(width)
        self._height = int(height)

        # create offscreen renderer when rgb_array is requested
        if self.render_mode == "rgb_array":
            # Requires MUJOCO_GL=egl in headless
            self.renderer = mujoco.Renderer(self.model, self._width, self._height)

        act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        obs_high = np.inf * np.ones(self.get_obs().shape[0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.frame_skip = 10
        self.max_steps = 1000
        self.step_count = 0

        self.episode_x_reward = 0.0
        self.vel_reward = 0.0

        # ✅ expose dt for downstream (video timebase etc.)
        self.dt = float(self.model.opt.timestep) * self.frame_skip

    def get_obs(self):
        root_height = self.data.qpos[1]
        root_speed = self.data.qvel[:2]
        joint_qpos = self.data.qpos[2:]
        joint_qvel = self.data.qvel[2:]
        obs = np.concatenate(
            [[root_height], root_speed, joint_qpos, joint_qvel, self.prev_action]
        )
        return obs.astype(np.float32)

    def compute_reward(self, obs):
        survival_reward = 1.0
        self.vel_reward = float(self.data.qvel[0])  # forward velocity
        action_rate_pen = action_rate_penalty_norm_j_range_num_action(
            self, self.data.ctrl[:], self.prev_action
        ) * (-0.1)
        return float(survival_reward + self.vel_reward + action_rate_pen)

    def step(self, action):
        q_mid = (self.model.actuator_ctrlrange[:, 1] + self.model.actuator_ctrlrange[:, 0]) / 2.0
        q_des = q_mid + action
        self.data.ctrl[:] = q_des

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        reward = self.compute_reward(obs)
        self.prev_action = action.copy()

        done = False
        if obs[0] < 0.1:  # torso too low
            done = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        # Contact termination
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == ground_id or g2 == ground_id:
                other = g2 if g1 == ground_id else g1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name != "foot_sphere":
                    done = True
                    break

        info = {}
        if done:
            reward -= 1.0
            info["vel_reward"] = self.vel_reward

        return obs, float(reward), done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0
        self.data.qvel[0] = float(self.np_random.uniform(0.0, self.reset_speed))
        self.data.qpos[1] = self.reset_height
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs(), {}

    def render(self):
        # ✅ Return an RGB array when requested
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, self._width, self._height)
            self.renderer.update_scene(self.data)
            # returns (H,W,3) uint8
            return self.renderer.render()

        # Human-rendered interactive viewer (requires windowing)
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            time.sleep(1 / self.metadata.get("render_fps", 50))

        # If neither, return None (Gymnasium convention)
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        # ✅ clean offscreen renderer
        if self.renderer is not None:
            # mujoco.Renderer doesn't need explicit close, just drop reference
            self.renderer = None
gym.register(
    id="SingleLeg-v0",
    entry_point="leg_env:SingleLegEnv",
)


def action_rate_penalty_norm_j_range_num_action(env, action, prev_action):
    """Compute action rate penalty normalized by joint range and number of actions."""
    joint_ranges = (
        env.model.actuator_ctrlrange[:, 1] - env.model.actuator_ctrlrange[:, 0]
    )
    action_rate = np.abs(action - prev_action)
    penalty = np.sum(action_rate / joint_ranges) / env.action_space.shape[0]
    return penalty