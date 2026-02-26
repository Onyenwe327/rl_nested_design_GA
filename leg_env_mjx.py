# single_leg_mjx_env.py
import os
import jax
import jax.numpy as jnp
import mujoco
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

class SingleLegMJX(PipelineEnv):
    def __init__(self, xml_path, link_length_sum=0.66, **kwargs):
        # Load MuJoCo XML
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 10
        kwargs['backend'] = 'mjx'
        kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)

        super().__init__(sys, **kwargs)

        # your constants
        self.reset_height = link_length_sum + 0.16 + 0.1
        self.reset_speed = 0.5
        self._minimum_z_height = 0.1
        self._forward_reward_weight = 1.0
        self.healthy_reward = 1.0
        self._terminate_when_unhealthy = True
        self.action_scale = 1.0
        self._ground_geom_id = 0
        self._foot_geom_id = 5
        self._action_rate_penalty_weight = -0.01


    def reset(self, rng):
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos0 = self.sys.qpos0
        # set the root height
        qpos = qpos0.at[1].set(self.reset_height)

        # sample initial qvel: do NOT use sys.qvel0 (not guaranteed to exist)
        low, hi = -1e-3, 1e-3
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        # forward velocity bias if you want your “reset_speed” behavior
        # Set first reset qvel as 1.0
        # qvel = qvel.at[0].set(self.reset_speed)
        qvel = qvel.at[0].set(jax.random.uniform(rng1, (), minval=0.0, maxval=self.reset_speed))

        # create pipeline state
        data = self.pipeline_init(qpos, qvel)

        # build observation (action is zeros at reset)
        zero_action = jnp.zeros(self.sys.nu)
        obs = self._get_obs(data, zero_action)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'vel_reward': zero,
            'reward_alive': zero,
            'action_rate_penalty': zero,
        }

        return State(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info={'last_action': jnp.zeros(self.sys.nu)})

    def step(self, state: State, action: jnp.ndarray) -> State:
        data0 = state.pipeline_state
        ctrl_range = jnp.array(self.sys.mj_model.actuator_ctrlrange)
        ctrl_min, ctrl_max = ctrl_range[:, 0], ctrl_range[:, 1]
        scaled_action = ctrl_min + (0.5 * (action + 1.0) * (ctrl_max - ctrl_min))
        scaled_action = jnp.clip(scaled_action, ctrl_min, ctrl_max)

        data  = self.pipeline_step(data0, scaled_action)
        x, xd = data.x, data.xd

        velocity = data.qvel[0]  # forward velocity
        forward_reward = self._forward_reward_weight * velocity

        min_z = self._minimum_z_height
        is_healthy = jnp.where(data.q[1] < min_z, 0.0, 1.0)
        if self._terminate_when_unhealthy:
            healthy_reward = self.healthy_reward
        else:
            healthy_reward = self.healthy_reward * is_healthy

        unwanted = self._unwanted_contact(data)

        prev_action = state.info['last_action']
        act_diff = action - prev_action
        act_penalty = self._action_rate_penalty_weight * jnp.sum(act_diff ** 2)
        obs = self._get_obs(data, prev_action)
        reward = forward_reward + healthy_reward + act_penalty
        # done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        done = jnp.maximum(1.0 - is_healthy, unwanted)

        state.metrics.update({
            'vel_reward': forward_reward,
            'reward_alive': healthy_reward,
            'action_rate_penalty': act_penalty,
        })

        new_info = {**state.info, 'last_action': action}

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, info=new_info)

    def _get_obs(self, data, last_action):
        root_height = data.qpos[1:2]
        root_speed = data.qvel[:2]
        joint_qpos = data.qpos[2:]
        joint_qvel = data.qvel[2:]
    
        return jnp.concatenate([root_height, root_speed, joint_qpos, joint_qvel, last_action])

    def _unwanted_contact(self, data):
        """Return 1.0 if any non-foot geom is close (<0.05 m) or touching the ground."""
        ncon = int(data._impl.ncon)
        if ncon == 0:
            return 0.0

        geom1 = jnp.array(data.contact.geom1[:ncon])
        geom2 = jnp.array(data.contact.geom2[:ncon])
        dist  = jnp.array(data.contact.dist[:ncon])

        ground_id = self._ground_geom_id
        foot_id   = self._foot_geom_id

        # consider only contacts where ground is involved
        ground_contact = jnp.logical_or(geom1 == ground_id, geom2 == ground_id)
        other_geom = jnp.where(geom1 == ground_id, geom2, geom1)

        # proximity threshold: any contact closer than 5 cm
        near_ground = dist < 0.01

        # combine masks
        valid = jnp.logical_and(ground_contact, near_ground)
        bad_contact = jnp.any(jnp.logical_and(valid, other_geom != foot_id))

        return jnp.where(bad_contact, 1.0, 0.0)



# Write main function to visualize a rollout
def main(args=None):
    xml_path = os.path.join(os.path.dirname(__file__), "single_leg_robstride02.xml")
    env = SingleLegMJX(xml_path, link_length_sum=0.6)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    # state = env.reset(jax.random.PRNGKey(0))
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    for i in range(200):
        ctrl = -0.1 * jp.ones(env.sys.nu)  # some constant action
        # state = env.step(state, ctrl)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        # data = state.pipeline_state

        # ncon = int(data.ncon)
        # if ncon > 0:
        #     g1 = np.array(data.contact.geom1[:ncon])
        #     g2 = np.array(data.contact.geom2[:ncon])
        #     dist = np.array(data.contact.dist[:ncon])
        #     print(f"\nStep {i:03d}: {ncon} contacts")
        #     print("geom1:", g1)
        #     print("geom2:", g2)
        #     print("dist :", dist)
        # else:
        #     print(f"Step {i:03d}: no contacts")

        # print(f"done={state.done}, reward={float(state.reward):.3f}")

    print("Rendering rollout...")
    frames = render_with_camera(env, rollout, camera="side_track_cam", width=720, height=1280)
    media.write_video("single_rollout_trackcam.mp4", frames, fps=1.0/env.dt)
    print("env dt is ", env.dt)
    print("Video saved as single_rollout.mp4")



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

if __name__ == "__main__":
    main()
