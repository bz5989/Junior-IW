from collections import defaultdict

import numpy as np
from minigrid.envs import FourRoomsEnv
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from envs.mujoco.mujoco_utils import MujocoTrait


class FourRoomsWrapper(MujocoTrait):
    def __init__(self, env: FourRoomsEnv, continuous: bool = False):
        self.env = env
        self.continuous = continuous

        self.observation_space = Box(env.observation_space['image'].low.flatten(), env.observation_space['image'].high.flatten(), (np.prod(env.observation_space['image'].shape),))
        self.action_space = Discrete(3)
        self.reward_range = env.env.reward_range
        self.metadata = env.env.metadata

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        return obs['image'].flatten()

    def step(self, action, render: bool = False):
        if self.continuous:
            action = np.argmax(action)

        action = int(action)
        x_before, y_before = self.env.env.agent_pos
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_after, y_after = self.env.env.agent_pos

        info['coordinates'] = np.array([x_before, y_before])
        info['next_coordinates'] = np.array([x_after, y_after])

        if render:
            info['render'] = self.env.render().transpose(2, 0, 1)

        return obs['image'].flatten(), reward, terminated or truncated, info

    def close(self, *args, **kwargs):
        pass

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = {}
        
        return eval_metrics