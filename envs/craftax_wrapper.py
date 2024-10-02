from collections import defaultdict

import numpy as np
import jax
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax_classic.renderer import render_craftax_pixels


class CraftaxWrapper:
    def __init__(self, env: CraftaxSymbolicEnv):
        self.env = env
        self.env_params = env.default_params

        self.observation_space = env.observation_space(None)
        self.action_space = env.action_space(None)
        self.reward_range = None
        self.metadata = None
        
        self.state = None
        self.rng = jax.random.PRNGKey(0)

    def reset(self):
        rng, _rng = jax.random.split(self.rng)
        obs, state = self.env.reset(_rng, self.env_params)

        self.state = state
        self.rng = rng

        return obs

    def step(self, action, render: bool = False):
        rng, _rng = jax.random.split(self.rng)
        obs, state, reward, done, info = self.env.step(_rng, self.state, action, self.env_params)
        info['achievements'] = state.achievements

        if render:
            pixels = render_craftax_pixels(state, block_pixel_size=16).transpose(2, 0, 1)
            info['render'] = pixels / 255.0

        self.state = state
        self.rng = rng

        return obs, reward, done, info

    def close(self, *args, **kwargs):
        pass

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = defaultdict(int)
        for trajectory in trajectories:
            for k, v in trajectory['env_infos'].items():
                if k.startswith('Achievements/'):
                    eval_metrics[k] += np.any(v)

        # normalize
        for k in eval_metrics:
            eval_metrics[k] /= len(trajectories)
        
        return eval_metrics