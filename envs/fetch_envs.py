
import gym
from gym.envs.robotics.fetch import reach
import numpy as np

class FetchReachEnv(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment."""

  def __init__(self):
    super(FetchReachEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((20,), -np.inf),
        high=np.full((20,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchReachEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action, render: bool = False):
    s, _, _, _ = super(FetchReachEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)  # Default from Fetch environment.
    info = {}
    if render:
      info['render'] = self.render(mode='rgb_array', width=100, height=100).transpose(2, 0, 1)
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 0
    end_index = 3
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)
  
  def calc_eval_metrics(self, trajectories, is_option_trajectories: bool):
    successes = []
    for t in trajectories:
      successes.append(np.sum(t['rewards']) >= 1.0)

    return {
        'success_rate': np.mean(successes),
    }   