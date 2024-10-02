
import gym
import numpy as np
import metaworld

class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self):
    self._goal = np.zeros(3)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    t = np.random.random()
    self._goal = t * pos1 + (1 - t) * pos2
    self._goal[2] = np.random.uniform(0.03, 0.12)
    return self._get_obs()

  def step(self, action, render: bool = False):
    super(SawyerBin, self).step(action)
    dist = np.linalg.norm(self._goal - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
    if render:
      info['render'] = self.render(offscreen=True, resolution=(128, 96)).transpose(2, 0, 1)
    
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)
  
  def calc_eval_metrics(self, trajectories, is_option_trajectories: bool):
    successes = []
    for t in trajectories:
      successes.append(np.sum(t['rewards']) >= 1.0)

    return {
        'success_rate': np.mean(successes),
    }   
