import copy
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians
from iod.sac_utils import _clip_actions

class MetraSf(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            pixel_shape=None,
            self_normalizing: bool = False,
            log_sum_exp: bool = False,
            fixed_lam: float = None,
            no_diff_in_rep: bool = False,
            use_discrete_sac: bool = False,
            turn_off_dones: bool = False,
            goal_range: float = None,
            frame_stack: int = None, 
            eval_goal_metrics: int = None,
            sample_new_z: bool = False,
            num_negative_z: int = 256,
            infonce_lam: float = 1.0,
            metra_mlp_rep: bool = False,
            f_encoder: torch.nn.Module = None,
            num_zero_shot_goals: int = 50,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.target_te = copy.deepcopy(self.traj_encoder)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape
        self.self_normalizing = self_normalizing
        self.log_sum_exp = log_sum_exp
        self.fixed_lam = fixed_lam
        self.no_diff_in_rep = no_diff_in_rep
        self.turn_off_dones = turn_off_dones
        self.use_discrete_sac = use_discrete_sac
        self.goal_range = goal_range
        self.frame_stack = frame_stack
        self.eval_goal_metrics = eval_goal_metrics
        self.sample_new_z = sample_new_z
        self.num_negative_z = num_negative_z
        self.infonce_lam = infonce_lam
        self.metra_mlp_rep = metra_mlp_rep
        if self.metra_mlp_rep:
            self.f_encoder = f_encoder.to(self.device)
        self.num_zero_shot_goals = num_zero_shot_goals

        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data: Dict[str, List[np.ndarray]]) -> Dict[str, torch.tensor]:
        """
        Joins all trajectories together per key.

        Args:
            data (Dict[str, List[np.ndarray]]): Dict where each key has a list of paths / trajectories.

        Returns:
            Dict[str, torch.tensor]: Dict where each key is a torch tensor of the joined paths.
        """
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data: Dict[str, List[np.ndarray]]) -> None:
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                # Every i iteration extracts one path (trajectory) from the data
                path: Dict[str, np.ndarray] = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data: Dict[str, List[np.ndarray]]):
        self._update_replay_buffer(path_data)

        epoch_data: Dict[str, torch.tensor] = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data: Dict[str, torch.tensor]):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}
        
        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_te(tensors, v)
            self._optimize_sf(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=(['traj_encoder'] if not self.metra_mlp_rep else ['f_encoder'])
        )

        if self.dual_reg and not self.log_sum_exp:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_sf(self, tensors, internal_vars):
        self._update_loss_sf_td(tensors, internal_vars)        

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_targets()

    def _optimize_op(self, tensors, internal_vars):
        states = self._get_concat_obs(self.option_policy.process_observations(internal_vars['obs']), internal_vars['options'])
        action_dists, *_ = self.option_policy(states)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = _clip_actions(self, new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)

        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

        sf1_feature = self.qf1(states, new_actions)
        sf2_feature = self.qf2(states, new_actions)

        q1_values = (sf1_feature * internal_vars['options']).sum(dim=-1)
        q2_values = (sf2_feature * internal_vars['options']).sum(dim=-1)
        q_values = torch.min(q1_values, q2_values)

        logits = -1 * q_values + alpha * new_action_log_probs

        loss_op = logits.mean()

        tensors.update({
            'LossOp': loss_op,
        })

        internal_vars.update({
            'new_action_log_probs': new_action_log_probs
        })

        self._gradient_descent(
            loss_op,
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v
        )

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.no_diff_in_rep:
                target_z = cur_z

            if self.self_normalizing:
                target_z = target_z / target_z.norm(dim=-1, keepdim=True)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=v['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = target_z @ new_z.t()
                else:
                    pairwise_scores = target_z @ v['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options']).sum(dim=1)
                rewards = inner

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        elif self.metra_mlp_rep:
            # unneccessary but avoids key errors for now
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })

            rep = self.f_encoder(obs, next_obs)
            rewards = (rep * v['options']).sum(dim=1)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=v['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = rep @ new_z.t()
                else:
                    pairwise_scores = rep @ v['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards
        if self.log_sum_exp:
            v['log_sum_exp'] = log_sum_exp

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError

            inside_l2 = phi_y - phi_x

            cst_penalty = cst_dist - torch.square(inside_l2).sum(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

            if self.self_normalizing:
                te_obj = rewards
            elif self.log_sum_exp:
                te_obj = rewards - self.infonce_lam * v['log_sum_exp']
            elif self.fixed_lam is not None:
                te_obj = rewards + self.fixed_lam * cst_penalty
            else:
                te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_sf_td(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']
        actions = v['actions']
        options = v['options']
        next_options = v['next_options']
        dones = v['dones']
        assert torch.allclose(options, next_options)
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(obs), options)
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(next_obs),
                                                      next_options)

        cur_repr = self.traj_encoder(obs).mean
        next_repr = self.traj_encoder(next_obs).mean

        sf1_pred = self.qf1(processed_cat_obs, actions)
        sf2_pred = self.qf2(processed_cat_obs, actions)

        next_action_dists, *_ = self.option_policy(next_processed_cat_obs)
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            _, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = _clip_actions(self, new_next_actions)

        target_next_sf1 = self.target_qf1(next_processed_cat_obs, new_next_actions)
        target_next_sf2 = self.target_qf2(next_processed_cat_obs, new_next_actions)
        target_next_sf = torch.stack([target_next_sf1, target_next_sf2], dim=1)
        next_qf1_values = (target_next_sf1 * next_options).sum(dim=-1)
        next_qf2_values = (target_next_sf2 * next_options).sum(dim=-1)
        next_qf_values = torch.stack([next_qf1_values, next_qf2_values], dim=1)
        min_idxs = torch.argmin(next_qf_values, dim=1)
        target_next_sf_min = target_next_sf[torch.arange(self._trans_minibatch_size, device=self.device), min_idxs]
        target_next_sf = target_next_sf_min

        with torch.no_grad():
            if self.turn_off_dones:
                dones[...] = 0
            
            if self.metra_mlp_rep:
                sf_target = self.f_encoder(v['obs'], v['next_obs']) + self.discount * (1. - dones[:, None]) * target_next_sf
            elif self.no_diff_in_rep:
                sf_target = cur_repr + self.discount * (1. - dones[:, None]) * target_next_sf
            else:
                sf_target = (next_repr - cur_repr) + self.discount * (1. - dones[:, None]) * target_next_sf

        loss_sf1 = F.mse_loss(sf1_pred, sf_target)
        loss_sf2 = F.mse_loss(sf2_pred, sf_target)

        tensors.update({
            'Qf1Mean': sf1_pred.mean(),
            'Qf2Mean': sf2_pred.mean(),
            'QfTargetsMean': sf_target.mean(),
            'QfTdErrsMean': ((sf_target - sf1_pred).mean() + (sf_target - sf2_pred).mean()) / 2,
            'LossQf1': loss_sf1,
            'LossQf2': loss_sf2,
        })

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_targets(self):
        # target_sfs = [self.target_sf1, self.target_sf2]
        # sfs = [self.sf1, self.sf2]
        target_sfs = [self.target_qf1, self.target_qf2]
        sfs = [self.qf1, self.qf2]
        for target_sf, sf in zip(target_sfs, sfs):
            for t_param, param in zip(target_sf.parameters(), sf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)

    
    def _update_target_te(self):
        for t_param, param in zip(self.target_te.parameters(), self.traj_encoder.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                param.data * self.tau)

    @torch.no_grad()
    def _evaluate_policy(self, runner):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}
        if self.eval_goal_metrics:
            env = runner._env
            goals = []  # list of (goal_obs, goal_info)
            goal_metrics = defaultdict(list)

            if self.env_name == 'kitchen':
                goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']
                for i in range(self.num_zero_shot_goals):
                    goal_idx = np.random.randint(len(goal_names))
                    goal_name = goal_names[goal_idx]
                    goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))

            elif self.env_name == 'robobin_image':
                goal_names = ['ReachLeft', 'ReachRight', 'PushFront', 'PushBack']
                for i in range(self.num_zero_shot_goals):
                    goal_idx = np.random.randint(len(goal_names))
                    goal_name = goal_names[goal_idx]
                    goal_obs = env.render_goal(goal_idx=goal_idx).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_idx': goal_idx, 'goal_name': goal_name}))

            elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid']:
                for i in range(self.num_zero_shot_goals):
                    env.reset()
                    state = env.physics.get_state().copy()
                    if self.env_name == 'dmc_cheetah':
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                    env.physics.set_state(state)
                    if self.env_name == 'dmc_humanoid':
                        for _ in range(50):
                            env.step(np.zeros_like(env.action_space.sample()))
                    else:
                        env.step(np.zeros_like(env.action_space.sample()))
                    goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                    goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    goals.append((goal_obs, {'goal_loc': goal_loc}))

            elif self.env_name in ['ant', 'ant_pixel', 'half_cheetah']:
                for i in range(self.num_zero_shot_goals):
                    env.reset()
                    state = env.unwrapped._get_obs().copy()
                    if self.env_name in ['half_cheetah']:
                        goal_loc = (np.random.rand(1) * 2 - 1) * self.goal_range
                        state[:1] = goal_loc
                        env.set_state(state[:9], state[9:])
                    else:
                        goal_loc = (np.random.rand(2) * 2 - 1) * self.goal_range
                        state[:2] = goal_loc
                        env.set_state(state[:15], state[15:])
                    for _ in range(5):
                        env.step(np.zeros_like(env.action_space.sample()))
                    if self.env_name == 'ant_pixel':
                        goal_obs = env.render(mode='rgb_array', width=64, height=64).copy().astype(np.float32)
                        goal_obs = np.tile(goal_obs, self.frame_stack or 1).flatten()
                    else:
                        goal_obs = env._apply_normalize_obs(state).astype(np.float32)
                    goals.append((goal_obs, {'goal_loc': goal_loc}))

            if self.unit_length:
                mean_length = 1.
            else:
                mean_length = np.linalg.norm(np.random.randn(1000000, self.dim_option), axis=1).mean()
            for method in ['Single', 'Adaptive'] if (self.discrete and self.inner) else ['']:
                for goal_obs, goal_info in goals:
                    obs = env.reset()
                    step = 0
                    done = False
                    success = 0
                    staying_time = 0

                    hit_success_3 = 0
                    end_success_3 = 0
                    at_success_3 = 0

                    hit_success_1 = 0
                    end_success_1 = 0
                    at_success_1 = 0

                    option = None
                    while step < self.max_path_length and not done:
                        if self.inner:
                            if self.no_diff_in_rep: 
                                te_input = torch.from_numpy(goal_obs[None, ...]).to(self.device)
                                phi = self.traj_encoder(te_input).mean[0]

                                if self.self_normalizing:
                                    phi = phi / phi.norm(dim=-1, keepdim=True)

                                phi = phi.detach().cpu().numpy()
                                if self.discrete:
                                    option = np.eye(self.dim_option)[phi.argmax()]
                                else:
                                    option = phi
                            else:
                                te_input = torch.from_numpy(np.stack([obs, goal_obs])).to(self.device)
                                phi_s, phi_g = self.traj_encoder(te_input).mean
                                phi_s, phi_g = phi_s.detach().cpu().numpy(), phi_g.detach().cpu().numpy()
                                if self.discrete:
                                    if method == 'Adaptive':
                                        option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                                    else:
                                        if option is None:
                                            option = np.eye(self.dim_option)[(phi_g - phi_s).argmax()]
                                else:
                                    option = (phi_g - phi_s) / np.linalg.norm(phi_g - phi_s) * mean_length
                        else:
                            te_input = torch.from_numpy(goal_obs[None, ...]).to(self.device)
                            phi = self.traj_encoder(te_input).mean[0]
                            phi = phi.detach().cpu().numpy()
                            if self.discrete:
                                option = np.eye(self.dim_option)[phi.argmax()]
                            else:
                                option = phi
                        action, agent_info = self.option_policy.get_action(np.concatenate([obs, option]))
                        next_obs, _, done, info = env.step(action)
                        obs = next_obs

                        if self.env_name == 'kitchen':
                            _success = env.compute_success(goal_info['goal_idx'])[0]
                            success = max(success, _success)
                            staying_time += _success

                        if self.env_name == 'robobin_image':
                            success = max(success, info['success'])
                            staying_time += info['success']

                        if self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                            if self.env_name in ['dmc_cheetah']:
                                cur_loc = env.physics.get_state()[:1]
                            elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                                cur_loc = env.physics.get_state()[:2]
                            elif self.env_name in ['half_cheetah']:
                                cur_loc = env.unwrapped._get_obs()[:1]
                            else:
                                cur_loc = env.unwrapped._get_obs()[:2] 

                            if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 3:
                                hit_success_3 = 1.
                                at_success_3 += 1.

                            if np.linalg.norm(cur_loc - goal_info['goal_loc']) < 1:
                                hit_success_1 = 1.
                                at_success_1 += 1.

                        step += 1

                    if self.env_name == 'kitchen':
                        goal_metrics[f'Kitchen{method}Goal{goal_info["goal_name"]}'].append(success)
                        goal_metrics[f'Kitchen{method}GoalOverall'].append(success * len(goal_names))
                        goal_metrics[f'Kitchen{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                        goal_metrics[f'Kitchen{method}GoalStayingTimeOverall'].append(staying_time)

                    elif self.env_name == 'robobin_image':
                        goal_metrics[f'Robobin{method}Goal{goal_info["goal_name"]}'].append(success)
                        goal_metrics[f'Robobin{method}GoalOverall'].append(success * len(goal_names))
                        goal_metrics[f'Robobin{method}GoalStayingTime{goal_info["goal_name"]}'].append(staying_time)
                        goal_metrics[f'Robobin{method}GoalStayingTimeOverall'].append(staying_time)

                    elif self.env_name in ['dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'ant', 'ant_pixel', 'half_cheetah']:
                        if self.env_name in ['dmc_cheetah']:
                            cur_loc = env.physics.get_state()[:1]
                        elif self.env_name in ['dmc_quadruped', 'dmc_humanoid']:
                            cur_loc = env.physics.get_state()[:2]
                        elif self.env_name in ['half_cheetah']:
                            cur_loc = env.unwrapped._get_obs()[:1]
                        else:
                            cur_loc = env.unwrapped._get_obs()[:2]
                        distance = np.linalg.norm(cur_loc - goal_info['goal_loc'])
                        squared_distance = distance ** 2
                        if distance < 3:
                            end_success_3 = 1.
                        if distance < 1:
                            end_success_1 = 1.
                    
                        goal_metrics[f'HitSuccess3{method}'].append(hit_success_3)
                        goal_metrics[f'EndSuccess3{method}'].append(end_success_3)
                        goal_metrics[f'AtSuccess3{method}'].append(at_success_3 / step)

                        goal_metrics[f'HitSuccess1{method}'].append(hit_success_1)
                        goal_metrics[f'EndSuccess1{method}'].append(end_success_1)
                        goal_metrics[f'AtSuccess1{method}'].append(at_success_1 / step)

                        goal_metrics[f'Goal{method}Distance'].append(distance)
                        goal_metrics[f'Goal{method}SquaredDistance'].append(squared_distance)

            goal_metrics = {key: np.mean(value) for key, value in goal_metrics.items()}
            eval_option_metrics.update(goal_metrics)

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
