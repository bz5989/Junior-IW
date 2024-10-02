import os
from typing import Dict, List

import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians


class CRL(IOD):
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
            log_eval_return: bool = False,
            use_l2_penalty: bool = False,
            self_normalizing: bool = False,
            log_sum_exp: bool = False,
            symmetrize_log_sum_exp: bool = False,
            add_log_sum_exp_to_rewards: bool = False,
            record_corr_m: bool = False,
            contrastive: bool = False,
            contrastive_every: bool = False,
            phi_encoder: torch.nn.Module = None,
            fixed_lam: float = None,
            use_mse: bool = False,
            metra_rep: bool = False,
            use_metra_penalty: bool = False,
            crl_standardize_output: bool = False,
            metra_include_actions: bool = False,
            use_info_nce: bool = False,
            use_oracle_goals: bool = False,
            use_half_random_goals: bool = False,
            add_penalty_to_rewards: bool = False,
            no_diff_in_penalty: bool = False,
            no_diff_in_rep: bool = False,
            include_one_minus_gamma: bool = False,
            use_positive_log_sum_exp: bool = False,
            scale_std: float = 1.0,
            use_next_state: bool = False,
            turn_off_dones: bool = False,
            use_discrete_sac: bool = False,
            use_goal_rep: bool = False,
            goal_dim: int = 2,
            add_norm_exploration: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.phi_encoder = phi_encoder.to(self.device)

        self.param_modules.update(
            phi_encoder=self.phi_encoder
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
        self.log_eval_return = log_eval_return
        self.use_l2_penalty = use_l2_penalty
        self.self_normalizing = self_normalizing
        self.log_sum_exp = log_sum_exp
        self.symmetrize_log_sum_exp = symmetrize_log_sum_exp
        self.add_log_sum_exp_to_rewards = add_log_sum_exp_to_rewards
        self.record_corr_m = record_corr_m
        self.contrastive = contrastive
        self.contrastive_every = contrastive_every
        self.use_mse = use_mse
        self.fixed_lam = fixed_lam
        self.metra_rep = metra_rep
        self.use_metra_penalty = use_metra_penalty
        self.crl_standardize_output = crl_standardize_output
        self.use_info_nce = use_info_nce
        self.use_oracle_goals = use_oracle_goals
        self.use_half_random_goals = use_half_random_goals
        self.use_positive_log_sum_exp = use_positive_log_sum_exp
        self.turn_off_dones = turn_off_dones
        self.use_goal_rep = use_goal_rep
        self.goal_dim = goal_dim
        self.add_norm_exploration = add_norm_exploration

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

            if self.use_oracle_goals:
                endpoints = np.linspace(self.eval_plot_axis[0], self.eval_plot_axis[1], 400)
                coord = np.random.choice(endpoints, size=runner._train_args.batch_size)
                sides = np.random.choice(np.array([0, 1, 2, 3]), size=runner._train_args.batch_size)

                for i in range(runner._train_args.batch_size):
                    if sides[i] == 0:
                        random_options[i][0] = coord[i]
                        random_options[i][1] = self.eval_plot_axis[0]
                    elif sides[i] == 1:
                        random_options[i][0] = coord[i]
                        random_options[i][1] = self.eval_plot_axis[1]
                    elif sides[i] == 2:
                        random_options[i][0] = self.eval_plot_axis[0]
                        random_options[i][1] = coord[i]
                    else:
                        random_options[i][0] = self.eval_plot_axis[1]
                        random_options[i][1] = coord[i]

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

            self._optimize_critic(tensors, v)
            self._optimize_actor(tensors, v)

        return tensors

    def _optimize_critic(self, tensors, v):
        criterion = torch.nn.BCEWithLogitsLoss()

        if self.env_name == 'fetch_reach' or self.env_name == 'ant_with_goals' or self.env_name == 'sawyer_bin' or self.env_name == 'point':
            self.traj_encoder = self.traj_encoder.to(self.device)
            half_idx = v['obs'].shape[1] - self.goal_dim
            states = v['obs'][:, :half_idx] # cut the goal
            future_states = v['future_obs'][:, :self.goal_dim] # cut the goal
        else:
            states = v['obs']
            next_states = v['next_obs']
            future_states = v['future_obs']

        actions = v['actions']

        sa_repr = self.phi_encoder(states, actions)
        g_repr = self.traj_encoder(future_states).mean

        if self.crl_standardize_output:
            g_repr_mean = torch.mean(g_repr, dim=0)
            g_repr_std = torch.std(g_repr, dim=0)
            g_repr = (g_repr - g_repr_mean) / (g_repr_std + 1e-5)

        if self.use_mse:
            sa_repr_m = sa_repr.unsqueeze(1).repeat((1, g_repr.shape[0], 1))
            g_repr_m = g_repr.unsqueeze(1).permute(1, 0, 2).repeat((sa_repr.shape[0], 1, 1))
            logits = -1 * torch.square(sa_repr_m - g_repr_m).sum(dim=-1)
        else:
            if self.metra_rep:
                next_sa_repr = self.phi_encoder(next_states, actions)
                logits = torch.einsum('ik, jk -> ij', (next_sa_repr - sa_repr), g_repr)
            else:
                logits = torch.einsum('ik, jk -> ij', sa_repr, g_repr)

        if self.use_info_nce:
            # take diagonal of logits matrix
            align = torch.diag(logits)
            uniformity = torch.logsumexp(logits, dim=-1)
            if self.symmetrize_log_sum_exp:
                uniformity = (uniformity + torch.logsumexp(logits.t(), dim=-1)) / 2.0

            loss = -1 * (align - uniformity).mean()
        else:
            loss = criterion(logits, torch.eye(logits.shape[0]).to(self.device))  

        if self.use_l2_penalty:
            l2_penalty = torch.square(g_repr).mean(dim=-1)
            loss += self.fixed_lam * l2_penalty.mean()

        if self.use_metra_penalty:
            dual_lam = self.dual_lam.param.exp()
            x = states
            y = next_states
            phi_x = self.traj_encoder(states).mean
            phi_y = self.traj_encoder(next_states).mean

            cst_dist = torch.ones_like(x[:, 0])
            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

            loss += dual_lam.detach().item() * cst_penalty.mean()


        tensors.update({
            'LossCritic': loss,
        })

        self._gradient_descent(
            loss,
            optimizer_keys=['phi_encoder', 'traj_encoder'],
        )

    def _optimize_actor(self, tensors, v):
        if self.env_name == 'fetch_reach' or self.env_name == 'ant_with_goals' or self.env_name == 'sawyer_bin'  or self.env_name == 'point':
            self.traj_encoder = self.traj_encoder.to(self.device)
            half_idx = v['obs'].shape[1] - self.goal_dim
            states = v['obs'][:, :half_idx]
            goals = v['future_obs'][:, :self.goal_dim]

            if self.use_half_random_goals:
                states = torch.cat([states, states], dim=0)
                goals = torch.cat([goals, torch.roll(goals, 1, dims=0)], dim=0)
            else:
                goals = torch.roll(goals, 1, dims=0)
        else:
            states = v['obs']
            goals = v['future_obs']

            if self.metra_rep:
                next_states = v['next_obs']
                next_actions = v['next_actions']

            goals = torch.roll(goals, 1, dims=0)

        g_repr = self.traj_encoder(goals).mean

        if self.crl_standardize_output:
            g_repr_mean = torch.mean(g_repr, dim=0)
            g_repr_std = torch.std(g_repr, dim=0)
            g_repr = (g_repr - g_repr_mean) / (g_repr_std + 1e-5)

        if self.env_name == 'fetch_reach' or self.env_name == 'ant_with_goals' or self.env_name == 'sawyer_bin' or self.env_name == 'point':
            if self.use_goal_rep:
                policy_input = torch.cat([states, g_repr], dim=1)
            else:
                policy_input = torch.cat([states, goals], dim=1)
        else:
            if self.use_oracle_goals:
                policy_input = torch.cat([states, v['future_coordinates']], dim=1)
            else:
                policy_input = torch.cat([states, g_repr], dim=1)

        action_dists, *_ = self.option_policy(policy_input)
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            _, new_actions = action_dists.rsample_with_pre_tanh_value()
        else:
            new_actions = action_dists.rsample()

        sa_repr = self.phi_encoder(states, new_actions)

        if self.use_mse: 
            if self.add_log_sum_exp_to_rewards:
                sa_repr_m = sa_repr.unsqueeze(1).repeat((1, g_repr.shape[0], 1))
                g_repr_m = g_repr.unsqueeze(1).permute(1, 0, 2).repeat((sa_repr.shape[0], 1, 1))
                logits = -1 * torch.square(sa_repr_m - g_repr_m).sum(dim=-1)
                if self.use_info_nce:
                    # take diagonal of logits matrix
                    align = torch.diag(logits)
                    uniformity = torch.logsumexp(logits, dim=-1)
                    if self.symmetrize_log_sum_exp:
                        uniformity = (uniformity + torch.logsumexp(logits.t(), dim=-1)) / 2.0

                    if self.use_positive_log_sum_exp:
                        logits = align + uniformity
                    else:
                        logits = align - uniformity
            else:
                logits = -1 * torch.square(sa_repr - g_repr).sum(dim=1)

                if self.add_norm_exploration:
                    logits += 0.5 * torch.norm(sa_repr, dim=-1)
        else:
            if self.metra_rep:
                next_sa_repr = self.phi_encoder(next_states, new_actions)
                logits = torch.einsum('ik, ik -> i', (next_sa_repr - sa_repr), g_repr)
            else:
                logits = torch.einsum('ik, ik -> i', sa_repr, g_repr)


        loss = -logits.mean()

        tensors.update({
            'LossActor': loss,
        })

        self._gradient_descent(
            loss,
            optimizer_keys=['option_policy'],
        )

    def _evaluate_policy(self, runner):
        if self.env_name == 'fetch_reach' or self.env_name == 'ant_with_goals' or self.env_name == 'sawyer_bin' or self.env_name == 'point':
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option) # not used
            random_option_colors = get_option_colors(random_options * 4)
            if self.use_goal_rep:
                extras = [{'traj_encoder':self.traj_encoder}]* self.num_random_trajectories
            else:
                extras = None
            random_trajectories = self._get_trajectories(
                runner,
                sampler_key='option_policy',
                extras=extras,
                batch_size=self.num_random_trajectories,
                worker_update=dict(
                    _render=False,
                    _deterministic_policy=True,
                ),
                env_update=dict(_action_noise_std=None),
            )

            if self.env_name == 'ant_with_goals':
                with FigManager(runner, 'TrajPlot_RandomZ') as fm:
                    runner._env.render_trajectories(
                        random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
                    )

            eval_option_metrics = {}

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
                    
                if self.use_goal_rep:
                    extras = [{'traj_encoder':self.traj_encoder}]* video_options.shape[0]
                else:
                    extras = None
                video_trajectories = self._get_trajectories(
                    runner,
                    sampler_key='local_option_policy',
                    extras=extras,
                    batch_size=video_options.shape[0],
                    worker_update=dict(
                        _render=True,
                        _deterministic_policy=True,
                    ),
                )
                record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)

            if self.use_oracle_goals:
                endpoints = np.linspace(self.eval_plot_axis[0], self.eval_plot_axis[1], 400)
                coord = np.random.choice(endpoints, size=self.num_random_trajectories)
                sides = np.random.choice(np.array([0, 1, 2, 3]), size=self.num_random_trajectories)

                for i in range(self.num_random_trajectories):
                    if sides[i] == 0:
                        random_options[i][0] = coord[i]
                        random_options[i][1] = self.eval_plot_axis[0]
                    elif sides[i] == 1:
                        random_options[i][0] = coord[i]
                        random_options[i][1] = self.eval_plot_axis[1]
                    elif sides[i] == 2:
                        random_options[i][0] = self.eval_plot_axis[0]
                        random_options[i][1] = coord[i]
                    else:
                        random_options[i][0] = self.eval_plot_axis[1]
                        random_options[i][1] = coord[i]

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

                if self.use_oracle_goals:
                    endpoints = np.linspace(self.eval_plot_axis[0], self.eval_plot_axis[1], 400)
                    coord = np.random.choice(endpoints, size=video_options.shape[0])
                    sides = np.random.choice(np.array([0, 1, 2, 3]), size=video_options.shape[0])

                    for i in range(video_options.shape[0]):
                        if sides[i] == 0:
                            video_options[i][0] = coord[i]
                            video_options[i][1] = self.eval_plot_axis[0]
                        elif sides[i] == 1:
                            video_options[i][0] = coord[i]
                            video_options[i][1] = self.eval_plot_axis[1]
                        elif sides[i] == 2:
                            video_options[i][0] = self.eval_plot_axis[0]
                            video_options[i][1] = coord[i]
                        else:
                            video_options[i][0] = self.eval_plot_axis[1]
                            video_options[i][1] = coord[i]
                
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
