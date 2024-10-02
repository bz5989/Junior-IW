import os
from typing import Dict, List
from collections import defaultdict

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
from sklearn.neighbors import KernelDensity

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

# from envs.craftax_wrapper import CraftaxWrapper


class METRA(IOD):
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
            metra_rep: bool = False,
            use_metra_penalty: bool = False,
            use_mse: bool = False,
            metra_include_actions: bool = False,
            crl_standardize_output: bool = False,
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
            use_discrete_sac: bool = False,
            turn_off_dones: bool = False,
            f_encoder: torch.nn.Module = None,
            metra_mlp_rep: bool = False,
            log_logp_vs_l2: bool = False,
            reward_lam: float = None,
            add_diff_norm_bonus: bool = False,
            add_angle_bonus: bool = False,
            add_norm_bonus: bool = False,
            eval_goal_metrics: bool = False,
            goal_range: float = None,
            frame_stack: int = None,
            relabel_actor_z: bool = False,
            relabel_critic_z: bool = False,
            use_bessel_penalty: bool = False,
            sample_new_z: bool = False,
            num_negative_z: int = 256,
            use_fp: bool = False,
            infonce_lam: float = 1.0,
            rep_temp: float = 1.0,
            actor_temp: float = 1.0,
            diayn_include_baseline: bool = False,
            uniform_z: bool = False,
            num_zero_shot_goals: int = 50,
            scale_radius: float = 1.0,
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

        self.use_discrete_sac = use_discrete_sac    
        self._reward_scale_factor = scale_reward
        if self.use_discrete_sac:
            self._target_entropy = np.log(self._env_spec.action_space.n) * target_coef
        else:
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
        self.fixed_lam = fixed_lam
        self.metra_include_actions = metra_include_actions
        self.use_oracle_goals = use_oracle_goals
        self.use_half_random_goals = use_half_random_goals
        self.add_penalty_to_rewards = add_penalty_to_rewards
        self.no_diff_in_penalty = no_diff_in_penalty
        self.no_diff_in_rep = no_diff_in_rep
        self.include_one_minus_gamma = include_one_minus_gamma
        self.scale_std = scale_std
        self.use_next_state = use_next_state
        self.turn_off_dones = turn_off_dones
        self.metra_mlp_rep = metra_mlp_rep
        if self.metra_mlp_rep:
            self.f_encoder = f_encoder.to(self.device)
        self.log_logp_vs_l2 = log_logp_vs_l2
        self.reward_lam = reward_lam
        self.add_diff_norm_bonus = add_diff_norm_bonus
        self.add_angle_bonus = add_angle_bonus
        self.add_norm_bonus = add_norm_bonus
        self.eval_goal_metrics = eval_goal_metrics
        self.goal_range = goal_range
        self.frame_stack = frame_stack
        self.relabel_actor_z = relabel_actor_z
        self.relabel_critic_z = relabel_critic_z
        self.use_bessel_penalty = use_bessel_penalty
        self.sample_new_z = sample_new_z
        self.num_negative_z = num_negative_z
        self.use_fp = use_fp
        self.infonce_lam = infonce_lam
        self.rep_temp = rep_temp
        self.actor_temp = actor_temp
        self.diayn_include_baseline = diayn_include_baseline
        self.uniform_z = uniform_z
        self.num_zero_shot_goals = num_zero_shot_goals
        self.scale_radius = scale_radius

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
            if self.uniform_z:
                random_options = np.random.uniform(low=-1.0, high=1.0, size=(runner._train_args.batch_size, self.dim_option))
            else:
                random_options = np.random.randn(runner._train_args.batch_size, self.dim_option) * self.scale_std
                if self.unit_length:
                    random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
                    random_options *= self.scale_radius
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

    def _sample_replay_buffer(self, batch: int = None):
        if batch is None:
            batch = self._trans_minibatch_size

        samples = self.replay_buffer.sample_transitions(batch)
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
            self._update_rewards(tensors, v, relabel_rewards=(self.relabel_critic_z or self.relabel_actor_z), use_fp=self.use_fp, temp=self.actor_temp)
            self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=(['traj_encoder'] if not self.metra_mlp_rep else ['f_encoder']),
        )

        if self.dual_reg and self.fixed_lam is None and not self.log_sum_exp:
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

        if self.use_fp:
            self._update_target_te()

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v, relabel_rewards: bool = False, use_fp: bool = False, temp: bool = 1.0):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            if self.metra_include_actions:
                cur_z = self.traj_encoder(obs, v['actions'])
                next_z = self.traj_encoder(next_obs, v['actions'])
                target_z = next_z - cur_z
            else:
                if use_fp:
                    cur_z = self.target_te(obs).mean
                    next_z = self.target_te(next_obs).mean
                else:
                    cur_z = self.traj_encoder(obs).mean
                    next_z = self.traj_encoder(next_obs).mean

                target_z = next_z - cur_z

            if self.no_diff_in_rep:
                target_z = cur_z

            target_z /= temp

            if self.self_normalizing:
                target_z = target_z / target_z.norm(dim=-1, keepdim=True)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=v['options'].device) * self.scale_std
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                        new_z *= self.scale_radius
                    pairwise_scores = target_z @ new_z.t()
                else:
                    pairwise_scores = target_z @ v['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

                if self.symmetrize_log_sum_exp:
                    log_sum_exp = (log_sum_exp + torch.logsumexp(pairwise_scores.t(), dim=-1)) / 2.0

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)

                if relabel_rewards:
                    relabeled_options = torch.from_numpy(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, self._trans_minibatch_size)]).to(v['options'].device).float()
                    relabeled_masks = (relabeled_options - relabeled_options.mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                    relabeled_rewards = (target_z * relabeled_masks).sum(dim=1)

            else:
                inner = (target_z * v['options']).sum(dim=1)
                rewards = inner

                if relabel_rewards:
                    relabeled_options = np.random.randn(self._trans_minibatch_size, self.dim_option) * self.scale_std
                    if self.unit_length:
                        relabeled_options /= np.linalg.norm(relabeled_options, axis=-1, keepdims=True)
                        relabeled_options *= self.scale_radius
                    relabeled_options = torch.from_numpy(relabeled_options).to(v['options'].device).float()
                    relabeled_rewards = (target_z * relabeled_options).sum(dim=1)

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
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=v['options'].device) * self.scale_std
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                        new_z *= self.scale_radius
                    pairwise_scores = rep @ new_z.t()
                else:
                    pairwise_scores = rep @ v['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)
        else:
            target_dists = self.traj_encoder(next_obs) # NOTE: breaks if not inner product and using actions

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

            if self.diayn_include_baseline:
                rewards -= torch.log(torch.tensor(1/self.dim_option))

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards
        if relabel_rewards:
            v['relabeled_rewards'] = relabeled_rewards
            v['relabeled_options'] = relabeled_options
        if self.log_sum_exp:
            v['log_sum_exp'] = log_sum_exp

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v, temp=self.rep_temp)
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

            if self.no_diff_in_penalty:
                inside_l2 = phi_x
            else:
                inside_l2 = phi_y - phi_x

            cst_penalty = cst_dist - torch.square(inside_l2).sum(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)

            if self.use_l2_penalty:
                cst_penalty = -1 * torch.square(inside_l2).sum(dim=1)

            if self.use_bessel_penalty:
                if self.dim_option == 2:
                    gamma_fn = 1
                    numerator = 1
                    bessel_fn = torch.special.i0
                    denominator = 1
                elif self.dim_option == 4:
                    gamma_fn = 1
                    numerator = 2
                    bessel_fn = torch.special.i1
                    denominator = torch.norm(inside_l2, dim=-1)
                else:
                    raise NotImplementedError("Bessel function cannot be computed for dim_option != 2 or 4")
                
                cst_penalty = -1 * torch.log(gamma_fn * numerator * bessel_fn(torch.norm(inside_l2, dim=-1)) / denominator)

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
                'phi_diff_l2': torch.square(phi_y - phi_x).sum(dim=1).mean(),
                'phi_l2': torch.square(phi_x).sum(dim=1).mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        if self.relabel_critic_z:
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['relabeled_options'])
            next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['relabeled_options'])

            rewards = v['relabeled_rewards'] * self._reward_scale_factor
        else:
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
            next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

            rewards = v['rewards'] * self._reward_scale_factor

        if self.add_log_sum_exp_to_rewards:
            # recompute log sum exp since traj encoder has been updated
            target_z = v['next_z'] - v['cur_z']
            if self.sample_new_z:
                new_z = torch.randn(self.num_negative_z, self.dim_option, device=v['options'].device) * self.scale_std
                if self.unit_length:
                    new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    new_z *= self.scale_radius
                pairwise_scores = target_z @ new_z.t()
            else:
                pairwise_scores = target_z @ v['options'].t()
            log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            rewards -= self.infonce_lam * log_sum_exp

        if self.add_penalty_to_rewards:
            if self.reward_lam is not None:
                rewards += (self.reward_lam * v['cst_penalty']).detach()
            elif self.fixed_lam is not None:
                rewards += (self.fixed_lam * v['cst_penalty']).detach()

                if self.add_diff_norm_bonus:
                    rewards += self.fixed_lam * (torch.norm(v['next_z'], dim=-1) - torch.norm(v['cur_z'], dim=-1))**2

                if self.add_angle_bonus:
                    cos_theta = (v['cur_z'] * v['next_z']).sum(dim=-1) / (torch.norm(v['cur_z'], dim=-1) * torch.norm(v['next_z'], dim=-1))
                    eps = torch.norm(v['next_z'], dim=-1) - torch.norm(v['cur_z'], dim=-1)
                    rewards += self.fixed_lam * (2 * torch.square(v['cur_z']).sum(dim=-1) * (1 - cos_theta) + 2 * torch.norm(v['cur_z'], dim=-1) * eps * (1 - cos_theta))

                if self.add_norm_bonus:
                    rewards += (1 - self.discount) * torch.norm(v['cur_z'], dim=-1)

            else:
                x = v['obs']
                y = v['next_obs']
                phi_x = v['cur_z']
                phi_y = v['next_z']
                cst_dist = torch.ones_like(x[:, 0])
                cst_penalty = cst_dist - torch.square(phi_y - phi_x).sum(dim=1)
                cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
                rewards += (self.dual_lam.param.exp() * cst_penalty).detach()

        if self.contrastive:
            rewards[v['dones'] != 1] = 0
            rewards[v['dones'].bool()] = (v['next_z'] * v['options']).sum(dim=1)[v['dones'].bool()]

        if self.contrastive_every:
            if self.include_one_minus_gamma:
                if self.use_next_state:
                    rewards = (1 - self.discount) * (v['next_z'] * v['options']).sum(dim=1)
                else:
                    rewards = (1 - self.discount) * (v['cur_z'] * v['options']).sum(dim=1)
            else:
                if self.use_next_state:
                    rewards = (v['next_z'] * v['options']).sum(dim=1)
                else:
                    rewards = (v['cur_z'] * v['options']).sum(dim=1)

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=rewards,
            policy=self.option_policy,
            contrastive_every=self.contrastive_every,
            turn_off_dones=self.turn_off_dones,
            use_discrete_sac=self.use_discrete_sac
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        if self.relabel_actor_z:
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['relabeled_options'])
        else:
            processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
            contrastive_every=self.contrastive_every,
            use_discrete_sac=self.use_discrete_sac
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
            use_discrete_sac=self.use_discrete_sac
        )

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
            if self.uniform_z:
                random_options = np.random.uniform(low=-1.0, high=1.0, size=(self.num_random_trajectories, self.dim_option))
            else:
                random_options = np.random.randn(self.num_random_trajectories, self.dim_option) * self.scale_std
                if self.unit_length:
                    random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
                    random_options *= self.scale_radius
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

        # if not isinstance(runner._env.env, CraftaxWrapper):
        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        if self.metra_include_actions:
            last_actions = torch.stack([torch.from_numpy(ac[-1]).to(self.device) for ac in data['actions']])
            option_dists = self.traj_encoder(last_obs, last_actions)
            if self.inner:
                option_stddevs = torch.ones_like(option_dists.detach().cpu()).numpy()
            else:
                option_stddevs = option_dists.stddev.detach().cpu().numpy()
            option_means = option_dists.detach().cpu().numpy()
            option_samples = option_dists.detach().cpu().numpy()
        else:
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
                mean_length = 1. * self.scale_radius
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
                    radius = 1. * self.scale_radius if self.unit_length else 1.5
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
                        video_options *= self.scale_radius
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

        with torch.no_grad():
            if self.log_eval_return:
                epoch_data = self._flatten_data(data)
                tensors = {}
                self._update_rewards(tensors, epoch_data, temp=self.actor_temp)
                for key in tensors:
                    eval_option_metrics[key] = tensors[key].item()

                if self.record_corr_m:
                    corr_m = np.zeros((29, 2))
                    for s_dim in range(epoch_data['obs'].shape[-1]):
                        for z_dim in range(epoch_data['cur_z'].shape[-1]):
                            corr_m[s_dim, z_dim] = np.corrcoef(x=epoch_data['obs'][:, s_dim].cpu().numpy(), y=epoch_data['cur_z'][:, z_dim].cpu().numpy())[0, 1]

                    # save
                    np.save(os.path.join(runner._snapshotter.snapshot_dir, f'corr_m_{runner.step_itr}.npy'), corr_m)

                    YLABELS = [
                        'torso z coord',
                        'torso x orient',
                        'torso y orient',
                        'torso z orient',
                        'torso w orient',
                        'angle torso, first link front left',
                        'angle two links, front left',
                        'angle torso, first link front right',
                        'angle two links, front right',
                        'angle torso, first link back left',
                        'angle two links, back left',
                        'angle torso, first link back right',
                        'angle two links, back right',
                        'torso x coord vel',
                        'torso y coord vel',
                        'torso z coord vel',
                        'torso x coord ang vel',
                        'torso y coord ang vel',
                        'torso z coord ang vel',
                        'angle torso, front left link, av',
                        'angle front left links, av',
                        'angle torso, front right link, av',
                        'angle front right links, av',
                        'angle torso, back left link, av',
                        'angle back left links, av',
                        'angle torso, back right link, av',
                        'angle back right links, av',
                        'torso x coord',
                        'torso y coord',
                    ]

                    # plot
                    sns.heatmap(corr_m, cmap='coolwarm', center=0)
                    plt.yticks(ticks=list(map(lambda x: x + 0.5, range(29))), labels=YLABELS, rotation=0)

                    plt.xlabel('Z')
                    plt.ylabel('Observations')
                    plt.tight_layout()

                    plt.savefig(os.path.join(runner._snapshotter.snapshot_dir, 'plots', f'corr_m_{runner.step_itr}.pdf'))
                    plt.clf()

            if self.log_logp_vs_l2 and self.replay_buffer is not None and self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                # From replay buffer
                replay_samples = self._sample_replay_buffer(1000)

                self._plot_kde(runner, replay_samples, kde_type='phi', data_type='replay', bandwidth=1.0)
                self._plot_kde(runner, replay_samples, kde_type='phi', data_type='replay', bandwidth=0.5)
                self._plot_kde(runner, replay_samples, kde_type='phi', data_type='replay', bandwidth=2.0)

                self._plot_kde(runner, replay_samples, kde_type='s', data_type='replay', bandwidth=1.0)
                self._plot_kde(runner, replay_samples, kde_type='s', data_type='replay', bandwidth=0.5)
                self._plot_kde(runner, replay_samples, kde_type='s', data_type='replay', bandwidth=2.0)

                # From rollouts
                rollout_data = self._flatten_data(data)

                self._plot_kde(runner, rollout_data, kde_type='phi', data_type='rollout', bandwidth=1.0)
                self._plot_kde(runner, rollout_data, kde_type='phi', data_type='rollout', bandwidth=0.5)
                self._plot_kde(runner, rollout_data, kde_type='phi', data_type='rollout', bandwidth=2.0)

                self._plot_kde(runner, rollout_data, kde_type='s', data_type='rollout', bandwidth=1.0)
                self._plot_kde(runner, rollout_data, kde_type='s', data_type='rollout', bandwidth=0.5)
                self._plot_kde(runner, rollout_data, kde_type='s', data_type='rollout', bandwidth=2.0)
                
        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)

    def _plot_kde(self, runner, samples, kde_type: str = 'phi', bandwidth: float = 1.0, data_type: str = 'replay'):
        s1 = samples['obs'].to(self.device)
        s2 = samples['next_obs'].to(self.device)

        phi1 = self.traj_encoder(s1).mean
        phi2 = self.traj_encoder(s2).mean

        if kde_type == 'phi':
            phi1_phi2 = torch.cat([phi1, phi2], axis=-1).cpu().numpy()
            kde = KernelDensity(bandwidth=bandwidth).fit(phi1_phi2)
            ys = -1 * kde.score_samples(phi1_phi2)
            xs = torch.square(phi2 - phi1).sum(dim=-1).cpu().numpy()
            ylabel = "$-\log \hat{p}(\phi(s), \phi(s'))$"
            
        elif kde_type == 's':
            s1_s2 = torch.cat([s1, s2], axis=-1).cpu().numpy()
            kde = KernelDensity(bandwidth=bandwidth).fit(s1_s2)
            ys = -1 * kde.score_samples(s1_s2)
            xs = torch.square(phi2 - phi1).sum(dim=-1).cpu().numpy()
            ylabel = "$-\log \hat{p}(s, s')$"

        sns.regplot(x=xs, y=ys)

        plt.xlabel("$||\phi(s') - \phi(s)||^2_2$")
        plt.ylabel(ylabel)

        file_name = os.path.join(runner._snapshotter.snapshot_dir, 'plots', f'logp_vs_l2_{data_type}_{kde_type}_B{bandwidth}_{runner.step_itr}.pdf')
        plt.savefig(file_name)
        plt.clf()