#!/usr/bin/env python3
import tempfile

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import argparse
import datetime
import functools
import os
import sys
import platform
import torch.multiprocessing as mp
import gym as gymnasium
# import minigrid

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal
# from garaged.src.garage.torch.modules import MLPModule

# from minigrid.envs import FourRoomsEnv, EmptyEnv
# from minigrid.wrappers import FullyObsWrapper

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.envs.child_policy_env import ChildPolicyEnv
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx
from garagei.torch.modules.categorical_mlp_module_ex import CategoricalMLPModuleEx
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.q_functions.discrete_mlp_q_function import DiscreteMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from iod.metra import METRA
from iod.metra_sf import MetraSf
from iod.dads import DADS
from iod.crl import CRL
from iod.ppo import PPO
from iod.cic import CIC
from iod.utils import get_normalizer_preset
from envs.fetch_envs import FetchReachEnv
# from envs.sawyer_bin import SawyerBin
from envs.point_env import PointEnv
from iod.sac_jens import SAC as JensSAC
from iod.sac import SAC
from iod.utils import get_normalizer_preset
# from envs.fourrooms_wrapper import FourRoomsWrapper


EXP_DIR = 'exp'
if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'


def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--wandb_entity', type=str, default='jtuyls')
    parser.add_argument('--wandb_mode', type=str, default='offline')
    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=0)

    parser.add_argument('--env', type=str, default='maze', choices=[
        'maze', 'half_cheetah', 'ant', 'dmc_cheetah', 'dmc_quadruped', 'dmc_humanoid', 'kitchen', 'fetch_reach', 'ant_with_goals', 'sawyer_bin', 'point', 'fourrooms', 'empty', 'empty_16x16', 'empty_random_6x6', 'half_cheetah_hurdle', 'ant_nav_prime', 'half_cheetah_goal', 'dmc_quadruped_goal', 'dmc_humanoid_goal', 'robobin_image'
    ])
    parser.add_argument('--frame_stack', type=int, default=None)

    parser.add_argument('--max_path_length', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_minibatch_size', type=int, default=256)
    parser.add_argument('--trans_optimization_epochs', type=int, default=200)

    parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    parser.add_argument('--n_epochs_per_log', type=int, default=25)
    parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_random_trajectories', type=int, default=48)
    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)

    parser.add_argument('--dim_option', type=int, default=2)

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--algo', type=str, default='metra', choices=['metra', 'dads', 'crl', 'sac', 'metra_sf', 'jens_sac', 'ppo', 'cic'])

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=300000)

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    parser.add_argument('--num_alt_samples', type=int, default=100)
    parser.add_argument('--split_group', type=int, default=65536)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])  # Only for continuous skills

    parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one'])
    parser.add_argument('--dual_lr', type=float, default=None)

    parser.add_argument('--log_eval_return', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_l2_penalty', type=int, default=0, choices=[0, 1])
    parser.add_argument('--self_normalizing', type=int, default=0, choices=[0, 1])
    parser.add_argument('--log_sum_exp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--symmetrize_log_sum_exp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--add_log_sum_exp_to_rewards', type=int, default=0, choices=[0, 1])
    parser.add_argument('--record_corr_m', type=int, default=0, choices=[0, 1])
    parser.add_argument('--contrastive', type=int, default=0, choices=[0, 1])
    parser.add_argument('--contrastive_every', type=int, default=0, choices=[0, 1])
    parser.add_argument('--fixed_lam', type=float, default=None)
    parser.add_argument('--use_mse', type=int, default=0, choices=[0, 1])
    parser.add_argument('--metra_rep', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_metra_penalty', type=int, default=0, choices=[0, 1])
    parser.add_argument('--metra_include_actions', type=int, default=0, choices=[0, 1])
    parser.add_argument('--crl_standardize_output', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_layer_norm', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_info_nce', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_oracle_goals', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_half_random_goals', default=0, type=int, choices=[0, 1])
    parser.add_argument('--add_penalty_to_rewards', default=0, type=int, choices=[0, 1])
    parser.add_argument('--no_diff_in_penalty', default=0, type=int, choices=[0, 1])
    parser.add_argument('--no_diff_in_rep', default=0, type=int, choices=[0, 1])
    parser.add_argument('--include_one_minus_gamma', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_positive_log_sum_exp', default=0, type=int, choices=[0, 1])
    parser.add_argument('--scale_std', default=1.0, type=float)
    parser.add_argument('--use_next_state', default=0, type=int, choices=[0, 1])
    parser.add_argument('--put_remaining_mass_on_last_state', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_discrete_sac', type=int, default=0, choices=[0, 1])
    parser.add_argument('--turn_off_dones', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_goal_rep', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ant_fix_goal', type=int, default=0, choices=[0, 1])
    parser.add_argument('--ant_model_path', type=str, default='ant.xml', choices=['ant.xml', 'ant_d4rl.xml'])
    parser.add_argument('--ant_coord_only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--goal_dim', default=2, type=int)
    parser.add_argument('--add_norm_exploration', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_diff_term', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_use_td', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_use_infonce_repr', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_geometric_entropy', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_exp_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--metra_mlp_rep', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_use_contrastive', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_freeze_traj_encoder', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sf_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--minigrid_use_po', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_recurrent_sac', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seq_model_hdim', type=int, default=256)
    parser.add_argument('--seq_model_num_layers', type=int, default=6)
    parser.add_argument('--sampled_seq_len', type=int, default=10)
    parser.add_argument('--seq_model_type', type=str, default='mamba', choices=['mamba', 'lstm'])
    parser.add_argument('--log_logp_vs_l2', type=int, default=0, choices=[0, 1])
    parser.add_argument('--reward_lam', type=float, default=None)
    parser.add_argument('--add_diff_norm_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--add_angle_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--add_norm_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_zero_shot_goals', type=int, default=50)
    parser.add_argument('--scale_radius', type=float, default=1.0)

    # Hierarchical control flags
    parser.add_argument('--cp_path', type=str, default=None)
    parser.add_argument('--cp_path_idx', type=int, default=None)  # For exp name
    parser.add_argument('--cp_multi_step', type=int, default=1)
    parser.add_argument('--cp_unit_length', type=int, default=0)

    parser.add_argument('--downstream_reward_type', type=str, default='esparse')
    parser.add_argument('--downstream_num_goal_steps', type=int, default=50)

    parser.add_argument('--goal_range', type=float, default=50)
    parser.add_argument('--policy_type', type=str, default='gaussian', choices=['gaussian', 'categorical'])

    parser.add_argument('--eval_goal_metrics', type=int, default=0, choices=[0, 1])

    parser.add_argument('--relabel_critic_z', type=int, default=0, choices=[0, 1])
    parser.add_argument('--relabel_actor_z', type=int, default=0, choices=[0, 1])

    parser.add_argument('--use_bessel_penalty', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample_new_z', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_negative_z', type=int, default=256)

    parser.add_argument('--use_fp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--no_actor_ent_bonus', type=int, default=0, choices=[0, 1])
    parser.add_argument('--infonce_lam', type=float, default=1.0)

    parser.add_argument('--rep_temp', type=float, default=1.0)
    parser.add_argument('--actor_temp', type=float, default=1.0)
    parser.add_argument('--diayn_include_baseline', type=int, default=0, choices=[0, 1])

    # only for DADS
    parser.add_argument('--uniform_z', default=0, type=int, choices=[0, 1])

    # only for CIC
    parser.add_argument('--cic_temp', type=float, default=0.5)
    parser.add_argument('--cic_alpha', type=float, default=1.0)
    parser.add_argument('--apt_knn_k', type=int, default=16)
    parser.add_argument('--apt_use_icm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--apt_rms', type=int, default=1, choices=[0, 1])

    # only for LSD, CIC
    parser.add_argument('--alive_reward', type=float, default=None)
    parser.add_argument('--dual_dist_scaling', type=str, default='geom', choices=['none', 'geom'])
    parser.add_argument('--const_scaler', type=float, default=1.)
    parser.add_argument('--wdm', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_cpc', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_idz', type=int, default=1, choices=[0, 1])
    parser.add_argument('--wdm_ids', type=int, default=0, choices=[0, 1])
    parser.add_argument('--wdm_diff', type=int, default=1, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--joint_train', type=int, default=1, choices=[0, 1])


    return parser


args = get_argparser().parse_args()
g_start_time = int(datetime.datetime.now().timestamp())


def get_exp_name():
    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name += '_' + args.env
    exp_name += '_' + args.algo

    return exp_name, exp_name_prefix


def get_log_dir():
    exp_name, exp_name_prefix = get_exp_name()
    assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f'The following path already exists: {log_dir}'

    return log_dir


def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     const_std=False,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,
                                     min_std=1e-6,
                                     max_std=None,
                                     spectral_normalization=False,
                                     **kwargs):
    module_kwargs = dict()
    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(dict(
            learn_std=False,
            init_std=init_std,
        ))
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(dict(
            std_hidden_sizes=hidden_sizes,
            std_hidden_nonlinearity=hidden_nonlinearity,
            std_hidden_w_init=w_init,
            std_output_w_init=w_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
        ))

    module_kwargs.update(dict(
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=w_init,
        output_w_init=w_init,
        std_parameterization='exp',
        bias=True,
        spectral_normalization=args.spectral_normalization or spectral_normalization,
        **kwargs,
    ))
    return module_cls, module_kwargs


def make_env(args, max_path_length: int):
    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)
    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100, model_path=args.ant_model_path)
    elif args.env == 'ant_with_goals':
        from envs.mujoco.ant_with_goals_env import AntWithGoalsEnv
        env_kwargs = {
            'render_hw': 100,
            'goal_limit': args.eval_plot_axis[1],
            'model_path': args.ant_model_path,
            'coord_only': args.ant_coord_only,
        }
        if args.ant_fix_goal:
            env_kwargs['fixed_goal'] = [args.eval_plot_axis[1], args.eval_plot_axis[1]]
        env = AntWithGoalsEnv(**env_kwargs)
    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  # Only support pixel-based environments
        if 'dmc_cheetah' in args.env:
            env = dmc.make('cheetah_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif 'dmc_quadruped' in args.env:
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif 'dmc_humanoid' in args.env:
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
        
        if args.env in ['dmc_quadruped_goal', 'dmc_humanoid_goal']:
            from envs.custom_dmc_tasks.goal_wrappers import GoalWrapper

            env = GoalWrapper(
                env,
                max_path_length=max_path_length,
                goal_range=args.goal_range,
                num_goal_steps=args.downstream_num_goal_steps,
            )
            cp_num_truncate_obs = 2

    elif args.env.startswith('robobin'):
        sys.path.append('lexa')
        from envs.lexa.robobin import MyRoboBinEnv
        if args.env == 'robobin':
            env = MyRoboBinEnv(log_per_goal=True)
        elif args.env == 'robobin_image':
            env = MyRoboBinEnv(obs_type='image', log_per_goal=True)

    elif args.env == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder  # Only support pixel-based environments
        env = MyKitchenEnv(log_per_goal=True)
    elif args.env == 'fetch_reach':
        env = FetchReachEnv()
    elif args.env == 'sawyer_bin':
        env = SawyerBin()
    elif args.env == 'point':
        env = PointEnv()
    # elif args.env == 'fourrooms':
    #     env = FourRoomsEnv(render_mode='rgb_array')
    #     if not args.minigrid_use_po:
    #         env = FullyObsWrapper(env)
    #     env = FourRoomsWrapper(env)
    # elif args.env == 'empty':
    #     env = gymnasium.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
    #     if not args.minigrid_use_po:
    #         env = FullyObsWrapper(env)
    #     env = FourRoomsWrapper(env, continuous=(not args.use_discrete_sac))
    # elif args.env == 'empty_16x16':
    #     env = gymnasium.make('MiniGrid-Empty-16x16-v0', render_mode='rgb_array', agent_start_pos=(8, 8))
    #     if not args.minigrid_use_po:
    #         env = FullyObsWrapper(env)
    #     env = FourRoomsWrapper(env, continuous=(not args.use_discrete_sac))
    # elif args.env == 'empty_random_6x6':
    #     env = gymnasium.make('MiniGrid-Empty-Random-6x6-v0', render_mode='rgb_array')
    #     if not args.minigrid_use_po:
    #         env = FullyObsWrapper(env)
    #     env = FourRoomsWrapper(env, continuous=(not args.use_discrete_sac))
    elif args.env == 'ant_nav_prime':
        from envs.mujoco.ant_nav_prime_env import AntNavPrimeEnv

        env = AntNavPrimeEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            num_goal_steps=args.downstream_num_goal_steps,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2
    elif args.env == 'half_cheetah_goal':
        from envs.mujoco.half_cheetah_goal_env import HalfCheetahGoalEnv
        env = HalfCheetahGoalEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 1
    elif args.env == 'half_cheetah_hurdle':
        from envs.mujoco.half_cheetah_hurdle_env import HalfCheetahHurdleEnv

        env = HalfCheetahHurdleEnv(
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2
    
    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        additional_dim = 0
        if args.env in ['ant_nav_prime']:
            normalizer_name = 'ant'
            additional_dim = cp_num_truncate_obs
        elif args.env in ['half_cheetah_goal', 'half_cheetah_hurdle']:
            normalizer_name = 'half_cheetah'
            additional_dim = cp_num_truncate_obs
        else:
            normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset', args.ant_coord_only)
        if additional_dim > 0:
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    if args.cp_path is not None:
        cp_path = args.cp_path
        if not os.path.exists(cp_path):
            import glob
            cp_path = glob.glob(cp_path)[0]
        cp_dict = torch.load(cp_path, map_location='cpu')

        env = ChildPolicyEnv(
            env,
            cp_dict,
            cp_action_range=1.5,
            cp_unit_length=args.cp_unit_length,
            cp_multi_step=args.cp_multi_step,
            cp_num_truncate_obs=cp_num_truncate_obs,
        )

    return env


@wrap_experiment(log_dir=get_log_dir(), name=get_exp_name()[0])
def run(ctxt=None):
    wandb.init(project='scaling-unsup-rl', entity=args.wandb_entity, group=args.run_group, name=get_exp_name()[0],
                config=vars(args), mode=args.wandb_mode)

    dowel.logger.log('ARGS: ' + str(args))
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)
    runner = OptionLocalRunner(ctxt)
    max_path_length = args.max_path_length
    if args.cp_path is not None:
        max_path_length *= args.cp_multi_step
    contextualized_make_env = functools.partial(make_env, args=args, max_path_length=max_path_length)
    env = contextualized_make_env()
    example_ob = env.reset()

    if args.encoder:
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        else:
            pixel_shape = (64, 64, 3)
            # if args.env == 'fourrooms':
            #     pixel_shape = (7, 7, 3)
            if args.env == 'empty':
                pixel_shape = (5, 5, 3)
    else:
        pixel_shape = None

    device = torch.device('cuda' if args.use_gpu else 'cpu')

    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim
    
    if args.encoder:
        def make_encoder(**kwargs):
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                kwargs = {}
                # if args.env == 'fourrooms':
                #     kwargs['cnn_kernels'] = (3, 3)
                encoder = make_encoder(**kwargs)

            return WithEncoder(encoder=encoder, module=module)

        kwargs = {}
        # if args.env == 'fourrooms':
        #     kwargs['cnn_kernels'] = (3, 3)
        example_encoder = make_encoder(**kwargs)
        module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        module_obs_dim = obs_dim

    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )
    module_kwargs = dict(
        hidden_sizes=master_dims,
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    if args.use_discrete_sac:
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    elif args.policy_type == 'categorical':
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    else:
        module_cls = GaussianMLPTwoHeadedModuleEx
        module_kwargs.update(dict(
            max_std=np.exp(2.),
            normal_distribution_cls=TanhNormal,
            output_w_init=functools.partial(xavier_normal_ex, gain=1.),
            init_std=1.,
            recurrent=args.use_recurrent_sac,
            seq_model_hdim=args.seq_model_hdim,
            seq_model_num_layers=args.seq_model_num_layers,
            seq_model_type=args.seq_model_type,
        ))

    policy_q_input_dim = module_obs_dim + args.dim_option
    if args.env == 'fetch_reach' or args.env == 'ant_with_goals' or args.env == 'sawyer_bin' or args.env == 'point':
        if args.use_goal_rep:
            policy_q_input_dim = module_obs_dim - args.goal_dim + args.dim_option # when using goal rep
        else:
            policy_q_input_dim = module_obs_dim # when using raw goals that equal state dimension

    if args.algo in ['sac', 'ppo']:
        policy_q_input_dim = module_obs_dim

    # if args.use_recurrent_sac:
    #     policy_q_input_dim = args.seq_model_hdim

    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )
    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    policy_kwargs['recurrent'] = args.use_recurrent_sac
    option_policy = PolicyEx(**policy_kwargs)

    output_dim = args.dim_option

    # NOTE: we repurpose the trajectory encoder as the psi() function in CRL
    traj_encoder_obs_dim = module_obs_dim
    if args.env == 'fetch_reach' or args.env == 'ant_with_goals' or args.env == 'sawyer_bin' or args.env == 'point':
        traj_encoder_obs_dim = args.goal_dim
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=traj_encoder_obs_dim,
        output_dim=output_dim,
        layer_normalization=args.use_layer_norm
    )
    traj_encoder = module_cls(**module_kwargs)
    if args.encoder:
        if args.spectral_normalization:
            te_encoder = make_encoder(spectral_normalization=True)
        else:
            te_encoder = None
        traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)    

    if args.metra_include_actions:
        traj_encoder = ContinuousMLPQFunctionEx(
            obs_dim=traj_encoder_obs_dim,
            action_dim=action_dim,
            output_dim=output_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            traj_encoder = with_encoder(traj_encoder)

    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    # Skill dynamics do not support pixel obs
    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )
    if args.algo == 'dads':
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    if args.algo == 'cic':
        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option * 2,
            output_dim=args.dim_option,
            layer_normalization=False, #args.phi_layer_norm,
        )
        pred_net = module_cls(**module_kwargs)

        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option,
            output_dim=output_dim,
            layer_normalization=False, #args.phi_layer_norm,
            spectral_normalization=True,
        )
        z_encoder = module_cls(**module_kwargs)
    else:
        pred_net = None

    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    if pred_net is not None:
        te_params = list(traj_encoder.parameters()) + list(z_encoder.parameters()) + list(pred_net.parameters())
    else:
        te_params = list(traj_encoder.parameters())

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': te_params, 'lr': _finalize_lr(args.lr_te)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args.dual_lr)},
        ]),
    }

    if skill_dynamics is not None:
        optimizers.update({
            'skill_dynamics': torch.optim.Adam([
                {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })
    if dist_predictor is not None:
        optimizers.update({
            'dist_predictor': torch.optim.Adam([
                {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(args.sac_max_buffer_size), 
        pixel_shape=pixel_shape,
        sample_goals=(args.algo == 'crl' or args.algo == 'metra_sf'),
        discount=args.sac_discount,
        put_remaining_mass_on_last_state=args.put_remaining_mass_on_last_state,
        recurrent=args.use_recurrent_sac,
        sampled_seq_len=args.sampled_seq_len
    )

    qf1 = None
    qf2 = None
    log_alpha = None
    phi_encoder = None
    if args.algo in ['metra', 'dads', 'sac', 'cic']:
        if args.use_discrete_sac:
            qf1 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf1 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
                recurrent=args.use_recurrent_sac,
                seq_model_hdim=args.seq_model_hdim,
                seq_model_num_layers=args.seq_model_num_layers,
                seq_model_type=args.seq_model_type,
            )
        if args.encoder:
            qf1 = with_encoder(qf1)

        if args.use_discrete_sac:
            qf2 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf2 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
                recurrent=args.use_recurrent_sac,
                seq_model_hdim=args.seq_model_hdim,
                seq_model_num_layers=args.seq_model_num_layers,
                seq_model_type=args.seq_model_type,
            )
        if args.encoder:
            qf2 = with_encoder(qf2)
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })
    
    elif args.algo == 'metra_sf':
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)

        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)

        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    elif args.algo == 'crl':
        phi_input_dim = module_obs_dim if (args.env != 'fetch_reach' and args.env != 'ant_with_goals' and args.env != 'sawyer_bin' and args.env != 'point') else module_obs_dim - args.goal_dim
        phi_encoder = ContinuousMLPQFunctionEx(
            obs_dim=phi_input_dim, # no goal
            action_dim=action_dim,
            output_dim=args.dim_option,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            phi_encoder = with_encoder(phi_encoder)

        optimizers.update({
            'phi_encoder': torch.optim.Adam([
                {'params': list(phi_encoder.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
        })

    # elif args.algo == 'ppo':
    #     # TODO: Currently not support pixel obs
    #     vf = MLPModule(
    #         input_dim=policy_q_input_dim,
    #         output_dim=1,
    #         hidden_sizes=master_dims,
    #         hidden_nonlinearity=nonlinearity or torch.relu,
    #         layer_normalization=False,
    #     )
    #     optimizers.update({
    #         'vf': torch.optim.Adam([
    #             {'params': vf.parameters(), 'lr': _finalize_lr(args.lr_op)},
    #         ]),
    #     })

    f_encoder = None
    if args.metra_mlp_rep:
        f_encoder = ContinuousMLPQFunctionEx(
            obs_dim=obs_dim,
            action_dim=obs_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )

        optimizers.update({
            'f_encoder': torch.optim.Adam([
                {'params': list(f_encoder.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
        })

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log,
        n_epochs_per_tb=args.n_epochs_per_log,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
        dim_option=args.dim_option,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name='METRA',
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,

        pixel_shape=pixel_shape
    )

    if args.algo == 'metra':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            log_logp_vs_l2=args.log_logp_vs_l2,
            reward_lam=args.reward_lam,
            add_diff_norm_bonus=args.add_diff_norm_bonus,
            add_angle_bonus=args.add_angle_bonus,
            add_norm_bonus=args.add_norm_bonus,
            log_eval_return=args.log_eval_return,
            use_l2_penalty=args.use_l2_penalty,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            symmetrize_log_sum_exp=args.symmetrize_log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            record_corr_m=args.record_corr_m,
            contrastive=args.contrastive,
            contrastive_every=args.contrastive_every,
            fixed_lam=args.fixed_lam,
            use_mse=args.use_mse,
            metra_include_actions=args.metra_include_actions,
            use_info_nce=args.use_info_nce,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_penalty=args.no_diff_in_penalty,
            no_diff_in_rep=args.no_diff_in_rep,
            include_one_minus_gamma=args.include_one_minus_gamma,
            scale_std=args.scale_std,
            use_next_state=args.use_next_state,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            relabel_actor_z=args.relabel_actor_z,
            relabel_critic_z=args.relabel_critic_z,
            use_bessel_penalty=args.use_bessel_penalty,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            use_fp=args.use_fp,
            infonce_lam=args.infonce_lam,
            rep_temp=args.rep_temp,
            actor_temp=args.actor_temp,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,
            scale_radius=args.scale_radius,
        )
        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = METRA(
            **algo_kwargs,
            **skill_common_args,
        )
    elif args.algo == 'metra_sf':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            use_diff_term=args.use_diff_term,
            sf_use_td=args.sf_use_td,
            sf_use_infonce_repr=args.sf_use_infonce_repr,
            sf_geometric_entropy=args.sf_geometric_entropy,
            sf_exp_bonus=args.sf_exp_bonus,
            sf_use_contrastive=args.sf_use_contrastive,
            sf_freeze_traj_encoder=args.sf_freeze_traj_encoder,
            sf_bonus=args.sf_bonus,
            log_eval_return=args.log_eval_return,
            use_l2_penalty=args.use_l2_penalty,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            symmetrize_log_sum_exp=args.symmetrize_log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            record_corr_m=args.record_corr_m,
            contrastive=args.contrastive,
            contrastive_every=args.contrastive_every,
            fixed_lam=args.fixed_lam,
            use_mse=args.use_mse,
            metra_rep=args.metra_rep,
            use_info_nce=args.use_info_nce,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_penalty=args.no_diff_in_penalty,
            no_diff_in_rep=args.no_diff_in_rep,
            include_one_minus_gamma=args.include_one_minus_gamma,
            scale_std=args.scale_std,
            use_next_state=args.use_next_state,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            relabel_actor_z=args.relabel_actor_z,
            relabel_critic_z=args.relabel_critic_z,
            use_bessel_penalty=args.use_bessel_penalty,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            use_fp=args.use_fp,
            no_actor_ent_bonus=args.no_actor_ent_bonus,
            infonce_lam=args.infonce_lam,
            rep_temp=args.rep_temp,
            actor_temp=args.actor_temp,
            num_zero_shot_goals=args.num_zero_shot_goals,
            scale_radius=args.scale_radius,
        )
        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = MetraSf(
            **algo_kwargs,
            **skill_common_args,
        )
    elif args.algo == 'cic':
        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = CIC(
            **algo_kwargs,
            **skill_common_args,

            pred_net=pred_net,
            z_encoder=z_encoder,
            cic_temp=args.cic_temp,
            cic_alpha=args.cic_alpha,
            knn_k=args.apt_knn_k,
            rms=args.apt_rms,
            alive_reward=args.alive_reward,

            dual_dist_scaling=args.dual_dist_scaling,
            const_scaler=args.const_scaler,
            wdm=args.wdm,
            wdm_cpc=args.wdm_cpc,
            wdm_idz=args.wdm_idz,
            wdm_ids=args.wdm_ids,
            wdm_diff=args.wdm_diff,
            aug=args.aug,
            joint_train=args.joint_train,
        )
    elif args.algo == 'jens_sac':
        algo_kwargs.update(
            use_recurrent_sac=args.use_recurrent_sac,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
        )
        algo = JensSAC(
            **algo_kwargs,
            **skill_common_args
        )
    elif args.algo == 'sac':
        algo_kwargs.update(
            use_discrete_sac=args.use_discrete_sac,
        )

        algo = SAC(
            **algo_kwargs,
            **skill_common_args
        )
    elif args.algo == 'dads': # TODO: check args here if we do run it ourselves
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            log_logp_vs_l2=args.log_logp_vs_l2,
            reward_lam=args.reward_lam,
            add_diff_norm_bonus=args.add_diff_norm_bonus,
            add_angle_bonus=args.add_angle_bonus,
            add_norm_bonus=args.add_norm_bonus,
            log_eval_return=args.log_eval_return,
            use_l2_penalty=args.use_l2_penalty,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            symmetrize_log_sum_exp=args.symmetrize_log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            record_corr_m=args.record_corr_m,
            contrastive=args.contrastive,
            contrastive_every=args.contrastive_every,
            fixed_lam=args.fixed_lam,
            use_mse=args.use_mse,
            metra_include_actions=args.metra_include_actions,
            use_info_nce=args.use_info_nce,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_penalty=args.no_diff_in_penalty,
            no_diff_in_rep=args.no_diff_in_rep,
            include_one_minus_gamma=args.include_one_minus_gamma,
            scale_std=args.scale_std,
            use_next_state=args.use_next_state,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            relabel_actor_z=args.relabel_actor_z,
            relabel_critic_z=args.relabel_critic_z,
            use_bessel_penalty=args.use_bessel_penalty,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            use_fp=args.use_fp,
            infonce_lam=args.infonce_lam,
            rep_temp=args.rep_temp,
            actor_temp=args.actor_temp,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
        )

        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = DADS(
            **algo_kwargs,
            **skill_common_args,
        )
    elif args.algo == 'crl':
        algo_kwargs.update(
            use_goal_rep=args.use_goal_rep,
            goal_dim=args.goal_dim,
            add_norm_exploration=args.add_norm_exploration,
            use_metra_penalty=args.use_metra_penalty,
            metra_rep=args.metra_rep,
            crl_standardize_output=args.crl_standardize_output,
            use_info_nce=args.use_info_nce,
            use_oracle_goals=args.use_oracle_goals,
            use_half_random_goals=args.use_half_random_goals,
            include_one_minus_gamma=args.include_one_minus_gamma,
            use_positive_log_sum_exp=args.use_positive_log_sum_exp,
        )
        skill_common_args.update(
            phi_encoder=phi_encoder
        )
        algo = CRL(
            **algo_kwargs,
            **skill_common_args,
        )  
    elif args.algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )
    else:
        raise NotImplementedError

    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)
    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )
    algo.option_policy.to(device)
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)


if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    run()
