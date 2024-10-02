import torch
from torch.nn import functional as F


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)

    return actions + clip

def update_loss_qf_recurrent(
        algo, 
        tensors, 
        observs,
        actions,
        dones,
        rewards,
        policy,
        masks,
        turn_off_dones: bool = False,
):
    num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
    with torch.no_grad():
        # get new_actions, new_log_probs
        # (T+1, B, dim) including reaction to last obs
        new_action_dists, *_ = policy(observs, prev_actions=actions)
        if hasattr(new_action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = new_action_dists.rsample_with_pre_tanh_value()
            new_log_probs = new_action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = new_action_dists.rsample()
            new_actions = _clip_actions(algo, new_actions)
            new_log_probs = new_action_dists.log_prob(new_actions)

        next_q1 = algo.target_qf1(observs, new_actions, prev_actions=actions) # (T + 1, B, 1)
        next_q2 = algo.target_qf2(observs, new_actions, prev_actions=actions) # (T + 1, B, 1)

        min_next_q_target = torch.min(next_q1, next_q2)
        min_next_q_target += alpha * (-new_log_probs.unsqueeze(-1))  # (T+1, B, 1)

        # q_target: (T, B, 1)
        if turn_off_dones:
            dones[...] = 0
        q_target = rewards + (1.0 - dones) * algo.discount * min_next_q_target  # next q
        q_target = q_target[1:]  # (T, B, 1)

    # Q(h(t), a(t)) (T, B, 1)
    q1_pred = algo.qf1(observs, actions[1:], prev_actions=actions)  # (T, B, 1)
    q2_pred = algo.qf2(observs, actions[1:], prev_actions=actions)  # (T, B, 1)

    # masked Bellman error: masks (T,B,1) ignore the invalid error
    # this is not equal to masks * q1_pred, cuz the denominator in mean()
    # 	should depend on masks > 0.0, not a constant B*T
    q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
    q_target = q_target * masks
    qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid * 0.5 # TD error
    qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid * 0.5 # TD error

    tensors.update({
        'QTargetsMean': q_target.sum() / num_valid,
        'QTdErrsMean': ((q_target - q1_pred).sum() / num_valid + (q_target - q2_pred).sum() / num_valid) / 2,
        'LossQf1': qf1_loss,
        'LossQf2': qf2_loss,
    })

def update_loss_qf(
        algo, tensors, v,
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy,
        contrastive_every: bool = False,
        use_discrete_sac: bool = False,
        turn_off_dones: bool = False,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    if use_discrete_sac:
        action_ids = action_ids = torch.argmax(actions.long(),dim=-1)
        q1_pred = algo.qf1(obs).gather(1, action_ids.view(-1, 1)).squeeze()
        q2_pred = algo.qf2(obs).gather(1, action_ids.view(-1, 1)).squeeze()
    else:
        q1_pred = algo.qf1(obs, actions).flatten()
        q2_pred = algo.qf2(obs, actions).flatten()

    if contrastive_every:
        q1_pred += (v['cur_z'] * v['options']).sum(dim=1).detach()
        q2_pred += (v['cur_z'] * v['options']).sum(dim=1).detach()

    next_action_dists, *_ = policy(next_obs)
    if use_discrete_sac:
        act_probs = next_action_dists.probs
        logits = next_action_dists.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        target_q_values = torch.min(
            algo.target_qf1(next_obs),
            algo.target_qf2(next_obs),
        )
        target_q_values = target_q_values - alpha * log_probs
        target_q_values = (act_probs * target_q_values).sum(dim=-1)
        target_q_values = target_q_values * algo.discount
    else:
        if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
            new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        else:
            new_next_actions = next_action_dists.rsample()
            new_next_actions = _clip_actions(algo, new_next_actions)
            new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

        if contrastive_every:
            reward_shaping_term = (v['next_z'] * v['options']).sum(dim=1).detach()
            target_q_values = torch.min(
                algo.target_qf1(next_obs, new_next_actions).flatten() + reward_shaping_term,
                algo.target_qf2(next_obs, new_next_actions).flatten() + reward_shaping_term,
            )
        else:
            target_q_values = torch.min(
                algo.target_qf1(next_obs, new_next_actions).flatten(),
                algo.target_qf2(next_obs, new_next_actions).flatten(),
            )

        target_q_values = target_q_values - alpha * new_next_action_log_probs
        target_q_values = target_q_values * algo.discount

    with torch.no_grad():
        if turn_off_dones:
            dones[...] = 0
        q_target = rewards + target_q_values * (1. - dones)

    # critic loss weight: 0.5
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    tensors.update({
        'QTargetsMean': q_target.mean(),
        'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
    })

def update_loss_sacp_recurrent(
        algo, tensors, v,
        observs,
        actions,
        policy,
        masks
):
    num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    # get new_actions, log_probs
    # (T+1, B, A)
    action_dists, *_ = policy(observs, prev_actions=actions)
    if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        log_probs = action_dists.log_prob(new_actions)
    log_probs = log_probs.unsqueeze(-1)

    q1 = algo.qf1(observs, new_actions, prev_actions=actions)  # (T+1, B, 1)
    q2 = algo.qf2(observs, new_actions, prev_actions=actions)  # (T+1, B, 1)
    
    min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

    policy_loss = -min_q_new_actions
    policy_loss += alpha * log_probs
    policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

    # masked policy_loss
    policy_loss = (policy_loss * masks).sum() / num_valid

    tensors.update({
        'SacpNewActionLogProbMean': (log_probs[:-1] * masks).sum() / num_valid,
        'LossSacp': policy_loss,
    })

    v.update({
        'new_action_log_probs': ((log_probs[:-1] * masks).sum() / num_valid).item(),
    })

def update_loss_sacp(
        algo, tensors, v,
        obs,
        policy,
        contrastive_every: bool = False,
        use_discrete_sac: bool = False,
        use_recurrent_sac: bool = False
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    if use_recurrent_sac:
        action_dists, *_ = policy(v['obs'], past_obs=v['obs_history'], past_actions=v['act_history'], option=v['options'])
    else:
        action_dists, *_ = policy(obs)
    act_probs = None
    if use_discrete_sac:
        act_probs = action_dists.probs
        logits = action_dists.logits
        new_action_log_probs = torch.log_softmax(logits, dim=-1)

        min_q_values = torch.min(
            algo.qf1(obs),
            algo.qf2(obs),
        )
        loss_sacp = (act_probs * (alpha * new_action_log_probs - min_q_values)).sum(dim=-1).mean()
        
    else:
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
            new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = action_dists.rsample()
            new_actions = _clip_actions(algo, new_actions)
            new_action_log_probs = action_dists.log_prob(new_actions)

        if contrastive_every:
            reward_shaping_term = (v['cur_z'] * v['options']).sum(dim=1).detach()
            min_q_values = torch.min(
                algo.qf1(obs, new_actions).flatten() + reward_shaping_term,
                algo.qf2(obs, new_actions).flatten() + reward_shaping_term,
            )
        else:
            if use_recurrent_sac:
                min_q_values = torch.min(
                    algo.qf1(v['obs'], new_actions, past_obs=v['obs_history'], past_actions=v['act_history'], option=v['options']).flatten(),
                    algo.qf2(v['obs'], new_actions, past_obs=v['obs_history'], past_actions=v['act_history'], option=v['options']).flatten(),
                )
            else:
                min_q_values = torch.min(
                    algo.qf1(obs, new_actions).flatten(),
                    algo.qf2(obs, new_actions).flatten(),
                )

        loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    tensors.update({
        'SacpNewActionLogProbMean': new_action_log_probs.mean(),
        'LossSacp': loss_sacp,
    })

    v.update({
        'new_action_log_probs': new_action_log_probs,
        'act_probs': act_probs
    })


def update_loss_alpha(
        algo, tensors, v, use_discrete_sac: bool = False, use_recurrent_sac: bool = False
):
    if use_discrete_sac:
        loss_alpha = (v['act_probs'].detach() * (-algo.log_alpha.param * (
                v['new_action_log_probs'].detach() + algo._target_entropy
        ))).sum(dim=-1).mean()
    else:
        if use_recurrent_sac:
            loss_alpha = -algo.log_alpha.param * (
                v['new_action_log_probs'] + algo._target_entropy
            )
        else:
            loss_alpha = (-algo.log_alpha.param * (
                v['new_action_log_probs'].detach() + algo._target_entropy
            )).mean()

    tensors.update({
        'Alpha': algo.log_alpha.param.exp(),
        'LossAlpha': loss_alpha,
    })


def update_targets(algo):
    """Update parameters in the target q-functions."""
    target_qfs = [algo.target_qf1, algo.target_qf2]
    qfs = [algo.qf1, algo.qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) +
                               param.data * algo.tau)
