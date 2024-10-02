"""This modules creates a continuous Q-function network."""

import torch
from mamba_ssm.models.config_mamba import MambaConfig

from garage.torch.modules import MLPModule

from networks.mamba_core import MambaCore
from networks.action_embedder import ActionEmbedder
from networks.state_embedder import StateEmbedder

class ContinuousMLPQFunctionEx(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, obs_dim, action_dim, output_dim: int = 1, recurrent: bool = False, seq_model_hdim: int = 256, seq_model_num_layers: int = 3, seq_model_type: str = 'mamba', **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.recurrent = recurrent

        MLPModule.__init__(self,
                           input_dim=self.obs_dim + self.action_dim if not recurrent else seq_model_hdim + self.action_dim,
                           output_dim=output_dim,
                           **kwargs)
        
        if self.recurrent:
            self.action_embedder = ActionEmbedder(action_dim, seq_model_hdim // 4)
            self.obs_embedder = StateEmbedder(obs_dim, seq_model_hdim // 4 * 3)
            # self.obs_embedder = StateEmbedder(obs_dim, seq_model_hdim)

            if seq_model_type == 'mamba':
                mamba_config = MambaConfig(
                    d_model=seq_model_hdim,
                    n_layer=seq_model_num_layers
                )
                self.seq_model = MambaCore(mamba_config)
            elif seq_model_type == 'lstm':
                self.seq_model = torch.nn.LSTM(
                    input_size=seq_model_hdim,
                    hidden_size=seq_model_hdim,
                    num_layers=seq_model_num_layers,
                    batch_first=False,
                )
            else:
                raise NotImplementedError

    def forward(self, observations, actions, prev_actions=None):
        """Return Q-value(s)."""

        if self.recurrent:
            observs = observations
            ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
            # return the hidden states (T+1, B, dim)
            hidden_states = self.get_hidden_states(
                prev_actions=prev_actions, observs=observs
            )

            if actions.shape[0] == observs.shape[0]:
                return super().forward(torch.cat([hidden_states, actions], -1))
                # return super().forward(torch.cat([observs, actions], -1))
            else:
                return super().forward(torch.cat([hidden_states[:-1], actions], -1))
                # return super().forward(torch.cat([observs[:-1], actions], -1))

        return super().forward(torch.cat([observations, actions], 1))
    
    def get_hidden_states(self, prev_actions, observs):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_s = self.obs_embedder(observs)
        inputs = torch.cat((input_a, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        inputs = inputs.transpose(0, 1)
        output = self.seq_model(inputs).last_hidden_state
        output = output.transpose(0, 1)

        return output
