"""This modules creates a continuous Q-function network."""

import torch
from torch import nn

from garage.torch.modules import MLPModule


class PlasticityInjectionBranch(nn.Module):
    """Function-preserving fresh MLP branch with selectable hidden width."""

    def __init__(self, input_dim, max_width, active_width, seed):
        super().__init__()
        self.max_width = int(max_width)
        self.register_buffer(
            'active_width_tensor', torch.tensor(int(active_width)))

        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        input_bound = (6. / (input_dim + self.max_width)) ** 0.5
        hidden_bound = (3. / self.max_width) ** 0.5
        output_bound = (6. / (self.max_width + 1)) ** 0.5

        input_weight = torch.empty(self.max_width, input_dim)
        input_weight.uniform_(-input_bound, input_bound, generator=generator)
        input_bias = torch.zeros(self.max_width)
        hidden_weight = torch.empty(self.max_width, self.max_width)
        hidden_weight.uniform_(
            -hidden_bound, hidden_bound, generator=generator)
        hidden_bias = torch.zeros(self.max_width)
        output_weight = torch.empty(1, self.max_width)
        output_weight.uniform_(
            -output_bound, output_bound, generator=generator)

        self.input_weight = nn.Parameter(input_weight)
        self.input_bias = nn.Parameter(input_bias)
        self.hidden_weight = nn.Parameter(hidden_weight)
        self.hidden_bias = nn.Parameter(hidden_bias)
        self.output_weight = nn.Parameter(output_weight)

        self.register_buffer('frozen_input_weight', input_weight.clone())
        self.register_buffer('frozen_input_bias', input_bias.clone())
        self.register_buffer('frozen_hidden_weight', hidden_weight.clone())
        self.register_buffer('frozen_hidden_bias', hidden_bias.clone())
        self.register_buffer('frozen_output_weight', output_weight.clone())

    @property
    def active_width(self):
        """Return the active prefix width."""
        return int(self.active_width_tensor.item())

    def set_active_width(self, width):
        """Select the prefix of fresh hidden units used by this branch."""
        self.active_width_tensor.fill_(int(width))

    def trainable_parameters(self):
        """Return only the trainable half of the injection pair."""
        return [
            self.input_weight,
            self.input_bias,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
        ]

    @staticmethod
    def _mlp(inputs, width, input_weight, input_bias, hidden_weight,
             hidden_bias, output_weight):
        hidden = torch.relu(
            torch.matmul(inputs, input_weight[:width].t()) +
            input_bias[:width])
        hidden = torch.relu(
            torch.matmul(hidden, hidden_weight[:width, :width].t()) +
            hidden_bias[:width])
        return torch.matmul(hidden, output_weight[:, :width].t())

    def forward(self, inputs):
        width = self.active_width
        trainable = self._mlp(
            inputs, width, self.input_weight, self.input_bias,
            self.hidden_weight, self.hidden_bias, self.output_weight)
        frozen = self._mlp(
            inputs, width, self.frozen_input_weight, self.frozen_input_bias,
            self.frozen_hidden_weight, self.frozen_hidden_bias,
            self.frozen_output_weight)
        return trainable - frozen


class ContinuousMLPQFunction(MLPModule):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, infer = False, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """

        self._multi_input = False
        self._env_spec = env_spec
        if isinstance(env_spec, list):
            self._multi_input = True
            self._obs_dim = 0
            self._action_dim = 0
            for spec in env_spec:
                self._obs_dim += spec.observation_space.flat_dim
                self._action_dim += spec.action_space.flat_dim
            
            self._total_dim = self._obs_dim + self._action_dim
            
            self._zero_pad_per_task = []
            prev_dim = 0
            for spec in env_spec:
                obs_dim = spec.observation_space.flat_dim
                action_dim = spec.action_space.flat_dim
                total_dim = obs_dim + action_dim
                
                self._zero_pad_per_task.append(nn.ConstantPad1d((prev_dim, self._total_dim- (prev_dim + total_dim)),0))
                prev_dim += total_dim



        else:
            self._obs_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=1,
                           **kwargs)
        
        self._feature = None
        self._plasticity_injection_branches = nn.ModuleList()
        feature_size = kwargs['hidden_sizes'][-1]
        if infer:
            self._infer = nn.Linear(feature_size, feature_size)
        

    # pylint: disable=arguments-differ
    def forward(self, observations, actions, seq_idx = None):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        input = torch.cat([observations, actions], 1)
        if self._multi_input:
            
            if isinstance(seq_idx, int):
                zero_pad = self._zero_pad_per_task[seq_idx]
                input = zero_pad(input)
            else:
                batch_size = len(observations)
                new_input = []
                for i in range(batch_size):
                    idx = seq_idx[i]
                    zero_padding = self._zero_pad_per_task[idx]
                    obs = observations[i]
                    new_obs = zero_padding(obs)
                    new_input.append(new_obs)
                input = torch.stack(new_input)
                
        ret = super().forward(input)
        if isinstance(ret, list):
            ret = ret[0]
        for branch in self._plasticity_injection_branches:
            ret = ret + branch(input)
        return ret

    def install_plasticity_injection(self, max_width, active_width, seed):
        """Append a zero-output fresh-minus-frozen critic branch."""
        device = next(self.parameters()).device
        branch = PlasticityInjectionBranch(
            self._obs_dim + self._action_dim,
            max_width=max_width,
            active_width=active_width,
            seed=seed).to(device)
        self._plasticity_injection_branches.append(branch)
        return branch

    def append_plasticity_injection(self, branch):
        """Append an already constructed plasticity-injection branch."""
        device = next(self.parameters()).device
        self._plasticity_injection_branches.append(branch.to(device))

    def plasticity_injection_inputs(self, observations, actions,
                                    seq_idx=None):
        """Return the state-action input seen by residual branches."""
        inputs = torch.cat([observations, actions], 1)
        if self._multi_input and isinstance(seq_idx, int):
            inputs = self._zero_pad_per_task[seq_idx](inputs)
        return inputs

    # InFeR: must be called after forward()
    def get_feature_prediction(self):
        return self._infer(self._feature)
    
    def update_kb(self, policy_kb):

        kb_state_dict = policy_kb.state_dict().clone()
        self.policy_kb.load_state_dict(kb_state_dict)

        print('(garage/torch/q_functions/continuous_mlp_q_functino.py) Knowledge Base Updated!')
