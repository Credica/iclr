import numpy as np
import torch


from garage.torch.algos import MTSAC, PPO
from garage.torch import as_torch_dict

import wandb

class Finetuning_SAC(MTSAC):

    def __init__(self, **sac_kwargs):
        super().__init__(**sac_kwargs)


class SpectralRegularizedSAC(Finetuning_SAC):
    """Finetuning SAC with ICLR 2025 layer spectral regularization."""

    def __init__(self, actor_coef=1e-4, critic_coef=1e-4,
                 power_iterations=1, **sac_kwargs):
        super().__init__(
            spectral_regularization=True,
            spectral_actor_coef=actor_coef,
            spectral_critic_coef=critic_coef,
            spectral_power_iterations=power_iterations,
            **sac_kwargs)
        
    
class Finetuning_PPO(PPO):

    def __init__(self, **ppo_kwargs):
        super().__init__(**ppo_kwargs)
