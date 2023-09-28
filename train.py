"""Train all models."""

from offline.offline_train import HIVES
from utilities.utils import params_extraction, load_pretrained_models
from utilities.optimization import set_optimizers
from rl.agent import VaLS
from rl.sampler import Sampler, ReplayBuffer
from datetime import datetime
from models.nns import Critic, SkillPolicy, StateEncoder, StateDecoder
import wandb
import os
import torch
import numpy as np
import copy
import pickle



# When using kitchen, remember in D4RL the tasks are open microwave,
# move kettle, flip light switch, and open (slide) cabinet.

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

import pdb

os.environ['WANDB_SILENT'] = "true"

wandb.login()

# The ids for envs are:

ANT = 'antmaze-medium-diverse-v2'
KITCHEN = 'kitchen-mixed-v0'
RELOCATE = 'relocate-expert-v1'
PEN = 'pen-cloned-v1'

ENV_NAME = PEN
EXP_NAME = 'Layernorm'

PARENT_FOLDER = f'checkpoints/{ENV_NAME}'        
CASE_FOLDER = 'Baseline'


if 'ant' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(1.6e5) + 1,
                         'buffer_size': int(1.6e5) + 1,
                         'reset_frequency': 20000 if 'SERENE' in EXP_NAME else 40000,
                         'skill_length': 40,
                         'delta_skill': 12,
                         'test_freq': 40000}

elif 'relocate' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(4e4) + 1,
                         'buffer_size': int(4e4) + 1,
                         'reset_frequency': 5000 if 'SERENE' in EXP_NAME else 10000,
                         'skill_length': 10,
                         'delta_skill': 32,
                         'test_freq': int(40000)}

elif 'pen' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(4e4) + 1,
                         'buffer_size': int(4e4) + 1,
                         'reset_frequency': 5000 if 'SERENE' else 10000,
                         'skill_length': 10,
                         'delta_skill': 32,
                         'test_freq': 50000}
    
elif 'kitchen' in ENV_NAME:
    hyperparams_dict  = {'max_iterations': int(8e4) + 1,
                         'buffer_size': int(8e4) + 1,
                         'reset_frequency': 10000 if 'SERENE' else 20000,
                         'skill_length': 20,
                         'delta_skill': 32,
                         'test_freq': 80000}
    
else:
    raise ValueError('This environment is not registered in the code')


config = {
    # General hyperparams
    'device': 'cuda',
    'hidden_dim': 128,
    'env_id': ENV_NAME,
    
    # Offline hyperparams
    'vae_batch_size': 256,
    'vae_lr': 6e-4,
    'priors_lr': 6e-4,
    'epochs': 501,
    'beta': 0.01,
    'z_skill_dim': 12,

    # Online hyperparams  
    'batch_size': 256,
    'action_range': 4,
    'learning_rate': 3e-4,
    'discount': 0.97,
    'sing_val_factor': 2, 
    'gradient_steps': 4,
    'singular_val_k': 1,

    # Algo selection params
    'SERENE': True if 'SERENE' in EXP_NAME else False ,
    'Replayratio': True if 'Replayratio' in EXP_NAME else False,
    'Underparameter': True if 'Underparameter' in EXP_NAME else False,
    'SPiRL': True if 'SPiRL' in EXP_NAME else False,
    'Layernorm': True if 'Layernorm' in EXP_NAME else False,

    'folder_sing_vals': EXP_NAME,
    
    # Run params
    'train_offline': False,
    'train_rl': True,
    'load_offline_models': True,
    'load_rl_models': False,
}


config.update(hyperparams_dict)


def main(config=None):
    """Train all modules."""
    offline = 'Offline' if config['train_offline'] else 'Online'
    with wandb.init(project=f'V3-{ENV_NAME}-{offline}', config=config,
                    notes='Training.',
                    name=EXP_NAME):

        config = wandb.config

        path = PARENT_FOLDER
        hives = HIVES(config)

        if not config.train_rl:
            hives.dataset_loader()

        skill_policy = SkillPolicy(hives.state_dim, hives.action_range,
                                   latent_dim=hives.z_skill_dim).to(hives.device)

        critic = Critic(hives.state_dim, hives.z_skill_dim,
                        layer_norm=config.Layernorm).to(hives.device)
        
        sampler = Sampler(skill_policy, hives.models['Decoder'], hives.evaluate_decoder, config)

        experience_buffer = ReplayBuffer(hives.buffer_size, sampler.env,
                                         hives.z_skill_dim, config.reset_frequency,
                                         config)

        vals = VaLS(sampler,
                    experience_buffer,
                    hives,
                    skill_policy,
                    critic,
                    config)
        
        hives_models = list(hives.models.values())

        models = [*hives_models, vals.skill_policy,
                  vals.critic, vals.critic,
                  vals.critic, vals.critic]
        
        names = [*hives.names, 'SkillPolicy', 'Critic', 'Target_critic']

        # Load params path
        params_path = None
        for root, dirs, files in os.walk(f'{PARENT_FOLDER}/Prior'):
            if len(files) > 1:
                raise ValueError('More than one params file.')
            for filename in files:
                params_path = os.path.join(root, filename)
                
        pretrained_params = load_pretrained_models(config, params_path)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))
        
        params = params_extraction(models, names, pretrained_params)
            
        keys_optims = ['VAE_skills']
        keys_optims.extend(['SkillPrior', 'SkillPolicy'])
        keys_optims.extend(['Critic'])

        optimizers = set_optimizers(params, keys_optims, config.learning_rate)

        print('Training is starting')
    
        if config.train_offline:
            for e in range(config.epochs):
                params = hives.train_vae(params,
                                         optimizers,
                                         config.vae_lr,
                                         config.beta)
                
            hives.set_skill_lookup(params)
            for i in range(config.epochs):
                params = hives.train_prior(params, optimizers,
                                           config.priors_lr)

            folder = 'Prior'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            fullpath = f'{path}/{folder}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
                
            torch.save(params, f'{path}/{folder}/params_{dt_string}_offline.pt')

        if config.train_rl:
            params = vals.training(params, optimizers, path, CASE_FOLDER)

            
main(config=config)

