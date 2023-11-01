"""Training RL algorithm."""

import sys
sys.path.insert(0, '../')

import torch
from utilities.optimization import GD_full_update, Adam_update
from utilities.utils import hyper_params, process_frames, reset_params
from torch.func import functional_call
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import wandb
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.autograd as autograd
import os
import pdb
import pickle
import time
from stable_baselines3.common.utils import polyak_update
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import math


MAX_SKILL_KL = 100
INIT_LOG_ALPHA = 0

class VaLS(hyper_params):
    def __init__(self,
                 sampler,
                 test_sampler,
                 experience_buffer,
                 vae,
                 skill_policy,
                 critic,
                 args):

        super().__init__(args)

        self.sampler = sampler
        self.test_sampler = test_sampler
        self.critic = critic
        self.skill_policy = skill_policy
        self.vae = vae
        self.experience_buffer = experience_buffer
        
        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True,
                                            device=self.device)
        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.learning_rate)

        self.reward_per_episode = 0
        self.steps_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = []
        self.log_data = 0
        POINTS = 128
        self.log_data_freq = (self.max_iterations + 1) // POINTS
        self.folder_svd = None

        
    def training(self, params, optimizers, path, name):
        self.iterations = 0
        ref_params = copy.deepcopy(params)

        obs = None    # These two lines are to setup the RL env.
        done = False  # Only need to be called once.

        while self.iterations < self.max_iterations:

            params, obs, done = self.training_iteration(params, done,
                                                        optimizers,
                                                        self.learning_rate,
                                                        ref_params,
                                                        obs=obs)

            if self.iterations % self.test_freq == 0 and self.iterations > 0:
                dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                print(f'Current iteration is {self.iterations}')
                print(dt_string)
                fullpath = f'{path}/{name}'
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)
                filename = f'{path}/{name}/params_rl_{dt_string}_iter{self.iterations}.pt'
                torch.save(params, filename)
                with open(f'{fullpath}/class_{dt_string}_{self.iterations}', 'wb') as file:
                    pickle.dump(self, file)

            if self.iterations % self.log_data_freq == 0:
                wandb.log({'Iterations': self.iterations})
            self.iterations += 1

            if self.SERENE or self.Replayratio:
                if self.iterations % self.reset_frequency == 0:
                    if self.only_critic:
                        keys = ['Critic']
                    else:
                        keys = ['SkillPolicy', 'Critic']
                    ref_params = copy.deepcopy(params)
                    
                    if self.Replayratio:
                        params, optimizers = reset_params(params, keys, optimizers, self.learning_rate)
                        params['Target_critic'] = copy.deepcopy(params['Critic'])
                    elif self.SERENE:
                        params, optimizers = self.rescale_singular_vals(params, keys, optimizers, self.learning_rate)
                        params['Target_critic'] = copy.deepcopy(params['Critic'])                        
                        self.singular_val_k = self.sing_val_factor * self.singular_val_k                        
                        
                    self.log_alpha_skill = torch.tensor(self.log_alpha_skill.item(), dtype=torch.float32,
                                                        requires_grad=True,
                                                        device=self.device)
                    self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=self.learning_rate)
                    
                
        return params

    def training_iteration(self,
                           params,
                           done,
                           optimizers,
                           lr,
                           ref_params,
                           obs=None):
               
        obs, data = self.sampler.skill_iteration(params, done, obs)

        next_obs, rew, z, next_z, done = data

        self.reward_per_episode += rew
        self.steps_per_episode += 1

        self.experience_buffer.add(obs, next_obs, z, next_z, rew, done)

        if done:
            if self.total_episode_counter > 2:
                self.experience_buffer.update_tracking_buffers(self.reward_per_episode)
            wandb.log({'Reward per episode': self.reward_per_episode,
                       'Steps per episode': self.skill_length * self.steps_per_episode,
                       'Total episodes': self.total_episode_counter})

            self.reward_logger.append(self.reward_per_episode)
            self.reward_per_episode = 0
            self.steps_per_episode = 0
            self.total_episode_counter += 1

        log_data = True if self.log_data % self.log_data_freq == 0 else False
        self.log_data = (self.log_data + 1) % self.log_data_freq

        if len(self.reward_logger) > 15 and log_data:
            step = self.iterations * self.skill_length
            wandb.log({'Cumulative reward dist': wandb.Histogram(np.array(self.reward_logger))})
            wandb.log({'Average reward over 100 eps': np.mean(self.reward_logger[-100:])}, step=step)

        if log_data:
            step = self.iterations * self.skill_length
            self.test_reward = self.testing(params)
            wandb.log({'Test average reward': self.test_reward}, step=step)            
            
        if self.experience_buffer.size >= self.batch_size or log_data:
            for i in range(self.gradient_steps):
                log_data = log_data if i == 0 else False # Only log data once for multi grad steps.
                policy_losses, critic_loss = self.losses(params, log_data, ref_params)
                if self.experience_buffer.size < self.batch_size:
                    continue
                losses = [*policy_losses, critic_loss]
                names = ['SkillPolicy', 'Critic']
                params = Adam_update(params, losses, names, optimizers, lr)
                polyak_update(params['Critic'].values(),
                              params['Target_critic'].values(), 0.005)

                if log_data:
                    with torch.no_grad():
                        dist_init1 = self.distance_to_params(params, ref_params, 'Critic', 'Critic')
                        dist_init_pol = self.distance_to_params(params, ref_params, 'SkillPolicy', 'SkillPolicy')
                    
                        wandb.log({'Critic/Distance to init weights': dist_init1,
                                   'Policy/Distance to init weights Skills': dist_init_pol}) 
           
        return params, next_obs, done

    def losses(self, params, log_data, ref_params):
        batch = self.experience_buffer.sample(batch_size=self.batch_size)

        obs = torch.from_numpy(batch.observations).to(self.device)
        next_obs = torch.from_numpy(batch.next_observations).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        next_z = torch.from_numpy(batch.next_z).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        cum_reward = torch.from_numpy(batch.cum_reward).to(self.device)
        norm_cum_reward = torch.from_numpy(batch.norm_cum_reward).to(self.device)

        if log_data:
            svd = self.compute_singular_svd(params)
            for log_name, log_val in svd.items():
                wandb.log({log_name: wandb.Histogram(log_val['S'])})

            svd['reward'] = self.test_reward
            path = f'results/{self.env_key}/{self.folder_sing_vals}/run-{self.run}'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(f'{path}/{self.iterations * self.skill_length}.npy', svd, allow_pickle=True)

            # # Critic analysis
            critic_test_arg = torch.cat([obs, z], dim=1)

            trials = 32

            new_z = z.reshape(z.shape[0], 1, -1).repeat(1, trials, 1)
            new_z = new_z.reshape(-1, new_z.shape[-1])
            z_rand = torch.rand(new_z.shape).to(self.device)
            new_z = new_z + torch.randn(new_z.shape).to(self.device) / 5

            new_obs = obs.reshape(obs.shape[0], 1, -1).repeat(1, trials, 1)
            new_obs = new_obs.reshape(-1, new_obs.shape[-1])

            new_critic_arg = torch.cat([new_obs, new_z], dim=1)
            new_critic_arg_rand = torch.cat([new_obs, z_rand], dim=1)
        
            with torch.no_grad():
                q_r, _ = self.eval_critic(critic_test_arg, params)
                
                q_rep, _ = self.eval_critic(new_critic_arg, params)
                q_rep = q_rep.reshape(-1, trials, 1)

                q_rand, _ = self.eval_critic(new_critic_arg_rand, params)
                q_rand = q_rand.reshape(-1, trials, 1)

                mean_diff_rand = q_r - q_rand.mean(1)

            eval_test_ave = self.log_scatter_3d(q_r, q_rand.mean(1), cum_reward, rew,
                                                'Q val', 'Q random', 'Cum reward', 'Reward')

            wandb.log({'Critic/Mean diff dist rand': wandb.Histogram(mean_diff_rand.cpu()),
                       'Critic/Mean diff average rand': mean_diff_rand.mean().cpu(),
                       'Policy/Eval policy critic_random': eval_test_ave,
                       'Reward batch': wandb.Histogram(rew.cpu())
                       })
                                                                 
        ####

        target_critic_arg = torch.cat([next_obs, next_z], dim=1)
        critic_arg = torch.cat([obs, z], dim=1)
        
        with torch.no_grad():                                
            z_prior = self.eval_skill_prior(obs, params)

            q_target, _ = self.eval_critic(target_critic_arg, params,
                                           target_critic=True)

        q, features = self.eval_critic(critic_arg, params)
        
        if log_data:
            with torch.no_grad():
                dist1 = self.distance_to_params(params, params, 'Critic', 'Target_critic')

            bellman_terms = self.log_scatter_3d(q, q_target, rew, cum_reward,
                                                'Q val', 'Q target', 'Reward', 'Cum reward')
            
            wandb.log({'Critic/Distance critic to target 1': dist1,
                       'Critic/Bellman terms': bellman_terms})

        q_target = rew + (self.discount * q_target).reshape(-1, 1) * (1 - dones)
        q_target = torch.clamp(q_target, min=-250, max=250)

        critic_loss = F.mse_loss(q.squeeze(), q_target.squeeze(),
                                 reduction='none')

        if self.SERENE and 3 < 1: # This is to not run weight for control experiment.
            with torch.no_grad():
                weights = F.sigmoid(self.sigma_max * norm_cum_reward).squeeze()
        else:
            weights = torch.ones_like(critic_loss)

        critic_loss = critic_loss * weights
                
        critic_loss = critic_loss.mean()

        if self.Underparameter:
            sing_vals_loss = self.compute_singular_vals_loss(features)
            critic_loss = critic_loss + 0.001 * sing_vals_loss            
        
        if log_data:
            wandb.log({'Critic/Critic Grad Norm': self.get_gradient(critic_loss, params, 'Critic')})
        
        z_sample, pdf, mu, std = self.eval_skill_policy(obs, params)

        q_pi_arg = torch.cat([obs, z_sample], dim=1)
        
        q_pi, _ = self.eval_critic(q_pi_arg, params)
        
        skill_prior = torch.clamp(kl_divergence(pdf, z_prior), max=MAX_SKILL_KL).mean()
        
        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        skill_prior_loss = alpha_skill * skill_prior

        q_pi = q_pi.squeeze() * weights
        
        q_val_policy = -torch.mean(q_pi)
        skill_policy_loss = q_val_policy + skill_prior_loss

        policy_losses = [skill_policy_loss]
            
        loss_alpha_skill = torch.exp(self.log_alpha_skill) * \
            (self.delta_skill - skill_prior).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss_alpha_skill.backward()
        self.optimizer_alpha_skill.step()
          
        if log_data:
            with torch.no_grad():
                mu_diff_as = F.l1_loss(mu, z, reduction='none').mean(1)
            
            pi_reward = self.log_scatter_3d(q_pi.reshape(-1, 1), rew, mu_diff_as.unsqueeze(dim=1), cum_reward,
                                            'Q pi', 'Reward', 'Diff mu pi and z', 'Cum reward')
                
            wandb.log(
                {'Policy/current_q_values': wandb.Histogram(q_pi.detach().cpu()),
                 'Policy/current_q_values_average': q_pi.detach().mean().cpu(),
                 'Policy/Z abs value mean': z_sample.abs().mean().detach().cpu(),
                 'Policy/Z std': z_sample.std(0).mean().detach().cpu(),
                 'Policy/Z distribution': wandb.Histogram(z_sample.detach().cpu()),
                 'Policy/Mean STD': std.mean().detach().cpu(),
                 'Policy/Standard dev of STD': std.std(0).mean().detach().cpu(),
                 'Policy/Mu dist': wandb.Histogram(mu.detach().cpu()),
                 'Policy/Pi reward': pi_reward,
                 'Policy/Skill Prior grad': self.get_gradient(skill_prior_loss, params, 'SkillPolicy'),
                 'Policy/Q function grad': self.get_gradient(q_val_policy, params, 'SkillPolicy')})

            wandb.log(
                {'Priors/Alpha skill': alpha_skill.detach().cpu(),
                 'Priors/skill_prior_loss': skill_prior.detach().cpu()})

            wandb.log(
                {'Critic/Critic loss': critic_loss,
                 'Critic/Q values': wandb.Histogram(q.detach().cpu())})
        
        return policy_losses, critic_loss 

    def eval_skill_prior(self, state, params):
        """Evaluate the policy.

        It takes the current state and params. It evaluates the
        policy.

        Parameters
        ----------
        state : Tensor
            The current observation of agent
        params : dictionary with all parameters for models
            It contains all relevant parameters, e.g., policy, critic,
            etc.
        """
        z_prior = functional_call(self.vae.models['SkillPrior'],
                                  params['SkillPrior'], state)
        return z_prior
    

    def eval_skill_policy(self, state, params):
        sample, pdf, mu, std = functional_call(self.skill_policy,
                                               params['SkillPolicy'],
                                               state)
        return sample, pdf, mu, std

    def eval_critic(self, arg, params, target_critic=False):
        if target_critic:
            name = 'Target_critic'
        else:
            name = 'Critic'

        q, features = functional_call(self.critic, params[name], arg)

        return q, features

    def log_histogram_2d(self, x, y, xlabel, ylabel):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        data = np.concatenate([x, y], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel])

        fig_heatmap = px.density_heatmap(df, x=xlabel, y=ylabel,
                                         marginal_x='histogram',
                                         marginal_y='histogram',
                                         nbinsx=60,
                                         nbinsy=60)

        return fig_heatmap

    def log_scatter_3d(self, x, y, z, color, xlabel, ylabel, zlabel, color_label):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

        data = np.concatenate([x, y, z, color], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel, zlabel, color_label])
        
        fig_scatter = px.scatter_3d(df, x=xlabel, y=ylabel,
                                    z=zlabel, color=color_label)
        fig_scatter.update_layout(scene=dict(aspectmode='cube'))

        return fig_scatter

    def compute_singular_svd(self, params):
        models = ['Critic', 'SkillPolicy']
        nicknames = ['Critic', 'Policy']

        svd = {}
        
        with torch.no_grad():
            for name, mods in zip(nicknames, models):
                for key, param in params[mods].items():
                    if len(param.shape) < 2:
                        continue
                    U, S, Vh = torch.linalg.svd(param)
                    svd_dict = {'U': U.cpu(), 'S': S.cpu(), 'Vh': Vh.cpu()}
                    svd[f'{name}/{key}-svd'] = svd_dict

        return svd

    def rescale_singular_vals(self, params, keys, optimizers, lr):
        k = self.singular_val_k
        
        with torch.no_grad():
            for model in keys:
                for key, param in params[model].items():
                    if len(param.shape) < 2:
                        continue
                    U, S, Vh = torch.linalg.svd(param, full_matrices=False)
                    # aux_mat = torch.randn(param.shape).to(self.device)                                      
                    # U_aux, S_aux, Vh_aux = torch.linalg.svd(aux_mat, full_matrices=False)
                    # new_param = U_aux @ torch.diag(S) @ Vh_aux
                    bounded_S = k * (1 - torch.exp(-S / k))
                    new_param = U @ torch.diag(bounded_S) @ Vh
                    params[model][key] = nn.Parameter(new_param)

                optimizers[model] = Adam(params[model].values(), lr)

        return params, optimizers

    def compute_singular_vals_loss(self, features):
        S = torch.linalg.svdvals(features)
        return torch.square(S[0]) - torch.square(S[-1])

    def get_gradient(self, x, params, key):
        grads = autograd.grad(x, params[key].values(), retain_graph=True,
                              allow_unused=True)

        grads = [grad for grad in grads if grad is not None]
        
        try:
            grads_vec = nn.utils.parameters_to_vector(grads)
        except RuntimeError:
            pdb.set_trace()
        return torch.norm(grads_vec).detach().cpu()

    def distance_to_params(self, params1, params2, name1, name2):
        with torch.no_grad():
            vec1 = nn.utils.parameters_to_vector(params1[name1].values())
            target_vec1 = nn.utils.parameters_to_vector(params2[name2].values())
        return torch.norm(vec1 - target_vec1)

    def testing(self, params):
        done = False
        obs = None

        rewards = []
        test_episodes = 100

        for j in range(test_episodes):
            while not done:
                _, data = self.test_sampler.skill_iteration(params, done, obs)
                obs, reward, _, _, done = data                
                rewards.append(reward)
            done = False
            obs = None

        average_reward = sum(rewards) / test_episodes

        return average_reward
