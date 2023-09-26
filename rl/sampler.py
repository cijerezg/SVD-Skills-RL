"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict, compute_cum_rewards
#import gymnasium as gym
import gym
import d4rl
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal

import pdb

WIDTH = 4 * 640
HEIGHT = 4 * 480


class Sampler(hyper_params):
    def __init__(self, skill_policy, decoder, eval_decoder, args):
        super().__init__(args)

        self.skill_policy = skill_policy
        self.decoder = decoder
        self.eval_decoder = eval_decoder

        self.env = gym.make(self.env_id)

        if 'relocate' in self.env_id:
            self.env._max_episode_steps = 40
        elif 'pen' in self.env_id:
            self.env._max_episode_steps = 40
        
    def skill_execution(self, actions, frames=None):
        obs_trj, rew_trj, done_trj = [], [], []
        aux_frames = []
        
        for i in range(actions.shape[0]):
            next_obs, rew, done, info = self.env.step(actions[i, :])
            if frames is not None:
                if self.env_key != 'kitchen':
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT,
                                                mode='offscreen',
                                                camera_name='vil_camera')
                    aux_frames.append(frame)
                else:
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT)
                    aux_frames.append(frame)
                    
            if self.env_key != 'kitchen':
                done = info['goal_achieved'] if len(info) == 1 else True
            obs_trj.append(next_obs)
            rew_trj.append(rew)
            done_trj.append(done)
        if frames is not None:
            frames.append(aux_frames)

        return obs_trj, rew_trj, done_trj, frames

    def skill_step(self, params, obs, frames=None):
        obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)
        obs_t = obs_t.reshape(1, -1)
        obs_trj, rew_trj, done_trj = [], [], []

        with torch.no_grad():
            z_sample, _, _, _ = functional_call(self.skill_policy,
                                                params['SkillPolicy'],
                                                obs_t)

            self.decoder.reset_hidden_state(z_sample)
            self.decoder.func_embed_z(z_sample)

            for i in range(self.skill_length):
                action = self.eval_decoder(obs_t, params)
                action = action.cpu().detach().numpy()
                action = action.squeeze()
                obs, rew, done, info = self.env.step(action)

                # Relocate environment does not use done. It uses info.
                obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)

                if 'adroit' in self.env_key:
                    done = True if done or info['goal_achieved'] else False
                # Collect trajectories
                obs_trj.append(obs)
                rew_trj.append(rew)
                done_trj.append(done)
                if done:                    
                    break
            # distance = np.linalg.norm(self.env.get_xy() - self.env.target_goal)
            # print(info)
            # print(distance)
        if frames is not None:
            done = True if sum(done_trj) > 0 else False
            return obs_trj[-1], done, frames

        next_obs_t = torch.from_numpy(obs_trj[-1]).to(self.device).to(torch.float32)
        next_obs_t = next_obs_t.reshape(1, -1)

        with torch.no_grad():
            next_z_sample, _, _, _ = functional_call(self.skill_policy,
                                                     params['SkillPolicy'],
                                                     next_obs_t)
           
        next_obs = obs_trj[-1]
        rew = sum(rew_trj)

        z = z_sample.cpu().numpy()
        next_z = next_z_sample.cpu().numpy()
        done = True if sum(done_trj) > 0 else False

        return next_obs, rew, z, next_z, done

    def skill_iteration(self, params, done=False, obs=None):
        if done or obs is None:
            obs = self.env.reset()

        return obs, self.skill_step(params, obs)

    def skill_iteration_with_frames(self, params, done=False, obs=None, frames=None):
        if done or obs is None:
            obs = self.env.reset()

        frames = self.skill_step(params, obs, frames)

        return frames
    
       
class ReplayBuffer(hyper_params):
    def __init__(self, size, env, lat_dim, reset_ratio, args):
        super().__init__(args)

        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.tracker = np.zeros((size,), dtype=bool)
        self.cum_reward = np.zeros((size, 1), dtype=np.float32)
        self.norm_cum_reward = np.zeros((size, 1), dtype=np.float32)        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.threshold = 0.0

        self.sampling_ratio = reset_ratio
        self.env = env
        self.lat_dim = lat_dim

    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tracker[self.ptr] = True
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs],
                         cum_reward=self.cum_reward[idxs],
                         norm_cum_reward=self.norm_cum_reward[idxs])
        return batch


    def update_tracking_buffers(self, ep_reward):
        last_ep_idx = np.where(self.done_buf[0:self.ptr - 1])[0].max() + 1
        self.cum_reward[last_ep_idx:self.ptr, :] = ep_reward
        mean = self.cum_reward[0:self.ptr, :].mean()
        std = self.cum_reward[0:self.ptr, :].std()
        self.norm_cum_reward[0:self.ptr, :] = (self.cum_reward[0:self.ptr, :] - mean) / (std + 1e-4)


