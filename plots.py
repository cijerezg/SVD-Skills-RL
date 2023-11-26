import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
import math
import pdb
import wandb
import pandas as pd
import os
from matplotlib.lines import Line2D
import copy

sns.set_theme(style='whitegrid')


COLORS = {'purple': '#AA4499',
          'rose': '#CC6677',
          'olive': '#999933',
          'teal': '#44AA99',
          'indigo': '#332288'}


def create_dir(path):
     if not os.path.exists(path):
        os.makedirs(path)

nested_dict = lambda: collections.defaultdict(nested_dict)

class SVD_to_erank:
    def __init__(self, path, savepath):
        self.path = path
        self.delta = 0.01
        self.savepath = savepath

        self.critic_layers = {'Critic/embed.weight-svd': 'Embedding layer',
                             'Critic/layer1.weight-svd': 'Hidden layer 1',
                             'Critic/layer2.weight-svd': 'Hidden layer 2',
                             'Critic/layer3.weight-svd': 'Preoutput layer'}

        self.policy_layers = {'Policy/embed_obs.weight-svd': 'Embedding layer',
                              'Policy/latent_policy.0.weight-svd': 'Hidden layer 1',
                              'Policy/latent_policy.2.weight-svd': 'Hidden layer 2',
                              'Policy/latent_policy.4.weight-svd': 'Hidden layer 3',
                              'Policy/mu.weight-svd': 'Mean layer',
                              'Policy/log_std.weight-svd': 'Variance layer'}

        self.models = {'Critic': self.critic_layers,
                       'Policy': self.policy_layers}

    def process_data(self):
        data = nested_dict()

        # Run all environments
        for env in os.listdir(self.path):
            env_dir = os.path.join(self.path, env)
            #Run all experiments
            for exp in os.listdir(env_dir):
                print(exp)
                run_dir = os.path.join(env_dir, exp)
                # Run all runs
                for run in os.listdir(run_dir):
                    aux_data = nested_dict()
                    folder_dir = os.path.join(run_dir, run)
                    # Load run
                    for file in os.listdir(folder_dir):                    
                        iteration = int(file.replace('.npy', ""))
                        svd = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                        for model in self.models:
                            for layer in self.models[model]:
                                id_layer = self.models[model][layer]
                                aux_data[model][id_layer][iteration] = np.array(svd[layer]['S'])
                    
                    # Compute erank for that run for all layers
                    for model in aux_data:
                        for layer in aux_data[model]:                             
                            erank = self.compute_erank(aux_data[model][layer])
                            data[env][exp][model][layer][run] = erank

        create_dir(self.savepath)
        self.save_data(data)
                                   
    def compute_erank(self, data):
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.sort_index()
        x_np = np.array(df)
        x_np = np.sqrt(np.cumsum(np.square(x_np), axis=1))
        x0 = x_np[:, -1]
        x_np = x_np / x0.reshape(-1, 1)
        erank = x_np >= 1 - self.delta
        erank = np.argmax(erank, axis=1)
        norm_erank = erank / erank[0]
        df = pd.DataFrame(norm_erank, columns=['erank'], index=df.index)

        return df

    def save_data(self, data):
        path = f'{self.savepath}'
        for env in data:
            for exp in data[env]:
                for model in data[env][exp]:
                    for layer in data[env][exp][model]:
                        runs = data[env][exp][model][layer]
                        vals = list(runs.values())
                        df = pd.concat(vals, axis=1)
                        full_path = f'{path}/{env}/{exp}/{model}/{layer}/'
                        create_dir(full_path)
                        df.to_csv(f'{full_path}/erank.csv')                

# Creating all plots

class Plots:
    def __init__(self, savepath, pre):
        self.savepath = savepath
        self.api = wandb.Api()
        self.pre = pre

    def load_runs(self, env_name, env_id, key, exps):
        runs = self.api.runs(f'{self.pre}{env_id}')

        all_runs = []

        for run in runs:
            if run.name in exps:
                run_history = run.history(keys=[key], pandas=True, x_axis="_step")
                run_history.set_index('_step', inplace=True)
                run_history.columns = [run.name] # This assigns experiment name
                all_runs.append(run_history)
                
        full_df = pd.concat(all_runs, axis=1)
        full_df.index = full_df.index / 1e5

        return full_df

    def erank_analysis(self, envs, path, no_reward=False):
        erank_data = self.load_erank_data(path)
        model = 'Policy' if no_reward else 'Critic'

        # For envs, 0 is wandb name, 1 is svd name, and 2 (-1) is exp names.

        for env in envs:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 16), sharex=True)
            axes = axes.flatten()
            for id_exp, exp in enumerate(envs[env][-1]):
                env_svd = envs[env][1]
                exp_svd = envs[env][-1][exp][0]
                for ax, layer in zip(axes, erank_data[env_svd][exp_svd][model]):
                    self.erank_plot(erank_data[env_svd][exp_svd][model][layer], ax, envs[env][-1][exp])
                    if id_exp == 0:
                         self.adjust_plot(ax=ax, title=layer, yaxis='n-erank')

            if not no_reward: # Add reward and critic loss                 
                exp_auxs = list(envs[env][-1].values())
                exp_names = [val[0] for val in exp_auxs]
                reward = self.load_runs(env, envs[env][0], 'Test average reward', exp_names)
                critic_loss = self.load_runs(env, envs[env][0], 'Critic/Critic loss', exp_names)
                self.vanilla_plot(reward, axes[-2], exp_auxs)
                self.adjust_plot(axes[-2], title='Reward', yaxis='Reward')
                self.vanilla_plot(critic_loss, axes[-1], exp_auxs)
                self.adjust_plot(axes[-1], title='Critic loss', yaxis='MSE')

            create_dir(f'{self.savepath}/{env}')
            self.adjust_and_save_plot(fig, f'{self.savepath}/{env}/erank-analysis',
                                      'Environment steps (1e5)', envs[env][-1],
                                      bbox_to_anchor=(.31, -.45, .5, .5),
                                      wspace=.2, hspace=0.18)
            
    def load_erank_data(self, path):
        erank_data = nested_dict()

        for env in os.listdir(path):
            env_dir = os.path.join(path, env)
            for exp in os.listdir(env_dir):
                exp_dir = os.path.join(env_dir, exp)
                for model in os.listdir(exp_dir):
                    model_dir = os.path.join(exp_dir, model)
                    for layer in os.listdir(model_dir):
                        erank = pd.read_csv(f'{model_dir}/{layer}/erank.csv', index_col=0)
                        erank_data[env][exp][model][layer] = erank

        return erank_data
                
    def erank_plot(self, eranks_data, ax, exp):         
        df = pd.melt(eranks_data, value_vars=eranks_data.columns, ignore_index=False)
        df['hue'] = 'hue'
        df['index'] = df.index / 1e5
        
        sns.lineplot(data=df, x='index', y='value', hue='hue',
                     palette=[exp[-1]], ax=ax,
                     errorbar='sd', legend=False, err_kws={'alpha': 0.05})

    def vanilla_plot(self, data, ax, exp_data):
        df = pd.melt(data, value_vars=data.columns, ignore_index=False, var_name='melt')
        df['index'] = df.index
        hue_order = [val[0] for val in exp_data]
        palette = [val[1] for val in exp_data]
        sns.lineplot(data=df, x='index', y='value', hue='melt',
                     err_kws={'alpha':0.05}, ax=ax,
                     hue_order=hue_order, palette=palette,
                     errorbar='se', legend=False)

        ax.set_xlabel('')

    def adjust_plot(self, ax, title, yaxis):
         ax.grid(True)
         ax.set_ylabel(yaxis, fontsize=16)
         ax.set_title(title, fontsize=18)

    def adjust_and_save_plot(self, fig, path, xlabel, ylabel=None, labels=None,
                             y_subx=0.07, bbox_to_anchor=(0.16, -.45, .5, .5),
                             wspace=0.35, hspace=0.05):
     
        fig.supxlabel(xlabel, y=y_subx, fontsize=18)
        if ylabel is not None:
            fig.text(0.08, 0.5, ylabel, va='center', rotation='vertical', fontsize=18)

        if labels is not None:
            colors = list(labels.values())
            colors = [val[-1] for val in colors]
            colors.reverse()

            labels = list(labels.keys())
            labels.reverse()

            handles = []
            labels_ = []

            for label, color in zip(labels, colors):
               handle = Line2D([0], [0], label=label, color=color)
               handles.append(handle)
               labels_.append(label)

            fig.legend(handles, labels_, ncol=len(handles), bbox_to_anchor=bbox_to_anchor,
                       fancybox=True, shadow=True, prop={'size': 20})

        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(path, bbox_inches='tight', dpi=450)
        plt.close()

    def reward_plot(self, envs, filename):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 6.5))
        axes = axes.flatten()

        for idx, env in enumerate(envs):
            exp_aux = list(envs[env][-1].values())
            exp_names = [val[0] for val in exp_aux]
            reward = self.load_runs(env, envs[env][0], 'Test average reward', exp_names)
            self.vanilla_plot(reward, axes[idx], exp_aux)
            self.adjust_plot(axes[idx], title=env, yaxis='')
            
        create_dir(f'{self.savepath}/reward')
        self.adjust_and_save_plot(fig, f'{self.savepath}/reward/{filename}',
                                  'Environment steps (1e5)', ylabel='Reward', labels=envs[env][-1],
                                  y_subx=0.01, bbox_to_anchor=(.3, -.5, .5, .5),
                                  wspace=0.18, hspace=0.35)

    def trade_off_plot(self, envs, filename, path):
        erank_data = self.load_erank_data(path) 
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 6.5))
        axes = axes.flatten()

        for idx, env in enumerate(envs):
            erank = self.collect_erank(erank_data, envs, env)
            exp_names = list(envs[env][-1].values())
            reward = self.load_runs(env, envs[env][0], 'Test average reward', exp_names)
            critic_loss = self.load_runs(env, envs[env][0], 'Critic/Critic loss', exp_names)
            self.plot_trade_off(erank, reward, critic_loss, axes[idx], envs[env][-1])
            pdb.set_trace()

    def collect_erank(self, erank_data, envs, env):
        exp_names = list(envs[env][-1].values())
        env_svd = envs[env][1]        
        layers = list(erank_data[env_svd][exp_names[0]]['Critic'].keys())
        
        erank = []

        for exp in exp_names:
            layer_data = []
            for layer in layers:
                erank_aux = erank_data[env_svd][exp]['Critic'][layer]
                erank_aux = erank_aux.iloc[-1, :]
                layer_data.append(erank_aux)
            layer_data = pd.concat(layer_data, axis=1)            
            layer_data = layer_data.product(1)
            layer_data.index = [exp] * len(layer_data.index)
            erank.append(layer_data)

        erank = pd.concat(erank, axis=0)
            
        return erank

    def plot_trade_off(self, erank, reward, critic_loss, ax, xlabels):
        reward = reward.iloc[-1, :]
        critic_loss = critic_loss.iloc[-1, :]

        erank.index = erank.index.map({v: k for k, v in xlabels.items()})
        reward.index = reward.index.map({v: k for k, v in xlabels.items()})
        critic_loss.index = critic_loss.index.map({v: k for k, v in xlabels.items()})

        erank = erank.to_frame(name='value')
        erank['index'] = erank.index
        erank['hue'] = 'placeholder1'
        
        reward = reward.to_frame(name='value')
        reward['index'] = reward.index
        reward['hue'] = 'placeholder2'
        
        critic_loss = critic_loss.to_frame(name='value')
        critic_loss['index'] = critic_loss.index
        critic_loss['hue'] = 'placeholder3'

        ax.set_xscale('log', base=5)
        ax.set_xlabel('')
        ax1, ax2 = ax.twinx(), ax.twinx()
        
        ax.set_ylabel('Reward')
        ax1.set_ylabel('n-erank')
        ax2.set_ylabel('MSE')

        sns.lineplot(data=reward, x='index', y='value', hue='hue', markers=True, style='hue',
                     ax=ax, palette=[COLORS['indigo']], legend=False)
        sns.lineplot(data=erank, x='index', y='value', hue='hue', markers=True, style='hue',
                     ax=ax1, palette=[COLORS['olive']], legend=False)
        sns.lineplot(data=critic_loss, x='index', y='value', hue='hue', markers=True, style='hue',
                     ax=ax2, palette=[COLORS['purple']], legend=False)

        ax2.spines['right'].set_position(('outward', 60))

        ax1.grid(False)
        ax2.grid(False)

        
# load = SVD_to_erank('results', 'results_svd')
# load.process_data()

SAVEPATH = 'Figures'
PRE = 'cijerezg/camera-ready-SERENE-'

plots = Plots(SAVEPATH, PRE)

# Overall results

BASE = {'SPiRL': ['SPiRL', COLORS['purple']],
        'U-SPiRL': ['Underparameter', COLORS['rose']],
        'N-SPiRL': ['Layernorm', COLORS['olive']],
        'SR-SPiRL': ['Replayratio', COLORS['teal']]}

EXPS_PEN = copy.deepcopy(BASE)
EXPS_PEN['SERENE'] = ['SERENE-S-1', COLORS['indigo']]

EXPS_REC = copy.deepcopy(BASE)
EXPS_REC['SERENE'] = ['SERENE-S-4', COLORS['indigo']]

EXPS_ANT = copy.deepcopy(BASE)
EXPS_ANT['SR-SPiRL'] = ['Replayratio-v4', COLORS['teal']]
EXPS_ANT['SERENE'] = ['SERENE-S-4-E-0.0001', COLORS['indigo']]

EXPS_KIT = copy.deepcopy(BASE)
EXPS_KIT['SERENE'] = ['SERENE-S-1-E-0.0001', COLORS['indigo']]

ENVS = {'Pen': ['AdroitHandPenSparse-v1-Online', 'adroit_pen', EXPS_PEN],
        'Relocate': ['relocate-expert-v1-Online', 'adroit_relocate', EXPS_REC],
        'Antmaze': ['antmaze-medium-diverse-v2-Online', 'ant', EXPS_ANT],
        'Kitchen': ['kitchen-mixed-v0-Online', 'kitchen', EXPS_KIT]}

PATH = 'results_svd'

# plots.erank_analysis(ENVS, PATH)
# plots.reward_plot(ENVS, 'main')

EXPS_PEN_T = {0.0002: 'SERENE-S-1-E-0.0002',
              0.00004: 'SERENE-S-1-E-0.00004',
              0.001: 'SERENE-S-1',
              0.005: 'SERENE-S-1-E-0.005',
              0.025: 'SERENE-S-1-E-0.025'}

EXPS_REC_T = {0.0002: 'SERENE-S-4-E-0.0002',
              0.00004: 'SERENE-S-4-E-0.00004',
              0.001: 'SERENE-S-4',
              0.005: 'SERENE-S-4-E-0.005',
              0.025: 'SERENE-S-4-E-0.025'}

EXPS_ANT_T = {0.00002: 'SERENE-S-4-E-0.00002',
              0.000004: 'SERENE-S-4-E-0.000004',
              0.0001: 'SERENE-S-4-E-0.0001',
              0.0005: 'SERENE-S-4-E-0.0005',
              0.0025: 'SERENE-S-4-E-0.0025'}

EXPS_KIT_T = {0.00002: 'SERENE-S-1-E-0.00002',
              0.000004: 'SERENE-S-1-E-0.000004',
              0.0001: 'SERENE-S-1-E-0.0001',
              0.0005: 'SERENE-S-1-E-0.0005',
              0.0025: 'SERENE-S-1-E-0.0025'}


ENVS_T = {'Pen': ['AdroitHandPenSparse-v1-Online', 'adroit_pen', EXPS_PEN_T],
          'Relocate': ['relocate-expert-v1-Online', 'adroit_relocate', EXPS_REC_T],
          'Antmaze': ['antmaze-medium-diverse-v2-Online', 'ant', EXPS_ANT_T],
          'Kitchen': ['kitchen-mixed-v0-Online', 'kitchen', EXPS_KIT_T]}


plots.trade_off_plot(ENVS_T, 'trade-off', PATH)
    

    
