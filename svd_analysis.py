"""Run full svd analysis."""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import pdb
import math
from matplotlib.lines import Line2D

sns.set_theme(style='whitegrid')

COLORS = {'indigo': '#332288',
          'teal': '#44AA99',
          'olive': '#999933',
          'rose': '#CC6677',
          'purple': '#AA4499'}


class SVD_analysis:
    """Run full SVD analysis."""
    
    def __init__(self, path, experiments, labels, exp_name):
        self.path = path
        self.experiments = experiments
        self.load_data()
        self.save_path = f'figures/{exp_name}'
        self.labels = labels
        self.nn_layers = {'Critic/embed.weight-svd': 'Embedding layer',
                          'Critic/layer1.weight-svd': 'Hidden layer 1',
                          'Critic/layer2.weight-svd': 'Hidden layer 2',
                          'Critic/layer3.weight-svd': 'Preoutput layer'}
        

    def load_data(self):
        """Load all data in nested dict."""
        nested_dict = lambda: collections.defaultdict(nested_dict)
        self.data = nested_dict()
        self.reward_dict = nested_dict()

        for env in os.listdir(self.path):
            for exp in self.experiments:
                env_dir = os.path.join(self.path, env, exp)
                if os.path.isdir(env_dir):
                    for i, run in enumerate(os.listdir(env_dir)):
                        folder_dir = os.path.join(env_dir, run)
                        for j, file in enumerate(os.listdir(folder_dir)):
                            iteration = int(file.replace('.npy', ''))
                            if file.endswith('npy'):
                                svd = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                                self.reward_dict[env][exp][run][iteration] = svd['reward']
                                for layer in svd:
                                    if 'out' in layer or 'Policy' in layer:
                                        continue
                                    elif 'reward' in layer:
                                        continue
                                    for element in svd[layer]:
                                        self.data[env][exp][layer][run][element][iteration] = np.array(svd[layer][element])
                                    

    def plots(self):
        path = f'{self.save_path}/plots'
        self.create_folder(path)
        self.delta = 0.01
                
        for env in self.data:
            fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
            for exp_i, exp in enumerate(self.data[env]):
                for idx, layer in enumerate(self.data[env][exp]):
                    ax = axes[idx]
                    eranks = []

                    for run in self.data[env][exp][layer]:
                        erank_df = self.compute_erank_array(self.data[env][exp][layer][run]['S'])
                        eranks.append(erank_df)
                    self.plot_explicit_eranks(eranks, exp_i, ax)
                    if exp_i == 1:
                        self.axis_adjust(ax, title=self.nn_layers[layer], yaxis='n-erank')

                self.plot_reward(env, exp, exp_i, axes[-1])

            self.axis_adjust(axes[-1], yaxis='Reward')

            self.adjust_and_save_plot(fig, f'{path}/{env}_explicit')

    def sing_vecs_computation(self, vecs, levels, case):
        angles = self.delta_theta_sing_vec(vecs, case=case)
        angles = list(angles.values())
        angles = np.stack(angles)

        # Explicit vecs
        arrays = np.split(angles, levels, axis=1)
        arrays = [np.mean(arr, axis=1) for arr in arrays]
        arrays = np.vstack(arrays)
        arrays = np.transpose(arrays)
        index = list(vecs.keys())
        index.sort()
        index.pop()
        df_exp = pd.DataFrame(arrays, columns=['1-3', '4-30', 'rest'],
                              index=index)

        # Implicit vecs
        imp_array = np.mean(angles, axis=1)
        imp_array = np.transpose(imp_array)
        df_imp = pd.DataFrame(imp_array, columns=['angle'], index=index)
        
        return df_imp, df_exp

    def compute_erank_array(self, data):
        df = pd.DataFrame.from_dict(data, orient='index')
        return self.compute_erank(df, self.delta)

    def plot_explicit_sing_vecs(self, vecs, ax):
        runs = pd.concat(vecs, axis=1)
        runs = pd.melt(runs, value_vars=runs.columns, ignore_index=False)
        runs['index'] = runs.index / 1e5

        #runs = runs.reset_index() Comment this in if get index error
        sns.lineplot(runs, x='index', y='value', hue='variable', ax=ax,
                     size=3, errorbar='se')

        ax.get_legend().remove()

    def plot_explicit_eranks(self, eranks, exp_i, ax):
        eranks = pd.concat(eranks, axis=0)
        eranks['index'] = eranks.index / 1e5
        color = list(COLORS.values())[exp_i]
        eranks['hue'] = 'test'
        sns.lineplot(data=eranks, x='index', y='erank', hue='hue', ax=ax,
                     errorbar='sd', palette=[color], legend=False)

    def plot_implicit_sing_vecs(self, vecs, ax):
        dfs = []
        for element in vecs:
            aux_df = pd.concat(vecs[element], axis=0)
            aux_df.columns = [element]
            dfs.append(aux_df)

        dfs = pd.concat(dfs, axis=1)
        sns.lineplot(data=dfs, ax=ax)

    def plot_implicit_eranks(self, eranks, ax):
        dfs = []
        for element in eranks:
            aux_df = pd.concat(eranks[element], axis=0)
            aux_df.columns = [element]
            dfs.append(aux_df)

        dfs = pd.concat(dfs, axis=1)
        sns.lineplot(data=dfs, ax=ax)

    def plot_reward(self, env, exp, exp_i, ax):
        data = pd.DataFrame.from_dict(self.reward_dict[env][exp])
        data.columns = ['Reward'] * data.shape[1] 
        data.index = data.index / 1e5
        color = list(COLORS.values())[exp_i]

        sns.lineplot(data=data, ax=ax, errorbar='se', palette=[color])
        ax.get_legend().remove()

    def axis_adjust(self, ax, title=None, yaxis='Test'):
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel(yaxis, fontsize=16)
        ax.set_xlabel('')
        

        if title is not None:
            ax.set_title(title, fontsize=18)

    def adjust_and_save_plot(self, fig, savepath):
        fig.supxlabel('Environment steps (1e5)', y=0.07, fontsize=16)

        handles = []
        labels = []

        for label, color in zip(self.labels, COLORS.values()):
            aux_label = Line2D([0], [0], label=label, color=color)
            handles.append(aux_label)
            labels.append(label)
        
        fig.legend(handles, labels, ncol=len(handles), bbox_to_anchor=(0.16, -.45, .5, .5),
                   fancybox=True, shadow=True, prop={'size':20})

        plt.subplots_adjust(hspace=0.35)
        
        plt.savefig(f'{savepath}.png', bbox_inches='tight', dpi=340)
        plt.close()

    def explicit_plot(self):
        path = f'{self.save_path}/explicit_plots'
        self.create_folder(path)
        self.delta = 0.01
        levels = [3, 30]

        for env in self.data:
            fig, axes = plt.subplots(4, 2, figsize=(9, 16))
                        
            for idx, layer in enumerate(self.data[env]):
                ax = axes[idx, :]

                eranks = []
                sing_vecs = []
                for run in self.data[env][layer]:
                    # Singular vectors
                    array = self.data[env][layer][run]['Vh']
                    angles = self.delta_theta_sing_vec(array, case='Vh')
                    pdb.set_trace()
                    angles = np.stack(angles)
                    arrays = np.split(angles, levels, axis=1)
                    arrays = [np.mean(arr, axis=1) for arr in arrays]
                    arrays = np.vstack(arrays)
                    arrays = np.transpose(arrays)
                    index = list(array.keys())
                    index.sort()
                    index.pop()
                    df_vec = pd.DataFrame(arrays,
                                          columns=[f'{run} - Top1', f'{run} - Top2', f'{run} - Top3'],
                                          index=index)
                    sing_vecs.append(df_vec)
                    
                    # Erank values
                    df_erank = pd.DataFrame.from_dict(self.data[env][layer][run]['S'], orient='index')
                    erank = self.compute_erank(df_erank, self.delta)
                    eranks.append(erank)

                # Singular vectors
                sing_vecs = pd.concat(sing_vecs, axis=1)
                sing_vecs = pd.melt(sing_vecs, value_vars=sing_vecs.columns, ignore_index=False)
                sing_vecs[['run', 'top']] = sing_vecs['variable'].str.split(' - ', expand=True)

                sns.lineplot(data=sing_vecs, x='index', y='value', hue='top', ax=ax[1])

                # Erank values
                eranks = pd.concat(eranks, axis=0)

                sns.lineplot(data=eranks, x='index', y='erank', ax=ax[0])

                for i in range(2):
                    ax[i].grid(True)
                    ax[i].tick_params(axis='both', labelsize=16)
                    ax[i].set_ylabel('')
                    ax[i].set_xlabel('')
                    ax[i].set_title(env, fontsize=18)

            fig.supxlabel('Environment steps (1e5)', fontsize=20,y=0.02)
            fig.supylabel('Singular values', fontsize=20, x=0.06)
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            plt.savefig(f'{path}/{env}.png', bbox_inches='tight', dpi=340)
            plt.close()

                    
    def delta_theta_sing_vec(self, U, case='U'):
        """Compute angles between consecutive singular vectors"""
        
        angles = {}
        iters = list(U.keys())
        iters.sort()
        
        for i in range(len(iters) - 1):
            U_1 = U[iters[i]]
            U_2 = U[iters[i + 1]]
            if case == 'Vh':
                U_1 = U_1.T
                U_2 = U_2.T
            U_1 = U_1
            U_2 = U_2
            dot_product = np.clip(np.einsum('ij,ij->j', U_1, U_2), -1, 1)
            angle = np.arccos(dot_product)
            angle_pi = math.pi - angle
            angle = np.minimum(angle, angle_pi)
            angles[iters[i]] = angle

        return angles
        
    def plot_sing_vals_distribution(self, layer):
        """Plot singular values distribution."""

        fig, axes = plt.subplots(2, 2, figsize=(27, 9))
        axes = axes.flatten()

        for env, ax in zip(self.data, axes):
            runs = []
            for run in self.data[env][layer]:                
                array = self.data[env][layer][run]['S']
                df = pd.DataFrame.from_dict(array, orient='index')
                runs.append(df)
            data = pd.concat(runs, axis=1)

            bins_x = []
            sorted_index = np.sort(data.index)        
            idx_diff = sorted_index[1] - sorted_index[0]
            init_val = sorted_index[0] - idx_diff / 2
            bins_x.append(init_val)
            for i in range(len(data.index)):
                init_val += idx_diff
                bins_x.append(init_val)
            bins_x = np.array(bins_x)
            
            df = data.stack().reset_index(level=0) # This is to convert to the column format
            df.columns = ['index', 'value']

            scaling = 100000

            df['index'] = df['index'] / scaling
            bins_x = bins_x / scaling

            vals, bins_y = np.histogram(df['value'], bins=40)
            
            sns.histplot(df, x='index', y='value', ax=ax,
                         bins=(bins_x, bins_y), cbar=True,
                         cmap=cmap, norm=LogNorm(), vmin=None, vmax=None)

            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title(env, fontsize=18)

        fig.supxlabel('Environment steps (1e5)', fontsize=20,y=0.02)
        fig.supylabel('Singular values', fontsize=20, x=0.06)
        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        name = layer.replace('/', "").replace(" ", "").replace(".","")
        plt.savefig(f'{self.save_path}/svals_{name}.png', bbox_inches='tight', dpi=400)
        plt.close()

    def erank_plots(self):
        """Get erank plots for each layer for all runs."""
        path = f'{self.save_path}/erank_plots'
        self.create_folder(path)

        for env in self.data:
            fig, axes = plt.subplots(1, 4, figsize=(27, 9))               

            for layer, ax in zip(self.data[env], axes):
                eranks = []
                for run in self.data[env][layer]:
                    df = pd.DataFrame.from_dict(self.data[env][layer][run]['S'], orient='index')
                    erank = self.compute_erank(df, self.delta)
                    erank['reward'] = self.reward_dict[env][run]
                    eranks.append(erank)
                eranks = pd.concat(eranks, axis=0)

                sns.lineplot(data=eranks, x='index', y='erank', hue='reward',
                             ax=ax)
                        
                ax.grid(True)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(env, fontsize=18)
                                        
            fig.supxlabel('Environment steps (1e5)', fontsize=20,y=0.02)
            fig.supylabel('Singular values', fontsize=20, x=0.06)
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            plt.savefig(f'{path}/{env}_erank.png', bbox_inches='tight', dpi=340)
            plt.close()
            
    def compute_erank(self, run, delta):
        x = run.sort_index()
        x_np = np.array(x)
        x_np = np.sqrt(np.cumsum(np.square(x_np), axis=1))
        x0 = x_np[:, -1]
        x_np = x_np / x0.reshape(-1, 1)
        erank = x_np >= 1 - delta
        erank = np.argmax(erank, axis=1)
        norm_erank = erank / erank[0]
        df = pd.DataFrame(norm_erank, columns=['erank'], index=x.index)

        return df

    def create_folder(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
        
            
PATH = 'results'
EXPERIMENTS = ['Replayratio', 'SPiRL']
LABELS = ['SC-SPiRL', 'SPiRL']

analysis = SVD_analysis(PATH, EXPERIMENTS, LABELS, 'benchmark')

analysis.plots()
