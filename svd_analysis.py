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

sns.set_theme(style='whitegrid')


colors = [(0, '#ff8888'), (0.5, '#fad7a0'), (1, '#0000ff')]#(.1, '#5e5ef2'), (1, '#0000ff')]
cmap = LinearSegmentedColormap.from_list('CustomCMap', colors)


class SVD_analysis:
    """Run full SVD analysis."""
    
    def __init__(self, path, experiment):
        self.path = path
        self.experiment = experiment
        self.load_data()
        self.save_path = f'figures/{experiment}'
        self.nn_layers = {'Critic/embed.weight-svd': 'Embedding layer',
                          'Critic/layer1.weight-svd': 'Hidden layer 1',
                          'Critic/layer2.weight-svd': 'Hidden layer 2',
                          'Critic/layer3.weight-svd': 'Preoutput layer 2'}
        

    def load_data(self):
        """Load all data in nested dict."""
        nested_dict = lambda: collections.defaultdict(nested_dict)
        self.data = nested_dict()
        self.reward_dict = nested_dict()

        for env in os.listdir(self.path):
            env_dir = os.path.join(self.path, env, self.experiment)
            if os.path.isdir(env_dir):
                for i, run in enumerate(os.listdir(env_dir)):
                    folder_dir = os.path.join(env_dir, run)
                    for j, file in enumerate(os.listdir(folder_dir)):
                        iteration = int(file.replace('.npy', ''))
                        if file.endswith('npy'):
                            svd = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                            self.reward_dict[env][run][iteration] = svd['reward']
                            for layer in svd:
                                if 'out' in layer or 'Policy' in layer:
                                    continue
                                elif 'reward' in layer:
                                    continue
                                for element in svd[layer]:
                                    self.data[env][layer][run][element][iteration] = np.array(svd[layer][element])
                                    

    def plots(self):
        path = f'{self.save_path}/plots'
        self.create_folder(path)
        self.delta = 0.01
        levels = [3, 30]
        CASE = 'Vh'

        for env in self.data:
            # if 'ant' == env or 'kitchen' == env or 'adroit_relocate' == env:
            #     continue
            fig1, axes1 = plt.subplots(3, 1, figsize=(9, 16))
            fig2, axes2 = plt.subplots(5, 2, figsize=(14, 16))
            gs = axes2[-1, 0].get_gridspec()

            for ax in axes2[-1, :]:
                ax.remove()
                
            axbig = fig2.add_subplot(gs[-1, :])

            all_eranks = {}
            all_sing_vecs = {}

            for idx, layer in enumerate(self.data[env]):
                ax2 = axes2[idx, :]
                eranks = []
                sing_vecs_implicit = []
                sing_vecs_explicit = []

                for run in self.data[env][layer]:
                    implicit, explicit = self.sing_vecs_computation(self.data[env][layer][run][CASE],
                                                                    levels, CASE)

                    sing_vecs_implicit.append(implicit)
                    sing_vecs_explicit.append(explicit)

                    erank_df = self.compute_erank_array(self.data[env][layer][run]['S'])
                    eranks.append(erank_df)

                self.plot_explicit_sing_vecs(sing_vecs_explicit, ax2[1])
                self.plot_explicit_eranks(eranks, ax2[0])

                self.axis_adjust(ax2[0], title=self.nn_layers[layer], yaxis='n-erank')
                self.axis_adjust(ax2[1], title=self.nn_layers[layer], yaxis=r'$\theta_{i}-\theta_{i+1}$')

                all_eranks[layer] = eranks
                all_sing_vecs[layer] = sing_vecs_implicit

            self.plot_reward(env, axbig)
            self.axis_adjust(axbig, yaxis='Reward')

            self.adjust_and_save_plot(fig2, f'{path}/{env}_explicit')
            self.plot_implicit_sing_vecs(all_sing_vecs, axes1[1])
            self.plot_implicit_eranks(all_eranks, axes1[0])
            self.plot_reward(env, axes1[2])

            for i in range(3):
                self.axis_adjust(axes1[i])

            self.adjust_and_save_plot(fig1, f'{path}/{env}_implicit')

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

    def plot_explicit_eranks(self, eranks, ax):
        eranks = pd.concat(eranks, axis=0)
        eranks['index'] = eranks.index / 1e5
        sns.lineplot(data=eranks, x='index', y='erank', ax=ax,
                     errorbar='se')

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

    def plot_reward(self, env, ax):
        data = pd.DataFrame.from_dict(self.reward_dict[env])
        data.columns = ['Reward'] * data.shape[1]
        data.index = data.index / 1e5
        sns.lineplot(data=data, ax=ax, errorbar='se')
        ax.get_legend().remove()

    def axis_adjust(self, ax, title=None, yaxis='Test'):
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=10)
        ax.set_ylabel(yaxis)
        ax.set_xlabel('')
        if title is not None:
            ax.set_title(title, fontsize=12)

    def adjust_and_save_plot(self, fig, savepath):
        fig.supxlabel('Environment steps (1e5)', y=0.08, fontsize=14)
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

    def implicit_plots(self):
        path = f'{self.save_path}/explicit_plots'
        self.create_folder(path)
        self.delta = 0.01

        for env in self.data:
            fig, axes = plt.subplots(3, 1, figsize=(9, 16))

            # Erank vals
            all_eranks = []
            all_sing_vecs = []
            
            for layer in self.data[env]:
                eranks = []
                sing_vecs = []
                for run in self.data[env][layer]:
                    df = pd.DataFrame.from_dict(self.data[env][layer][run]['S'], orient='index')
                    erank = self.compute_erank(df, self.delta)
                    erank['reward'] = self.reward_dict[env][run]
                    eranks.append(erank)
                eranks = pd.concat(eranks, axis=0)
                all_eranks.append(eranks)

            all_eranks = pd.concat(all_eranks, axis=0)

            # Singular vectors
            

                
                

        

                                    
    def singular_vals(self):
        """Plot all singular values plots."""
        self.delta = 0.01

        envs = list(self.data.keys())
        layers = list(self.data[envs[0]].keys())
        self.erank_plots()

        for layer in layers:
            self.plot_sing_vals_distribution(layer)

        name = f'{self.save_path}/singular_vecs'
        self.create_folder(name)

    def singular_vecs_plot(self, case='Vh'):
        path = f'{self.save_path}/singular_vecs'
        self.create_folder(path)

        levels = [3, 30]
                
        for env in self.data:
            fig, axes = plt.subplots(1, 4, figsize=(27, 9))
            axes = axes.flatten()
            
            for layer, ax in zip(self.data[env], axes):
                if 'Policy' in layer:
                    continue

                runs = []
                for run in self.data[env][layer]:
                    array_dict = self.data[env][layer][run][case]
                    angles = self.delta_theta_sing_vec(array_dict, case=case)
                    angles = list(angles.values())
                    angles = np.stack(angles)
                    arrays = np.split(angles, levels, axis=1)
                    arrays = [np.mean(arr, axis=1) for arr in arrays]
                    arrays = np.vstack(arrays)
                    arrays = np.transpose(arrays)
                    index = list(array_dict.keys())
                    index.sort()
                    index.pop()
                    df = pd.DataFrame(arrays, columns=[f'{run} - Top1', f'{run} - Top2', f'{run} - Top3'],
                                      index=index)
                    runs.append(df)
                runs = pd.concat(runs, axis=1)
                runs = pd.melt(runs, value_vars=runs.columns, ignore_index=False)
                runs[['run', 'top']] = runs['variable'].str.split(' - ', expand=True)
                runs.drop(columns=['variable'], inplace=True)

                runs.replace({'run': self.reward_dict[env]}, inplace=True)
                runs['index'] = runs.index
                try:
                    sns.lineplot(runs, x='index', y='value', hue='run', style='top', ax=ax,
                                 size=2)
                except ValueError:
                    pdb.set_trace()

                ax.grid(True)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(env, fontsize=18)

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
EXPERIMENT = 'SERENE-v16-Freq-4000'

analysis = SVD_analysis(PATH, EXPERIMENT)

analysis.plots()

# analysis.singular_vecs_plot()
# analysis.singular_vals()

