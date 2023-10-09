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
                    iters = os.listdir(folder_dir)
                    iters = [int(it.replace('.npy', "")) for it in iters]
                    iters.sort()
                    max_iter = iters[-2]
                    for j, file in enumerate(os.listdir(folder_dir)):
                        if file.endswith('npy'):
                            svd = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                            if int(file.replace(".npy", "")) == max_iter:
                                aux_rew = svd['test reward'] + np.random.rand(1) / 20
                                self.reward_dict[env][run] = aux_rew.item()
                            for layer in svd:
                                if 'out' in layer or 'Policy' in layer:
                                    continue
                                elif 'reward' in layer:
                                    continue
                                for element in svd[layer]:
                                    iteration = int(file.replace('.npy', ''))
                                    self.data[env][layer][run][element][iteration] = np.array(svd[layer][element])
                                    
        
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
        df = pd.DataFrame(norm_erank, columns=['erank'])
        df['index'] = x.index

        return df

    def create_folder(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
        
            
PATH = 'results'
EXPERIMENT = 'Replayratio-16'

analysis = SVD_analysis(PATH, EXPERIMENT)

analysis.singular_vecs_plot()
analysis.singular_vals()
