import collections
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb
from matplotlib.colors import LinearSegmentedColormap, LogNorm

sns.set_theme(style='whitegrid')

colors = [(0, '#ff8888'), (0.5, '#fad7a0'), (1, '#0000ff')]#(.1, '#5e5ef2'), (1, '#0000ff')]
cmap = LinearSegmentedColormap.from_list('CustomCMap', colors)



def load_data(root_dir, experiment, case, combine=True):
    data = {}
    FREQ = 2 # This is the total point along x axis (time) to be plotted.
    for env in os.listdir(root_dir):
        env_dir = os.path.join(root_dir, env, experiment)
        if os.path.isdir(env_dir):
            aux_data = {}
            folders = [int(i) for i in list(os.listdir(env_dir))]
            folders.sort()
            if len(folders) % FREQ != 0:
                raise ValueError(f'The frequency {FREQ} does not evenly divide the {len(folders)} points. Select another frequency')           
            idx_array = np.arange(len(folders))
            idxs = idx_array[0::FREQ]
            folders = [str(folders[idx]) for idx in idxs]
            for i, folder in enumerate(folders):
                folder_dir = os.path.join(env_dir, folder)
                if os.path.isdir(folder_dir):
                    arrays = []
                    for file in os.listdir(folder_dir):
                        if file.endswith('.npy'):
                            array = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                            array = array[case]
                            arrays.append(array)
                    if combine:
                        arrays = np.hstack(arrays)
                    aux_data[int(folder)] = arrays
        if combine:
            df = pd.DataFrame.from_dict(aux_data, orient='index', dtype=np.float32)
            data[env] = df
        else:
            data[env] = aux_data

    return data


def plot_singular_vals_distribution(ROOT_DIR, EXPERIMENT, CASE):
    data = load_data(ROOT_DIR, EXPERIMENT, CASE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for env, ax in zip(data, axes):
        bins_x = []
        sorted_index = np.sort(data[env].index)        
        idx_diff = sorted_index[1] - sorted_index[0]
        init_val = sorted_index[0] - idx_diff / 2
        bins_x.append(init_val)
        for i in range(len(data[env].index)):
            init_val += idx_diff
            bins_x.append(init_val)
        bins_x = np.array(bins_x)
            
        df = data[env].stack().reset_index(level=0) # This is to convert to the column format
        df.columns = ['index', 'value']

        scaling = 100000

        df['index'] = df['index'] / scaling
        bins_x = bins_x / scaling

        vals, bins_y = np.histogram(df['value'], bins=40)

        sns.histplot(df, x='index', y='value', ax=ax,
                     bins=(bins_x, bins_y), cbar=True,
                     cmap=cmap, norm=LogNorm(), vmin=None, vmax=None)

        # Leave it in case it useful for later
        # cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
        # cbar.set_ticks([max_val, mid_val, low_val])
        # cbar.set_ticklabels([1, 0.1, 0.01])

        ax.grid(True)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(env, fontsize=18)
        

    fig.supxlabel('Environment steps (1e5)', fontsize=20,y=0.02)
    fig.supylabel('Singular values', fontsize=20, x=0.06)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    name = CASE.replace('/',"").replace(" ","").replace(".","")

    path = f'figures/{EXPERIMENT}/'

    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.savefig(f'{path}/svals_{name}.png', bbox_inches='tight', dpi=400)
    plt.close()


def compute_erank(x, delta, case):
    x = x.sort_index()
    x_np = np.array(x)
    x_np = np.sqrt(np.cumsum(np.square(x_np), axis=1))
    x0 = x_np[:, -1]
    x_np = x_np / x0.reshape(-1, 1)
    erank = x_np >= 1 - delta
    erank = np.argmax(erank, axis=1)
    norm_erank = erank / erank[0]
    df = pd.DataFrame(norm_erank, index=x.index, columns=[f'Layer {case}'])

    return df


def erank_plots(ROOT_DIR, EXPERIMENT):

    DELTA = 0.01
    nested_dict = lambda: collections.defaultdict(nested_dict)
    ndict = nested_dict()
    
    for CASE in CASES:
        data = load_data(ROOT_DIR, EXPERIMENT, CASE, combine=False)
        for env in data:
            for idx in data[env]:
                for run in range(len(data[env][idx])):
                    ndict[env][CASE][run][idx] = np.array(data[env][idx][run])

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()

    for env, ax in zip(ndict, axes):
        eranks_dfs = []
        
        for idx, case in enumerate(ndict[env]):
            for run in range(len(ndict[env][case])):
                df = pd.DataFrame.from_dict(ndict[env][case][run], orient='index')
                eranks = compute_erank(df, DELTA, idx)
                eranks_dfs.append(eranks)

        eranks_dfs = pd.concat(eranks_dfs, axis=1)
        eranks_dfs = pd.melt(eranks_dfs, value_vars=eranks_dfs.columns, ignore_index=False)
        eranks_dfs['Index'] = eranks_dfs.index
        eranks_dfs.reset_index(drop=True, inplace=True)

        sns.lineplot(data=eranks_dfs, x='Index', y='value', hue='variable',
                     errorbar='se', ax=ax)

        ax.grid(True)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(env, fontsize=18)

    fig.supxlabel('Environment steps (1e5)', fontsize=20,y=0.02)
    fig.supylabel('Singular values', fontsize=20, x=0.06)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.show()


        
ROOT_DIR = 'results'
EXPERIMENT = 'SPiRL'
CASES = ['Critic/embed.weight - singular vals',
         'Critic/layer1.weight - singular vals',
         'Critic/layer2.weight - singular vals',
         'Critic/layer3.weight - singular vals',
         ]

# Singular vals
# for CASE in CASES:
#     plot_singular_vals_distribution(ROOT_DIR, EXPERIMENT, CASE)

# Erank plots
erank_plots(ROOT_DIR, EXPERIMENT)
    
