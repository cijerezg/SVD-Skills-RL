"""Run full svd analysis."""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import collections
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import pdb

sns.set_theme(style='whitegrid')

colors = [(0, '#ff8888'), (0.5, '#fad7a0'), (1, '#0000ff')]#(.1, '#5e5ef2'), (1, '#0000ff')]
cmap = LinearSegmentedColormap.from_list('CustomCMap', colors)


class SVD_analysis:
    def __init__(self, path, experiment):
        self.path = path
        self.experiment = experiment

    def load_data(self):
        nested_dict = lambda: collections.defaultdict(nested_dict)
        self.data = nested_dict()

        for env in os.listdir(self.path):
            env_dir = os.path.join(self.path, env, self.experiment)
            if os.path.isdir(env_dir):
                for i, folder in enumerate(os.listdir(env_dir)):
                    folder_dir = os.path.join(env_dir, folder)
                    for file in os.listdir(folder_dir):
                        if file.endswith('npy'):
                            data = np.load(os.path.join(folder_dir, file), allow_pickle=True).item()
                            for element in array:
                            
                            pdb.set_trace()
                
