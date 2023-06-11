# %%
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# select data folder name
data_folder = os.path.join(os.getcwd(), 'data', 'processed_evokeds')

file_locs = {
    'del2019':os.path.join(data_folder, 'del2019' + '.pickle'),         # delogu 2019
    'del2021':os.path.join(data_folder, 'del2021' + '.pickle'),         # delogu 2021
    'aurn2021':os.path.join(data_folder, 'aurn2021' + '.pickle'),       # aurnhammer 2021 EEG epochs
    'aurn2023':os.path.join(data_folder, 'aurn2023' + '.pickle')        # aurnhammer 2023
}


# check if output directory already exists. if not, make it
output_folder = os.path.join(os.getcwd(), 'plots', 'grand_averages')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

# The channel that we'll be plotting the grand avg ERP of
channel = 'Cz'

# %%
i = 0
titles = {'del2019': 'Delogu et al (2019)',
          'del2021': 'Delogu et al (2021)',
          'aurn2021': 'Aurnhammer et al (2021)',
          'aurn2023': 'Aurnhammer et al (2023)'}

# loop through datasets, process if available
for dataset_key, path in file_locs.items():
    try:
        # import the processed data
        print(f'Importing data for {dataset_key}...')
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Defining contrasts and order of plotting
        # These are the same contrasts as those taken in the original papers
        if dataset_key == 'del2019':
            contrasts = [('control', 'script-related'), ('control', 'script-unrelated')]
            order = ['control', 'script-related', 'script-unrelated']
        elif dataset_key == 'del2021':
            contrasts = [('baseline', 'plausible'), ('baseline', 'implausible'), ('plausible', 'implausible')]
            order = ['baseline', 'plausible', 'implausible']
        elif dataset_key == 'aurn2021':
            contrasts = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('C', 'D')]
            order = ['A', 'B', 'C', 'D']
        elif dataset_key == 'aurn2023':
            contrasts = [('A', 'B'), ('A', 'C')]
            order = ['A', 'B', 'C']

        first_subject = sorted(data.keys())[0]
        ERP = dict((cond, data[first_subject][cond].pick(channel).data) for cond in order)
        N = len(list(data.keys())) # no. of subjects
        x = data[first_subject][order[0]].times
        
        plt.figure(i)
        plt.gca().invert_yaxis()
        for condition in order:
            for subject in sorted(data.keys())[1:]:
                ERP[condition] += data[subject][condition].pick(channel).data

            ERP[condition] = ERP[condition]/N
            plt.plot(x, ERP[condition][0])
            plt.title(titles[dataset_key])
            plt.legend(order, loc='upper right')
            plt.xlim(x[0], x[-1])
            plt.xlabel('t (s)')
            # plt.ylabel(r'$\mu V$')
            # plt.ylim(1.1*min(ERP[condition][0]), 1.1*max(ERP[condition][0]))
            plt.tight_layout()
            plt.grid(visible=True)
            plt.savefig(os.path.join(output_folder, f"{dataset_key}.png"), dpi=600)

        i += 1

    
    except Exception as e:
        print("Exception occurred:")
        print(e)
        print(f"Skipping dataset with key {dataset_key} ...")
    # %%