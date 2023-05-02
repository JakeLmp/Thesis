# %% imports
import mne
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn

# check if we're running in interactive python (notebook or IPython shell)
# https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/4
try:
    # get_ipython() should be automatically available in any IPython shell/notebook, but not in other versions
    ip = get_ipython()
    if ip is None:
        # we have IPython installed but not running from IPython
        interactive_mode = False
    else:
        interactive_mode = True
except:
    # We do not even have IPython installed
    interactive_mode = False



# %%
# Defining ROIs

# get subject directory
subjects_dir = mne.datasets.sample.data_path(download=False) / 'subjects'

# download the 'aparc' parcellation data
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)

# read in the labels
labels = mne.read_labels_from_annot(
    'fsaverage', 'aparc', 'lh', subjects_dir=subjects_dir)

# uncomment the following line to print all available labels
# print([label.name for label in labels])

# these are the subsets of filenames from what we just downloaded 
parcels_temporal = ['middletemporal-lh',    # middle temporal gyrus
                    'superiortemporal-lh',  # superior temporal gyrus
                    'bankssts-lh'           # banks of the superior temporal sulcus
                    ]

parcels_frontal = ['parsopercularis-lh',    # xxx
                'parsorbitalis-lh',      # xxx
                'parstriangularis-lh'    # xxx
                ]

# get all the ROI label objects
ROIs_temporal = [label for label in labels 
                if label.name in parcels_temporal]

ROIs_frontal = [label for label in labels 
                if label.name in parcels_frontal]

# we don't need the two ROIs as separate sublabels, so we add them together
ROI_temp = ROIs_temporal[0]
for ROI in ROIs_temporal[1:]:
    ROI_temp += ROI

ROI_front = ROIs_frontal[0]
for ROI in ROIs_frontal[1:]:
    ROI_front += ROI

# %%
# visualise the selected regions
if interactive_mode:
    brain = mne.viz.Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
                cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation('aparc')

    for ROI in ROIs_temporal + ROIs_frontal:
        brain.add_label(ROI, borders=False)

    brain = mne.viz.Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
                cortex='low_contrast', background='white', size=(800, 600))
    brain.add_annotation('aparc')
    brain.add_label(ROI_front, borders=False)
    brain.add_label(ROI_temp, borders=False)
else:
    print("\nSkipping ROI visualisation\n")


# Relevant time windows
N400_window = (.300, .500)      # most commonly seen in my literature review
P600_window = (.600, .800)      # 600 ms as a start is common, end of window debatable


# Main loop over all available datasets
keys = ['del2019', 'del2021', 'aurn2021', 'aurn2023']
for dataset_key in keys: 

    # Importing preprocessed data, forward solution and averaged source estimates
    file_loc = os.path.join(os.getcwd(), 'data', 'processed_evokeds', dataset_key + '.pickle')
    with open(file_loc, 'rb') as f:
        data = pickle.load(f)

    file_loc = os.path.join(os.getcwd(), 'data', 'forward_solutions', dataset_key + '-fwd.fif')
    fwd = mne.read_forward_solution(file_loc)

    file_loc = os.path.join(os.getcwd(), 'data', 'source_estimates', dataset_key + '.stc_dict.pickle')
    with open(file_loc, 'rb') as f:
        average_stcs = pickle.load(f)


    # %%
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


    # %%
    # Getting agglomerated activation time series for ROIs

    # we need the source space for this (stored as attribute of forward solution)
    src = fwd['src']

    # tcs_temporal = average_stcs['control'].extract_label_time_course(ROI_temp, src, mode='mean')
    tcs_temporal = dict((cond, stc.extract_label_time_course(ROI_temp, src, mode='mean')[0]) 
                        for cond, stc in average_stcs.items())
    tcs_frontal = dict((cond, stc.extract_label_time_course(ROI_front, src, mode='mean')[0])
                    for cond, stc in average_stcs.items())


    # %%
    # Plotting the activation time series (separate plots, one figure)

    # don't display figures in interactive mode
    if interactive_mode: plt.ioff()

    x = np.arange(start=-0.2, stop=1.2, step=0.002)
    save_loc = os.path.join(os.getcwd(), 'plots', 'activation_plots')

    # options: 'mean', 'linear regression', 'moving window'
    threshold_type = 'linear regression'

    # range of standard deviations we want to compare against
    std_multipliers = np.array([1.0, 2.0])

    # moving window of size 200 ms
    window = min_count = 100

    # leave this string empty if you don't want extra info in the filenam
    # optional_filename_string = ''
    optional_filename_string = '_' + threshold_type

    # less copy-pasting of code if we put it in a loop
    time_course_dict = {'temporal':tcs_temporal, 'frontal':tcs_frontal}

    for lobe_name, tcs in time_course_dict.items():
        # just the activations
        fig, axs = plt.subplots(ncols=1,
                                nrows=len(order),
                                sharex=True,
                                dpi=800)

        # I want the y-axis to be consistent between the plots
        # so we're finding the max/min values of all plots together to set the ylimits
        maxval, minval = 0, 0

        for ax, key in zip(axs, order):
            y = tcs[key]

            # simple mean with std interval
            if threshold_type == 'mean':
                error = std_multipliers * np.std(np.array(y))
                ax.hlines(y=np.mean(y), xmin=x[0], xmax=x[-1], linestyle='dotted')
                for e in error: ax.fill_between(x, np.mean(y)-e, np.mean(y)+e, alpha=0.25, color='#1f77b4', linewidth=0)

            # linear regression with std calculated on detrended data
            if threshold_type == 'linear regression':
                m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                error = std_multipliers * np.std(np.array(y) - m*x + c)
                ax.plot(x, m*x + c, linestyle='dotted')
                for e in error: ax.fill_between(x, m*x+c + e, m*x+c - e, alpha=0.25, color='#1f77b4', linewidth=0)

            # moving average and std
            if threshold_type == 'moving window':
                window_avg = bn.move_mean(y, window=window, min_count=min_count)
                window_std = np.array([std_multipliers]).T * bn.move_std(y, window=window, min_count=min_count)

                # we shift the x-axis by half the window size so it lines up with the actual data
                ax.plot(x - window*0.002/2, window_avg, linestyle='dotted', color='#1f77b4')
                for std in window_std: ax.fill_between(x - window*0.002/2, window_avg-std, window_avg+std, alpha=0.25, color='#1f77b4', linewidth=0)

            ax.plot(x, y, label=key, color='#1f77b4')

            # storing the max/min values
            if max(y) > maxval: maxval = max(y)
            if min(y) < minval: minval = min(y)

        for ax in axs:
            ax.legend(loc='upper left')
            ax.axvspan(*N400_window, alpha=0.2, color='grey')
            ax.axvspan(*P600_window, alpha=0.2, color='grey')

            ax.grid(visible=True)
            ax.set_ylim(bottom=minval*1.1, top=maxval*1.1)
            ax.set_xlim(left=x[0], right=x[-1])

        fig.suptitle(f'Estimated activation in {lobe_name} lobe')

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity_separate' + optional_filename_string + '.png')
        fig.savefig(file_loc)

        # the contrasts
        fig, axs = plt.subplots(ncols=1,
                                nrows=len(contrasts),
                                sharex=True,
                                dpi=800)

        maxval, minval = 0, 0

        for ax, (c1, c2) in zip(axs, contrasts):
            y = tcs[c2] - tcs[c1]

            # simple mean with std interval
            if threshold_type == 'mean':
                error = std_multipliers * np.std(np.array(y))
                ax.hlines(y=np.mean(y), xmin=x[0], xmax=x[-1], linestyle='dotted')
                for e in error: ax.fill_between(x, np.mean(y)-e, np.mean(y)+e, alpha=0.25, color='#1f77b4', linewidth=0)

            # linear regression with std calculated on detrended data
            if threshold_type == 'linear regression':
                m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                error = std_multipliers * np.std(np.array(y) - m*x + c)
                ax.plot(x, m*x + c, linestyle='dotted')
                for e in error: ax.fill_between(x, m*x+c + e, m*x+c - e, alpha=0.25, color='#1f77b4', linewidth=0)

            # moving average and std
            if threshold_type == 'moving window':
                window_avg = bn.move_mean(y, window=window, min_count=min_count)
                window_std = np.array([std_multipliers]).T * bn.move_std(y, window=window, min_count=min_count)

                # we shift the x-axis by half the window size so it lines up with the actual data
                ax.plot(x - window*0.002/2, window_avg, linestyle='dotted', color='#1f77b4')
                for std in window_std: ax.fill_between(x - window*0.002/2, window_avg-std, window_avg+std, alpha=0.25, color='#1f77b4', linewidth=0)

            ax.plot(x, y, label=f'{c2} - {c1}', color='#1f77b4')

            # storing the max/min values
            if max(y) > maxval: maxval = max(y)
            if min(y) < minval: minval = min(y)

        for ax in axs:
            ax.legend(loc='upper left')
            ax.axvspan(*N400_window, alpha=0.2, color='grey')
            ax.axvspan(*P600_window, alpha=0.2, color='grey')

            ax.grid(visible=True)
            ax.set_ylim(bottom=minval*1.1, top=maxval*1.1)
            ax.set_xlim(left=x[0], right=x[-1])

        fig.suptitle(f'Contrasted estimated activation in {lobe_name} lobe')

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity_contrasts_separate' + optional_filename_string + '.png')
        fig.savefig(file_loc)
