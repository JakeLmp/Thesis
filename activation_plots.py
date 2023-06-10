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


# %%
# Relevant time windows
N400_window = (.300, .500)      # most commonly seen in my literature review
P600_window = (.600, .800)      # 600 ms as a start is common, end of window debatable

# don't display figures in interactive mode
if interactive_mode: plt.ioff()

x = np.arange(start=-0.2, stop=1.2, step=0.002)
save_loc = os.path.join(os.getcwd(), 'plots', 'activation_plots')

if not os.path.isdir(save_loc):
    os.mkdir(save_loc)

# options: 'mean', 'linear regression', 'moving window'
threshold_type = 'linear regression'

# range of standard deviations we want to compare against
std_multipliers = np.array([1.0, 2.0])

# moving window of size 200 ms
window = min_count = 100

# leave this string empty if you don't want extra info in the filenam
# optional_filename_string = ''
optional_filename_string = '_' + threshold_type

# kwargs for threshold bands
fill_between_kwargs = {'alpha':0.5/len(std_multipliers), 
                       'color':'#1f77b4', 
                       'linewidth':0}

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

    # in the aurnhammer 2023 study, sampling frequency was doubled
    # we subsample the returned arrays to match the 500 Hz frequency
    if dataset_key == 'aurn2023':
        tcs_temporal = dict((cond, stc[0:-1:2]) for cond, stc in tcs_temporal.items())
        tcs_frontal = dict((cond, stc[0:-1:2]) for cond, stc in tcs_frontal.items())


    # %%
    # Plotting the activation time series (separate plots, one figure)

    # less copy-pasting of code if we put it in a loop
    time_course_dict = {'temporal':tcs_temporal, 'frontal':tcs_frontal}

    for lobe_name, tcs in time_course_dict.items():
        # just the activations
        fig, axs = plt.subplots(ncols=1,
                                nrows=len(order),
                                sharex=True,
                                dpi=600)

        # I want the y-axis to be consistent between the plots
        # so we're finding the max/min values of all plots together to set the ylimits
        maxval, minval = 0, 0

        for ax, key in zip(axs, order):
            y = tcs[key]

            # simple mean with std interval
            if threshold_type == 'mean':
                error = std_multipliers * np.std(np.array(y))
                ax.hlines(y=np.mean(y), xmin=x[0], xmax=x[-1], linestyle='dotted')
                for e in error: ax.fill_between(x, np.mean(y)-e, np.mean(y)+e, **fill_between_kwargs)

            # linear regression with std calculated on detrended data
            if threshold_type == 'linear regression':
                m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                error = std_multipliers * np.std(np.array(y) - m*x + c)
                ax.plot(x, m*x + c, linestyle='dotted')
                for e in error: ax.fill_between(x, m*x+c + e, m*x+c - e, **fill_between_kwargs)

            # moving average and std
            if threshold_type == 'moving window':
                window_avg = bn.move_mean(y, window=window, min_count=min_count)
                window_std = np.array([std_multipliers]).T * bn.move_std(y, window=window, min_count=min_count)

                # we shift the x-axis by half the window size so it lines up with the actual data
                ax.plot(x - window*0.002/2, window_avg, linestyle='dotted', color='#1f77b4')
                for std in window_std: ax.fill_between(x - window*0.002/2, window_avg-std, window_avg+std, **fill_between_kwargs)

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
            
        axs[-1].set_xlabel('t (s)')
        fig.suptitle('', y=0.5) # spoof location of suptitle to reduce bboxes
        # fig.suptitle(f'Estimated activation in {lobe_name} lobe')

        plt.tight_layout()

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity_separate' + optional_filename_string + '.png')
        fig.savefig(file_loc)

        # the contrasts
        fig, axs = plt.subplots(ncols=1,
                                nrows=len(contrasts),
                                sharex=True,
                                dpi=600)

        # now the same thing, but lines in one plot and without the regression/bands
        plt.figure(np.random.randint(10, high=100000))
        for key in order:
            plt.plot(x, tcs[key], label=key)
        plt.legend(loc='upper left')
        # plt.title(f'Estimated activation in {lobe_name} lobe')
        plt.ylim(minval*1.1, maxval*1.1)
        plt.xlim(x[0], x[-1])
        plt.xlabel('t (s)')
        plt.axvspan(*N400_window, alpha=0.2, color='grey')
        plt.axvspan(*P600_window, alpha=0.2, color='grey')
        plt.grid(visible=True)

        plt.tight_layout()

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity.png')
        plt.savefig(file_loc, dpi=600)
        
        # and then the contrasts
        maxval, minval = 0, 0

        for ax, (c1, c2) in zip(axs, contrasts):
            y = tcs[c2] - tcs[c1]

            # simple mean with std interval
            if threshold_type == 'mean':
                error = std_multipliers * np.std(np.array(y))
                ax.hlines(y=np.mean(y), xmin=x[0], xmax=x[-1], linestyle='dotted')
                for e in error: ax.fill_between(x, np.mean(y)-e, np.mean(y)+e, **fill_between_kwargs)

            # linear regression with std calculated on detrended data
            if threshold_type == 'linear regression':
                m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                error = std_multipliers * np.std(np.array(y) - m*x + c)
                ax.plot(x, m*x + c, linestyle='dotted')
                for e in error: ax.fill_between(x, m*x+c + e, m*x+c - e, **fill_between_kwargs)

            # moving average and std
            if threshold_type == 'moving window':
                window_avg = bn.move_mean(y, window=window, min_count=min_count)
                window_std = np.array([std_multipliers]).T * bn.move_std(y, window=window, min_count=min_count)

                # we shift the x-axis by half the window size so it lines up with the actual data
                ax.plot(x - window*0.002/2, window_avg, linestyle='dotted', color='#1f77b4')
                for std in window_std: ax.fill_between(x - window*0.002/2, window_avg-std, window_avg+std, **fill_between_kwargs)

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

        axs[-1].set_xlabel('t (s)')
        fig.suptitle('', y=0.5) # spoof location of suptitle to reduce bboxes
        # fig.suptitle(f'Contrasted estimated activation in {lobe_name} lobe')

        plt.tight_layout()

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity_contrasts_separate' + optional_filename_string + '.png')
        fig.savefig(file_loc)

        # now the same thing, but lines in one plot and without the regression/bands
        plt.figure(np.random.randint(10, high=100000))
        for c1, c2 in contrasts:
            plt.plot(x, tcs[c2] - tcs[c1], label=f'{c2} - {c1}')
        # plt.plot(x, np.vstack([tcs[c2] - tcs[c1] for c1, c2 in contrasts]))
        plt.legend(loc='upper left')
        # plt.title(f'Contrasted estimated activation in {lobe_name} lobe')
        plt.ylim(minval*1.1, maxval*1.1)
        plt.xlim(x[0], x[-1])
        plt.xlabel('t (s)')
        plt.axvspan(*N400_window, alpha=0.2, color='grey')
        plt.axvspan(*P600_window, alpha=0.2, color='grey')
        plt.grid(visible=True)

        plt.tight_layout()

        file_loc = os.path.join(save_loc, dataset_key + f'_{lobe_name}_activity_contrasts.png')
        plt.savefig(file_loc, dpi=600)

    # close all figures and move on to the next dataset
    plt.close('all')
