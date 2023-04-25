# %% imports
import mne
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn

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
# Specifying some stuff

plot_kwargs = {'linewidth':20, 'linestyle':'-', 'solid_capstyle':'butt', 'alpha':0.5}

x = np.arange(start=-0.2, stop=1.2, step=0.002)
y_plotlevels = [0, 2, 5, 9]
save_loc = os.path.join(os.getcwd(), 'plots', 'significance_plots')

# options: 'mean', 'linear regression', 'moving window'
threshold_type = 'linear regression'

# range of standard deviations we want to compare against
std_multipliers = np.array([0.5, 1.5])
std_multipliers = np.flip(np.sort(std_multipliers)) # this needs to be in descending order

# moving window of size 200 ms
window = min_count = 100

# leave this string empty if you don't want extra info in the filenam
# optional_filename_string = ''
optional_filename_string = '_' + threshold_type

# Relevant time windows
N400_window = (.300, .500)      # most commonly seen in my literature review
# N400_window = (.350, .450)      # narrow N400 window
P600_window = (.600, .800)      # 600 ms as a start is common, end of window debatable
# P600_window = (.600, 1.)        # broad P600 window

# one subplot for each ROI
fig, axs = plt.subplots(ncols=2,
                        nrows=1,
                        # sharex=True,'
                        sharey=True,
                        figsize=(15,5),
                        dpi=800)

keys = ['del2019', 'del2021', 'aurn2021', 'aurn2023']
for y_plot, dataset_key in zip(y_plotlevels, keys):
    print(f'\nWorking on dataset {dataset_key}\n')

    # Importing preprocessed data, forward solution
    file_loc = os.path.join(os.getcwd(), 'data', 'processed_evokeds', dataset_key + '.pickle')

    with open(file_loc, 'rb') as f:
        data = pickle.load(f)

    # now the forward solution
    file_loc = os.path.join(os.getcwd(), 'data', 'forward_solutions', dataset_key + '-fwd.fif')

    fwd = mne.read_forward_solution(file_loc)


    # lastly, the averaged source estimates
    file_loc = os.path.join(os.getcwd(), 'data', 'source_estimates', dataset_key + '.stc_dict.pickle')

    with open(file_loc, 'rb') as f:
        average_stcs = pickle.load(f)


    # Defining contrasts
    # These are the same contrasts as those taken in the original papers

    if dataset_key == 'del2019':
        A = 'control'
        B = 'script-related'
        C = 'script-unrelated'

        contrasts = [(A, B), (A, C)]
        order = [A, B, C]

    if dataset_key == 'del2021':
        A = 'baseline'
        B = 'plausible'
        C = 'implausible'

        contrasts = [(A, B), (A, C), (B, C)]
        order = [A, B, C]

    if dataset_key == 'aurn2021':
        A = 'A'
        B = 'B'
        C = 'C'
        D = 'D'

        contrasts = [(A, B), (A, C), (A, D), (C, D)]
        order = [A, B, C, D]

    if dataset_key == 'aurn2023':
        A = 'A'
        B = 'B'
        C = 'C'

        contrasts = [(A, B), (A, C)]
        order = [A, B, C]


    # Getting agglomerated activation time series for ROIs
    # we need the source space for this (stored as attribute of forward solution)
    src = fwd['src']

    # tcs_temporal = average_stcs['control'].extract_label_time_course(ROI_temp, src, mode='mean')
    tcs_temporal = dict((cond, stc.extract_label_time_course(ROI_temp, src, mode='mean')[0]) 
                        for cond, stc in average_stcs.items())
    tcs_frontal = dict((cond, stc.extract_label_time_course(ROI_front, src, mode='mean')[0])
                    for cond, stc in average_stcs.items())

    # less copy-pasting of code if we put it in a loop
    time_course_dict = {'temporal':tcs_temporal, 'frontal':tcs_frontal}
    
    # for both ROIs
    for ax, (lobe_name, tcs) in zip(axs, time_course_dict.items()):
        # we only do this for the contrasted time courses
        for j, (c1, c2) in enumerate(contrasts):
            y = tcs[c2] - tcs[c1]

            # these are going to contain 0's (i.e. 'False') wherever the threshold is reached
            # this is going to be used as a masking array, where 'False' is transparent
            positive_significance_bool = np.ones(shape=(len(std_multipliers), len(y)), dtype=int)
            negative_significance_bool = np.ones(shape=(len(std_multipliers), len(y)), dtype=int)

            # simple mean with std interval
            if threshold_type == 'mean':
                error = std_multipliers * np.std(np.array(y))

                for i, e in enumerate(error): 
                    positive_significance_bool[i] -= (y > np.mean(y) + e)
                    negative_significance_bool[i] -= (y < np.mean(y) - e)

            # linear regression with std calculated on detrended data
            if threshold_type == 'linear regression':
                m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
                error = std_multipliers * np.std(np.array(y) - m*x + c)

                for i, e in enumerate(error): 
                    positive_significance_bool[i] -= (y > m*x+c + e)
                    negative_significance_bool[i] -= (y < m*x+c - e)

            # moving average and std
            # if threshold_type == 'moving window':
            #     window_avg = bn.move_mean(y, window=window, min_count=min_count)
            #     window_std = np.array([std_multipliers]).T * bn.move_std(y, window=window, min_count=min_count)

                # we shift the x-axis by half the window size so it lines up with the actual data
                # x_shift = x - window*0.002/2
                # for i, e in enumerate(error): 
                #     positive_significance_bool[i] += np.ones(shape=y.shape, dtype=int) * (y > m*x+c + e)
                #     negative_significance_bool[i] += np.ones(shape=y.shape, dtype=int) * (y < m*x+c - e)
            
            # let's plot thick lines at the intervals
            for arr in positive_significance_bool:
                intervals = np.ma.array(x, mask=arr)
                ax.plot(intervals, np.full_like(intervals, fill_value=y_plot+j), c="green",
                        **plot_kwargs)
            
            for arr in negative_significance_bool:
                intervals = np.ma.array(x, mask=arr)
                ax.plot(intervals, np.full_like(intervals, fill_value=y_plot+j), c="red",
                        **plot_kwargs)

for ax, lobe_name in zip(axs, ['Temporal', 'Frontal']):
    ax.set_title(f'{lobe_name}')
    ax.set_xlim(left=x[0], right=x[-1])
    ax.axvspan(*N400_window, alpha=0.2, color='grey')
    ax.axvspan(*P600_window, alpha=0.2, color='grey')
    ax.grid(visible=True)

axs[0].invert_yaxis()

y_labels = ['Delogu (2019) B-A',
            'Delogu (2019) C-A',
            'Delogu (2021) B-A',
            'Delogu (2021) C-A',
            'Delogu (2021) C-B',
            'Aurnhammer (2021) B-A',
            'Aurnhammer (2021) C-A',
            'Aurnhammer (2021) D-A',
            'Aurnhammer (2021) D-C',
            'Aurnhammer (2023) B-A',
            'Aurnhammer (2023) C-A']
axs[0].set_yticks(list(range(len(y_labels))), labels=y_labels)


# %%