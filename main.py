# %% imports
import mne
import os
import pickle

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
# Importing preprocessed data, forward solution

# Manually select the key of the dataset to use
# available keys: del2019, del2021, aurn2021, aurn2023
dataset_key = 'del2019'

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




# %%
# Relevant time windows
N400_window = (.300, .500)      # most commonly seen in my literature review
# N400_window = (.350, .450)      # narrow N400 window
P600_window = (.600, .800)      # 600 ms as a start is common, end of window debatable
# P600_window = (.600, 1.)        # broad P600 window

# crop the stcs to the specified windows, then take the mean activation
# do this on a copy, so we don't have to recalculate the average activations
N400_window_average = dict((cond, stc.copy().crop(tmin=N400_window[0], 
                                           tmax=N400_window[1]).mean())
                        for cond, stc in average_stcs.items())
P600_window_average = dict((cond, stc.copy().crop(tmin=P600_window[0], 
                                           tmax=P600_window[1]).mean())
                        for cond, stc in average_stcs.items())



# %%
# Defining contrasts
# These are the same contrasts as those taken in the original papers

if dataset_key == 'del2019':
    A = 'control'
    B = 'script-related'
    C = 'script-unrelated'

    contrasts = [(A, B), (A, C)]

if dataset_key == 'del2021':
    A = 'baseline'
    B = 'plausible'
    C = 'implausible'

    contrasts = [(A, B), (A, C), (B, C)]

if dataset_key == 'aurn2021':
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'

    contrasts = [(A, B), (A, C), (A, D), (C, D)]

if dataset_key == 'aurn2023':
    A = 'A'
    B = 'B'
    C = 'C'

    contrasts = [(A, B), (A, C)]


# %%
# Calculating contrasted source estimates

N400_stc_contrasts = []
P600_stc_contrasts = []

for c1, c2 in contrasts:
    x = N400_window_average[c2] - N400_window_average[c1]
    N400_stc_contrasts.append(x.copy())

    x = P600_window_average[c2] - P600_window_average[c1]
    P600_stc_contrasts.append(x.copy())


# %%
# Interactive visualisation of estimates

# check if running in interactive python, skip if no
if interactive_mode:
    from mne.datasets import fetch_fsaverage
    import os.path as op

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    kwargs = dict(initial_time=0.0, hemi='lh', subjects_dir=subjects_dir,
                size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
                smoothing_steps=7)
    
    # this is where you select what to visualise 
    # change the stc object to whatever you want to see

    i=1
    for result in N400_stc_contrasts:
        brain = result.plot(figure=i, **kwargs); i+=1
    # brain = (P600_window_average[C] - P600_window_average[A]).plot(figure=i, **kwargs); i+=1
    
    # brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

else:
    print("\nSkipping visualisation of inverse result\n")



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
# Getting agglomerated activation time series for ROIs

# we need the source space for this (stored as attribute of forward solution)
src = fwd['src']

# tcs_temporal = average_stcs['control'].extract_label_time_course(ROI_temp, src, mode='mean')
tcs_temporal = dict((cond, stc.extract_label_time_course(ROI_temp, src, mode='mean')) 
                    for cond, stc in average_stcs.items())
tcs_frontal = dict((cond, stc.extract_label_time_course(ROI_front, src, mode='mean'))
                   for cond, stc in average_stcs.items())

# %%
