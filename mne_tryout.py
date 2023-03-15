# %% imports
import pandas as pd
import mne
from data_preprocessing import evoked_pipeline

# check if we're running in interactive python (notebook or IPython shell)
# https://discourse.jupyter.org/t/find-out-if-my-code-runs-inside-a-notebook-or-jupyter-lab/6935/4
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is None:
        # we have IPython installed but not running from IPython
        interactive_mode = False
    else:
        from IPython.core.interactiveshell import InteractiveShell
        format = InteractiveShell.instance().display_formatter.format
        if len(format(_checkhtml, include="text/html")[0]):
            # TODO: need to check for qtconsole here!
            pass
        else:
            interactive_mode = False
except:
    # We do not even have IPython installed
    interactive_mode = False


# %% select which dataset to use
data_folder = 'C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\'

f = data_folder + 'dbc_2019\\dbc_data.csv'
# f = data_folder + 'dbc_2021\\dbc_data.csv'
# f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer EEG epochs
# f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer reading times

df_data = pd.read_csv(f)
evokeds = dict((i, evoked_pipeline(df_data, i)) for i in sorted(df_data['Subject'].unique()))




# %% Forward operator with template head MRI
# https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html

import os.path as op
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # source space
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # bem surfaces

# using the first subject's info object, should work for every subject in dataset
info = evokeds[sorted(evokeds.keys())[0]][0]['control'].info



# %% opens external window if executed as a jupyter code cell
# Check that the locations of EEG electrodes is correct with respect to MRI

# check if running in interactive python, skip if no
if interactive_mode:
    mne.viz.plot_alignment(
            info, src=src, eeg=['original', 'projected'], trans=trans,
            show_axes=True, mri_fiducials=True, dig='fiducials')
else:
    print("\nSkipping visualisation of montage\n")


# %% make forward solution for this dataset
# set n_jobs = -1 to use all available cores for parallel processing
fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=-1)



# %% Computing inverse solutions
# https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

i = 5
evoked = evokeds[i][0]['control']
cov = evokeds[i][1]

# inverse operator
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, 
                                             loose=1.,              # loose=0. fixed orientations, loose=1. free orientations
                                             depth=0.8,             # how to weight (or normalize) the forward using a depth prior. default is 0.8
                                             verbose=True)

# apply inverse
snr = 3.0
lambda2 = 1.0 / snr ** 2
kwargs = dict(initial_time=0.08, hemi='lh', subjects_dir=subjects_dir,
              size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
              smoothing_steps=7)

stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))




# %% opens external window if executed as jupyter code cell

# check if running in interactive python, skip if no
if interactive_mode:
    brain = stc.plot(figure=1, **kwargs)
    brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)
else:
    print("\nSkipping visualisation of inverse result\n")

# %%
