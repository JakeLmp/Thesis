# %% imports
import pandas as pd
import mne
from data_preprocessing import evoked_pipeline



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
import numpy as np

from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# using the first subject's control condition and info object,
# this is just for metadata, no actual experimental data is used for forward operator
raw = evokeds[2]['control']
info = raw.info

# Clean channel names to be able to use a standard 1005 montage
new_names = dict(
    (ch_name,
     ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
    for ch_name in raw.ch_names)
raw.rename_channels(new_names)

# Read and set the EEG electrode locations, which are already in fsaverage's
# space (MNI space) for standard_1020:
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling




# %% opens external window if executed as a jupyter code cell
# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')



# %% 
# set n_jobs = -1 to use all available cores for parallel processing
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=-1)



# %% Computing inverse solutions
# https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

evoked = evokeds[2]['control']

# Ad-hoc covariance matrix
cov = mne.make_ad_hoc_cov(info)

# inverse operator
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, 
                                             loose=0.,              # loose=0. fixed orientations, loose=1. free orientations
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
brain = stc.plot(figure=1, **kwargs)
brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)
