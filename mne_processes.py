# %% everything in a cell
import mne

def forward_solution(info, mindist=5.0, n_jobs=-1):
    """
    Forward operator with template head MRI
    https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html

    Args:
    info : mne.Info object containing channel names etc.
    mindist : 5.0 : mindist argument of mne.make_forward_solution
    n_jobs : -1 : n_jobs argument of mne.make_forward_solution

    Returns:
        - mne.Forward object : return value of mne.make_forward_solution
    """

    from mne.datasets import fetch_fsaverage
    import os.path as op

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # source space
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # bem surfaces

    # make forward solution for this dataset
    # set n_jobs = -1 to use all available cores for parallel processing
    fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=mindist, n_jobs=n_jobs)

    return fwd


# putting this globally because it's constant anyway
from mne.datasets import fetch_fsaverage
import os.path as op

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


def inverse_solution(evoked, cov, fwd):
    """
    Computing inverse solutions
    https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

    Args:
    evoked : mne.Evoked object containing to-fit-on data
    cov : mne.Covariance object that is accessory to the provided evoked
    fwd : mne.Forward object containing forward solution to use in inverse calculation

    Returns: 
        - abs() of mne.SourceEstimate object : absolute values of return value of mne.minimum_norm.apply_inverse
    """

    # inverse operator
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, 
                                                 loose=1.,              # loose=0. fixed orientations, loose=1. free orientations
                                                 depth=0.8,             # how to weight (or normalize) the forward using a depth prior. default is 0.8
                                                 verbose=True)

    # apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    stc = abs(mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'MNE', verbose=True))

    return stc


if __name__ == '__main__':
    import pandas as pd
    from data_preprocessing import evoked_pipeline

    # select which dataset to use
    data_folder = 'C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\'

    f = data_folder + 'dbc_2019\\dbc_data.csv'
    # f = data_folder + 'dbc_2021\\dbc_data.csv'
    # f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer EEG epochs
    # f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer reading times

    df_data = pd.read_csv(f)

    # run on df
    evokeds = dict((i, evoked_pipeline(df_data, i)) for i in sorted(df_data['Subject'].unique()))
    
    # get forward operator
    fwd = forward_solution(info = evokeds[sorted(evokeds.keys())[0]][0]['control'].info)

    # get inverse solution
    i = 5
    evoked = evokeds[i][0]['script-unrelated']
    cov = evokeds[i][1]

    stc = inverse_solution(evoked, cov, fwd)

    # %% assuming we're running in interactive mode
    kwargs = dict(initial_time=0.08, hemi='lh', subjects_dir=subjects_dir,
                size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
                smoothing_steps=7)

    brain = stc.plot(figure=1, **kwargs)
    brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)
# %%
