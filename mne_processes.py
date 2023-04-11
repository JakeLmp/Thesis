# %% everything in a cell
import mne

# putting this globally because it's constant anyway
from mne.datasets import fetch_fsaverage
import os.path as op

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

def forward_solution(info, verbose=False, forward_kwargs=None):
    """
    Forward operator with template head MRI
    https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html

    Args:
    info : mne.Info object containing channel names etc.
    verbose : bool : controls verbosity of forward solution method
    forward_kwargs : dict : keyword arguments to be passed along to the mne.make_forward_solution method

    Returns:
        - mne.Forward object : return value of mne.make_forward_solution
    """

    from mne.datasets import fetch_fsaverage
    import os.path as op

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # source space
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # bem surfaces

    # make forward solution for this dataset
    # set n_jobs = -1 to use all available cores for parallel processing
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, verbose=verbose, **forward_kwargs)

    return fwd


def inverse_solution(evoked, cov, fwd, snr=3.0, verbose=False, make_inverse_kwargs=None, apply_inverse_kwargs=None):
    """
    Computing inverse solutions
    https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

    Args:
    evoked : mne.Evoked object containing to-fit-on data
    cov : mne.Covariance object that is accessory to the provided evoked
    fwd : mne.Forward object containing forward solution to use in inverse calculation
    snr : float (3.0) : signal-to-noise ratio used in inverse operator application
    verbose : bool : controls verbosity of output of mne methods
    make_inverse_kwargs : dict : keyword arguments to be passed along to the mne.minimum_norm.make_inverse_operator method
    apply_inverse_kwargs : dict : keywords arguments to be passed along to the mne.minimum_norm.apply_inverse method

    Returns: 
        - mne.SourceEstimate object : return value of mne.minimum_norm.apply_inverse
    """

    # inverse operator
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, 
                                                 verbose=verbose,
                                                 **make_inverse_kwargs)

    # apply inverse
    lambda2 = 1.0 / snr ** 2
    
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, verbose=verbose, **apply_inverse_kwargs)

    return stc


if __name__ == '__main__':
    import pandas as pd
    import os
    import pickle

    # select which dataset to use
    data_folder = os.path.join(os.getcwd(), 'data', 'processed_evokeds')

    file_locs = {
        'del2019':os.path.join(data_folder, 'del2019' + '.pickle'),         # delogu 2019
        'del2021':os.path.join(data_folder, 'del2021' + '.pickle'),         # delogu 2021
        'aurn2021':os.path.join(data_folder, 'aurn2021' + '.pickle'),       # aurnhammer 2021 EEG epochs
        'aurn2023':os.path.join(data_folder, 'aurn2023' + '.pickle')        # aurnhammer 2023
    }

    # check if output directory already exists. if not, make it
    output_folder = os.path.join(os.getcwd(), 'data', 'forward_solutions')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # loop through datasets, process if available
    for key, path in file_locs.items():
        try:
            # import the processed data
            print(f'Importing data for {key}...')
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # first subject's first condition info object
            first_subject = sorted(data.keys())[0]
            first_condition = list(data[first_subject].keys())[0]
            info = data[first_subject][first_condition].info

            # forward operator kwargs
            fwd_kwargs = dict(mindist=5.0, 
                            n_jobs=-1)    # set n_jobs = -1 to use all available cores for parallel processing

            # get forward operator
            print(f'Calculating forward solution for {key}...')
            fwd = forward_solution(info, forward_kwargs=fwd_kwargs)

            # write to file
            output_file = os.path.join(output_folder, key + '-fwd.fif')
            print(f'Writing forward solution to file {output_file} ...')
            mne.write_forward_solution(output_file, fwd, overwrite=True)
        
        except Exception as e:
            print("Exception occurred:")
            print(e)
            print(f"Skipping forward solution with key {key} ...")
