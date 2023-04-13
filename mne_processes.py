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




# kwargs for mne.minimum_norm.make_inverse_operator and mne.minimum_norm.apply_inverse
make_inverse_kwargs = dict(loose=0.2,       # loose=0. fixed orientations, loose=1. free orientations
                           depth=2,         # how to weight (or normalize) the forward using a depth prior. default is 0.8, but [2.0 , 5.0] is a better range for EEG
                           )
apply_inverse_kwargs = dict(method='dSPM')

SNR = 3.0

# generator function, so we can halt calculation of stcs to preserve memory
def stc_task(data):
    # all conditions in this dataset
    first_subject = sorted(data.keys())[0]
    conditions = list(set(data[first_subject].keys()) - {'noise_covariance'})

    # for all subjects
    for i, val in data.items():
        stc_conditions = []
        # for all conditions
        for cond in conditions:
            # get the evoked for this condition, and the covariance matrix
            evoked, cov = val[cond], val['noise_covariance']
            # append source time course (taking the abs() means we're only looking at magnitude)
            stc_conditions.append(abs(inverse_solution(evoked, cov, fwd, 
                                        snr=SNR, 
                                        make_inverse_kwargs=make_inverse_kwargs, 
                                        apply_inverse_kwargs=apply_inverse_kwargs)))
        
        # halt rest of loop to reduce memory demand
        yield dict(zip(conditions, stc_conditions))



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
    forward_output_folder = os.path.join(os.getcwd(), 'data', 'forward_solutions')
    if not os.path.isdir(forward_output_folder):
        os.mkdir(forward_output_folder)

    stc_output_folder = os.path.join(os.getcwd(), 'data', 'source_estimates')
    if not os.path.isdir(stc_output_folder):
        os.mkdir(stc_output_folder)

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

            # write forward solution to file
            forward_output_file = os.path.join(forward_output_folder, key + '-fwd.fif')
            print(f'Writing forward solution to file {forward_output_file} ...')
            mne.write_forward_solution(forward_output_file, fwd, overwrite=True)

            # create stc worker function
            stc_worker = stc_task(data)

            # number of subjects
            N = len(data.keys())

            # initialise average stc dict (placing the first subject's stcs in there)
            print("Calculating first subject's source estimate...")
            average_stcs = next(stc_worker)

            # divide by N to get actual contribution to the average
            for cond, stc in average_stcs.items():
                stc = stc/N

            print("Calculating remaining source estimates...")

            # it's nice to know how long this takes
            from tqdm import tqdm
            pbar = tqdm(total = N-1)

            # for all remaining subjects
            for subject_stcs in stc_worker:
                # for all condition/stc pairs
                for cond, stc in average_stcs.items():
                    # add contribution to average
                    stc += subject_stcs[cond]/N     # maybe not good practice to alter the elements we're looping over, but it works and I don't care

                pbar.update(1)

            # write source estimate/time course dictionary to file
            stc_output_file = os.path.join(stc_output_folder, key + '.stc_dict.pickle')
            print(f'Writing source time course to file {stc_output_file}')
            with open(stc_output_file, 'wb') as f:
                pickle.dump(subject_stcs, f)

        except Exception as e:
            print("Exception occurred:")
            print(e)
            print(f"Skipping forward solution with key {key} ...")