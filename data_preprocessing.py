# imports
import pandas as pd
import mne

def evoked_pipeline(df, subject_no, all_electrodes, bad_electrodes, verbose=False):
    """
    Pipeline from raw EEG data to evoked objects

    Args:
    df : dataframe containing EEG trial data
    subject_no : integer of subject of interest
    all_electrodes : list of str : list of electrode names present in the header of df
    bad_electrodes : list of str : list of electrode names that should not be considered good data
    verbose : bool (default False) : controls verbosity of MNE methods

    Returns: dict containing
        - mne.Evoked object for this subject, for each condition
        - mne.Covariance baseline covariance matrix for this subject
    """

    # the conditions in this dataset (enumerate in dict like 'condition':i)
    conditions = dict((cond, i) for i, cond in enumerate(df['Condition'].unique()))
    
    # subselection of dataset --- rest of process only with data for this subject
    df_subject = df[df['Subject'] == subject_no]

    # split into per-condition dataframes
    dfs_conditions = dict((cond, df_subject[df_subject['Condition'] == cond]) for cond in conditions.keys())
    del df_subject

    # create info object
    info = mne.create_info(ch_names = all_electrodes, 
                        sfreq = 500,
                        ch_types = 'eeg')
    
    info['bads'] = bad_electrodes

    # create raw objects per-condition
    raws_conditions = dict((cond, mne.io.RawArray(df[all_electrodes].transpose(), info, verbose=verbose)) 
                                for cond, df in dfs_conditions.items())
    del dfs_conditions

    # create epoch object per-condition, shift timelines so that events are at t=0
    epoch_conditions = dict((cond, mne.make_fixed_length_epochs(raw, 
                                                                duration=1.4, 
                                                                id=conditions[cond],
                                                                preload=True, 
                                                                verbose=verbose).shift_time(-0.2, relative=False)) 
                                                                    for cond, raw in raws_conditions.items())
    del raws_conditions

    # data is already baseline corrected, but MNE apparently wants this really badly
    for ep in epoch_conditions.values():
        ep.apply_baseline(baseline=(None, 0), verbose=verbose) # use beginning of epoch up to and including 0s

    # compute covariance matrix ('auto' option commands automatic selection of best covariance estimator)
    # noise matrix is calculated on pre-stimulus interval, which is condition-independent brain activity
    # concatenate the per-condition epochs objects into a single object 
    epochs_concat = mne.concatenate_epochs(list(epoch_conditions.values()), verbose=verbose)
    noise_cov = mne.compute_covariance(epochs_concat, tmax=0, method='auto', verbose=verbose)

    # create evokeds object
    evokeds = dict((cond, epochs.average(by_event_type = False)) # only 1 event type in here 
                        for cond, epochs in epoch_conditions.items())
    del epoch_conditions

    # Read and set the EEG electrode locations, for use with fsaverage's
    # space (MNI space) for standard_1020:
    montage = mne.channels.make_standard_montage('standard_1005') # 1005 is just higher-density 1020, unused electrode positions fall away
    for evoked in evokeds.values():
        evoked.set_montage(montage, verbose=verbose)
        evoked.set_eeg_reference(projection=True, verbose=verbose)  # needed for inverse modeling, is not applied to data here

    # add the noise covariance to the dict
    data = evokeds
    data['noise_covariance'] = noise_cov

    return data

if __name__ == '__main__':
    import os
    import pickle
    from tqdm import tqdm

    # define channel names --- SHOULD THIS BE HARD-CODED??? --- it shouldnt, not the same set for each dataset
    electrodes = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
                'T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10',
                    'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2']
    good_electrodes = ["Fz", "Cz", "Pz", "F3", "FC1", "FC5", "F4", "FC2", "FC6",
                        "P3", "CP1", "CP5", "P4", "CP2", "CP6", "O1", "Oz", "O2"]

    other_electrodes = list(set(electrodes) - set(good_electrodes))


    # select data folder
    data_folder = os.path.abspath('C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\')

    # CSV file locations
    file_locs = {
        'del2019':os.path.join(data_folder, 'dbc2019data', 'dbc_data.csv'),                        # delogu 2019
        'del2021':os.path.join(data_folder, 'dbc2021data', 'dbc_data.csv'),                        # delogu 2021
        'aurn2021':os.path.join(data_folder, 'PLOSONE21lmerERP_ObservedData', 'CAPExp.csv'),       # aurnhammer 2021 EEG epochs
        # 'aurn2021b':os.path.join(data_folder, 'PLOSONE21lmerERP_ObservedData', 'CAPSPR.csv'),   # aurnhammer 2021 reading times
        'aurn2023':os.path.join(data_folder, 'adbc23_data', 'adbc23_erp.csv')                      # aurnhammer 2023
    }

    # check if output directory already exists. if not, make it
    output_folder = os.path.join(data_folder, 'processed_evokeds')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    # loop through datasets, process if available
    for key, path in file_locs.items():
        try:
            # import the df
            print(f"Importing dataset {key}...")
            df_data = pd.read_csv(path)
            # run evoked calc on df
            print("Processing evokeds...")
            data = dict((i, evoked_pipeline(df_data, i, electrodes, other_electrodes)) for i in tqdm(sorted(df_data['Subject'].unique())))

            # pickle the data dict
            output_file = os.path.join(output_folder, key + '.pickle')
            print(f"Pickling into {output_file}...")
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
        
        except Exception as e:
            print("Exception occurred:")
            print(e)
            print(f"Skipping dataset with key {key} ...")

    
