# imports
import pandas as pd
import mne

def evoked_pipeline(df, subject_no, verbose=False):
    """
    Pipeline from raw EEG data to evoked objects

    Args:
    df : dataframe containing EEG trial data
    subject_no : integer of subject of interest

    Returns: tuple containing
        - mne.Evoked object for each condition
        - mne.Covariance baseline covariance matrix for each epoch and condition
    """


    # subselection of dataset --- rest of function only with data for this subject
    df_subject = df[df['Subject'] == subject_no]

    # define channel names
    electrodes = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6',
                'T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10',
                    'P7','P3','Pz','P4','P8','PO9','O1','Oz','O2']
    good_electrodes = ["Fz", "Cz", "Pz", "F3", "FC1", "FC5", "F4", "FC2", "FC6",
                        "P3", "CP1", "CP5", "P4", "CP2", "CP6", "O1", "Oz", "O2"]

    other_electrodes = list(set(electrodes) - set(good_electrodes))
    
    # the three conditions in this dataset
    conditions = {'control': 0, 'script-related': 1, 'script-unrelated': 2}

    # split into per-condition dataframes
    dfs_conditions = dict((cond, df_subject[df_subject['Condition'] == cond]) for cond in conditions.keys())
    del df_subject

    # create info object
    info = mne.create_info(ch_names = electrodes, 
                        sfreq = 500,
                        ch_types = 'eeg')
    
    info['bads'] = other_electrodes

    # create raw objects per-condition
    raws_conditions = dict((cond, mne.io.RawArray(df[electrodes].transpose(), info, verbose=verbose)) 
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

    # compute covariance matrices ('auto' option commands automatic selection of best covariance estimator)
    # one matrix per condition -> CHECK THAT
    noise_cov = dict((cond, mne.compute_covariance(epochs, tmax=0, method='auto', verbose=verbose)) 
                        for cond, epochs in epoch_conditions.items())

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

    return dict((cond, [evokeds[cond], noise_cov[cond]]) for cond in conditions.keys())

if __name__ == '__main__':
    # select which dataset to use
    data_folder = 'C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\'

    f = data_folder + 'dbc_2019\\dbc_data.csv'
    # f = data_folder + 'dbc_2021\\dbc_data.csv'
    # f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer EEG epochs
    # f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer reading times

    df_data = pd.read_csv(f)

    # run on df
    evokeds = dict((i, evoked_pipeline(df_data, i)) for i in sorted(df_data['Subject'].unique()))
    print(evokeds[sorted(df_data['Subject'].unique())[0]])