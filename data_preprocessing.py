# %% imports
import pandas as pd
import mne

# %% data pipeline from df to evoked
def evoked_pipeline(df, subject_no):
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
    dfs_conditions = [df_subject[df_subject['Condition'] == cond] for cond in conditions.keys()]
    del df_subject

    # create info object
    info = mne.create_info(ch_names = electrodes, 
                        sfreq = 500,
                        ch_types = 'eeg')

    # create raw objects per-condition
    raws_conditions = [mne.io.RawArray(df[electrodes].transpose(), info) for df in dfs_conditions]
    del dfs_conditions

    # create epoch object per-condition
    epoch_conditions = [mne.make_fixed_length_epochs(raw, duration=1.4, id=conditions[cond]) for raw, cond in zip(raws_conditions, conditions.keys())]
    del raws_conditions

    # concatenate the per-condition epochs objects into a single object 
    epochs = mne.concatenate_epochs(epoch_conditions)

    # create evokeds object
    evokeds = epochs.average(by_event_type = True)
    evokeds = dict(zip(conditions.keys(), evokeds))

    return evokeds

if __name__ == '__main__':
    pass
    # %% select which dataset to use
    data_folder = 'C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\'

    f = data_folder + 'dbc_2019\\dbc_data.csv'
    # f = data_folder + 'dbc_2021\\dbc_data.csv'
    # f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer EEG epochs
    # f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer reading times

    df_data = pd.read_csv(f)

    # %% run on df
    evokeds = dict((i, evoked_pipeline(df_data, i)) for i in sorted(df_data['Subject'].unique()))
    print(evokeds[sorted(df_data['Subject'].unique())[0]])