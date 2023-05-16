# %%
import os
from tqdm import tqdm
import pandas as pd
import mne
from copy import deepcopy

from data_preprocessing import evoked_pipeline
from mne_processes import inverse_solution

# select data folder
data_folder = os.path.join(os.getcwd(), 'data')

# CSV file locations
file_locs = {
    'del2019':os.path.join(data_folder, 'dbc2019data', 'dbc_data.csv'),                        # delogu 2019
    'del2021':os.path.join(data_folder, 'dbc2021data', 'dbc_data.csv'),                        # delogu 2021
    'aurn2021':os.path.join(data_folder, 'PLOSONE21lmerERP_ObservedData', 'CAPExp.csv'),       # aurnhammer 2021 EEG epochs
    'aurn2023':os.path.join(data_folder, 'adbc23_data', 'adbc23_erp.csv')                      # aurnhammer 2023
}

# define 'good' channel names (taken from papers)
# only these channels will be considered in the resulting Evokeds
good_electrodes = {
    'del2019': ["Fz", "Cz", "Pz", "F3", "FC1", "FC5", "F4", "FC2", "FC6",
                "P3", "CP1", "CP5", "P4", "CP2", "CP6", "O1", "Oz", "O2"],
    'del2021': ["F3" , "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "C3",  "Cz", "C4",
                "CP5", "CP1", "CP2", "CP6", "P3","Pz", "P4", "O1",  "Oz", "O2"],
    'aurn2021':["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
                "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1",
                "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1",
                "Oz", "O2"],
    'aurn2023': ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5",
                    "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1",
                    "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1",
                    "Oz", "O2"]
}

def evoked_task(data):
    first_subject = sorted(data.keys())[0]
    conditions = list(set(data[first_subject].keys()) - {'noise_covariance'})

    for i, val in data.items():
        yield dict((condition, val[condition]) for condition in conditions)


# kwargs for mne.minimum_norm.make_inverse_operator and mne.minimum_norm.apply_inverse
make_inverse_kwargs = dict(loose=0.2,       # loose=0. fixed orientations, loose=1.0 free orientations. default is 0.2
                           depth=2,         # how to weight (or normalize) the forward using a depth prior. default is 0.8, but [2.0 , 5.0] is a better range for EEG
                           )
apply_inverse_kwargs = dict(method='dSPM')

SNR = 3.0

estimate_on_average = dict()

# loop through datasets, process if available
for dataset_key, path in file_locs.items():
    try:
        # import the df
        print(f"Importing dataset {dataset_key}...")
        df_data = pd.read_csv(path)
        
        # run evoked calc on df
        print("Processing evokeds...")
        data = dict((i, evoked_pipeline(df_data, i, good_electrodes[dataset_key])) for i in tqdm(sorted(df_data['Subject'].unique())))

        # grand average calc here
        evoked_worker = evoked_task(data)

        N = len(data.keys())
        average_evokeds = deepcopy(next(evoked_worker))

        for cond, evoked in average_evokeds.items():
            evoked = evoked/N
        
        for subject_evokeds in evoked_worker:
            for cond, evoked in average_evokeds.items():
                evoked += subject_evokeds[cond]/N
        
        # calculate the covariance on the average estimate pre-stim interval
        # pre-stimulus interval is condition-independent, get a concatenated object
        evoked_concat = mne.concatenate
        cov = mne.compute_covariance(epochs_concat, tmax=0, method='auto', verbose=verbose)
        
        # do estimation
        fwd_loc = os.path.join(os.getcwd(), 'data', 'forward_solutions', dataset_key + '-fwd.fif')
        fwd = mne.read_forward_solution(fwd_loc)

        estimate_on_average[dataset_key] = abs(inverse_solution(evoked, cov, fwd, 
                                                snr=SNR, 
                                                make_inverse_kwargs=make_inverse_kwargs, 
                                                apply_inverse_kwargs=apply_inverse_kwargs)
        


    except Exception as e:
        print("Exception occurred:")
        print(e)
        print(f"Skipping dataset with key {key} ...")




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

