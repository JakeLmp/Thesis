# %% imports
import pandas as pd
# import mne
from data_preprocessing import evoked_pipeline
from mne_processes import forward_solution, inverse_solution
from tqdm import tqdm

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


# %% select which dataset to use
data_folder = 'C:\\Users\\Jakob\\Downloads\\School - MSc\\Thesis\\Data\\'

f = data_folder + 'dbc_2019\\dbc_data.csv'
# f = data_folder + 'dbc_2021\\dbc_data.csv'
# f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer 2021 EEG epochs
# f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer 2021 reading times
# f = data_folder + 'adbc23_data\\adbc23_erp.csv'       # aurnhammer 2023 

df_data = pd.read_csv(f)
evokeds = dict((i, evoked_pipeline(df_data, i)) for i in tqdm(sorted(df_data['Subject'].unique())))



# %% Forward operator with template head MRI
info = evokeds[sorted(evokeds.keys())[0]]['control'][0].info

fwd_kwargs = dict(mindist=5.0, 
                  n_jobs=-1)

fwd = forward_solution(info, forward_kwargs=fwd_kwargs)


# %% Computing inverse solutions
# https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

make_inverse_kwargs = dict(loose=1.,              # loose=0. fixed orientations, loose=1. free orientations
                           depth=0.8,             # how to weight (or normalize) the forward using a depth prior. default is 0.8
                           )
apply_inverse_kwargs = dict(method='MNE')

i = list(evokeds.keys())[0] # first subject in set
# i = 5

condition = 'control'
evoked, cov = evokeds[i][condition]
stc1 = inverse_solution(evoked, cov, fwd, make_inverse_kwargs=make_inverse_kwargs, apply_inverse_kwargs=apply_inverse_kwargs)

condition = 'script-related'
evoked, cov = evokeds[i][condition]
stc2 = inverse_solution(evoked, cov, fwd, make_inverse_kwargs=make_inverse_kwargs, apply_inverse_kwargs=apply_inverse_kwargs)

condition = 'script-unrelated'
evoked, cov = evokeds[i][condition]
stc3 = inverse_solution(evoked, cov, fwd, make_inverse_kwargs=make_inverse_kwargs, apply_inverse_kwargs=apply_inverse_kwargs)


# %% Group analysis prep
from tqdm import tqdm
from copy import deepcopy

average_stcs = {'control':deepcopy(stc1), 
                'script-related':deepcopy(stc2), 
                'script-unrelated':deepcopy(stc3)}

subjects = list(evokeds.keys())[1:] # we already have the first subject's stcs

for condition in average_stcs.keys():
    # add stcs of all subjects
    for i in tqdm(subjects):
        evoked, cov = evokeds[i][condition]

        stc = inverse_solution(evoked, cov, fwd, make_inverse_kwargs=make_inverse_kwargs, apply_inverse_kwargs=apply_inverse_kwargs)

        average_stcs[condition] += stc

    # divide stc by nr of subjects
    average_stcs[condition] = average_stcs[condition] / len(evokeds)


# %%

stc_contrast_related = average_stcs['script-related'] - average_stcs['control']
stc_contrast_unrelated = average_stcs['script-unrelated'] - average_stcs['control']

# %% Get average stcs from saved files (NOT WORKING YET)

# import os

# stc_dir = os.path.abspath("C:\\Users\\Jakob\Downloads\\School - MSc\\Thesis\\Results\\")
# average_stcs = {'control':None, 'script-related':None, 'script-unrelated':None}

# for condition in average_stcs.keys():
#     # we're only taking the left hemisphere (indicated by '-lh')
#     fname = os.path.join(stc_dir, 'delogu_2019_average_stc_' + condition + '-lh.stc')
#     average_stcs[condition] = mne.read_source_estimate(fname)


# %% opens external window if executed as jupyter code cell

# check if running in interactive python, skip if no
if interactive_mode:
    from mne.datasets import fetch_fsaverage
    import os.path as op

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    kwargs = dict(initial_time=0.08, hemi='lh', subjects_dir=subjects_dir,
                size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
                smoothing_steps=7)
    
    # kwargs = dict(initial_time=0.08, subjects_dir=subjects_dir,
    #             size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
    #             smoothing_steps=7, views='flat')
    
    
    brain = stc2.plot(figure=1, **kwargs)
    brain = stc3.plot(figure=2, **kwargs)

    # brain = average_stcs['script-unrelated'].plot(figure=1, **kwargs)
    # brain = average_stcs['script-related'].plot(figure=2, **kwargs)
    
    brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

else:
    print("\nSkipping visualisation of inverse result\n")

# %%
