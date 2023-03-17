# %% imports
import pandas as pd
import mne
from data_preprocessing import evoked_pipeline
from mne_processes import forward_solution, inverse_solution

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
# f = data_folder + 'aurn_2021\\observed\\CAPExp.csv'  # aurnhammer EEG epochs
# f = data_folder + 'aurn_2021\\observed\\CAPSPR.csv'  # aurnhammer reading times

df_data = pd.read_csv(f)
evokeds = dict((i, evoked_pipeline(df_data, i)) for i in sorted(df_data['Subject'].unique()))



# %% Forward operator with template head MRI
info = evokeds[sorted(evokeds.keys())[0]][0]['control'].info

fwd = forward_solution(info)



# %% Computing inverse solutions
# https://mne.tools/stable/auto_tutorials/inverse/40_mne_fixed_free.html#free-orientation

i = 5
evoked = evokeds[i][0]['script-related']
cov = evokeds[i][1]

stc = inverse_solution(evoked, cov, fwd)



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
    
    brain = stc.plot(figure=1, **kwargs)
    brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

else:
    print("\nSkipping visualisation of inverse result\n")


# %%
