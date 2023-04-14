# %% imports
import mne
import os
import pickle

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



# %% 
# Importing preprocessed data, forward solution

# Manually select the key of the dataset to use
# available keys: del2019, del2021, aurn2021, aurn2023
dataset_key = 'del2019'

file_loc = os.path.join(os.getcwd(), 'data', 'processed_evokeds', dataset_key + '.pickle')

with open(file_loc, 'rb') as f:
    data = pickle.load(f)

# now the forward solution
file_loc = os.path.join(os.getcwd(), 'data', 'forward_solutions', dataset_key + '-fwd.fif')

fwd = mne.read_forward_solution(file_loc)


# lastly, the averaged source estimates
file_loc = os.path.join(os.getcwd(), 'data', 'source_estimates', dataset_key + '.stc_dict.pickle')

with open(file_loc, 'rb') as f:
    average_stcs = pickle.load(f)




# %%
# Relevant time windows
N400_window = (.300, .500)      # most commonly seen in my literature review
# N400_window = (.350, .450)      # narrow N400 window
P600_window = (.600, .800)      # 500 ms as a start is common, end of window debatable
# P600_window = (.600, 1.)        # broad P600 window

# crop the stcs to the specified windows, then take the mean activation
# do this on a copy, so we don't have to recalculate the average activations
N400_average = dict((cond, stc.copy().crop(tmin=N400_window[0], 
                                           tmax=N400_window[1]).mean())
                        for cond, stc in average_stcs.items())
P600_average = dict((cond, stc.copy().crop(tmin=P600_window[0], 
                                           tmax=P600_window[1]).mean())
                        for cond, stc in average_stcs.items())



# %%
# Defining contrasts
# These are the same contrasts as those taken in the original papers

if dataset_key == 'del2019':
    A = 'control'
    B = 'script-related'
    C = 'script-unrelated'

    contrasts = [(A, B), (A, C)]

if dataset_key == 'del2021':
    A = 'baseline'
    B = 'plausible'
    C = 'implausible'

    contrasts = [(A, B), (A, C), (B, C)]

if dataset_key == 'aurn2021':
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'

    contrasts = [(A, B), (A, C), (A, D), (C, D)]

if dataset_key == 'aurn2023':
    A = 'A'
    B = 'B'
    C = 'C'

    contrasts = [(A, B), (A, C)]


# %%
# Calculating contrasted source estimates

N400_results = []
P600_results = []

for c1, c2 in contrasts:
    x = N400_average[c2] - N400_average[c1]
    x += x.data.min()
    N400_results.append(x.copy())

    x = P600_average[c2] - P600_average[c1]
    x += x.data.min()
    P600_results.append(x.copy())

# %%
# Plotting the results

# import matplotlib.pyplot as plt

# # Download fsaverage files to use as head model
# from mne.datasets import fetch_fsaverage
# fs_dir = fetch_fsaverage(verbose=True)
# subjects_dir = os.path.dirname(fs_dir)

# # specify MNE plotting kwargs
# kwargs = dict(hemi='lh', 
#               subjects_dir=subjects_dir,
#               colormap='mne', 
#             #   clim='auto',
#               clim=dict(kind='value', lims=[70, 85, 99]),
#               smoothing_steps=7,
#               backend='matplotlib',
#               )

# # create figure with room for all the contrasts
# fig = plt.figure(layout='constrained', figsize=(5*len(contrasts), 6))
# subfigs = fig.subfigures(nrows = 1, ncols = len(contrasts), 
#                          wspace=0.07)

# # do the plotting for N400
# for sfig, result in zip(subfigs, N400_results):
#     result.plot(figure = sfig, 
#                 **kwargs) 

# # check if output directory already exists. if not, make it
# plot_folder = os.path.join(os.getcwd(), 'plots')
# if not os.path.isdir(plot_folder):
#     os.mkdir(plot_folder)

# # save figure
# fname = os.path.join(plot_folder, dataset_key + '_N400_contrasts.png')
# fig.savefig(fname = fname, dpi = 800)



# %%
# Interactive visualisation of estimates

# check if running in interactive python, skip if no
if interactive_mode:
    from mne.datasets import fetch_fsaverage
    import os.path as op

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)

    kwargs = dict(initial_time=0.0, hemi='lh', subjects_dir=subjects_dir,
                size=(600, 600), clim=dict(kind='percent', lims=[90, 95, 99]),
                smoothing_steps=7)
    
    # this is where you select what to visualise 
    # change the stc object to whatever you want to see

    i=1
    for result in P600_results:
        brain = result.plot(figure=i, **kwargs); i+=1
    # brain = (P600_average[C] - P600_average[A]).plot(figure=i, **kwargs); i+=1
    
    # brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

else:
    print("\nSkipping visualisation of inverse result\n")
# %%
