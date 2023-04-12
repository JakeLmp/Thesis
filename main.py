# %% imports
import mne
from mne_processes import inverse_solution
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
dataset_key = 'del2021'

file_loc = os.path.join(os.getcwd(), 'data', 'processed_evokeds', dataset_key + '.pickle')

with open(file_loc, 'rb') as f:
    data = pickle.load(f)

# now the forward solution
file_loc = os.path.join(os.getcwd(), 'data', 'forward_solutions', dataset_key + '-fwd.fif')

fwd = mne.read_forward_solution(file_loc)




# %%
# Calculating Source Time Courses

# all conditions in this dataset
first_subject = sorted(data.keys())[0]
conditions = list(set(data[first_subject].keys()) - {'noise_covariance'})

# kwargs for mne.minimum_norm.make_inverse_operator and mne.minimum_norm.apply_inverse
make_inverse_kwargs = dict(loose=0.2,       # loose=0. fixed orientations, loose=1. free orientations
                           depth=2,         # how to weight (or normalize) the forward using a depth prior. default is 0.8, but [2.0 , 5.0] is a better range for EEG
                           )
apply_inverse_kwargs = dict(method='dSPM')

SNR = 3.0

# generator function, so we can halt calculation of stcs to preserve memory
def stc_task():
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


# --- HOW TO USE THE WORKER : average stc over subjects example ---

# create stc worker function
stc_worker = stc_task()

# number of subjects
N = len(data.keys())

# initialise average stc dict (placing the first subject's stcs in there)
average_stcs = next(stc_worker)

# divide by N to get actual contribution to the average
for cond, stc in average_stcs.items():
    stc = stc/N

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
# Print the condition keywords for this dataset 
# ('noise_covariance' is not one of them)
print(data[first_subject].keys())

# %%
# Define your contrasts here
A = 'baseline'
B = 'plausible'
C = 'implausible'

c1, c2 = A, B
comp = N400_average
x = comp[c2] - comp[c1]     # this does the contrast
x += x.data.min()           # we can't see negative magnitude, how do we fix that?

association_1 = x

c1, c2 = A, C
comp = N400_average
x = comp[c2] - comp[c1]
x += x.data.min()

association_2 = x

c1, c2 = C, B
comp = P600_average
x = comp[c2] - comp[c1]
x += x.data.min()

plausibility = x



# %%
# visualisation of results

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
    brain = association_1.plot(figure=i, **kwargs); i+=1
    brain = association_2.plot(figure=i, **kwargs); i+=1
    brain = plausibility.plot(figure=i, **kwargs); i+=1
    # brain = (P600_average[C] - P600_average[A]).plot(figure=i, **kwargs); i+=1
    
    brain.add_text(0.1, 0.9, 'MNE', 'title', font_size=14)

else:
    print("\nSkipping visualisation of inverse result\n")
# %%
