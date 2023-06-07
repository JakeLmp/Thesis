# %% THIS FILE CAN ONLY BE RUN IN INTERACTIVE PYTHON
# (jupyter notebook/IPython shell/VS Code cells)

# check if we're running in interactive python
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
if interactive_mode:
    from mne.datasets import fetch_fsaverage
    import os
    import pickle

    # Download fsaverage files
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    # load pre-calc estimations (manually choose a dataset key)
    keys = ['del2019', 'del2021', 'aurn2021', 'aurn2023']
    dataset_key = 'del2019'

    file_loc = os.path.join(os.getcwd(), 'data', 'source_estimates', dataset_key + '.stc_dict.pickle')
    with open(file_loc, 'rb') as f:
        average_stcs = pickle.load(f)

    # all conditions in this dataset
    conditions = list(set(average_stcs.keys()) - {'noise_covariance'})

    # visualisation kwargs
    kwargs = dict(initial_time=0.0, 
                  hemi='lh', 
                  subjects_dir=subjects_dir,
                  size=(600, 600), 
                  clim=dict(kind='percent', lims=[90, 95, 99]),
                  smoothing_steps=7,
                  time_viewer=True,
                  colorbar=True)

    i = j = k = 0

    # %%
    # vis all conditions
    # you might have to play around with the "color limits" > "rescaling" options to get visible results
    for i, condition in enumerate(conditions):
        brain = average_stcs[condition].plot(figure=i, **kwargs)
        brain.add_text(0.1, 0.9, condition, 'title', font_size=14)

    # %%
    # Defining contrasts and order of plotting
    # These are the same contrasts as those taken in the original papers
    if dataset_key == 'del2019':
        contrasts = [('control', 'script-related'), ('control', 'script-unrelated')]
    elif dataset_key == 'del2021':
        contrasts = [('baseline', 'plausible'), ('baseline', 'implausible'), ('plausible', 'implausible')]
    elif dataset_key == 'aurn2021':
        contrasts = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('C', 'D')]
    elif dataset_key == 'aurn2023':
        contrasts = [('A', 'B'), ('A', 'C')]

    # vis all contrasts
    for j, (A, B) in enumerate(contrasts, start=i+1):
        brain = (average_stcs[B] - average_stcs[A]).plot(figure=j, **kwargs)
        brain.add_text(0.1, 0.9, f"{B} - {A}", 'title', font_size=14)

    # %%
    # or extract png's at multiple time stamps (change these variables manually to get what you want)
    output_loc = os.path.join(os.getcwd(), 'plots', 'brain_vis_plots')
    stamps = list(range(0,1200,100))
    kwargs['time_viewer'] = False
    for k, stamp in enumerate(stamps, start=j+1):
        brain = (average_stcs['script-related'] - average_stcs['control']).copy().crop(tmin=, tmax=stamp, include_tmax=True).plot(figure=k, **kwargs)
        brain.save_image(os.path.join(output_loc, f'{stamp}ms.png'))

# %%
# not in interactive mode
else:
    print("Run file in interactive mode, otherwise visualisation will fail.")
