# Thesis

This repository contains the code written for the master's thesis "On the Reliability of Minimum-Norm Source Estimation in Studies of the Event-Related Potential — An investigation of the generators of the N400 and P600 components of the ERP".

Data can be found at the following repositories, and should be unpacked in the ```data``` folder:

* Delogu et al. (2019) — https://github.com/hbrouwer/dbc2019rerps
* Delogu et al. (2021) — https://github.com/hbrouwer/dbc2021rerps
* Aurnhammer et al. (2021) — https://github.com/caurnhammer/PLOSONE21lmerERP
* Aurnhammer et al. (2023) — https://github.com/caurnhammer/psyp23rerps

Each python file can either be run as a script, or ```main.py``` can be run to execute all scripts in the correct order.

Description of files:

* ```main.py```:                    runs all other scripts in order (except for ```visualise_estimates.py```)
* ```data_preprocessing.py```:      preprocessing of csv files from datasets, stores results into pickled object files.
* ```mne_processes.py```:           imports pickled objects from files produced by ```data_preprocessing.py``` and runs MNE-related calculations. Results are stored in MNE ```.fif``` files and, again, in pickled object files.
* ```activation_plots.py```:        imports objects from files produced by ```mne_processes.py``` and calculates/plots the estimated activations of the regions of interest. Produced plots are saved in the ```plots``` folder of the local repository.
* ```significance_intervals.py```:  imports objects from files produced by ```mne_processes.py``` and calculates the contrasts of estimated activations. The regions in which these contrasts surpass predefined thresholds are then plotted. Produced plots are saved in the ```plots``` folder of the local repository.
* ```visualise_estimates.py```:     imports objects from files produces by ```mne_processes.py``` and presents an interactive window in which estimated activity can be inspected. Can only be run in interactive Python (IPython shell/Jupyter notebook/VS Code cells/etc).
* ```recreate_ERPs.py```:           imports objects from files produced by ```data_preprocessing.py``` and calculates grand average ERP waveforms. Produced plots are saved in the ```plots``` folder of the local repository
