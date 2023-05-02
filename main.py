import os
import subprocess

if __name__ == '__main__':
    cwd = os.getcwd()
    scripts = [
        'data_preprocessing.py',
        'mne_processes.py',
        'activation_plots.py',
        'significance_intervals.py'
    ]
    
    for script in scripts:
        f = os.path.join(cwd, script)
        try:
            print(f'\nAttempting execution of {f}\n')
            subprocess.run('python ' + f)
        except Exception:
            print(f'\nFailed execution of {f}\n')
            break