import csv
from pathlib import Path
from os.path import exists
from os import makedirs
from yaml import dump
from numpy import save as np_save

def append_test_mse(csv_path, model_name, test_name, mse, **hyperparameters):
    '''
    Custom function designed to append the mse for a specific test of a specific model to the given .csv-path
    '''
    
    # TODO: Check if file doesn't exist. If it doesnt, also add header!
    # TODO: Currently adds without newline! Remember to add newline to end of header when fixed automatic header-adding
    assert isinstance(csv_path, Path), 'Path given must be of the \'Path\' type!'
    row = [model_name, test_name, mse]

    if hyperparameters:
        for param in hyperparameters.values():
            row.append(param)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(row)

def make_parent_dir(dir):
    if not exists(dir):
        makedirs(dir)


def safe_save(path, data, filetype, create_parent=False, errmsgstr: str='___'):
    '''
    Custom function designed to save the given data to the given path, and safeguard for overwriting. 
    Filetype has to be specified, as different filetypes require different means of saving
    '''

    # TODO: Add bool for mkdir, and mkdir-check. Bonus: account for if dir exists, but contains file of same name, such that want to make new dir now, and not provide new names after
    assert isinstance(path, Path), 'Path given must be of the \'Path\' type!'

    # Define how to perform save
    if filetype == 'csv':
        def save(path, data):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)

                for row in data:
                    writer.writerow(row)

    elif filetype == 'yaml':
        def save(path, data):
            with open(path, "w", encoding = "utf-8") as yaml_file:
                yaml_file.write(dump(data, default_flow_style = False, allow_unicode = True, encoding = None))

    elif filetype == 'npy':
        def save(path, data):        
            np_save(path, data)

    elif filetype == 'fig':
        def save(path, fig):
            # TODO: Assert for \'fig\' to be a matplotlib-fib
            for suffix in ['.png', '.eps']:
                path = path.parent / (path.stem + suffix)
                fig.savefig(path, bbox_inches='tight')

    else:
        return ValueError('Invalid filetype specified. Options are \'csv\', \'yaml\', \'npy\' and \'fig\'')
    
    # Perform 
    if create_parent: make_parent_dir(path.parent)

    if not exists(path): # Safe to save; nothing can be overwritten
        save(path, data)
    
    else:
        filename = input(f"File ({errmsgstr}) already exists. Provide new name or \'y\' to overwrite ([enter] aborts. File-endings are automatic!): ")
        if filename != '': # Do _not_ abort save. Will overwrite if `filename=='y'`
            if filename != 'y': # Do _not_ overwrite
                path = path.parent / (filename + '.' + filetype)
            
            save(path, data)
            print(f"File written to \'{path}\'")

        else:
            print(f"File ({errmsgstr}) was not saved.")


    