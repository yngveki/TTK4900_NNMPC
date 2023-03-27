import csv
from pathlib import Path
from os.path import exists
from yaml import dump

def append_test_mse(csv_path, model_name, test_name, mse, **hyperparameters):
    '''
    Custom function designed to append the mse for a specific test of a specific model to the given .csv-path
    '''
    
    assert isinstance(csv_path, Path), 'Path given must be of the \'Path\' type!'
    row = [model_name, test_name, mse]

    if hyperparameters:
        for param in hyperparameters.values():
            row.append(param)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(row)


def safe_save(path, data, filetype):
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

    else:
        return ValueError('Invalid filetype specified. Options are \'csv\' and \'yaml\'')
    
    # Perform save
    if not exists(path): # Safe to save; nothing can be overwritten
        save(path, data)
    
    else:
        filename = input("File already exists. Provide new name or \'y\' to overwrite ([enter] aborts. File-endings are automatic!): ")
        if filename != '': # Do _not_ abort save. Will overwrite if `filename=='y'`
            if filename != 'y': # Do _not_ overwrite
                path = path.parent / (filename + '.' + filetype)
            
            save(path, data)
            print(f"File written to \'{path}\'")

        else:
            print("File was not saved.")