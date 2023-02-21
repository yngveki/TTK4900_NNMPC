# Explanation of the mess that is this collection of div data

This folder contains my custom framework for creating a data foundation on which to understand the Single Well system, as well as training and testing a neural network.

All csv-files that define input sequences define these in accordance with the `Timeseries` class as defined in TTK4900_NNMPC/src/utils/timeseries. The csv-files representing the output data are simple loggings of choke, gas lift, gas rate and oil rate for each timestep.

## Overview of data folders

The folders containing the generated input (.csv) and output (.csv, .eps, .png) data are kept in the folders `inputs` and `outputs`, respectively.

The folders kept within `inputs` and `outputs` are named after the type of input profile (kept in `inputs`) that gives rise to a certain output (kept in `outputs`), such that finding output corresponding to some input from `inputs` is simply a matter of finding files with matching names within `outputs`.

*Example*: `inputs/steps_choke/step_choke_50_52.csv` containts a .csv-file defining a step in choke (with constant gas lift), the output of which is logged in `outputs/steps_choke/csv/step_choke_50_52_output_clipped.csv` and illustrated by means of .eps and .png in `outputs/steps_choke/eps/step_choke_50_52_output_clipped.eps` and `outputs/steps_choke/png/step_choke_50_52_output_clipped.png`, respectively.

The suffixes mean:
- `_output`: the file is the output from a simulation with the same filename
- `_clipped`: the file contents are clipped in the beginning, such that the period ensuring a defined initial state (the so-called "warm start") becomes omitted.

### div
This folder contains different experimental input profiles.

This specific dataset is incomplete! This is due to development processes and undocumented experimentation, only the .csv-files containing the output data are kept.

### staircases {#staircases}
This folder contains various steps in both choke and gas lift in parallel, starting at the lower bounds (0 for choke, 2000 for gas lift). The steps are performed randomly, and will not exceed the bounds:

| input type | lower bound | upper bound |
| ----- | ----- | ----- |
| choke | 0 | 100 |
| gas lift | 2000 | 10000 |

Note that these input profiles are not representative for actual use.

### steps
Some simple hand-written sequences of more realistic steps in both choke and gas lift. These sequences are more realistic in that only choke is used when choke is not maxed out, and only then is gas lift employed.

These input sequences' major weaknesses are that the step discretely from 0 to 5000 in gas lift (which is highly unrealistic, as well as the wrong lower bound for gas lift, which should instead be 2000) and that the steps in both choke and gas lift are pseudo-random, and in many cases likely too large to represent actual use-cases.

### steps_choke {#steps_choke}
This folder contains different steps in choke, while gas lift is kept at a constant 0, the rationale of which is that in the actual system, it is not desirable to use gas lift before choke is maxed out at 100% opening, and can contribute no further to increased production.

### steps_GL {#steps_gl}
This folder contains different steps in gas lift, while choke is kept at a constant 100, the rationale of which is tha tin the actual system, it is not desirable to use gas lift before choke is maxed out at 100% opening, and can contribute no further to increased production.

## Overview of python scripts
The python files kept within this directory are scripts used to either create the data within this directory, or handle it. See the descriptions below for specific details.

### generate_csvs.py
generate_csvs.py contains different segments of the format `if __name__ == '__main__' and sequences == some_name:`, which are used to select what block of the script is to be executed, by means of setting the variable `some_name` appropriately. Each block of code generates .csv-files with specific attributes, e.g. {#steps_gl}, {#steps_choke} and {#staircases}.

### generate_data.py
This file simulates the Single Well FMU for some specified input profile. Note that the .csv-file used to generate an input profile is fed through the `Timeseries` class, meaning it should be compatible/created with this in mind.

The output is a log of the choke, gas lift, gas rate and oil rate values for the full simulation, as well as a .eps and .png file illustrating the simulation results.

### load_input_data.py
This file contains the interface necessary to extract an output .csv-file as a dataloader-object, usable in training neural networks.