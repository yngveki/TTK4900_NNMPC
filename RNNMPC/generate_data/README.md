# Explanation of the mess that is this collection of div data

This folder contains my custom framework for creating a data foundation on which to understand the Single Well system, as well as training and testing a neural network.

All csv-files that define input sequences define these in accordance with the `Timeseries` class as defined in TTK4900_NNMPC/src/utils/timeseries. The csv-files representing the output data are simple loggings of choke, gas lift, gas rate and oil rate for each timestep.

## Overview folders

Folders are in general named after what sort of input data they represent. The input is defined in csv-files. Each folder has a corresponding equally named folder within data, which holds csv, png and eps filer for all output, in their respective folders. See sections for specific output types for more detailed information.

### steps_choke
