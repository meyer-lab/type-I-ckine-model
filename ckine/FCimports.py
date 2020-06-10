"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
import os
from os.path import dirname
from pathlib import Path
import numpy as np
import pandas as pd
from FlowCytometryTools import FCMeasurement

path_here = dirname(dirname(__file__))


def combineWells(samples):
    """Accepts sample array returned from importF, and array of channels, returns combined well data"""
    combinedSamples = samples[0]
    for sample in samples[1:]:
        combinedSamples.data = combinedSamples.data.append(sample.data, ignore_index=True)
    return combinedSamples


def importF(date, plate, wellRow, panel, channels, wellNum=None):
    """
    Import FCS files. Variable input: date in format mm-dd, plate #, panel #, and well letter. Output is a list of Data File Names in FCT Format
    Title/file names are returned in the array file --> later referenced in other functions as title/titles input argument
    """
    path_ = os.path.abspath("")

    pathname = path_ + "/ckine/data/flow/" + date + " Live PBMC Receptor Data/Plate " + plate + "/Plate " + plate + " - Panel " + str(panel) + " IL2R/"

    # Declare arrays and int
    file = []
    sample = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")

    for path in pathlist:
        wellID = path.name.split("_")[1]
        if wellID[0] == wellRow:
            file.append(str(path))
        else:
            unstainedWell = FCMeasurement(ID="Unstained Sample", datafile=str(path))  # Stores data from unstainedWell separately
    file.sort()
    assert file != []
    # Go through each file and assign the file contents to entry in the array sample
    for entry in file:
        sample.append(FCMeasurement(ID="Test Sample" + str(z), datafile=entry))
        z += 1
    # The array sample contains data of each file in folder (one file per entry in array)

    if wellNum is None:
        combinedSamples = combineWells(sample)  # Combines all files from samples
        combinedSamples = subtract_unstained_signal(combinedSamples, channels, unstainedWell)  # Subtracts background
        return combinedSamples.transform("hlog", channels=channels)  # Transforms and returns

    tsample = subtract_unstained_signal(sample[wellNum - 1], channels, unstainedWell)
    return tsample.transform("hlog", channels=channels)


def subtract_unstained_signal(sample, channels, unstainedWell):
    """ Subtract mean unstained signal from all input channels for a given sample. """
    for _, channel in enumerate(channels):
        meanBackground = np.mean(unstainedWell.data[channel])  # Calculates mean unstained signal for given channel
        sample[channel] = np.maximum(sample[channel] - meanBackground, 0.0)

    return sample


def compMatrix(date, plate, panel, invert=True):
    """Creates compensation matrix given parameters date in mm-dd, plate number and panel A, B, or C."""
    path = path_here + "/ckine/data/compensation/0" + date + "/Plate " + plate + "/Plate " + plate + " - " + panel + ".csv"
    # imports csv file with comp values as a dataframe
    header_names = ['Channel1', 'Channel2', 'Comp']
    df_comp = pd.read_csv(path, header=None, skiprows=1, names=header_names)
    # Add diangonal values of 100 to compensation values
    addedChannels = []
    for i in df_comp.index:
        channelName = df_comp.iloc[i]['Channel1']
        if channelName not in addedChannels:  # Ensures a diagonal value is only added once for each channel
            addedChannels.append(channelName)
            df2 = pd.DataFrame([[channelName, channelName, 100]], columns=['Channel1', 'Channel2', 'Comp'])  # Creates new row for dataframe
            df_comp = df_comp.append(df2, ignore_index=True)  # Adds row
    # create square matrix from compensation values
    df_matrix = pd.DataFrame(index=addedChannels, columns=addedChannels)  # df_matrix is now a square and has exactly one row and one column for each channel
    for i in df_matrix.index:
        for c in df_matrix.columns:
            df_matrix.at[i, c] = df_comp.loc[(df_comp['Channel1'] == c) & (df_comp['Channel2'] == i), 'Comp'].iloc[0]  # Fills in square matrix by finding corresponding comp value from csv
    # df_matrix now has all values in square matrix form
    df_matrix = df_matrix.div(100)
    if invert:  # true by default, inverts matrix before returning it
        a = np.matrix(df_matrix.values, dtype=float)  # Convert to np to allow for linalg usage
        df_matrix = pd.DataFrame(np.linalg.inv(a), df_matrix.columns, df_matrix.index)  # Calculate inverse and put pack as dataframe
    return df_matrix


def applyMatrix(sample, matrix):
    """Multiples two matrices together in the order sample dot matrix"""
    holder = pd.DataFrame()  # Will hold columns not being compensated
    for c in sample.data.columns:
        if c not in matrix:  # If sample channel column is not found in matrix
            holder = holder.join(sample.data[[c]], how='right')  # Store for after calculation
            sample.data = sample.data.drop([c], axis=1)  # Removed column to allow for matrix multiplication

    cols = sample.data.columns
    matrix = matrix[cols]
    matrix = matrix.reindex(cols)
    sample.data = sample.data.dot(matrix)  # Use matrix multiplication to compensate the relevant data
    sample.data = sample.data.join(holder)  # Restore uncompensated channels to sample
    return sample
