"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
import os
from os.path import dirname  # , join
from pathlib import Path
#import pandas as pd
#import numpy as np
from FlowCytometryTools import FCMeasurement  # , PolyGate, ThresholdGate
path_here = dirname(dirname(__file__))


def combineWells(samples):
    """Accepts sample array returned from importF, and array of channels, returns combined well data"""
    combinedSamples = samples[0]
    for sample in samples[1:]:
        combinedSamples.data = combinedSamples.data.append(sample.data, ignore_index=True)
    return combinedSamples


def importF(date, plate, wellRow, panel, wellNum=None):
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
        # compSample = applyMatrix(combinedSamples, compMatrix(date, plate, wellRow))  # Applies compensation matrix
        return compSample, unstainedWell

    #compSample = applyMatrix(sample, compMatrix(date, plate, wellRow))
    return compSample, unstainedWell
