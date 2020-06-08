"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
import os
from os.path import dirname, join
from pathlib import Path
import pandas as pd
import numpy as np
from FlowCytometryTools import FCMeasurement, PolyGate, ThresholdGate
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
        compSample = applyMatrix(combinedSamples, compMatrix(date, plate, wellRow))  # Applies compensation matrix
        return compSample, unstainedWell

    compSample = applyMatrix(sample, compMatrix(date, plate, wellRow))
    return compSample, unstainedWell


def compMatrix(date, plate, panel, invert=True):
    """Applies compensation matrix given parameters date in mm-dd, plate number and panel A, B, or C."""
    path = path_here + "/ckine/data/compensation/0" + date + "/Plate " + plate + "/Plate " + plate + " - " + panel + ".csv"

    header_names = ['Channel1', 'Channel2', 'Comp']
    df_comp = pd.read_csv(path, header=None, skiprows=1, names=header_names)
    # Add diangonal values of 100 to compensation values
    addedChannels = []
    for i in df_comp.index:
        channelName = df_comp.iloc[i]['Channel1']
        if channelName not in addedChannels:
            addedChannels.append(channelName)
            df2 = pd.DataFrame([[channelName, channelName, 100]], columns=['Channel1', 'Channel2', 'Comp'])
            df_comp = df_comp.append(df2, ignore_index=True)
    # create square matrix from compensation values
    df_matrix = pd.DataFrame(index=addedChannels, columns=addedChannels)
    for i in df_matrix.index:
        for c in df_matrix.columns:
            df_matrix.at[i, c] = df_comp.loc[(df_comp['Channel1'] == c) & (df_comp['Channel2'] == i), 'Comp'].iloc[0]
            # switch i and c to transpose
    # df_matrix now has all values in square matrix form
    if invert:
        a = np.matrix(df_matrix.values, dtype=float)
        df_matrix = pd.DataFrame(np.linalg.inv(a), df_matrix.columns, df_matrix.index)
    return df_matrix


def applyMatrix(sample, matrix):
    """Multiples two matrices together in the order sample dot matrix"""
    holder = pd.DataFrame()
    for c in sample.data.columns:
        if c not in matrix:
            holder = holder.join(sample.data[[c]], how='right')
            sample.data = sample.data.drop([c], axis=1)
    sample.data = sample.data.dot(matrix)
    sample.data = sample.data.join(holder)
    return sample


def subtract_unstained_signal(sample, channels, unstainedWell):
    """ Subtract mean unstained signal from all input channels for a given sample. """
    meanBackground = np.mean(unstainedWell.data['RL1-H'])  # Calculates mean unstained signal

    def compare_background(signal, background):
        """ Compares signal to background: subtracts background from signal if greater or sets to zero if not. """
        if signal <= background:
            return 0
        return signal - background

    vfunc = np.vectorize(compare_background)

    for _, channel in enumerate(channels):
        sample[channel] = vfunc(sample[channel], meanBackground)

    return sample


def import_gates():
    """ Imports dataframe with gates for all cell types and replicates. """
    data = pd.read_csv(join(path_here, "ckine/data/fc_gates.csv"))
    data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    return data


def apply_gates(date, plate, gates_df, subpopulations=False):
    """ Constructs dataframe with channels relevant to receptor quantification. """
    df, unstainedWell = thelp_sample(date, plate, gates_df, mem_naive=subpopulations)
    df = df.append(treg_sample(date, plate, gates_df, mem_naive=subpopulations))
    df = df.append(nk_nkt_sample(date, plate, gates_df, nkt=subpopulations))
    df = df.append(cd8_sample(date, plate, gates_df, mem_naive=subpopulations))
    df = subtract_unstained_signal(df, ["VL1-H", "BL5-H", "RL1-H"], unstainedWell)
    #print(df)
    return df


def thelp_sample(date, plate, gates_df, mem_naive=False):
    """ Returns gated T-helper sample for a given date and plate. """
    # import data and create transformed df for gating
    panel1, unstainedWell = importF(date, plate, "A", 1)
    panel1_t = panel1.transform("tlog", channels=['VL6-H', 'VL4-H', 'BL1-H', 'VL1-H', 'BL3-H'])

    df = pd.DataFrame(columns=["Cell Type", "Date", "Plate", "VL1-H", "BL5-H", "RL1-H"])  # initialize dataframe for receptor quant channels

    # implement gating, revert tlog, and add to dataframe
    samplecd3cd4 = panel1_t.gate(eval(gates_df.loc[(gates_df["Name"] == 'CD3CD4') &
                                                   (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
    samplethelp = samplecd3cd4.gate(eval(gates_df.loc[(gates_df["Name"] == 'T-helper') &
                                                      (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
    gated_idx = np.array(samplethelp.data.index)
    panel1.set_data(panel1.data.loc[gated_idx])

    df_add = pd.DataFrame({"Cell Type": np.tile("T-helper", panel1.counts), "Date": np.tile(date, panel1.counts), "Plate": np.tile(plate, panel1.counts),
                           "VL1-H": panel1.data[['VL1-H']].values.reshape((panel1.counts,)), "BL5-H": panel1.data[['BL5-H']].values.reshape((panel1.counts,)),
                           "RL1-H": panel1.data[['RL1-H']].values.reshape((panel1.counts,))})
    df = df.append(df_add)

    # separates memory and naive populations and adds to dataframe
    if mem_naive:
        panel1_n = panel1.copy()
        samplenaive = samplethelp.gate(eval(gates_df.loc[(gates_df["Name"] == 'Naive Th') &
                                                         (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
        gated_idx = np.array(samplenaive.data.index)
        panel1_n.set_data(panel1.data.loc[gated_idx])
        df_add = pd.DataFrame({"Cell Type": np.tile("Naive Th", samplenaive.counts), "Date": np.tile(date, samplenaive.counts), "Plate": np.tile(plate, samplenaive.counts),
                               "VL1-H": panel1_n.data[['VL1-H']].values.reshape((samplenaive.counts,)), "BL5-H": panel1_n.data[['BL5-H']].values.reshape((samplenaive.counts,)),
                               "RL1-H": panel1_n.data[['RL1-H']].values.reshape((samplenaive.counts,))})
        df = df.append(df_add)
        panel1_m = panel1.copy()
        samplemem = samplethelp.gate(eval(gates_df.loc[(gates_df["Name"] == 'Mem Th') &
                                                       (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
        gated_idx = np.array(samplemem.data.index)
        panel1_m.set_data(panel1.data.loc[gated_idx])
        df_add = pd.DataFrame({"Cell Type": np.tile("Mem Th", samplemem.counts), "Date": np.tile(date, samplemem.counts), "Plate": np.tile(plate, samplemem.counts),
                               "VL1-H": panel1_m.data[['VL1-H']].values.reshape((samplemem.counts,)), "BL5-H": panel1_m.data[['BL5-H']].values.reshape((samplemem.counts,)),
                               "RL1-H": panel1_m.data[['RL1-H']].values.reshape((samplemem.counts,))})
        df = df.append(df_add)

    return df, unstainedWell
