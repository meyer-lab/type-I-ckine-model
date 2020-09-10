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


def importF(date, plate, wellRow, panel, receptorType, wellNum, comp=True):
    """
    Import FCS files. Variable input: date in format mm-dd, plate #, panel #, and well letter. Output is a list of Data File Names in FCT Format
    Title/file names are returned in the array file --> later referenced in other functions as title/titles input argument
    """
    path_ = os.path.abspath("")

    pathname = path_ + "/ckine/data/flow/" + date + " Live PBMC Receptor Data/Plate " + plate + "/Plate " + plate + " - Panel " + str(panel) + " " + receptorType + "/"
    isotypePathname = pathname + "/Isotypes/"
    # Declare arrays and int
    file = []
    sample = []
    isotypes = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    iso_pathlist = Path(r"" + str(isotypePathname)).glob("**/*.fcs")

    for iso in iso_pathlist:
        name = iso.name.split("_")[0]
        isoSample = FCMeasurement(ID=name, datafile=str(iso))
        compedIsoSample = applyMatrix(isoSample, compMatrix(date, plate, wellRow))
        isotypes.append(compedIsoSample)

    unstainedWell = None
    for path in pathlist:
        if path.name.split("_")[2] == "Isotype.fcs":
            continue
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
        if comp is False:
            combinedSamples = combineWells(sample)
            return combinedSamples, unstainedWell, isotypes
        combinedSamples = combineWells(sample)  # Combines all files from samples
        compSample = applyMatrix(combinedSamples, compMatrix(date, plate, wellRow))  # Applies compensation matrix
        return compSample, unstainedWell, isotypes

    if comp is False:
        return sample[wellNum - 1], unstainedWell, isotypes
    compSample = applyMatrix(sample[wellNum - 1], compMatrix(date, plate, wellRow))
    return compSample, unstainedWell, isotypes


def subtract_unstained_signal(sample, channels, receptors, unstainedWell, isotypes):
    """ Subtract larger of mean isotype signal and mean unstained signal from all input channels for a given sample. """
    for _, channel in enumerate(channels):
        fileFound = False
        if _ < len(isotypes):
            for i, iso in enumerate(isotypes):
                if iso.ID == receptors[_]:
                    assert(isotypes[i].ID == receptors[_])
                    fileFound = True
                    meanBackground = compareSignals(isotypes[i], unstainedWell, channel)  # Returns larger of two background signals
                    break
        if not fileFound:
            print("Isotype File Not Found")
            meanBackground = np.mean(unstainedWell.data[channel])
        sample[channel] = np.maximum(sample[channel] - meanBackground, 0.0)

    return sample


def compareSignals(isotype, unstained, channel):
    """ Compares a mean isotype and mean unstained signal and returns greater of the two"""
    meanIsotype = np.mean(isotype.data[channel])
    meanUnstained = np.mean(unstained.data[channel])
    if meanIsotype > meanUnstained:
        return meanIsotype
    else:
        return meanUnstained


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
    # Create square matrix from compensation values
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


def import_gates():
    """ Imports dataframe with gates for all cell types and replicates. """
    data = pd.read_csv(join(path_here, "ckine/data/fc_gates.csv"))
    data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    return data


def apply_gates(date, plate, gates_df, subpopulations=False):
    """ Constructs dataframe with channels relevant to receptor quantification. """
    if date == "5-16":
        receptors = ['CD127']
        channels = ['BL1-H']
    else:
        receptors = ['CD25', 'CD122', 'CD132']
        channels = ["VL1-H", "BL5-H", "RL1-H"]
    for i, r in enumerate(receptors):
        cellTypes = ['T-helper', 'T-reg', 'NK', 'CD8+']
        for j, cellType in enumerate(cellTypes):
            if i == 0 and j == 0:
                df, unstainedWell, isotypes = samp_Gate(date, plate, gates_df, cellType, receptor=r, subPop=subpopulations)
                df = subtract_unstained_signal(df, channels, receptors, unstainedWell, isotypes)
            else:
                df2, unstainedWell2, isotypes2 = samp_Gate(date, plate, gates_df, cellType, receptor=r, subPop=subpopulations)
                df2 = subtract_unstained_signal(df2, channels, receptors, unstainedWell2, isotypes2)
                df = df.append(df2)

    return df


def samp_Gate(date, plate, gates_df, cellType, receptor, subPop=False):
    """ Returns gated sample for a given date and plate. """
    # import data and create transformed df for gating
    Dict = {'CD127': 1, 'CD25': 1, 'CD122': 3, 'CD132': 5}
    wellNum = Dict[receptor]

    if receptor == 'CD127':
        rType = 'IL7R'
    else:
        rType = 'IL2R'

    tchannels, subPopName, row, panelNum = cellGateDat(cellType)
    if date == '5-16' and (row == 'C' or row == 'B'):
        row = 'A'
        panelNum = 1

    panel, unstainedWell, isotypes = importF(date, plate, row, panelNum, rType, wellNum)

    panel_t = panel.transform("tlog", channels=tchannels)  # Creates copy of panel to transform and gate

    df = pd.DataFrame(columns=["Cell Type", "Date", "Plate", "VL1-H", "BL5-H", "RL1-H", "BL1-H"])

    # Implement gating, revert tlog, and add to dataframe
    if cellType in ('T-reg', 'T-helper'):
        samplecd3cd4 = panel_t.gate(eval(gates_df.loc[(gates_df["Name"] == 'CD3CD4') &
                                                      (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
        sample = samplecd3cd4.gate(eval(gates_df.loc[(gates_df["Name"] == cellType) &
                                                     (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
    else:
        sample = panel_t.gate(eval(gates_df.loc[(gates_df["Name"] == cellType) &
                                                (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))

    # Gated signals based on gating values from csv
    gated_idx = np.array(sample.data.index)
    panel.set_data(panel.data.loc[gated_idx])  # Selects only the corresponding data points from panel1(untransformed) based on gated points from panel1_t
    df_add = pd.DataFrame({"Cell Type": np.tile(cellType, sample.counts), "Date": np.tile(date, sample.counts), "Plate": np.tile(plate, sample.counts),
                           "VL1-H": panel.data[["VL1-H"]].values.reshape((sample.counts,)), "BL5-H": panel.data[["BL5-H"]].values.reshape((sample.counts,)),
                           "RL1-H": panel.data[["RL1-H"]].values.reshape((sample.counts,)), "BL1-H": panel.data[["BL1-H"]].values.reshape((sample.counts,))})
    df = df.append(df_add)
    df['Receptor'] = str(receptor)

    # Separates memory and naive populations and adds to dataframe
    if subPop:
        for subpopulation in subPopName:
            sampleSub = sample.gate(eval(gates_df.loc[(gates_df["Name"] == subpopulation) &
                                                      (gates_df["Date"] == date) & (gates_df["Plate"] == float(plate))]["Gate"].values[0]))
            gated_idx = np.array(sampleSub.data.index)
            panel_S = panel.data.loc[gated_idx]
            df_add = pd.DataFrame({"Cell Type": np.tile(subpopulation, sampleSub.counts), "Date": np.tile(date, sampleSub.counts), "Plate": np.tile(plate, sampleSub.counts),
                                   "VL1-H": panel_S.data[["VL1-H"]].values.reshape((sampleSub.counts,)), "BL5-H": panel_S.data[["BL5-H"]].values.reshape((sampleSub.counts,)),
                                   "RL1-H": panel_S.data[["RL1-H"]].values.reshape((sampleSub.counts,))})
            df = df.append(df_add)

    return df, unstainedWell, isotypes


def cellGateDat(cellType):
    "Returns pertinent gating information for a given cell type"
    if cellType in ('T-reg', 'T-helper'):
        tchannels = ['VL6-H', 'VL4-H', 'BL1-H', 'VL1-H', 'BL3-H']
        row = 'A'
        panel = 1
        if cellType == "T-reg":
            subPopName = ["Mem Treg", "Naive Treg"]
        else:
            subPopName = ['Naive Th', 'Mem Th']
    elif cellType == "CD8+":
        tchannels = ['VL4-H', 'VL6-H', 'BL3-H']
        row = 'C'
        panel = 3
        subPopName = ['Naive CD8+', 'Mem CD8+']
    elif cellType == "NK":
        tchannels = ['VL4-H', 'BL3-H']
        row = 'B'
        panel = 2
        subPopName = ['NKT']

    assert tchannels != []

    return tchannels, subPopName, row, panel
