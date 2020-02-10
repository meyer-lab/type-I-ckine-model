"""
This file includes various methods for flow cytometry analysis.
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import QuadGate, ThresholdGate


def importF(pathname, WellRow):
    """
    Import FCS files. Variable input: name of path name to file. Output is a list of Data File Names in FCT Format
    Title/file names are returned in the array file --> later referenced in other functions as title/titles input argument
    """
    # Declare arrays and int
    file = []
    sample = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    for path in pathlist:
        wellID = path.name.split("_")[1]
        if wellID[0] == WellRow:
            file.append(str(path))
    file.sort()
    assert file != []
    # Go through each file and assign the file contents to entry in the array sample
    for entry in file:
        sample.append(FCMeasurement(ID="Test Sample" + str(z), datafile=entry))
        z += 1
    # Returns the array sample which contains data of each file in folder (one file per entry in array)
    return sample, file


def cd4():
    """ Function for gating CD4+ cells (generates T cells). """
    cd41 = ThresholdGate(6.514e+03, ('VL4-H'), region="above", name='cd41')
    cd42 = ThresholdGate(7.646e+03, ('VL4-H'), region="below", name='cd42')
    cd4_gate = cd41 & cd42
    return cd4_gate


vert = {}
vert['treg'] = vert['tregMem'] = vert['tregNaive'] = [(4.814e+03, 3.229e+03), (6.258e+03, 5.814e+03)]
vert['nonTreg'] = vert['THelpMem'] = vert['THelpN'] = [(5.115e+03, 3.470e+02), (2.586e+03, 5.245e+03)]
vert['nk'] = [(6.468e03, 4.861e03), (5.550e03, 5.813e03)]
vert['nkt'] = [(6.758e03, 6.021e03), (5.550e03, 7.013e03)]
vert['bnk'] = [(7.342e03, 4.899e03), (6.533e03, 5.751e03)]
vert['cd'] = [(9.016e03, 5.976e03), (6.825e03, 7.541e03)]

channels = {}
channels['treg'] = channels['tregMem'] = channels['tregNaive'] = channels['nonTreg'] = channels['THelpMem'] = channels['THelpN'] = ('BL1-H', 'VL1-H')
channels['nk'] = channels['nkt'] = channels['bnk'] = ("BL1-H", "VL4-H")
channels['cd'] = ("RL1-H", "VL4-H")

regionSpec = {}
regionSpec['treg'] = regionSpec['tregMem'] = regionSpec['tregNaive'] = ['top right', 'bottom left']
regionSpec['nonTreg'] = regionSpec['THelpMem'] = regionSpec['THelpN'] = regionSpec['nk'] = regionSpec['nkt'] = regionSpec['bnk'] = regionSpec['cd'] = ['top left', 'bottom right']

regionSpec_ = {}
regionSpec_['treg'] = regionSpec_['nonTreg'] = regionSpec_['nk'] = regionSpec_['nkt'] = regionSpec_['bnk'] = regionSpec_['cd'] = None
regionSpec_['tregMem'] = regionSpec_['THelpMem'] = "below"
regionSpec_['tregNaive'] = regionSpec_['THelpN'] = "above"


def gating(cell_type):
    """ Creates and returns the cell type gate on CD4+ cells. """
    cell1 = QuadGate(vert[cell_type][0], channels[cell_type], region=regionSpec[cell_type][0], name=(cell_type + '1'))
    cell2 = QuadGate(vert[cell_type][1], channels[cell_type], region=regionSpec[cell_type][1], name=(cell_type + '2'))
    if regionSpec_[cell_type] is not None:
        cd45 = ThresholdGate(6300, ('BL3-H'), region=regionSpec_[cell_type], name='cd45')
        gate = cell1 & cell2 & cd4() & cd45
    else:
        if cell_type in ('nk', 'nkt', 'bnk', 'cd'):
            gate = cell1 & cell2
        else:
            gate = cell1 & cell2 & cd4()
    return gate


def cellData(sample_i, gate, Tcells=True):
    """
    Function for returning the count of cells and raw data in a single .fcs. file of a single cell file. Arguments: single sample/.fcs file and the gate of the desired cell output
    """
    # Import single file and save data to a variable --> transform to logarithmic scale
    if Tcells:
        channels_ = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        channels_ = ["BL1-H", "RL1-H", "VL4-H"]
    smpl = sample_i.transform("hlog", channels=channels_)
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Number of events (AKA number of cells)
    cell_data = cells.get_data()
    cell_count = cells.get_data().shape[0]
    # print(cell_count)
    # print('Number of Treg cells:' + str(treg_count))
    return cell_data, cell_count


channel_data = {}
channel_data['tcells'] = ["VL4-H", "BL1-H", "VL1-H"]
channel_data['nk_bnk'] = ["BL1-H", "VL4-H", "RL1-H"]
channel_data['cd'] = ["BL1-H", "VL4-H", "RL1-H"]


def plot_cells(sample_i, gates, channels_, plot_channels, cell_names, title, plot_entire_sample=False):
    """ Plots specified cell types and gates. """
    smpl = sample_i.transform('hlog', channels=channels_)

    _, ax = plt.subplots()

    colors = ["y", "g", "b"]

    for i, gate in enumerate(gates):
        cells = smpl.gate(gate)
        cells.plot(plot_channels, color=colors[i])

    if plot_entire_sample:
        smpl.plot(plot_channels)

    legend_names = []
    legend_range = []
    bar_range = [np.arange(0, 0), np.arange(1, 11), np.arange(30, 40)]
    bar_ = 0

    for j, name in enumerate(cell_names):
        if name == 'CD4':
            continue
        bar_ = bar_ + 1
        legend_range.append(np.arange(0, 10), bar_range[bar_], bottom=bar_range[bar_ - 1], color=colors[j])
        legend_names.append(name)

    ax.set(title=str(title), xlabel="Foxp3", ylabel="CD25", fontsize=12)
    ax.legend(legend_range, legend_names, loc="upper left")


def count_data(sampleType, gate, Tcells=True):
    """
    Used to count the number of cells and store the data of all of these cells in a folder with multiple files --> automates the process sampleType
    is NK or T cell data, gate is the desired cell population.
    Sample type: is the overall importF assignment for T or NK (all the T cell files, all NK cell files)
    """
    # declare the arrays to store the data
    count_array = []
    data_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    for _, sample in enumerate(sampleType):
        rawData, cellCount = cellData(sample, gate, Tcells)
        count_array.append(cellCount)
        data_array.append(rawData)
    # returns the array for count of cells and the array where each entry is the data for the specific cell population in that .fcs file
    return count_array, data_array
