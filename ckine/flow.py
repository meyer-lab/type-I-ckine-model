"""
This file includes various methods for flow cytometry analysis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import QuadGate, ThresholdGate
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.optimize import least_squares


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


# *********************************** Gating Fxns *******************************************
# Treg and NonTreg

# add which channels relate to the proteins
def cd4():
    """Function for gating CD4+ cells (generates T cells)"""
    cd41 = ThresholdGate(6.514e+03, ('VL4-H'), region="above", name='cd41')
    cd42 = ThresholdGate(7.646e+03, ('VL4-H'), region="below", name='cd42')
    cd4_gate = cd41 & cd42
    return cd4_gate


def treg():
    """Function for creating and returning the T reg gate on CD4+ cells"""
    treg1 = QuadGate((4.814e+03, 3.229e+03), ('BL1-H', 'VL1-H'), region='top right', name='treg1')
    treg2 = QuadGate((6.258e+03, 5.814e+03), ('BL1-H', 'VL1-H'), region='bottom left', name='treg2')
    treg_gate = treg1 & treg2 & cd4()
    return treg_gate


def tregMem():
    """Function for creating and returning the T reg gate on CD4+ cells"""
    treg1 = QuadGate((4.814e+03, 3.229e+03), ('BL1-H', 'VL1-H'), region='top right', name='treg1')
    treg2 = QuadGate((6.258e+03, 5.814e+03), ('BL1-H', 'VL1-H'), region='bottom left', name='treg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="below", name='cd45')
    treg_gate = treg1 & treg2 & cd4() & cd45
    return treg_gate


def tregNaive():
    """Function for creating and returning the T reg gate on CD4+ cells"""
    treg1 = QuadGate((4.814e+03, 3.229e+03), ('BL1-H', 'VL1-H'), region='top right', name='treg1')
    treg2 = QuadGate((6.258e+03, 5.814e+03), ('BL1-H', 'VL1-H'), region='bottom left', name='treg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="above", name='cd45')
    treg_gate = treg1 & treg2 & cd4() & cd45
    return treg_gate


def nonTreg():
    """Function for creating and returning the non T reg gate on CD4+ cells"""
    nontreg1 = QuadGate((5.115e+03, 3.470e+02), ('BL1-H', 'VL1-H'), region="top left", name='nontreg1')
    nontreg2 = QuadGate((2.586e+03, 5.245e+03), ('BL1-H', 'VL1-H'), region="bottom right", name='nontreg2')
    nonTreg_gate = nontreg1 & nontreg2 & cd4()
    return nonTreg_gate


def THelpMem():
    """Function for creating and returning the non T reg gate on CD4+ cells"""
    nontreg1 = QuadGate((5.115e+03, 3.470e+02), ('BL1-H', 'VL1-H'), region="top left", name='nontreg1')
    nontreg2 = QuadGate((2.586e+03, 5.245e+03), ('BL1-H', 'VL1-H'), region="bottom right", name='nontreg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="below", name='cd45')
    nonTreg_gate = nontreg1 & nontreg2 & cd4() & cd45
    return nonTreg_gate


def THelpN():
    """Function for creating and returning the non T reg gate on CD4+ cells"""
    nontreg1 = QuadGate((5.115e+03, 3.470e+02), ('BL1-H', 'VL1-H'), region="top left", name='nontreg1')
    nontreg2 = QuadGate((2.586e+03, 5.245e+03), ('BL1-H', 'VL1-H'), region="bottom right", name='nontreg2')
    cd45 = ThresholdGate(6300, ('BL3-H'), region="above", name='cd45')
    nonTreg_gate = nontreg1 & nontreg2 & cd4() & cd45
    return nonTreg_gate


def nk():
    """Function for creating and returning the NK gate"""
    # NK cells: Take quad gates for NK cells and combine them to create single, overall NK gate
    nk1 = QuadGate((6.468e03, 4.861e03), ("BL1-H", "VL4-H"), region="top left", name="nk1")
    nk2 = QuadGate((5.550e03, 5.813e03), ("BL1-H", "VL4-H"), region="bottom right", name="nk2")
    nk_gate = nk1 & nk2
    return nk_gate


def nkt():
    """Function for creating and returning the NKT gate"""
    # Bright NK cells: Take quad gates for bright NK cells and combine them to create single, overall bright NK gate
    nkt1 = QuadGate((7.342e03, 4.899e03), ("BL1-H", "VL4-H"), region="top left", name="nkt1")
    nkt2 = QuadGate((6.533e03, 5.751e03), ("BL1-H", "VL4-H"), region="bottom right", name="nkt2")
    nkt_gate = nkt1 & nkt2
    return nkt_gate


def cd():
    """Function for creating and returning the CD gate"""
    # CD cells: Take quad gates for CD cells and combine them to create single, overall CD gate
    cd1 = QuadGate((9.016e03, 5.976e03), ("RL1-H", "VL4-H"), region="top left", name="cd1")
    cd2 = QuadGate((6.825e03, 7.541e03), ("RL1-H", "VL4-H"), region="bottom right", name="cd2")
    cd_gate = cd1 & cd2
    return cd_gate


def cellCount(sample_i, gate, Tcells=True):
    """
    Function for returning the count of cells in a single .fcs. file of a single cell file. Arguments: single sample/.fcs file and the gate of the
    desired cell output.
    """
    # Import single file and save data to a variable --> transform to logarithmic scale
    if Tcells:
        channels = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        channels = ["BL1-H", "RL1-H", "VL4-H"]
    smpl = sample_i.transform("hlog", channels=channels)
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Number of events (AKA number of cells)
    cell_count = cells.get_data().shape[0]
    # print(cell_count)
    # print('Number of Treg cells:' + str(treg_count))
    return cell_count


def rawData(sample_i, gate, Tcells=True):
    """
    Function that returns the raw data of certain cell population in a given file. Arguments: sample_i is a single entry/.fcs file and the gate
    of the desired cell population.
    """
    if Tcells:
        channels = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        channels = ["BL1-H", "RL1-H", "VL4-H"]
    smpl = sample_i.transform("hlog", channels=channels)
    # Apply T reg gate to overall data --> i.e. step that detrmines which cells are T reg
    cells = smpl.gate(gate)
    # Get raw data of t reg cells in file
    cell_data = cells.get_data()
    return cell_data


def tcells(sample_i, treg_gate, nonTreg_gate, title):
    """
    Function that is used to plot the Treg and NonTreg gates in CD4+ cells. Treg (yellow) and Non Treg (green). sample_i is an indivual flow cytommetry file/data.
    """
    # Data to use is on CD4+ cells
    # Apply new T reg and Non treg gate
    # Assign data of current file for analysis to variable smpl and transform to log scale
    smpl = sample_i.transform('hlog', channels=["VL4-H", "BL1-H", "VL1-H"])
    # Create data set to only include CD4 cells
    cd4_gate = cd4()
    cd4_cells = smpl.gate(cd4_gate)
    # CD25 v. Foxp33: VL1 v. BL1
    # Treg
    # Apply T reg gate to overall data --> step that determines which cells are Treg
    treg_cells = smpl.gate(treg_gate)
    # Non Tregs
    # Apply non T reg gate to overall data --> step that detrmines which cells are non T reg
    nonTreg_cells = smpl.gate(nonTreg_gate)

    # Declare figure and axis
    _, ax = plt.subplots()
    # Plot the treg gate
    treg_cells.plot(["BL1-H", "VL1-H"], color="teal")
    # Plot the non Treg gate
    nonTreg_cells.plot(["BL1-H", "VL1-H"], color="cyan")
    # Plot all of the cells in the file
    ax.set_title("T Reg + Non T Reg - Gating - " + str(title), fontsize=12)
    cd4_cells.plot(["BL1-H", "VL1-H"])
    plt.xlabel("Foxp3", fontsize=12)
    plt.ylabel("CD25", fontsize=12)
    # Set values for legend
    bar_T = ax.bar(np.arange(0, 10), np.arange(1, 11), color="teal")
    bar_NT = ax.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="cyan")
    ax.legend([bar_T, bar_NT], ("T Reg", "Non T Reg"), loc="upper left")


def nk_nkt_plot(sample_i, nk_gate, nkt_gate, title):
    """
    Function that plots the graph of NK and Bright NK cells (both are determined by same x, y-axis). Arguemnt 1: current sample (a single file).
    Argument 2: the gate for NK. Argument 3: the gate for bright NK.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])

    # CD3 v. CD56: VL4 v. BL1
    # NK
    # Apply NK gate to overall data --> step that determines which cells are NK
    nk_cells = smpl.gate(nk_gate)
    # CD56 Bright NK
    # Apply Bright NK gate to overall data --> step that determines which cells are Bright NK
    nkt_cells = smpl.gate(nkt_gate)

    _, ax1 = plt.subplots()
    ax1.set_title("CD56 BrightNK + NK - Gating - " + str(title), fontsize=12)
    nk_cells.plot(["BL1-H", "VL4-H"], color="y", label="NK")
    nkt_cells.plot(["BL1-H", "VL4-H"], color="g", label="Bright NK")
    smpl.plot(["BL1-H", "VL4-H"])

    bar_NK = ax1.bar(np.arange(0, 10), np.arange(1, 11), color="y")
    bar_NKT = ax1.bar(np.arange(0, 10), np.arange(30, 40), bottom=np.arange(1, 11), color="g")
    ax1.legend([bar_NK, bar_NKT], ("NK", "NKT"), loc="upper left")


def cd_plot(sample_i, cd_gate, title):
    """
    Function that plots the graph of CD cells. Argument 1: current sample (a single file). Argument 2: the gate for CD cells. Argument 3: the value
    of the current i in a for loop --> use
    when plotting multiple files.
    """
    smpl = sample_i.transform("hlog", channels=["BL1-H", "VL4-H", "RL1-H"])
    # CD3 v. CD8: VL4 v. RL1
    # CD3+CD8+
    # Apply CD cell gate to overall data --> step that determines which cells are CD
    cd_cells = smpl.gate(cd_gate)

    _, ax2 = plt.subplots()
    ax2.set_title("CD3+CD8+ - Gating - " + str(title), fontsize=20)
    cd_cells.plot(["RL1-H", "VL4-H"], color="b")
    smpl.plot(["RL1-H", "VL4-H"])

    bar_CD = ax2.bar(np.arange(0, 10), np.arange(1, 11), color="b")
    ax2.legend([bar_CD], ("CD3+8+"), loc="upper left")


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
        count_array.append(cellCount(sample, gate, Tcells))
        data_array.append(rawData(sample, gate, Tcells))
    # returns the array for count of cells and the array where each entry is the data for the specific cell population in that .fcs file
    return count_array, data_array


def plotAll(sampleType, check, gate1, gate2, titles):
    """
    Ask the user to input 't' for t cell, 'n' for nk cell, and 'c' for cd cell checks are used to determine if user input a T-cell, NK-cell, or
    CD-cell gate automates the process for plotting multiple files.
    """
    if check == "t":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            tcells(sample, gate1, gate2, title)
    elif check == "n":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            nk_nkt_plot(sample, gate1, gate2, title)
    elif check == "c":
        for i, sample in enumerate(sampleType):
            title = titles[i].split("/")
            title = title[len(title) - 1]
            cd_plot(sample, gate1, title)


# ********************************** PCA Functions****************************************************
def sampleT(smpl):
    """Output is the T cells data (the protein channels related to T cells)"""
    # Features are the protein channels of interest when analyzing T cells
    features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    # Transform to put on log scale
    tform = smpl.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "BL3-H"])
    # Save the data of each column of the protein channels
    data = tform.data[["BL1-H", "VL1-H", "VL4-H", "BL3-H"]][0:]
    # Save pSTAT5 data
    pstat = tform.data[["RL1-H"]][0:]
    return data, pstat, features


def sampleNK(smpl):
    """Output is the NK cells data (the protein channels related to NK cells)"""
    # For NK, the data consists of different channels so the data var. output will be different
    # Output is data specific to NK cells
    # Features for the NK file of proteins (CD3, CD8, CD56)
    features = ["VL4-H", "RL1-H", "BL1-H"]
    # Transform all proteins (including pSTAT5)
    tform = smpl.transform("hlog", channels=["VL4-H", "RL1-H", "BL1-H"])
    # Assign data of three protein channels AND pSTAT5
    data = tform.data[["VL4-H", "RL1-H", "BL1-H"]][0:]
    pstat = tform.data[["BL2-H"]][0:]
    return data, pstat, features


def fitPCA(data, Tcells=True):
    """
    Fits the PCA model to data, and returns the fit PCA model for future transformations
    (allows for consistent loadings plots between wells)
    """
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]
    # Apply PCA to the data set
    # setting values of data of selected features to data frame
    xi = data.loc[:, features].values
    # STANDARDIZE DATA --> very important to do before applying machine learning algorithm
    scaler = preprocessing.StandardScaler()
    xs = scaler.fit_transform(xi)
    xs = np.nan_to_num(xs)
    # setting how many components wanted --> PC1 and PC2
    pca = PCA(n_components=2)
    # apply PCA to standardized data set
    PCAobj = pca.fit(xs)
    # creates the loading array (equation is defintion of loading)
    loading = pca.components_.T
    return PCAobj, loading


def appPCA(data, PCAobj, Tcells=True):
    """Applies the PCA algorithm to the data set"""
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]
    # setting values of data of selected features to data frame
    xi = data.loc[:, features].values
    # STANDARDIZE DATA --> very important to do before applying machine learning algorithm
    scaler = preprocessing.StandardScaler()
    xs = scaler.fit_transform(xi)
    xs = np.nan_to_num(xs)
    # transform to the prefit pca object
    xf = PCAobj.transform(xs)
    return xf


def pcaPlt(xf, pstat, ax, Tcells=True):
    """
    Used to plot the score graph.
    Scattered point color gradients are based on range/abundance of pSTAT5 data. Light --> Dark = Less --> More Active
    """
    # PCA
    # Setting x and y values from xf
    x = xf[:, 0]
    y = xf[:, 1]
    # Saving numerical values for pSTAT5 data
    pstat_data = pstat.values
    # Creating a figure for both scatter and mesh plots for PCA
    # This is the scatter plot of the cell clusters colored by pSTAT5 data
    # lighter --> darker = less --> more pSTAT5 present
    # Creating correct dimensions
    pstat = np.squeeze(pstat_data)
    # Creating a data from of x, y, and pSTAT5 in order to graph using seaborn
    combined = np.stack((x, y, pstat)).T
    df = pd.DataFrame(combined, columns=["PC1", "PC2", "pSTAT5"])
    # Creating plot using seaborn. Cool note: virdis is visible for individuals who are colorblind.
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    if Tcells:
        sns.scatterplot(x="PC1", y="PC2", hue="pSTAT5", palette="viridis", data=df, s=5, ax=ax, legend=False, hue_norm=(0, 40000))
        points = ax.scatter(df["PC1"], df["PC2"], c=df["pSTAT5"], s=0, cmap="viridis", vmin=0, vmax=40000)  # set style options
    else:
        sns.scatterplot(x="PC1", y="PC2", hue="pSTAT5", palette="viridis", data=df, s=5, ax=ax, legend=False, hue_norm=(0, 6000))
        points = ax.scatter(df["PC1"], df["PC2"], c=df["pSTAT5"], s=0, cmap="viridis", vmin=0, vmax=6000)  # set style options
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    # add a color bar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('pSTAT Level')


def loadingPlot(loading, ax, Tcells=True):
    """Plot the loading data"""
    # Loading
    # Create graph for loading values
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]

    x_load = loading[:, 0]
    y_load = loading[:, 1]

    # Create figure for the loading plot
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.scatter(x_load, y_load)
    ax.grid()

    for z, feature in enumerate(features):
        # Please note: not the best logic, but there are three features in NK and four features in T cells
        if Tcells:
            if feature == "BL1-H":
                feature = "Foxp3"
            elif feature == "VL1-H":
                feature = "CD25"
            elif feature == "VL4-H":
                feature = "CD4"
            elif feature == "BL3-H":
                feature = "CD45RA"
        else:
            if feature == "VL4-H":
                feature = "CD3"
            if feature == "RL1-H":
                feature = "CD8"
            if feature == "BL1-H":
                feature = "CD56"
        ax.annotate(str(feature), xy=(x_load[z], y_load[z]), fontsize=8)


def pcaAll(sampleType, Tcells=True):
    """
    Use to plot the score and loading graphs for PCA. Assign protein and pstat5 arrays AND score and loading arrays
    This is all the data for each file.
    Want to use for both T and NK cells? Use it twice!
    sampleType is importF for T or NK
    check == "t" for T cells OR check == "n" for NK cells
    """
    # declare the arrays to store the data
    data_array = []
    pstat_array = []
    xf_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    if Tcells:
        for i, sample in enumerate(sampleType):
            data, pstat, _ = sampleT(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)

    else:
        for i, sample in enumerate(sampleType):
            data, pstat, _ = sampleNK(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
    return data_array, pstat_array, xf_array, loading

# ************************PCA by color (gating+PCA)******************************


def sampleTcolor(smpl):
    """Output is the T cells data (the protein channels related to T cells)"""
    # Features are the protein channels of interest when analyzing T cells
    features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    # Transform to put on log scale
    tform = smpl.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "BL3-H", "RL1-H"])
    # Save the data of each column of the protein channels
    data = tform.data[["BL1-H", "VL1-H", "VL4-H", "BL3-H"]][0:]
    # Save pSTAT5 data
    pstat = tform.data[["RL1-H"]][0:]
    colmat = [] * (len(data) + 1)
    for i in range(len(data)):
        if data.iat[i, 0] > 5.115e+03 and data.iat[i, 0] < 6.258e+03 and data.iat[i, 1] > 3.229e+03 and data.iat[i, 1] < 5.814e+03 and data.iat[i, 2] > 6.512e+03:
            if data.iat[i, 3] > 6300:
                colmat.append('r')  # Treg naive
            else:
                colmat.append('darkorange')  # Treg mem
        elif data.iat[i, 0] > 2.586e+03 and data.iat[i, 0] < 5.115e+03 and data.iat[i, 1] > 3.470e+02 and data.iat[i, 1] < 5.245e+03 and data.iat[i, 2] > 6.512e+03:
            if data.iat[i, 3] > 6300:
                colmat.append('g')  # Thelp naive
            else:
                colmat.append('darkorchid')  # Thelp mem
        else:
            colmat.append('c')
    return data, pstat, features, colmat


def sampleNKcolor(smpl):
    """Output is the NK cells data (the protein channels related to NK cells)"""
    # For NK, the data consists of different channels so the data var. output will be different
    # Output is data specific to NK cells
    # Features for the NK file of proteins (CD3, CD8, CD56)
    features = ["VL4-H", "RL1-H", "BL1-H"]
    # Transform all proteins (including pSTAT5)
    tform = smpl.transform("hlog", channels=["VL4-H", "RL1-H", "BL1-H"])
    # Assign data of three protein channels AND pSTAT5
    data = tform.data[["VL4-H", "RL1-H", "BL1-H"]][0:]
    pstat = tform.data[["BL2-H"]][0:]
    # Create a section for assigning colors to each data point of each cell population --> in this case NK cells
    colmat = [] * (len(data) + 1)
    
    for i in range(len(data)):
        if data.iat[i, 0] > 5.550e03 and data.iat[i, 0] < 6.468e03 and data.iat[i, 2] > 4.861e03 and data.iat[i, 2] < 5.813e03:
            colmat.append('r')  # nk
        elif data.iat[i, 0] > 6.533e03 and data.iat[i, 0] < 7.34e03 and data.iat[i, 2] > 4.899e03 and data.iat[i, 2] < 5.751e03:
            colmat.append('darkgreen')  # nkt
        elif data.iat[i, 0] > 5.976e03 and data.iat[i, 0] < 7.541e03 and data.iat[i, 1] > 6.825e03 and data.iat[i, 1] < 9.016e03:
            colmat.append('blueviolet') # cd8+
        else:
            colmat.append('c')
    return data, pstat, features, colmat


def pcaPltColor(xf, colormat, ax, Tcells=True):
    """
    Used to plot the score graph.
    Scattered point color gradients are based on range/abundance of pSTAT5 data. Light --> Dark = Less --> More Active
    """
    # PCA
    # Setting x and y values from xf
    x = xf[:, 0]
    y = xf[:, 1]
    # Working with pSTAT5 data --> setting min and max values
    # Creating a figure for both scatter and mesh plots for PCA
    ax.set_xlabel("PC 1", fontsize=15)
    ax.set_ylabel("PC 2", fontsize=15)
    ax.set(xlim=(-5, 5), ylim=(-5, 5))
    # This is the scatter plot of the cell clusters colored by pSTAT5 data
    # lighter --> darker = less --> more pSTAT5 present
    colormat = np.array(colormat)
    if Tcells:
        ax.scatter(x[colormat == "c"], y[colormat == "c"], s=0.5, c="c", label="Other", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "g"], y[colormat == "g"], s=0.5, c="g", label="T Helper Naive", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "darkorchid"], y[colormat == "darkorchid"], s=0.5, c="darkorchid", label="T Helper Memory", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "darkorange"], y[colormat == "darkorange"], s=0.5, c="darkorange", label="T Reg Memory", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "r"], y[colormat == "r"], s=0.5, c="r", label="T Reg Naive", alpha=0.3, edgecolors='none')
        ax.legend()
    else:
        ax.scatter(x[colormat == "c"], y[colormat == "c"], s=0.5, c="c", label="Other", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "darkgreen"], y[colormat == "darkgreen"], s=0.5, c="g", label="NKT", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "r"], y[colormat == "r"], s=0.5, c="r", label="NK", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "blueviolet"], y[colormat == "blueviolet"], s=0.5, c="blueviolet", label="CD8+", alpha=0.3, edgecolors='none')
        ax.legend()


def pcaAllCellType(sampleType, Tcells=True):
    """
    Use to plot the score and loading graphs for PCA. Assign protein and pstat5 arrays AND score and loading arrays
    This is all the data for each file.
    Want to use for both T and NK cells? Use it twice!
    sampleType is importF for T or NK
    check == "t" for T cells OR check == "n" for NK cells
    """
    # declare the arrays to store the data
    data_array = []
    pstat_array = []
    xf_array = []
    colormat_array = []

    # create the for loop to file through the datfa and save to the arrays
    # using the functions created above for a singular file
    if Tcells:
        for i, sample in enumerate(sampleType):
            data, pstat, _, colormat = sampleTcolor(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
            colormat_array.append(colormat)
    else:
        for i, sample in enumerate(sampleType):
            data, pstat, _, colormat = sampleNKcolor(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
            colormat_array.append(colormat)
    return data_array, pstat_array, xf_array, loading, colormat_array

# ************************Dose Response by PCA******************************


def PCADoseResponse(sampleType, PC1Bnds, PC2Bnds, gate, Tcells=True):
    """
    Given data from a time Point and two PC bounds, the dose response curve will be calculated and graphed
    (needs folder with FCS from one time point)
    """
    dosemat = np.array([84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474])
    pSTATvals = np.zeros([1, dosemat.size])
    if gate:
        gates = gate()
        _, alldata = count_data(sampleType, gates, Tcells)

    for i, sample in enumerate(sampleType):
        if Tcells:
            data, pstat, _ = sampleT(sample)  # retrieve data
            statcol = 'RL1-H'
        else:
            data, pstat, _ = sampleNK(sample)
            statcol = 'BL2-H'
        if gate:
            data = alldata[i]
            pstat = data[[statcol]]

        if i == 0:
            PCAobj, loading = fitPCA(data, Tcells)  # only fit to first set

        xf = appPCA(data, PCAobj, Tcells)  # get PC1/2 vals
        PCApd = PCdatTransform(xf, pstat)
        PCApd = PCApd[(PCApd['PC1'] >= PC1Bnds[0]) & (PCApd['PC1'] <= PC1Bnds[1]) & (PCApd['PC2'] >= PC2Bnds[0]) & (PCApd['PC2'] <= PC2Bnds[1])]  # remove data that that is not within given PC bounds
        pSTATvals[0, i] = PCApd.loc[:, "pSTAT"].mean()  # take average Pstat activity of data fitting criteria

    pSTATvals = pSTATvals.flatten()

    return pSTATvals, dosemat, loading


def PCdatTransform(xf, pstat):
    """Takes PCA Data and transforms it into a pandas dataframe (variable saver)"""
    PC1, PC2, pstat = np.transpose(xf[:, 0]), np.transpose(xf[:, 1]), pstat.to_numpy()
    PC1, PC2 = np.reshape(PC1, (PC1.size, 1)), np.reshape(PC2, (PC2.size, 1))
    PCAstat = np.concatenate((PC1, PC2, pstat), axis=1)
    PCApd = pd.DataFrame({'PC1': PCAstat[:, 0], 'PC2': PCAstat[:, 1], 'pSTAT': PCAstat[:, 2]})  # arrange into pandas datafrome
    return PCApd


def StatGini(sampleType, ax, gate, Tcells=True):
    """
    Define the Gini Coefficient of Pstat Vals Across a timepoint for either whole or gated population.
    Takes a folder of samples, a timepoint (string), a boolean check for cell type and an optional gate parameter.
    """
    alldata = []
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])
    ginis = np.zeros([2, dosemat.size])

    if Tcells:
        statcol = 'RL1-H'
    else:
        statcol = 'BL2-H'

    if gate:
        gate = gate()
        _, alldata = count_data(sampleType, gate, Tcells)  # returns array of dfs in case of gate or no gate

    else:
        for i, sample in enumerate(sampleType):
            if Tcells:
                _, pstat, _ = sampleT(sample)
                alldata.append(pstat)
            else:
                _, pstat, _ = sampleNK(sample)
                alldata.append(pstat)

    for i, sample in enumerate(sampleType):  # get pstat data and put it into list form
        dat_array = alldata[i]
        stat_array = dat_array[[statcol]]
        stat_array = stat_array.to_numpy()
        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
        stat_array.tolist()  # manipulate data to be compatible with gin calculation
        stat_sort = np.sort(np.hstack(stat_array))
        num = stat_array.size
        subconst = (num + 1) / num
        coef = 2 / num
        summed = sum([(j + 1) * stat for j, stat in enumerate(stat_sort)])
        ginis[0, i] = (coef * summed / (stat_sort.sum()) - subconst)\

    for i, sample in enumerate(sampleType):  # Get inverse Ginis
        dat_array = alldata[i]
        stat_array = dat_array[[statcol]]
        stat_array = stat_array.to_numpy()
        stat_array = stat_array.clip(min=1)
        stat_array = np.reciprocal(stat_array)
        stat_array.tolist()
        stat_sort = np.sort(np.hstack(stat_array))
        num = stat_array.size
        subconst = (num + 1) / num
        coef = 2 / num
        summed = sum([(j + 1) * stat for j, stat in enumerate(stat_sort)])
        ginis[1, i] = (coef * summed / (stat_sort.sum()) - subconst)

    ax.plot(dosemat, np.expand_dims(ginis[0, :], axis=0), ".--", color="navy", label="Gini Coefficients")
    ax.plot(dosemat, np.expand_dims(ginis[1, :], axis=0), ".--", color="darkorange", label="Inverse Gini Coefficients")
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel("Cytokine Dosage (log10[nM])")
    ax.set_ylabel("Gini Coefficient")
    ax.set(xlim=(0.0001, 100))
    ax.set(ylim=(0., 1))

    return ginis, dosemat


def nllsq_EC50(x0, xdata, ydata):
    """
    Performs nonlinear least squares on activity measurements to determine parameters of Hill equation and outputs EC50.
    """
    lsq_res = least_squares(residuals, x0, args=(xdata, ydata), bounds=([0., 0., 0., 0.], [10., 100., 10**5., 10**5]), jac='3-point')
    return lsq_res.x[0]


def residuals(x0, x, y):
    """ Residual function for Hill Equation. """
    return hill_equation(x, x0) - y


def hill_equation(x, x0, solution=0):
    """ Calculates EC50 from Hill Equation. """
    k = x0[0]
    n = x0[1]
    A = x0[2]
    floor = x0[3]
    xk = np.power(x / k, n)
    return (A * xk / (1.0 + xk)) - solution + floor


def EC50_PC_Scan(sampleType, min_max_pts, ax, gate, Tcells=True, PC1=True):
    """Scans along one Principal component and returns EC50 for slices along that Axis"""
    x0 = [1, 2., 5000., 3000.]  # would put gating here
    EC50s = np.zeros([1, min_max_pts[2]])
    scanspace = np.linspace(min_max_pts[0], min_max_pts[1], num=min_max_pts[2] + 1)
    axrange = np.array([-100, 100])

    for i in range(0, min_max_pts[2]):  # set bounds and calculate EC50s
        if PC1:
            PC1Bnds, PC2Bnds = np.array([scanspace[i], scanspace[i + 1]]), axrange
        else:
            PC2Bnds, PC1Bnds = np.array([scanspace[i], scanspace[i + 1]]), axrange
        pSTATs, doses, loading = PCADoseResponse(sampleType, PC1Bnds, PC2Bnds, gate, Tcells)
        doses = np.log10(doses.astype(np.float) * 1e4)
        EC50s[0, i] = nllsq_EC50(x0, doses, pSTATs)

    EC50s = EC50s.flatten() - 4  # account for 10^4 multiplication
    ax.plot(scanspace[:-1] + (min_max_pts[1] - min_max_pts[0]) / (2 * min_max_pts[2]), EC50s, ".--", color="navy")
    ax.grid()

    if PC1:
        ax.set_xlabel("PC1 Space")
    else:
        ax.set_xlabel("PC2 Space")
    ax.set_ylabel("log[EC50] (nM)")
    ax.set(ylim=(-3, 3))
    ax.set(xlim=(min_max_pts[0], min_max_pts[1]))
    ax.grid()

    return loading
