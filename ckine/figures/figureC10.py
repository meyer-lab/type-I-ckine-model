import os
import matplotlib.lines as mlines
import pandas as pds
import numpy as np
from scipy import stats
from .figureCommon import subplotLabel, getSetup
from ..flow import importF
from ..PCA import StatGini, sampleT, sampleNK
from ..flow import gating, count_data

from ..FCimports import compMatrix, applyMatrix, combineWells

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    StatMV()

    # global_legend(ax[7])

    return f


def global_legend(ax):
    """ Create legend for Inverse and Standard Gini """
    blue = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', markersize=6, label='Gini Coeff')
    orange = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', markersize=6, label='Inverse Gini Coeff')
    ax.legend(handles=[orange, blue], bbox_to_anchor=(0, 1), loc="upper left")


def StatMV():
    """
    Calculate mean and variance of a sample in a pandas dataframe, and plot.
    """

    dataFiles = ["/data/flow/2019-03-19 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate.zip",
                 "/data/flow/2019-03-27 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate.zip",
                 "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/",
                 "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip",
                 "/data/flow/2019-03-27 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip",
                 "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate - NEW PBMC LOT.zip"]
    dataFiles = ["/home/brianoj/Tplate15", "/home/brianoj/Tplate27", "/home/brianoj/Tplate418", "/home/brianoj/Nkplate15", "/home/brianoj/Nkplate27", "/home/brianoj/Nkplate418"]
    dates = ["3/15/2019", "3/27/2019", "4/18/2019", "3/15/2019", "3/27/2019", "4/18/2019"]
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cellTypesT = ['treg', 'nonTreg']
    cellTypesNK = ["nk", "cd"]
    TitlesT = ["Treg", "Thelper"]
    TitlesNK = ["NK", "CD8"]
    masterMVdf = pds.DataFrame(columns={"Date", "Time", "Cell", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov"})
    MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov"})
    alldata = []
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])
    repList = [0, 0, 0, 0, 0, 0]

    T_matrix = compMatrix("2019-11-08", "1", "A")  # Create matrix 1
    Cd8_NKmatrix = compMatrix("2019-11-08", "1", "B")  # Create matrix 2

    for i, filename in enumerate(dataFiles):
        if i < 3:
            Tcells = True
        else:
            Tcells = False

        if Tcells:
            statcol = "RL1-H"
            IL2RaCol = "VL1-H"
            for k, cell_type in enumerate(cellTypesT):
                for j, row in enumerate(rows):
                    print(filename)
                    sample, _ = importF(filename, row)
                    if cell_type:
                        for jj, subSample in enumerate(sample):
                            sample[jj] = applyMatrix(subSample, T_matrix)
                        gates = gating(cell_type, dates[i], True, repList[i])
                        _, alldata = count_data(sample, gates, Tcells, True)
                    else:
                        for jj, samplejj in enumerate(sample):
                            _, pstat, _ = sampleT(samplejj)
                            alldata.append(pstat)

                    for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                        dat_array = alldata[ii]
                        stat_array = dat_array[[statcol]]
                        stat_array = stat_array.to_numpy()
                        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                        IL2Ra_array = dat_array[[IL2RaCol]]
                        IL2Ra_array = IL2Ra_array.to_numpy()
                        IL2Ra_array = IL2Ra_array.clip(min=1)
                        IL2Ra_array = IL2Ra_array / 1.5
                        if stat_array.size == 0:
                            MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timeFunc(row), "Cell": TitlesT[k], "Ligand": cytFunc(row),
                                                                        "Dose": dosemat[0, ii], "Mean": [0], "Variance": [0], "Skew": [0], "Kurtosis": [0], "alphStatCov": [0], "Bivalent": [0]}))
                        else:
                            MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timeFunc(row), "Cell": TitlesT[k], "Ligand": cytFunc(row), "Dose": dosemat[0, ii], "Mean": np.mean(stat_array), "Variance": np.var(
                                stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array), "alphStatCov": [np.cov(stat_array.flatten(), IL2Ra_array.flatten())[1, 0]], "Bivalent": [0]}))

                    if j == 3 or j == 7:
                        MVdf['Mean'] = MVdf['Mean'] - MVdf['Mean'].min()
                        masterMVdf = masterMVdf.append(MVdf)
                        MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov", "Bivalent"})
        else:
            statcol = "BL2-H"
            for k, cell_type in enumerate(cellTypesNK):
                for j, row in enumerate(rows):
                    print(filename)
                    sample, _ = importF(filename, row)
                    if (row == 'H' and i == 4) is False:
                        if cell_type:
                            for jj, subSample in enumerate(sample):
                                sample[jj] = applyMatrix(subSample, Cd8_NKmatrix)
                            gates = gating(cell_type, dates[i], True, repList[i])
                            _, alldata = count_data(sample, gates, Tcells, True)
                        else:
                            for jj, samplejj in enumerate(sample):
                                print(row, jj)
                                _, pstat, _ = sampleNK(samplejj)
                                alldata.append(pstat)

                        for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                            dat_array = alldata[ii]
                            stat_array = dat_array[[statcol]]
                            stat_array = stat_array.to_numpy()
                            stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                            if stat_array.size == 0:
                                MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timeFunc(row), "Cell": TitlesNK[k], "Ligand": cytFunc(
                                    row), "Dose": dosemat[0, ii], "Mean": [0], "Variance": [0], "Skew": [0], "Kurtosis": [0], "alphStatCov": [0], "Bivalent": [0]}))
                            else:
                                MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timeFunc(row), "Cell": TitlesNK[k], "Ligand": cytFunc(row), "Dose": dosemat[0, ii], "Mean": np.mean(
                                    stat_array), "Variance": np.var(stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array), "alphStatCov": [0], "Bivalent": [0]}))
                    if j == 3 or j == 7:
                        MVdf['Mean'] = MVdf['Mean'] - MVdf['Mean'].min()
                        masterMVdf = masterMVdf.append(MVdf)
                        MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov", "Bivalent"})

    dataFiles = ["/home/brianoj/Muteins 060-062 T/2019-04-19 IL2-060 IL2-062 Treg plate",
                 "/home/brianoj/Muteins 088-097 T/2019-04-19 IL2-088 IL2-097 Treg plate",
                 "/home/brianoj/Muteins 060-088 T/2019-05-02 IL2-060 IL2-088 Treg plate",
                 "/home/brianoj/Muteins 062-097 T/2019-05-02 IL2-062 IL2-097 Treg plate",
                 "/home/brianoj/Muteins 060-062 Nk/2019-04-19 IL2-060 IL2-062 NK plate",
                 "/home/brianoj/Muteins 088-097 Nk/2019-04-19 IL2-088 IL2-097 NK plate",
                 "/home/brianoj/Muteins 060-088 Nk/2019-05-02 IL2-060 IL2-088 NK plate",
                 "/home/brianoj/Muteins 062-097 Nk/2019-05-02 IL2-062 IL2-097 NK plate"]
    dates = ["4/19/2019", "4/19/2019", "5/2/2019", "5/2/2019", "4/19/2019", "4/19/2019", "5/2/2019", "5/2/2019"]
    repList = [0, 1, 0, 1, 0, 1, 0, 1]

    print("Starting Muteins")

    for i, filename in enumerate(dataFiles):
        if i < 4:
            Tcells = True
        else:
            Tcells = False
        if Tcells:
            statcol = "RL1-H"
            for k, cell_type in enumerate(cellTypesT):
                for j, row in enumerate(rows):
                    print(filename)
                    sample, _ = importF(filename, row)
                    if cell_type:
                        for jj, subSample in enumerate(sample):
                            sample[jj] = applyMatrix(subSample, T_matrix)
                        gates = gating(cell_type, dates[i], True, repList[i])
                        _, alldata = count_data(sample, gates, Tcells, True)
                    else:
                        for jj, samplejj in enumerate(sample):
                            _, pstat, _ = sampleT(samplejj)
                            alldata.append(pstat)

                    for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                        dat_array = alldata[ii]
                        stat_array = dat_array[[statcol]]
                        stat_array = stat_array.to_numpy()
                        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                        IL2Ra_array = dat_array[[IL2RaCol]]
                        IL2Ra_array = IL2Ra_array.to_numpy()
                        IL2Ra_array = IL2Ra_array.clip(min=1)
                        timelig = mutFunc(row, filename)
                        if stat_array.size == 0:
                            MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timelig[0], "Cell": TitlesT[k], "Ligand": timelig[1], "Dose": dosemat[0, ii], "Mean": [
                                               0], "Variance": [0], "Skew": [0], "Kurtosis": [0], "alphStatCov": [0], "Bivalent": timelig[2]}))
                        else:
                            MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timelig[0], "Cell": TitlesT[k], "Ligand": timelig[1], "Dose": dosemat[0, ii], "Mean": np.mean(stat_array), "Variance": np.var(
                                stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array), "alphStatCov": [np.cov(stat_array.flatten(), IL2Ra_array.flatten())[1, 0]], "Bivalent": timelig[2]}))
                    if j == 3 or j == 7:
                        MVdf['Mean'] = MVdf['Mean'] - MVdf['Mean'].min()
                        masterMVdf = masterMVdf.append(MVdf)
                        MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov", "Bivalent"})
        else:
            statcol = "BL2-H"
            for k, cell_type in enumerate(cellTypesNK):
                for j, row in enumerate(rows):
                    print(filename)
                    sample, _ = importF(filename, row)
                    if (row == 'H' and i == 4) is False:
                        if cell_type:
                            for jj, subSample in enumerate(sample):
                                sample[jj] = applyMatrix(subSample, Cd8_NKmatrix)
                            gates = gating(cell_type, dates[i], True, repList[i])
                            _, alldata = count_data(sample, gates, Tcells, True)
                        else:
                            for jj, samplejj in enumerate(sample):
                                print(row, jj)
                                _, pstat, _ = sampleNK(samplejj)
                                alldata.append(pstat)

                        for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                            dat_array = alldata[ii]
                            stat_array = dat_array[[statcol]]
                            stat_array = stat_array.to_numpy()
                            stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                            timelig = mutFunc(row, filename)
                            if stat_array.size == 0:
                                MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timelig[0], "Cell": TitlesNK[k], "Ligand": timelig[1], "Dose": dosemat[0, ii], "Mean": [
                                                   0], "Variance": [0], "Skew": [0], "Kurtosis": [0], "alphStatCov": [0], "Bivalent": timelig[2]}))
                            else:
                                MVdf = MVdf.append(pds.DataFrame.from_dict({"Date": dates[i], "Time": timelig[0], "Cell": TitlesNK[k], "Ligand": timelig[1], "Dose": dosemat[0, ii], "Mean": np.mean(
                                    stat_array), "Variance": np.var(stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array), "alphStatCov": [0], "Bivalent": timelig[2]}))

                    if j == 3 or j == 7:
                        MVdf['Mean'] = MVdf['Mean'] - MVdf['Mean'].min()
                        masterMVdf = masterMVdf.append(MVdf)
                        MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov", "Bivalent"})

    masterMVdf.to_csv("WTDimericMutSingleCellData.csv", index=False)

    return MVdf


def timeFunc(letter):
    if letter == "A" or letter == "E":
        return 4.0
    elif letter == "B" or letter == "F":
        return 2.0
    elif letter == "C" or letter == "G":
        return 1.0
    elif letter == "D" or letter == "H":
        return 0.5


def cytFunc(letter):
    if letter == "A" or letter == "B" or letter == "C" or letter == "D":
        return "IL2"
    elif letter == "E" or letter == "F" or letter == "G" or letter == "H":
        return "IL15"


def mutFunc(letter, datafile):
    if datafile == "/home/brianoj/Muteins 060-062 T/2019-04-19 IL2-060 IL2-062 Treg plate" or datafile == "/home/brianoj/Muteins 060-062 Nk/2019-04-19 IL2-060 IL2-062 NK plate":
        if letter == "A":
            return [4.0, "WT N-term", 1]
        elif letter == "B":
            return [4.0, "WT N-term", 1]
        elif letter == "C":
            return [4.0, "WT N-term", 1]
        elif letter == "D":
            return [0.5, "WT N-term", 1]
        elif letter == "E":
            return [4.0, "H16N N-term", 1]
        elif letter == "F":
            return [2.0, "H16N N-term", 1]
        elif letter == "G":
            return [1.0, "H16N N-term", 1]
        elif letter == "H":
            return [2.0, "WT N-term", 1]

    elif datafile == "/home/brianoj/Muteins 088-097 T/2019-04-19 IL2-088 IL2-097 Treg plate" or datafile == "/home/brianoj/Muteins 088-097 Nk/2019-04-19 IL2-088 IL2-097 NK plate":
        if letter == "A":
            return [4.0, "R38Q N-term", 1]
        elif letter == "B":
            return [2.0, "R38Q N-term", 1]
        elif letter == "C":
            return [1.0, "R38Q N-term", 1]
        elif letter == "D":
            return [1.0, "WT N-term", 1]
        elif letter == "E":
            return [4.0, "R38Q/H16N", 1]
        elif letter == "F":
            return [2.0, "R38Q/H16N", 1]
        elif letter == "G":
            return [1.0, "R38Q/H16N", 1]
        elif letter == "H":
            return [0.5, "R38Q/H16N", 1]

    elif datafile == "/home/brianoj/Muteins 060-088 T/2019-05-02 IL2-060 IL2-088 Treg plate" or datafile == "/home/brianoj/Muteins 060-088 Nk/2019-05-02 IL2-060 IL2-088 NK plate":
        if letter == "A":
            return [4.0, "WT N-term", 1]
        elif letter == "B":
            return [4.0, "WT N-term", 1]
        elif letter == "C":
            return [4.0, "WT N-term", 1]
        elif letter == "D":
            return [0.5, "WT N-term", 1]
        elif letter == "E":
            return [4.0, "R38Q N-term", 1]
        elif letter == "F":
            return [2.0, "R38Q N-term", 1]
        elif letter == "G":
            return [1.0, "R38Q N-term", 1]
        elif letter == "H":
            return [2.0, "R38Q N-term", 1]

    elif datafile == "/home/brianoj/Muteins 062-097 T/2019-05-02 IL2-062 IL2-097 Treg plate" or datafile == "/home/brianoj/Muteins 062-097 Nk/2019-05-02 IL2-062 IL2-097 NK plate":
        if letter == "A":
            return [4.0, "H16N N-term", 1]
        elif letter == "B":
            return [2.0, "H16N N-term", 1]
        elif letter == "C":
            return [1.0, "H16N N-term", 1]
        elif letter == "D":
            return [1.0, "H16N N-term", 1]
        elif letter == "E":
            return [4.0, "R38Q/H16N", 1]
        elif letter == "F":
            return [2.0, "R38Q/H16N", 1]
        elif letter == "G":
            return [1.0, "R38Q/H16N", 1]
        elif letter == "H":
            return [0.5, "R38Q/H16N", 1]
