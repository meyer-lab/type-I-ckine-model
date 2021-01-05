import os
from os.path import join
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

    dataFiles = ["ckine/data/2019-11-08 monomer IL-2 Fc signaling/CD4 T cells - IL2-060 mono, IL2-060 dimeric",
                 "ckine/data/2019-11-08 monomer IL-2 Fc signaling/CD4 T cells - IL2-062 mono, IL2-118 mono",
                 "ckine/data/2019-11-27 monomer IL-2 Fc signaling/CD4 T cells - C-term IL2-060 mono, C-term V91K mono",
                 "ckine/data/2019-12-05 monomer IL-2 Fc signaling/CD4 T cells - IL2-109 mono, IL2-118 mono",
                 "ckine/data/2019-12-05 monomer IL-2 Fc signaling/CD4 T cells - IL2-110 mono, C-term N88D mono",
                 "ckine/data/2019-11-08 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-060 mono, IL2-060 dimeric",
                 "ckine/data/2019-11-08 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-062 mono, IL2-118 mono",
                 "ckine/data/2019-11-27 monomer IL-2 Fc signaling/NK CD8 T cells - C-term IL2-060 mono, C-term V91K mono",
                 "ckine/data/2019-12-05 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-109 mono, IL2-118 mono",
                 "ckine/data/2019-12-05 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-110 mono, C-term N88D mono"]
    dates = ["11/8/2019", "11/8/2019", "11/27/2019", "12/5/2019", "12/5/2019", "11/8/2019", "11/8/2019", "11/27/2019", "12/5/2019", "12/5/2019"]
    repList = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cellTypesT = ['treg', 'nonTreg']
    cellTypesNK = ["nk", "cd"]
    TitlesT = ["Treg", "Thelper"]
    TitlesNK = ["NK", "CD8"]
    masterMVdf = pds.DataFrame(columns={"Date", "Time", "Cell", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov"})
    MVdf = pds.DataFrame(columns={"Date", "Time", "Ligand", "Dose", "Mean", "Variance", "Skew", "Kurtosis", "alphStatCov"})
    alldata = []
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])
    # T_matrix = compMatrix("2019-11-08", "1", "A")  # Create matrix 1
    # Cd8_NKmatrix = compMatrix("2019-11-08", "1", "B")  # Create matrix 2

    print("Starting Muteins")

    for i, filename in enumerate(dataFiles):
        if i < 5:
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
                        # for jj, subSample in enumerate(sample):
                        #    sample[jj] = applyMatrix(subSample, T_matrix)
                        gates = gating(cell_type, dates[i], True, repList[i])
                        _, alldata = count_data(sample, gates, Tcells, True)
                    for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                        dat_array = alldata[ii]
                        stat_array = dat_array[[statcol]]
                        stat_array = stat_array.to_numpy()
                        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                        IL2Ra_array = dat_array[[IL2RaCol]]
                        IL2Ra_array = IL2Ra_array.to_numpy()
                        IL2Ra_array = IL2Ra_array.clip(min=1)
                        IL2Ra_array = IL2Ra_array / 1.5  # convert roughly to number from signal
                        IL2Ra_array = np.reshape(IL2Ra_array[stat_array != np.amax(stat_array)], (-1, 1))  # Remove random exploding value
                        stat_array = np.reshape(stat_array[stat_array != np.amax(stat_array)], (-1, 1))  # Remove random exploding value
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
                    if cell_type:
                        # for jj, subSample in enumerate(sample):
                        #    sample[jj] = applyMatrix(subSample, Cd8_NKmatrix)
                        gates = gating(cell_type, dates[i], True, repList[i])
                        _, alldata = count_data(sample, gates, Tcells, True)
                    for ii, sampleii in enumerate(sample):  # get pstat data and put it into list form
                        dat_array = alldata[ii]
                        stat_array = dat_array[[statcol]]
                        stat_array = stat_array.to_numpy()
                        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
                        stat_array = np.reshape(stat_array[stat_array != np.amax(stat_array)], (-1, 1))  # Remove random exploding value
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

    masterMVdf.to_csv("MonomericMutSingleCellData.csv", index=False)

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

# done done


def mutFunc(letter, datafile):
    if datafile == "ckine/data/2019-11-08 monomer IL-2 Fc signaling/CD4 T cells - IL2-060 mono, IL2-060 dimeric" or datafile == "ckine/data/2019-11-08 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-060 mono, IL2-060 dimeric":
        if letter == "A":
            return [4.0, "WT N-term", 0]
        elif letter == "B":
            return [2.0, "WT N-term", 0]
        elif letter == "C":
            return [1.0, "WT N-term", 0]
        elif letter == "D":
            return [0.5, "WT N-term", 0]
        elif letter == "E":
            return [4.0, "WT N-term", 1]
        elif letter == "F":
            return [2.0, "WT N-term", 1]
        elif letter == "G":
            return [1.0, "WT N-term", 1]
        elif letter == "H":
            return [0.5, "WT N-term", 1]
# done done
    elif datafile == "ckine/data/2019-11-08 monomer IL-2 Fc signaling/CD4 T cells - IL2-062 mono, IL2-118 mono" or datafile == "ckine/data/2019-11-08 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-062 mono, IL2-118 mono":
        if letter == "A":
            return [4.0, "H16N N-term", 0]
        elif letter == "B":
            return [2.0, "H16N N-term", 0]
        elif letter == "C":
            return [1.0, "H16N N-term", 0]
        elif letter == "D":
            return [0.5, "H16N N-term", 0]
        elif letter == "E":
            return [4.0, "H16L N-term", 0]
        elif letter == "F":
            return [2.0, "H16L N-term", 0]
        elif letter == "G":
            return [1.0, "H16L N-term", 0]
        elif letter == "H":
            return [0.5, "H16L N-term", 0]
# done done
    elif datafile == "ckine/data/2019-11-27 monomer IL-2 Fc signaling/CD4 T cells - C-term IL2-060 mono, C-term V91K mono" or datafile == "ckine/data/2019-11-27 monomer IL-2 Fc signaling/NK CD8 T cells - C-term IL2-060 mono, C-term V91K mono":
        if letter == "A":
            return [4.0, "WT C-term", 0]
        elif letter == "B":
            return [2.0, "WT C-term", 0]
        elif letter == "C":
            return [1.0, "WT C-term", 0]
        elif letter == "D":
            return [0.5, "WT C-term", 0]
        elif letter == "E":
            return [4.0, "V91K C-term", 0]
        elif letter == "F":
            return [2.0, "V91K C-term", 0]
        elif letter == "G":
            return [1.0, "V91K C-term", 0]
        elif letter == "H":
            return [0.5, "V91K C-term", 0]
# done not done
    elif datafile == "ckine/data/2019-12-05 monomer IL-2 Fc signaling/CD4 T cells - IL2-109 mono, IL2-118 mono" or datafile == "ckine/data/2019-12-05 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-109 mono, IL2-118 mono":
        if letter == "A":
            return [4.0, "R38Q N-term", 0]
        elif letter == "B":
            return [2.0, "R38Q N-term", 0]
        elif letter == "C":
            return [1.0, "R38Q N-term", 0]
        elif letter == "D":
            return [0.5, "R38Q N-term", 0]
        elif letter == "E":
            return [4.0, "H16L N-term", 0]
        elif letter == "F":
            return [2.0, "H16L N-term", 0]
        elif letter == "G":
            return [1.0, "H16L N-term", 0]
        elif letter == "H":
            return [0.5, "H16L N-term", 0]
# done
    elif datafile == "ckine/data/2019-12-05 monomer IL-2 Fc signaling/CD4 T cells - IL2-110 mono, C-term N88D mono" or datafile == "ckine/data/2019-12-05 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-110 mono, C-term N88D mono":
        if letter == "A":
            return [4.0, "F42Q N-Term", 0]
        elif letter == "B":
            return [2.0, "F42Q N-Term", 0]
        elif letter == "C":
            return [1.0, "F42Q N-Term", 0]
        elif letter == "D":
            return [0.5, "F42Q N-Term", 0]
        elif letter == "E":
            return [4.0, "N88D C-term", 0]
        elif letter == "F":
            return [2.0, "N88D C-term", 0]
        elif letter == "G":
            return [1.0, "N88D C-term", 0]
        elif letter == "H":
            return [0.5, "N88D C-term", 0]
