"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import matplotlib.lines as mlines
import pandas as pds
import numpy as np
import seaborn as sns
from scipy import stats
from .figureCommon import subplotLabel, getSetup, dosemat
from ..flow import importF
from ..PCA import sampleT, sampleNK
from ..flow import gating, count_data

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 4), multz={0: 1, 2: 1, 4: 1, 6: 1})

    subplotLabel(ax)

    gatesT = ["treg", "nonTreg"]
    TitlesT = ["T-regs", "T-helper"]
    gatesNK = ["nk", "cd"]
    TitlesNK = ["NK", "CD8+"]
    Tsample2, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "B")
    NKsample2, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "B")
    Tsample15, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "F")
    NKsample15, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "F")
    Tdate = "4/18/2019"
    NKdate = "3/15/2019"

    violinDist(Tsample2, Tsample15, ax[0], "treg", TitlesT[0], Tdate, Tcells=True)
    violinDist(Tsample2, Tsample15, ax[1], "nonTreg", TitlesT[1], Tdate, Tcells=True)
    violinDist(NKsample2, NKsample15, ax[2], "nk", TitlesNK[0], NKdate, Tcells=False)
    violinDist(NKsample2, NKsample15, ax[3], "cd", TitlesNK[1], NKdate, Tcells=False)

    for i, cell in enumerate(gatesT):
        StatMV(Tsample2, ax[i + 4], cell, "IL2", TitlesT[i], Tdate, Tcells=True)
        StatMV(Tsample15, ax[i + 8], cell, "IL15", TitlesT[i], Tdate, Tcells=True)
    for j, cell in enumerate(gatesNK):
        StatMV(NKsample2, ax[j + 6], cell, "IL2", TitlesNK[j], Tdate, Tcells=False)
        StatMV(NKsample15, ax[j + 10], cell, "IL15", TitlesNK[j], Tdate, Tcells=False)

    return f


def global_legend(ax):
    """ Create legend for Inverse and Standard Gini """
    blue = mlines.Line2D([], [], color="navy", marker="o", linestyle="None", markersize=6, label="Gini Coeff")
    orange = mlines.Line2D([], [], color="darkorange", marker="o", linestyle="None", markersize=6, label="Inverse Gini Coeff")
    ax.legend(handles=[orange, blue], bbox_to_anchor=(0, 1), loc="upper left")


def StatMV(sampleType, ax, cell_type, ligand, title, date, Tcells=True):
    """
    Calculate mean and variance of a sample in a pandas dataframe, and plot.
    """
    MVdf = pds.DataFrame(columns={"Dose", "Mean", "Variance", "Skew", "Kurtosis"})
    alldata = []

    if Tcells:
        statcol = "RL1-H"
    else:
        statcol = "BL2-H"

    if cell_type:
        gates = gating(cell_type, date, True)
        _, alldata = count_data(sampleType, gates, Tcells)  # returns array of dfs in case of gate or no gate

    else:
        for i, sample in enumerate(sampleType):
            if Tcells:
                _, pstat, _ = sampleT(sample)
                alldata.append(pstat)
            else:
                _, pstat, _ = sampleNK(sample)
                alldata.append(pstat)

    for i, _ in enumerate(sampleType):  # get pstat data and put it into list form
        dat_array = alldata[i]
        stat_array = dat_array[[statcol]]
        stat_array = stat_array.to_numpy()
        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
        if stat_array.size == 0:
            MVdf = MVdf.append(pds.DataFrame.from_dict({"Dose": dosemat[i], "Mean": [0], "Variance": [0], "Skew": [0], "Kurtosis": [0]}))
        else:
            MVdf = MVdf.append(
                pds.DataFrame.from_dict({"Dose": dosemat[i], "Mean": np.mean(stat_array), "Variance": np.var(stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array)})
            )

    MVdf["Mean"] = MVdf["Mean"] - MVdf["Mean"].min()
    MVdf.plot(x="Dose", y="Mean", ax=ax, color="dodgerblue", legend=False)
    ax.set_ylabel("Mean", color="dodgerblue")
    ax.tick_params(axis="y", labelcolor="dodgerblue")
    ax1 = ax.twinx()
    MVdf.plot(x="Dose", y="Variance", ax=ax1, color="orangered", legend=False)
    ax1.set_ylabel("Variance", color="orangered")
    ax1.tick_params(axis="y", labelcolor="orangered")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlabel(ligand + " (log10[nM])")

    return MVdf


def violinDist(sampleType2, sampleType15, ax, cell_type, title, date, Tcells=True):
    """
    Calculate mean and variance of a sample in a pandas dataframe, and plot.
    """
    distDF = pds.DataFrame(columns={"Dose", "Ligand", "pSTAT", "Mean"})
    alldata2, alldata15 = [], []
    ILs = ["IL-2", "IL-15"]

    if Tcells:
        statcol = "RL1-H"
    else:
        statcol = "BL2-H"

    if cell_type:
        gates = gating(cell_type, date, True)
        _, alldata2 = count_data(sampleType2, gates, Tcells)  # returns array of dfs in case of gate or no gate
        _, alldata15 = count_data(sampleType15, gates, Tcells)  # returns array of dfs in case of gate or no gate

    else:
        for i, sample in enumerate(sampleType2):
            if Tcells:
                _, pstat2, _ = sampleT(sample)
                alldata2.append(pstat2)
                _, pstat15, _ = sampleT(sampleType15[i])
                alldata15.append(pstat15)
            else:
                _, pstat2, _ = sampleNK(sample)
                alldata2.append(pstat2)
                _, pstat15, _ = sampleNK(sampleType15[i])
                alldata15.append(pstat15)

    for i, _ in enumerate(sampleType2):  # get pstat data and put it into list form
        dat_array2, dat_array15 = alldata2[i], alldata15[i]
        stat_array2, stat_array15 = dat_array2[[statcol]], dat_array15[[statcol]]
        stat_array2, stat_array15 = stat_array2.to_numpy(), stat_array15.to_numpy()
        stat_array2, stat_array15 = stat_array2.clip(min=1), stat_array15.clip(min=1)  # remove small percentage of negative pstat values

        while np.amax(stat_array2) > 100000:
            stat_array2 = np.reshape(stat_array2[stat_array2 != np.amax(stat_array2)], (-1, 1))
        while np.amax(stat_array15) > 100000:
            stat_array15 = np.reshape(stat_array15[stat_array15 != np.amax(stat_array15)], (-1, 1))

        for kk, pSTATArray in enumerate(np.array([stat_array2, stat_array15])):
            if pSTATArray.size == 0:
                distDF = distDF.append(pds.DataFrame.from_dict({"Dose": dosemat[i], "Ligand": ILs[kk], "pSTAT": [0]}))
            else:
                distDF = distDF.append(pds.DataFrame.from_dict({"Dose": np.tile(dosemat[i], (len(pSTATArray))), "Ligand": np.tile(ILs[kk], (len(pSTATArray))), "pSTAT": pSTATArray.flatten()}))

    sns.violinplot(x="Dose", y="pSTAT", hue="Ligand", data=distDF, split=True, palette={"IL-2": "darkorchid", "IL-15": "goldenrod"}, ax=ax)
    ax.set(xlabel="Ligand nM", ylabel="pSTAT Signal", ylim=(0, 100000), title=title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)

    return distDF
