"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import matplotlib.lines as mlines
import pandas as pds
import numpy as np
from scipy import stats
from .figureCommon import subplotLabel, getSetup
from ..flow import importF
from ..PCA import StatGini, sampleT, sampleNK
from ..flow import gating, count_data

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    gates = [False, 'treg', 'tregMem', 'tregNaive', 'nonTreg', 'THelpMem', 'THelpN']
    Titles = ["Tcells", "T-regs", "Mem Treg", "Naive Treg", "T-helper", "Mem Th", "Naive Th"]
    Tsample, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "B")
    Nksample, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "B")
    for i, cell in enumerate(gates):
        StatGini(Tsample, ax[i], cell, Tcells=True)
        ax[i].set_title(Titles[i])
        StatMV(Tsample, ax[i + 8], cell, Tcells=True)
        ax[i + 8].set_title(Titles[i])

    StatGini(Nksample, ax[7], 'nk', Tcells=False)
    ax[7].set_title("NK")
    StatMV(Nksample, ax[15], 'nk', Tcells=False)
    ax[15].set_title("NK")

    # global_legend(ax[7])

    return f


def global_legend(ax):
    """ Create legend for Inverse and Standard Gini """
    blue = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', markersize=6, label='Gini Coeff')
    orange = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', markersize=6, label='Inverse Gini Coeff')
    ax.legend(handles=[orange, blue], bbox_to_anchor=(0, 1), loc="upper left")


def StatMV(sampleType, ax, cell_type, Tcells=True):
    """
    Calculate mean and variance of a sample in a pandas dataframe, and plot.
    """
    MVdf = pds.DataFrame(columns={"Dose", "Mean", "Variance", "Skew", "Kurtosis"})
    alldata = []
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])

    if Tcells:
        statcol = "RL1-H"
    else:
        statcol = "BL2-H"

    if cell_type:
        gates = gating(cell_type)
        _, alldata = count_data(sampleType, gates, Tcells)  # returns array of dfs in case of gate or no gate

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
        MVdf = MVdf.append(pds.DataFrame.from_dict({"Dose": dosemat[0, i], "Mean": np.mean(stat_array), "Variance": np.var(
            stat_array), "Skew": stats.skew(stat_array), "Kurtosis": stats.kurtosis(stat_array)}))

    MVdf['Mean'] = MVdf['Mean'] - MVdf['Mean'].min()
    MVdf.plot.scatter(x='Dose', y='Mean', ax=ax, color="dodgerblue", legend=False)
    ax.set_xscale("log")
    ax.set_xlabel("Cytokine Dosage (log10[nM])")
    ax.set_ylabel("Mean", color="dodgerblue")
    ax.tick_params(axis='y', labelcolor="dodgerblue")
    ax1 = ax.twinx()
    MVdf.plot.scatter(x='Dose', y='Variance', ax=ax1, color="orangered", legend=False)
    ax1.set_ylabel("Variance", color="orangered")
    ax1.tick_params(axis='y', labelcolor="orangered")

    return MVdf
