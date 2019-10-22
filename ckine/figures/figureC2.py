"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import string
import matplotlib.lines as mlines
from .figureCommon import subplotLabel, getSetup
from ..flow import importF, treg, tregMem, tregNaive, nonTreg, THelpMem, THelpN, nk, StatGini

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 5), (2, 4))

    for i, item in enumerate(ax):
        if i < 8:
            subplotLabel(item, string.ascii_uppercase[i])

    gates = [False, treg, tregMem, tregNaive, nonTreg, THelpMem, THelpN]
    Titles = ["Tcells", "T-regs", "Mem Treg", "Naive Treg", "T-helper", "Mem Th", "Naive Th"]
    Tsample, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "B")
    Nksample, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "B")
    for i, gate in enumerate(gates):
        StatGini(Tsample, ax[i], gate, Tcells=True)
        ax[i].set_title(Titles[i])

    StatGini(Nksample, ax[7], nk, Tcells=False)
    ax[7].set_title("NK")

    global_legend(ax[7])

    return f


def global_legend(ax):
    """ Create legend for Inverse and Standard Gini """
    blue = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', markersize=6, label='Gini Coeff')
    orange = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', markersize=6, label='Inverse Gini Coeff')
    ax.legend(handles=[orange, blue], bbox_to_anchor=(0, 1), loc="upper left")
