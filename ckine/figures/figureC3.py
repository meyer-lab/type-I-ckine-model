"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import string
from .figureCommon import subplotLabel, getSetup
from ..flow import importF, treg, nonTreg, nk, EC50_PC_Scan, loadingPlot

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 7.5), (3, 4))
    PCscanVecT = [-2, 2, 6]
    PCscanVecNk = [-1, 1, 3]
    loadingT = []
    loadingNk = []

    for i, item in enumerate(ax):
        if i < 12:
            subplotLabel(item, string.ascii_uppercase[i])

    gates = [False, treg, nonTreg]
    Titles = ["Tcells", "T-regs", "T Helper"]
    Tsample, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "A")
    Nksample, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "A")

    for i, gate in enumerate(gates):
        EC50_PC_Scan(Tsample, PCscanVecT, ax[i], gate, Tcells=True, PC1=True)
        ax[i].set_title(Titles[i] + " PC1 Scan")
        loadingT = EC50_PC_Scan(Tsample, PCscanVecT, ax[i + 4], gate, Tcells=True, PC1=False)
        ax[i + 4].set_title(Titles[i] + " PC2 Scan")
        loadingPlot(loadingT, ax=ax[i + 8], Tcells=True)
        ax[i + 8].set_title(Titles[i] + " Loadings")

    EC50_PC_Scan(Nksample, PCscanVecNk, ax[3], nk, Tcells=False, PC1=True)
    ax[3].set_title("Nk PC1 Scan")
    loadingNk = EC50_PC_Scan(Nksample, PCscanVecNk, ax[7], nk, Tcells=False, PC1=False)
    ax[7].set_title("Nk PC2 Scan")
    loadingPlot(loadingNk, ax=ax[11], Tcells=False)
    ax[11].set_title("Nk Loadings")

    return f
