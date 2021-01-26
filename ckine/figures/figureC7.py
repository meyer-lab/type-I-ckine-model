"""
This creates Figure 7, tensor factorization of mutant and WT biv and monovalent ligands.
"""

import os
import numpy as np
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..imports import import_pstat_all
from ..tensorFac import makeTensor, factorTensor, R2Xplot, plot_tFac_Ligs, plot_tFac_Time, plot_tFac_Conc, plot_tFac_Cells

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 10), (4, 4))
    ax[3].axis("off")
    ax[11].axis("off")
    axLabel = ax.copy()
    del axLabel[3]
    del axLabel[10]
    subplotLabel(axLabel)

    # Imports receptor levels from .csv created by figC5
    respDF = import_pstat_all()
    respTensor = makeTensor(respDF)
    tFacAllM = factorTensor(respTensor, 3)
    tFacAllM.normalize()

    R2Xplot(ax[0], respTensor, 5)
    ligHandles, ligLabels = plot_tFac_Ligs(ax[1:3], tFacAllM, respDF)
    ax[3].legend(ligHandles, ligLabels, loc="center", prop={"size": 8}, title="Ligand Legend")
    plot_tFac_Time(ax[4], tFacAllM, respDF)
    plot_tFac_Conc(ax[5], tFacAllM, respDF)
    plot_tFac_Cells(ax[6:8], tFacAllM, respDF)

    respTensor = makeTensor(respDF, Variance=True)
    tFacAllV = factorTensor(respTensor, 3)
    tFacAllV.normalize()

    R2Xplot(ax[8], respTensor, 5)
    ligHandles, ligLabels = plot_tFac_Ligs(ax[9:11], tFacAllV, respDF)
    ax[11].legend(ligHandles, ligLabels, loc="center", prop={"size": 8}, title="Ligand Legend")
    plot_tFac_Time(ax[12], tFacAllV, respDF)
    plot_tFac_Conc(ax[13], tFacAllV, respDF)
    plot_tFac_Cells(ax[14:16], tFacAllV, respDF)

    return f
