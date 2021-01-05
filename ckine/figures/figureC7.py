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

    ax, f = getSetup((10, 5), (2, 4))
    ax[3].axis("off")
    axLabel = ax.copy()
    del axLabel[3]
    subplotLabel(axLabel)

    # Imports receptor levels from .csv created by figC5
    respDF = import_pstat_all()
    respTensor = makeTensor(respDF)
    tFacAll = factorTensor(respTensor, 3)
    tFacAll.normalize()

    R2Xplot(ax[0], respTensor, 5)
    ligHandles, ligLabels = plot_tFac_Ligs(ax[1:3], tFacAll, respDF)
    ax[3].legend(ligHandles, ligLabels, loc="center", prop={"size": 8}, title="Ligand Legend")
    plot_tFac_Time(ax[4], tFacAll, respDF)
    plot_tFac_Conc(ax[5], tFacAll, respDF)
    plot_tFac_Cells(ax[6:8], tFacAll, respDF)

    return f
