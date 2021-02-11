"""
This creates Figure 8, dose response of all IL-2 cytokines using binding model.
"""

import numpy as np
from .figureCommon import subplotLabel, getSetup, plotDoseResponses
from ..MBmodel import runFullModel


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    modelDF = runFullModel()

    ligands = modelDF.Ligand.unique()
    cells = modelDF.Cell.unique()
    ligList = np.array([])
    valList = np.array([])

    for lig in ligands:
        uniqueVals = modelDF.loc[(modelDF.Ligand == lig)].Valency.unique()
        valList = np.append(valList, uniqueVals)
        ligList = np.append(ligList, np.tile(lig, uniqueVals.size))

    ax, f = getSetup((10, 25), (ligList.size, 4))

    for i, lig in enumerate(ligList):
        for j, cell in enumerate(cells):
            plotDoseResponses(ax[4 * i + j], modelDF, lig, valList[i], cell)

    return f
