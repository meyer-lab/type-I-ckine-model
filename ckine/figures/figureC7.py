"""
This creates Figure 7, tensor factorization of mutant and WT biv and monovalent ligands.
"""

import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..imports import import_pstat_all

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 5), (2, 4))
    subplotLabel(ax)

    # Imports receptor levels from .csv created by figC5
    respDF = import_pstat_all()
    respTensor = makeTensor(respDF)

    return f


def makeTensor(sigDF):
    """Makes tensor of data with dimensions mutein x valency x time point x concentration x cell type"""
    ligands = sigDF.Ligand.unique()
    valency = sigDF.Bivalent.unique()
    tps = sigDF.Time.unique()
    concs = sigDF.Dose.unique()
    cellTypes = sigDF.Cell.unique()
    tensor = np.empty((len(ligands), len(valency), len(tps), len(concs), len(cellTypes)))
    tensor[:] = np.nan
    for i, lig in enumerate(ligands):
        for j, val in enumerate(valency):
            for k, tp in enumerate(tps):
                for ii, conc in enumerate(concs):
                    for jj, cell in enumerate(cellTypes):
                        entry = sigDF.loc[(sigDF.Ligand == lig) & (sigDF.Bivalent == val) & (sigDF.Time == tp) & (sigDF.Dose == conc) & (sigDF.Cell == cell)].Mean.values
                        if len(entry) >= 1:
                            tensor[i, j, k, ii, jj] = np.mean(entry)

    return tensor
