"""
This creates Figure 4 for Single Cell data analysis. Plots of flow intensity versus receptor quantification.
"""

import os
import numpy as np
from .figureCommon import subplotLabel, getSetup, plot_regression
from ..imports import channels, receptors
from ..flow import importF

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 5), (2, 4))

    subplotLabel(ax)

    print(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/")

    sampleD, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads", "D")
    sampleE, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "E")
    sampleF, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "F")
    sampleG, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "G")
    sampleH, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "H")

    recQuant1 = np.array([0.0, 4407, 59840, 179953, 625180])  # CD25, CD122, IL15
    recQuant2 = np.array([0.0, 7311, 44263, 161876, 269561])  # CD132

    plot_regression(ax[0], sampleD, channels["D"], receptors["D"], recQuant1)
    plot_regression(ax[1], sampleE, channels["E"], receptors["E"], recQuant2, 2, True)
    plot_regression(ax[2], sampleF, channels["F"], receptors["F"], recQuant1)
    plot_regression(ax[3], sampleG, channels["G"], receptors["G"], recQuant1)
    plot_regression(ax[4], sampleH, channels["H"], receptors["H"], recQuant1)

    return f
