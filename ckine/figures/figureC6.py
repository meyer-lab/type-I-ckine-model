"""
Figure 6. Optimization of Ligands
"""
import os
from os.path import dirname, join
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup
from ..MBmodel import polyc

path_here = dirname(dirname(__file__))


def makeFigure():
    """ Make figure 6. """
    # Get list of axis objects
    ax, f = getSetup((16, 8), (3, 6))
    subplotLabel(ax)
    optimizeDesign(ax[0:2], ["Treg"], ["Thelper", "NK", "CD8"])

    return f


optBnds = [(5, 11),  # Ka IL2Ra
           (5, 11),  # Ka IL2Rb
           (-13, -11)]  # Kx


def cytBindingModelOpt(x, val, cellType):
    """Runs binding model for a given mutein, valency, dose, and cell type. """
    recQuantDF = pd.read_csv(join(path_here, "data/RecQuantitation.csv"))

    affs = [[np.power(10, x[0]), 1e2], [1e2, np.power(10, x[1])]]
    Kx = np.power(10, x[2])

    recCount = recQuantDF[["Receptor", cellType]]
    recCount = [recCount.loc[(recCount.Receptor == "IL2Ra")][cellType].values, recCount.loc[(recCount.Receptor == "IL2Rb")][cellType].values]
    recCount = np.ravel(np.power(10, recCount))
    output = polyc(1e-9 / val, Kx, recCount, [[val, val]], [1.0], affs)[1][0][1]  # IL2RB binding only

    return output


def minSelecFunc(x, val, targCell, offTCells):
    """Provides the function to be minimized to get optimal selectivity"""
    offTargetBound = 0

    targetBound = cytBindingModelOpt(x, val, targCell[0])
    for cellT in offTCells:
        offTargetBound += cytBindingModelOpt(x, val, cellT)

    return (offTargetBound) / (targetBound)


def optimizeDesign(ax, targCell, offTcells):
    """ A more general purpose optimizer """
    X0 = [8, 8, -12]
    vals = np.logspace(0, 4, num=5, base=2)
    optDF = pd.DataFrame(columns={"Valency", "Selectivity", "IL2Ra", "IL2RBG"})

    for val in vals:
        optimized = minimize(minSelecFunc, X0, bounds=optBnds, args=(val, targCell, offTcells), jac="3-point")
        print(val)
        print(optimized.fun)
        IL2RaKD = 1e9 / np.power(10, optimized.x[0])
        IL2RBGKD = 1e9 / np.power(10, optimized.x[1])
        optDF = optDF.append(pd.DataFrame({"Valency": [val], "Selectivity": [len(offTcells) / optimized.fun], "IL2Ra": IL2RaKD, "IL2RBG": IL2RBGKD}))

    sns.barplot(x="Valency", y="Selectivity", data=optDF, ax=ax[0])

    affDF = pd.melt(optDF, id_vars=['Valency'], value_vars=['IL2Ra', 'IL2RBG'])
    sns.barplot(x="Valency", y="value", hue="variable", data=affDF, ax=ax[1])
    ax[1].set(yscale="log", ylabel=r"$K_D$ (nM)")

    return optimized
