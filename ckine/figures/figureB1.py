"""
This creates Figure 1.
"""
import string
import numpy as np
from scipy.optimize import brentq
from .figureCommon import subplotLabel, getSetup
from ..model import runIL2simple, receptor_expression
from ..make_tensor import rxntfR
from ..imports import import_Rexpr

df, _, _ = import_Rexpr()
df.reset_index(inplace=True)

_, numpy_data, cell_names = import_Rexpr()
numpy_data = receptor_expression(numpy_data, rxntfR[17], rxntfR[20], rxntfR[19], rxntfR[21])


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    ax, f = getSetup((11, 6), (2, 2))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    cellNames = ["T-reg", "T-helper"]
    cellReceptors = np.zeros((2, 3))

    for i, cellName in enumerate(cellNames):
        IL2Ra = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == 'IL-2R$\\alpha$'), "Count"].item()
        IL2Rb = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == 'IL-2R$\\beta$'), "Count"].item()
        gc = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == '$\\gamma_{c}$'), "Count"].item()
        cellReceptors[i, :] = receptor_expression(np.array([IL2Ra, IL2Rb, gc]).astype(np.float), rxntfR[17], rxntfR[20], rxntfR[19], rxntfR[21])

    halfMax_IL2RaAff(ax[0])
    activeReceptorComplexes(ax[1])
    halfMax_IL2RbAff(ax[2], cellNames, cellReceptors)
    halfMax_IL2RbAff_highIL2Ra(ax[3], cellNames, cellReceptors)

    return f


def dRespon(input_params, input_receptors=None):
    """ Calculate an IL2 dose response curve. """
    ILs = np.logspace(-3.0, 3.0)
    activee = np.array([runIL2simple(rxntfR, input_params, ii, input_receptors=input_receptors) for ii in ILs]).squeeze()

    return ILs, activee


def IC50global(input_params, input_receptors=None):
    """ Calculate half-maximal concentration w.r.t. wt. """
    tps = np.array([50000.0])
    halfResponse = runIL2simple(rxntfR, [1.0, 1.0, 5.0], 5000.0, input_receptors=input_receptors, tps=tps) / 2.0

    return brentq(lambda x: runIL2simple(rxntfR, input_params, x, input_receptors=input_receptors, tps=tps) - halfResponse, 0, 1000.0, rtol=1e-5)


def halfMax_IL2RaAff(ax):
    """ Plots half maximal IL2 concentration for varied IL2Ra and IL2Rb affinities. """
    changesA_a = np.logspace(-2, 1, num=20)
    changesB_a = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    output = np.zeros((changesA_a.size, changesB_a.size))
    for i, itemA in enumerate(changesA_a):
        for j, itemB in enumerate(changesB_a):
            output[i, j] = IC50global([itemA, itemB, 5.0])
    for ii in range(output.shape[1]):
        ax.loglog(changesA_a, output[:, ii], label=str(changesB_a[ii]))
    ax.loglog([0.01, 10.0], [0.17, 0.17], "k-")
    ax.set(ylabel="Half-Maximal IL2 Concentration [nM]", xlabel="IL2Ra-IL2 Kd (fold wt)", ylim=(0.01, 20))
    ax.legend(title="IL2Rb Kd v wt")


def activeReceptorComplexes(ax):
    """ Plots active receptor complexes per cell across increasing IL2 concentration for wild type and adjusted affinities. """
    wt = dRespon([1.0, 1.0, 5.0])
    ten = dRespon([0.1, 5.0, 5.0])

    ax.semilogx(wt[0], wt[1], label="wt")
    ax.semilogx(ten[0], ten[1], "r", label="10X higher/lower affinity IL2Ra/IL2Rb")
    ax.set(ylabel="Active Receptor Complexes (#/cell)", xlabel="IL2 [nM]")
    ax.legend()


changesA = np.logspace(-1, 1.5, num=20)


def halfMax_IL2RbAff(ax, cellName, receptorExpr):
    """ Plots half maximal IL2 concentration across decreasing IL2Rb affinity for varied IL2Ra expression levels using wild type IL2Ra affinity. """
    output = np.zeros((changesA.size, receptorExpr.shape[0]))
    for i, itemA in enumerate(changesA):
        for j, itemB in enumerate(receptorExpr):
            output[i, j] = IC50global([1.0, itemA, 5.0], input_receptors=itemB)

    for ii in range(output.shape[1]):
        ax.loglog(changesA, output[:, ii], label=str(cellName[ii]))

    ax.loglog([0.1, 10.0], [0.17, 0.17], "k-")
    ax.set(ylabel="Half-Maximal IL2 Concentration [nM]", xlabel="IL2Rb-IL2 Kd (relative to wt)", xlim=(0.1, 10))
    ax.legend(title="Cell Type")


def halfMax_IL2RbAff_highIL2Ra(ax, cellName, receptorExpr):
    """ Plots half maximal IL2 concentration across decreasing IL2Rb affinity for varied IL2Ra expression levels using 10x
    increased IL2Ra affinity. """
    output = np.zeros((changesA.size, receptorExpr.shape[0]))
    for i, itemA in enumerate(changesA):
        for j, itemB in enumerate(receptorExpr):
            output[i, j] = IC50global([0.1, itemA, 5.0], input_receptors=itemB)

    for ii in range(output.shape[1]):
        ax.loglog(changesA, output[:, ii], label=str(cellName[ii]))

    ax.loglog([0.1, 10.0], [0.17, 0.17], "k-")
    ax.set(ylabel="Half-Maximal IL2 Concentration [nM]", xlabel="IL2Rb-IL2 Kd (relative to wt)", xlim=(0.1, 10))
    ax.legend(title="Cell Type")
