"""
This creates Figure 2.
"""
import string
import numpy as np
from .figureCommon import subplotLabel, getSetup
from .figureB1 import runIL2simple
from ..model import receptor_expression
from ..make_tensor import rxntfR
from ..imports import import_Rexpr

df, _, _ = import_Rexpr()
df.reset_index(inplace=True)

changesAff = np.logspace(-2, 2, num=7)
cellNames = ["T-reg", "T-helper", "NK"]


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    ax, f = getSetup((8, 11), (3, 3))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    cellReceptors = np.zeros((3, 3))

    for i, cellName in enumerate(cellNames):
        IL2Ra = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == 'IL-2R$\\alpha$'), "Count"].values[0]
        IL2Rb = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == 'IL-2R$\\beta$'), "Count"].values[0]
        gc = df.loc[(df["Cell Type"] == cellName) & (df["Receptor"] == '$\\gamma_{c}$'), "Count"].values[0]
        cellReceptors[i, :] = receptor_expression(np.array([IL2Ra, IL2Rb, gc]).astype(np.float), rxntfR[17], rxntfR[20], rxntfR[19], rxntfR[21])

    for i, receptors in enumerate(cellReceptors):
        plot_dResp_2Ra(ax[i], receptors)
        ax[i].set_title(cellNames[i])
        plot_dResp_2Rb(ax[3 + i], receptors)
        ax[3 + i].set_title(cellNames[i])
        plot_dResp_2Rb_HIGH(ax[6 + i], receptors)
        ax[6 + i].set_title(cellNames[i])

    ax[0].legend(title="IL2Ra Kd vs wt")
    ax[3].legend(title="IL2Rb Kd vs wt")
    ax[6].legend(title="IL2Rb Kd vs wt")

    return f


def dRespon_loc(input_params, input_receptors):  # same as dRespon except with different ILs range
    """ Calculate an IL2 dose response curve. """
    ILs = np.logspace(-4.0, 3.0)
    activee = np.array([runIL2simple(rxntfR, input_params, ii, input_receptors=input_receptors) for ii in ILs])
    return ILs, activee


def plot_dResp_2Ra(ax, input_receptors):
    """ Plots dose response curves for various IL2Ra affinities given a CD25 relative expression rate. """
    for _, itemA in enumerate(changesAff):
        ILs, BB = dRespon_loc([itemA, 1.0, 5.0], input_receptors)
        ax.semilogx(ILs, BB, label=str(round(itemA, 2)))

    ax.set_ylabel('Active Receptor Complexes (#/cell)')
    ax.set_xlabel('IL2 [nM]')


def plot_dResp_2Rb(ax, input_receptors):
    """ Plots dose response curves for various IL2Rb affinities given a CD25 relative expression rate with wt IL2Ra affinity. """
    for _, itemB in enumerate(changesAff):
        ILs, BB = dRespon_loc([1.0, itemB, 5.0], input_receptors)
        ax.semilogx(ILs, BB, label=str(round(itemB, 2)))

    ax.set_ylabel('Active Receptor Complexes (#/cell)')
    ax.set_xlabel('IL2 [nM]')


def plot_dResp_2Rb_HIGH(ax, input_receptors):
    """ Plots dose response curves for various IL2Rb affinities given a CD25 relative expression rate with increased IL2Ra affinity. """
    for _, itemB in enumerate(changesAff):
        ILs, BB = dRespon_loc([0.1, itemB, 5.0], input_receptors)
        ax.semilogx(ILs, BB, label=str(round(itemB, 2)))

    ax.set_ylabel('Active Receptor Complexes (#/cell)')
    ax.set_xlabel('IL2 [nM]')
