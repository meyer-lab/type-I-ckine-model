"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import numpy as np
from .figureCommon import subplotLabel, getSetup
from ..flow import importF
from ..PCA import pcaAll, pcaPlt, appPCA, fitPCA, sampleT, sampleNK, pcaPltColor, pcaAllCellType, loadingPlot


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((13, 10), (3, 4))
    Titles = [" 84 nM IL-2", " 0.345 nM IL-2", " Zero Treatment"]
    ax[10].axis('off')
    ax[11].axis('off')

    subplotLabel(ax)

    dose_ind = np.array([0., 6., 11.])
    Tsample, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "C")
    _, pstat_arrayT, _, loadingT = pcaAll(Tsample, Tcells=True)  # take out titles req
    dataT, _, _ = sampleT(Tsample[0])
    PCAobjT, _ = fitPCA(dataT, Tcells=True)

    Nksample, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "C")
    _, pstat_arrayNk, _, loadingNk = pcaAll(Nksample, Tcells=False)  # take out titles req
    dataNk, _, _ = sampleNK(Nksample[0])
    PCAobjNk, _ = fitPCA(dataNk, Tcells=False)

    for i, col in enumerate(dose_ind):
        col = int(col)
        dataT, _, _ = sampleT(Tsample[col])
        xfT = appPCA(dataT, PCAobjT, Tcells=True)
        pcaPlt(xfT, pstat_arrayT[col], ax[i], Tcells=True)
        ax[i].set_title("T-reg" + Titles[i], fontsize=15)

        dataNk, _, _ = sampleNK(Nksample[col])
        xfNk = appPCA(dataNk, PCAobjNk, Tcells=False)
        pcaPlt(xfNk, pstat_arrayNk[col], ax[i + 4], Tcells=False)
        ax[i + 4].set_title("Nk" + Titles[i], fontsize=15)

    loadingPlot(loadingT, ax=ax[3], Tcells=True)
    ax[3].set_title("T-reg Loadings", fontsize=15)
    loadingPlot(loadingNk, ax=ax[7], Tcells=False)
    ax[7].set_title("T-reg Loadings", fontsize=15)

    ColPlot(Tsample, ax, 4, True)
    ColPlot(Nksample, ax, 4, False)

    return f


def ColPlot(sample, ax, col, Tcells=True):
    """Fills in an ax with a colored by gating PCA plot"""
    if Tcells:
        _, _, xf_arrayT, _ = pcaAll(sample, Tcells=True)
        _, _, _, _, colormatT = pcaAllCellType(sample, Tcells=True)
        pcaPltColor(xf_arrayT[col], colormatT[col], ax=ax[8], Tcells=True)
        ax[8].set_title("T-reg PCA by Gating", fontsize=15)
    else:
        _, _, xf_arrayNk, _ = pcaAll(sample, Tcells=False)
        _, _, _, _, colormatNk = pcaAllCellType(sample, Tcells=False)
        pcaPltColor(xf_arrayNk[col], colormatNk[col], ax=ax[9], Tcells=False)
        ax[9].set_title("Nk PCA by Gating", fontsize=15)
