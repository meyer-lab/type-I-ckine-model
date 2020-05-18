"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import numpy as np
import pandas as pds
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..flow import importF, gating, cellData
from ..PCA import pcaAll, pcaPlt, appPCA, fitPCA, sampleT, sampleNK, pcaPltColor, pcaAllCellType, loadingPlot


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((13, 10), (3, 4))
    Titles = [" 84 nM IL-2", " 0.345 nM IL-2", " Zero Treatment"]

    subplotLabel(ax)

    dose_ind = np.array([0.0, 6.0, 11.0])
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

    # ColPlot(Tsample, ax[8], 4, True)
    # ColPlot(Nksample, ax[9], 4, False)
    RecQuantResp(ax[8], Tsample)
    RecQuantResp(ax[9], Tsample, "treg")
    RecQuantResp(ax[10], Tsample, "nonTreg")
    RecQuantResp(ax[11], Tsample, "tregNaive")

    return f


def ColPlot(sample, ax, col, Tcells=True):
    """Fills in an ax with a colored by gating PCA plot"""
    if Tcells:
        _, _, xf_arrayT, _ = pcaAll(sample, Tcells=True)
        _, _, _, _, colormatT = pcaAllCellType(sample, Tcells=True)
        pcaPltColor(xf_arrayT[col], colormatT[col], ax=ax, Tcells=True)
        ax.set_title("T-reg PCA by Gating", fontsize=15)
    else:
        _, _, xf_arrayNk, _ = pcaAll(sample, Tcells=False)
        _, _, _, _, colormatNk = pcaAllCellType(sample, Tcells=False)
        pcaPltColor(xf_arrayNk[col], colormatNk[col], ax=ax, Tcells=False)
        ax.set_title("Nk PCA by Gating", fontsize=15)


def RecQuantResp(ax, samples, cellType=False):
    """Plots dose response curves for cells separated by their receptor expression levels"""
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])
    quartDF = pds.DataFrame(columns=["IL-2 Dose (nM)", "Activity", "Quartile"])
    if cellType:
        gates = gating(cellType)
    for i, sample in enumerate(samples):

        if cellType:
            df, _ = cellData(sample, gates)
            pstatdata = pds.DataFrame({"RL1-H": df["RL1-H"]})
            df = df["VL1-H"]

        else:
            df, pstatdata, _ = sampleT(sample)
            df = df["VL1-H"]

        quantiles = np.array([df.quantile(0.25), df.quantile(0.5), df.quantile(0.75)])
        quartDF = quartDF.append(pds.DataFrame({"IL-2 Dose (nM)": dosemat[0, i], "Activity": pstatdata[df < quantiles[0]].mean(), "Quartile": "IL-2Rα Quartile 1"}))
        quartDF = quartDF.append(pds.DataFrame({"IL-2 Dose (nM)": dosemat[0, i], "Activity": pstatdata.loc[(df < quantiles[1]) & (df > quantiles[0])].mean(), "Quartile": "IL-2Rα Quartile 2"}))
        quartDF = quartDF.append(pds.DataFrame({"IL-2 Dose (nM)": dosemat[0, i], "Activity": pstatdata.loc[(df < quantiles[2]) & (df > quantiles[1])].mean(), "Quartile": "IL-2Rα Quartile 3"}))
        quartDF = quartDF.append(pds.DataFrame({"IL-2 Dose (nM)": dosemat[0, i], "Activity": pstatdata[df > quantiles[2]].mean(), "Quartile": "IL-2Rα Quartile 4"}))

    sns.lineplot(x="IL-2 Dose (nM)", y="Activity", hue="Quartile", data=quartDF, ax=ax, palette="husl")
    ax.set(xscale="log", xticks=[1e-4, 1e-2, 1e0, 1e2], xlim=[1e-4, 1e2])
    if cellType:
        ax.set_title(cellType)
    else:
        ax.set_title("T-cells")
