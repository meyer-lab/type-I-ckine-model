"""
This creates Figure 2 for Single Cell FC data analysis. Gating, and receptor quant will go here.
"""

import os
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
from scipy import stats
from .figureCommon import subplotLabel, getSetup, dosemat, plot_regression
from ..imports import channels, receptors
from ..flow import importF, gating, count_data, bead_regression
from ..FCimports import import_gates, apply_gates
from ..PCA import sampleT, sampleNK

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    TitlesT = ["T-regs", "T-helper"]
    TitlesNK = ["NK", "CD8+"]
    Tsample2, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "B")
    NKsample2, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "B")
    Tsample15, _ = importF(path_here + "/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT/", "F")
    NKsample15, _ = importF(path_here + "/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate/", "F")
    Tdate = "4/18/2019"
    NKdate = "3/15/2019"

    violinDist(Tsample2, Tsample15, ax[0], "treg", TitlesT[0], Tdate, Tcells=True)
    violinDist(Tsample2, Tsample15, ax[1], "nonTreg", TitlesT[1], Tdate, Tcells=True)
    violinDist(NKsample2, NKsample15, ax[2], "nk", TitlesNK[0], NKdate, Tcells=False)
    violinDist(NKsample2, NKsample15, ax[3], "cd", TitlesNK[1], NKdate, Tcells=False)

    sampleD, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads", "D")
    sampleE, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "E")
    sampleF, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "F")
    sampleG, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "G")
    sampleH, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "H")
    sampleI, _ = importF(path_here + "/data/flow/2019-05-16 Receptor Quant - Beads/", "F")

    recQuant1 = np.array([0.0, 4407, 59840, 179953, 625180])  # CD25, CD122, IL15
    recQuant2 = np.array([0.0, 7311, 44263, 161876, 269561])  # CD132
    recQuant3 = np.array([4407, 59840, 179953, 625180, 0.0])  # CD127

    plot_regression(ax[4], sampleD, channels["D"], receptors["D"], recQuant1)
    plot_regression(ax[5], sampleE, channels["E"], receptors["E"], recQuant1, 2, True)
    plot_regression(ax[6], sampleF, channels["F"], receptors["F"], recQuant2)
    plot_regression(ax[7], sampleG, channels["G"], receptors["G"], recQuant1)
    plot_regression(ax[8], sampleH, channels["H"], receptors["H"], recQuant1)
    plot_regression(ax[9], sampleI, channels["I"], receptors["I"], recQuant3)

    # import bead data and run regression to get equations
    lsq_cd25, lsq_cd122, lsq_cd132, lsq_cd127 = run_regression()

    # create dataframe with gated samples (all replicates)
    df_gates = import_gates()
    df_signal = apply_gates("4-23", "1", df_gates)
    df_signal = df_signal.append(apply_gates("4-23", "2", df_gates))
    df_signal = df_signal.append(apply_gates("4-26", "1", df_gates))
    df_signal = df_signal.append(apply_gates("4-26", "2", df_gates))
    df_signal = df_signal.append(apply_gates("5-16", "1", df_gates))
    df_signal = df_signal.append(apply_gates("5-16", "2", df_gates))

    # make new dataframe for receptor counts
    df_rec = pd.DataFrame(columns=["Cell Type", "Receptor", "Count", "Date", "Plate"])
    cell_names = ["T-reg", "T-helper", "NK", "CD8+"]
    receptors_ = ["CD25", "CD122", "CD132", "CD127"]
    channels_ = ["VL1-H", "BL5-H", "RL1-H", "BL1-H"]
    lsq_params = [lsq_cd25, lsq_cd122, lsq_cd132, lsq_cd127]
    dates = ["4-23", "4-26", "5-16"]
    plates = ["1", "2"]

    # calculate receptor counts
    for _, cell in enumerate(cell_names):
        for j, receptor in enumerate(receptors_):
            for _, date in enumerate(dates):
                for _, plate in enumerate(plates):
                    data = df_signal.loc[(df_signal["Cell Type"] == cell) & (df_signal["Receptor"] == receptor) & (df_signal["Date"] == date) & (df_signal["Plate"] == plate)][channels_[j]]
                    data = data[data >= 0]
                    rec_counts = np.zeros(len(data))
                    for k, signal in enumerate(data):
                        A, B, C, D = lsq_params[j]
                        rec_counts[k] = C * (((A - D) / (signal - D)) - 1)**(1 / B)
                    df_add = pd.DataFrame({"Cell Type": np.tile(cell, len(data)), "Receptor": np.tile(receptor, len(data)),
                                           "Count": rec_counts, "Date": np.tile(date, len(data)), "Plate": np.tile(plate, len(data))})
                    df_rec = df_rec.append(df_add)
    # write to csv
    update_path = path_here + "/data/receptor_levels.csv"
    df_rec.to_csv(str(update_path), index=False, header=True)

    # calculate mean, variance, and skew for each replicate
    df_stats = calculate_moments(df_rec, cell_names, receptors_)

    # plot log10 of mean, variance, and skew
    celltype_pointplot(ax[10], df_stats, "Mean")
    celltype_pointplot(ax[11], df_stats, "Variance")
    celltype_pointplot(ax[12], df_stats, "Skew")

    return f


def violinDist(sampleType2, sampleType15, ax, cell_type, title, date, Tcells=True):
    """
    Calculate mean and variance of a sample in a pandas dataframe, and plot.
    """
    distDF = pd.DataFrame(columns={"Dose", "Ligand", "pSTAT", "Mean"})
    alldata2, alldata15 = [], []
    ILs = ["IL-2", "IL-15"]

    if Tcells:
        statcol = "RL1-H"
    else:
        statcol = "BL2-H"

    if cell_type:
        gates = gating(cell_type, date, True)
        _, alldata2 = count_data(sampleType2, gates, Tcells)  # returns array of dfs in case of gate or no gate
        _, alldata15 = count_data(sampleType15, gates, Tcells)  # returns array of dfs in case of gate or no gate

    else:
        for i, sample in enumerate(sampleType2):
            if Tcells:
                _, pstat2, _ = sampleT(sample)
                alldata2.append(pstat2)
                _, pstat15, _ = sampleT(sampleType15[i])
                alldata15.append(pstat15)
            else:
                _, pstat2, _ = sampleNK(sample)
                alldata2.append(pstat2)
                _, pstat15, _ = sampleNK(sampleType15[i])
                alldata15.append(pstat15)

    for i, _ in enumerate(sampleType2):  # get pstat data and put it into list form
        dat_array2, dat_array15 = alldata2[i], alldata15[i]
        stat_array2, stat_array15 = dat_array2[[statcol]], dat_array15[[statcol]]
        stat_array2, stat_array15 = stat_array2.to_numpy(), stat_array15.to_numpy()
        stat_array2, stat_array15 = stat_array2.clip(min=1), stat_array15.clip(min=1)  # remove small percentage of negative pstat values

        while np.amax(stat_array2) > 100000:
            stat_array2 = np.reshape(stat_array2[stat_array2 != np.amax(stat_array2)], (-1, 1))
        while np.amax(stat_array15) > 100000:
            stat_array15 = np.reshape(stat_array15[stat_array15 != np.amax(stat_array15)], (-1, 1))

        for kk, pSTATArray in enumerate(np.array([stat_array2, stat_array15])):
            if pSTATArray.size == 0:
                distDF = distDF.append(pd.DataFrame.from_dict({"Dose": dosemat[i], "Ligand": ILs[kk], "pSTAT": [0]}))
            else:
                distDF = distDF.append(pd.DataFrame.from_dict({"Dose": np.tile(dosemat[i], (len(pSTATArray))), "Ligand": np.tile(ILs[kk], (len(pSTATArray))), "pSTAT": pSTATArray.flatten()}))

    sns.violinplot(x="Dose", y="pSTAT", hue="Ligand", data=distDF, split=True, palette={"IL-2": "darkorchid", "IL-15": "goldenrod"}, ax=ax)
    ax.set(xlabel="Ligand nM", ylabel="pSTAT Signal", ylim=(0, 100000), title=title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)

    return distDF


def calculate_moments(df, cell_names, receptors):
    """ Calculates mean, variance, and skew for each replicate. """
    df_stats = pd.DataFrame(columns=["Cell Type", "Receptor", "Mean", "Variance", "Skew", "Date", "Plate"])
    for _, cell in enumerate(cell_names):
        for _, receptor in enumerate(receptors):
            for _, date in enumerate(["4-23", "4-26", "5-16"]):
                for _, plate in enumerate(["1", "2"]):
                    df_subset = df.loc[(df["Cell Type"] == cell) & (df["Receptor"] == receptor) & (df["Date"] == date) & (df["Plate"] == plate)]["Count"]
                    mean_ = np.log10(df_subset.mean())
                    var_ = np.log10(df_subset.var())
                    skew_ = np.log10(df_subset.skew())
                    df_new = pd.DataFrame(columns=["Cell Type", "Receptor", "Mean", "Variance", "Skew", "Date", "Plate"])
                    df_new.loc[0] = [cell, receptor, mean_, var_, skew_, date, plate]
                    df_stats = df_stats.append(df_new)

    return df_stats


def celltype_pointplot(ax, df, moment):
    """ Plots a given distribution moment with SD among replicates for all cell types and receptors. """
    sns.pointplot(x="Cell Type", y=moment, hue="Receptor", data=df, ci='sd', join=False, dodge=True, ax=ax, estimator=sp.stats.gmean)
    ax.set_ylabel("log(" + moment + ")")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, rotation_mode="anchor", ha="right", position=(0, 0.02), fontsize=7.5)


def run_regression():
    """ Imports bead data and runs regression to get least squares parameters for conversion of signal to receptor count. """
    sampleD, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads", "D")
    sampleE, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "E")
    sampleF, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "F")
    sampleI, _ = importF(path_here + "/data/flow/2019-05-16 Receptor Quant - Beads/", "F")

    recQuant1 = np.array([0., 4407, 59840, 179953, 625180])  # CD25, CD122
    recQuant2 = np.array([0., 7311, 44263, 161876, 269561])  # CD132
    recQuant3 = np.array([4407, 59840, 179953, 625180, 0.0])  # CD127

    _, lsq_cd25 = bead_regression(sampleD, channels['D'], recQuant1)
    _, lsq_cd122 = bead_regression(sampleE, channels['E'], recQuant1, 2, True)
    _, lsq_cd132 = bead_regression(sampleF, channels['F'], recQuant2)
    _, lsq_cd127 = bead_regression(sampleI, channels["I"], recQuant3)

    return lsq_cd25, lsq_cd122, lsq_cd132, lsq_cd127
