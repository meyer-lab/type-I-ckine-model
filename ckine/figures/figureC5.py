"""
This creates Figure 5 for Single Cell data analysis. Plots of mean, variance, and skew by cell type.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from .figureCommon import subplotLabel, getSetup
from ..imports import channels
from ..flow import importF, bead_regression
from ..FCimports import import_gates, apply_gates

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (3, 1))

    subplotLabel(ax)

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
    celltype_pointplot(ax[0], df_stats, "Mean")
    celltype_pointplot(ax[1], df_stats, "Variance")
    celltype_pointplot(ax[2], df_stats, "Skew")

    return f


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
