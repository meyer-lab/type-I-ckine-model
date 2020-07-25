"""
This creates Figure 6 for IL2Ra correlatoin data analysis.
"""

import os
import numpy as np
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..imports import channels
from ..flow import importF, bead_regression
from ..FCimports import import_gates, apply_gates

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 5), (2, 4))
    subplotLabel(ax)

    # Imports receptor levels from .csv created by figC5
    receptor_levels = getReceptors()
    cell_types = ['T-reg', 'T-helper', 'NK', 'CD8+']

    for index, cell_type in enumerate(cell_types):
        i = 2 * index

        alphaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD25')]
        betaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD122')]
        gammaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD132')]

        alphaCounts = alphaLevels['Count'].reset_index(drop=True)
        betaCounts = betaLevels['Count'].reset_index(drop=True)
        d = {'alpha': alphaCounts, 'beta': betaCounts}
        recepCounts = pd.DataFrame(data=d)
        recepCounts = recepCounts.dropna()
        recepCounts = recepCounts[(recepCounts[['alpha', 'beta']] != 0).all(axis=1)]

        hex1 = ax[i]
        hex1.hexbin(recepCounts['alpha'], recepCounts['beta'], xscale='log', yscale='log', mincnt=1, cmap='viridis')
        hex1.set_xlabel('CD25')
        hex1.set_ylabel('CD122')
        hex1.set_title(cell_type + ' Alpha-Beta correlation')

        alphaCounts = alphaLevels['Count'].reset_index(drop=True)
        gammaCounts = gammaLevels['Count'].reset_index(drop=True)
        d2 = {'alpha': alphaCounts, 'gamma': gammaCounts}
        recepCounts2 = pd.DataFrame(data=d2)
        recepCounts2 = recepCounts2.dropna()
        recepCounts2 = recepCounts2[(recepCounts2[['alpha', 'gamma']] != 0).all(axis=1)]

        hex2 = ax[i + 1]
        hex2.hexbin(recepCounts2['alpha'], recepCounts2['gamma'], xscale='log', yscale='log', mincnt=1, cmap='viridis')
        hex2.set_xlabel('CD25')
        hex2.set_ylabel('CD132')
        hex2.set_title(cell_type + ' Alpha-Gamma correlation')

    return f


def getReceptors():
    # import bead data and run regression to get equations
    lsq_cd25, lsq_cd122, lsq_cd132 = run_regression()

    # create dataframe with gated samples (all replicates)
    df_gates = import_gates()
    df_signal = apply_gates("4-23", "1", df_gates)
    df_signal = df_signal.append(apply_gates("4-23", "2", df_gates))
    df_signal = df_signal.append(apply_gates("4-26", "1", df_gates))
    df_signal = df_signal.append(apply_gates("4-26", "2", df_gates))

    # make new dataframe for receptor counts
    df_rec = pd.DataFrame(columns=["Cell Type", "Receptor", "Count", "Date", "Plate"])
    cell_names = ["T-reg", "T-helper", "NK", "CD8+"]
    receptors_ = ["CD25", "CD122", "CD132"]
    channels_ = ["VL1-H", "BL5-H", "RL1-H"]
    lsq_params = [lsq_cd25, lsq_cd122, lsq_cd132]
    dates = ["4-23", "4-26"]
    plates = ["1", "2"]

    # calculate receptor counts
    for _, cell in enumerate(cell_names):
        for j, receptor in enumerate(receptors_):
            for _, date in enumerate(dates):
                for _, plate in enumerate(plates):
                    data = df_signal.loc[(df_signal["Cell Type"] == cell) & (df_signal["Date"] == date) & (df_signal["Plate"] == plate)][channels_[j]]
                    rec_counts = np.zeros(len(data))
                    for k, signal in enumerate(data):
                        A, B, C, D = lsq_params[j]
                        rec_counts[k] = C * (((A - D) / (signal - D)) - 1)**(1 / B)
                    df_add = pd.DataFrame({"Cell Type": np.tile(cell, len(data)), "Receptor": np.tile(receptor, len(data)),
                                           "Count": rec_counts, "Date": np.tile(date, len(data)), "Plate": np.tile(plate, len(data))})
                    df_rec = df_rec.append(df_add)

    return df_rec


def run_regression():
    """ Imports bead data and runs regression to get least squares parameters for conversion of signal to receptor count. """
    sampleD, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads", "D")
    sampleE, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "E")
    sampleF, _ = importF(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "F")

    recQuant1 = np.array([0., 4407, 59840, 179953, 625180])  # CD25, CD122
    recQuant2 = np.array([0., 7311, 44263, 161876, 269561])  # CD132

    _, lsq_cd25 = bead_regression(sampleD, channels['D'], recQuant1)
    _, lsq_cd122 = bead_regression(sampleE, channels['E'], recQuant2, 2, True)
    _, lsq_cd132 = bead_regression(sampleF, channels['F'], recQuant1)

    return lsq_cd25, lsq_cd122, lsq_cd132
