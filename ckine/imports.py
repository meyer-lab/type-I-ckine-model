"""File that deals with everything about importing and sampling."""
import os
from os.path import join
import numpy as np
import scipy as sp
import pandas as pds

path_here = os.path.dirname(os.path.dirname(__file__))


def import_Rexpr():
    """ Loads CSV file containing Rexpr levels from Visterra data. """
    data = pds.read_csv(join(path_here, "ckine/data/final_receptor_levels.csv"))  # Every row in the data represents a specific cell
    df = data.groupby(["Cell Type", "Receptor"]).agg(sp.stats.gmean)  # Get the mean receptor count for each cell across trials in a new dataframe.
    cell_names, receptor_names = df.index.unique().levels  # gc_idx=0|IL15Ra_idx=1|IL2Ra_idx=2|IL2Rb_idx=3
    cell_names = cell_names[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2]]  # Reorder to match pstat import order
    receptor_names = receptor_names[[2, 3, 0, 1, 4]]  # Reorder so that IL2Ra_idx=0|IL2Rb_idx=1|gc_idx=2|IL15Ra_idx=3|IL7Ra_idx=4
    numpy_data = pds.Series(df["Count"]).values.reshape(cell_names.size, receptor_names.size)  # Rows are in the order of cell_names. Receptor Type is on the order of receptor_names
    numpy_data = numpy_data[:, [2, 3, 0, 1, 4]]  # Rearrange numpy_data to place IL2Ra first, then IL2Rb, then gc, then IL15Ra in this order
    numpy_data = numpy_data[[4, 0, 5, 1, 9, 7, 3, 8, 6, 2], :]  # Reorder to match cells
    return df, numpy_data, cell_names


def import_muteins():
    """ Import mutein data and return a normalized DataFrame and tensor. """
    data = pds.read_csv(join(path_here, "ckine/data/2019-07-mutein-timecourse.csv"))

    # Concentrations are across columns, so melt
    data = pds.melt(data, id_vars=["Cells", "Ligand", "Time", "Replicate"], var_name="Concentration", value_name="RFU")

    # Make the concentrations numeric
    data["Concentration"] = pds.to_numeric(data["Concentration"])

    # Subtract off the minimum signal
    data["RFU"] = data["RFU"] - data.groupby(["Cells", "Replicate"])["RFU"].transform("min")

    # Each replicate varies in its sensitivity, so correct for that
    replAvg = data[data["Time"] > 0.6].groupby(["Replicate"]).mean()
    ratio = replAvg.loc[2, "RFU"] / replAvg.loc[1, "RFU"]
    data.loc[data["Replicate"] == 1, "RFU"] *= ratio

    # Take the average across replicates
    dataMean = data.groupby(["Cells", "Ligand", "Time", "Concentration"]).mean()
    dataMean.drop("Replicate", axis=1, inplace=True)

    # Make a data tensor. Dimensions correspond to groupby above
    dataTensor = np.reshape(dataMean["RFU"].values, (8, 4, 4, 12))

    return dataMean, dataTensor


def import_pstat(combine_samples=True):
    """ Loads CSV file containing pSTAT5 levels from Visterra data. Incorporates only Replicate 1 since data missing in Replicate 2. """
    path = os.path.dirname(os.path.dirname(__file__))
    data = np.array(pds.read_csv(join(path, "ckine/data/pSTAT_data.csv"), encoding="latin1"))
    ckineConc = data[4, 2:14]
    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    # 4 time points, 10 cell types, 12 concentrations, 2 replicates
    IL2_data = np.zeros((40, 12))
    IL2_data2 = IL2_data.copy()
    IL15_data = IL2_data.copy()
    IL15_data2 = IL2_data.copy()
    cell_names = list()
    for i in range(10):
        cell_names.append(data[12 * i + 3, 1])
        # Subtract the zero treatment plates before assigning to returned arrays
        if i <= 4:
            zero_treatment = data[12 * (i + 1), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        else:
            zero_treatment = data[8 + (12 * i), 13]
            zero_treatment2 = data[8 + (12 * i), 30]
        # order of increasing time by cell type
        IL2_data[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 2:14].astype(np.float) - zero_treatment, 0)
        IL2_data2[4 * i: 4 * (i + 1), :] = np.flip(data[6 + (12 * i): 10 + (12 * i), 19:31].astype(np.float) - zero_treatment2, 0)
        IL15_data[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 2:14].astype(np.float) - zero_treatment, 0)
        IL15_data2[4 * i: 4 * (i + 1), :] = np.flip(data[10 + (12 * i): 14 + (12 * i), 19:31].astype(np.float) - zero_treatment2, 0)

    if combine_samples is False:
        return ckineConc, cell_names, IL2_data, IL2_data2, IL15_data, IL15_data2

    for i in range(IL2_data.shape[0]):
        for j in range(IL2_data.shape[1]):
            # take average of both replicates if specific entry isn't nan
            IL2_data[i, j] = np.nanmean(np.array([IL2_data[i, j], IL2_data2[i, j]]))
            IL15_data[i, j] = np.nanmean(np.array([IL15_data[i, j], IL15_data2[i, j]]))

    dataMean = pds.DataFrame(
        {
            "Cells": np.tile(np.repeat(cell_names, 48), 2),
            "Ligand": np.concatenate((np.tile(np.array("IL2"), 480), np.tile(np.array("IL15"), 480))),
            "Time": np.tile(np.repeat(tps, 12), 20),
            "Concentration": np.tile(ckineConc, 80),
            "RFU": np.concatenate((IL2_data.reshape(480), IL15_data.reshape(480))),
        }
    )

    return ckineConc, cell_names, IL2_data, IL15_data, dataMean


# Receptor Quant - Beads (4/23 & 4/26)


channels = {}
channels["A"] = ["VL1-H", "BL5-H", "RL1-H", "RL1-H", "RL1-H", "Width"]
channels["C"] = ["VL4-H", "VL6-H", "BL1-H", "BL3-H"]
channels["D"] = ["VL1-H", "VL1-H", "VL1-H", "VL1-H", "VL1-H"]
channels["E"] = ["VL6-H", "BL3-H", "BL5-H", "BL5-H", "BL5-H", "BL5-H", "BL5-H"]
channels["F"] = channels["G"] = channels["H"] = ["RL1-H", "RL1-H", "RL1-H", "RL1-H", "RL1-H"]
channels["I"] = ["BL1-H", "BL1-H", "BL1-H", "BL1-H", "BL1-H"]

receptors = {}
receptors["A"] = ["CD25", "CD122", "CD132", "IL15(1)", "IL15(2)", " "]
receptors["C"] = ["CD3", "CD4", "CD127", "CD45RA"]
receptors["D"] = ["CD25", "CD25", "CD25", "CD25", "CD25"]
receptors["E"] = ["CD8", "CD56", "CD122", "CD122", "CD122", "CD122", "CD122"]
receptors["F"] = ["CD132", "CD132", "CD132", "CD132", "CD132"]
receptors["G"] = ["IL15(1)", "IL15(1)", "IL15(1)", "IL15(1)", "IL15(1)"]
receptors["H"] = ["IL15(2)", "IL15(2)", "IL15(2)", "IL15(2)", "IL15(2)"]
receptors["I"] = ["CD127", "CD127", "CD127", "CD127", "CD127"]


def importMoments():
    path = os.path.dirname(os.path.dirname(__file__))
    momentDF = pds.read_csv(join(path, "ckine/data/pSTATMomentData.csv"), encoding="latin1")
    return momentDF


def import_pstat_all():
    """ Loads CSV file containing all WT and Mutein pSTAT responses and moments"""
    WTbivDF = pds.read_csv(join(path_here, "ckine/data/WTDimericMutSingleCellData.csv"), encoding="latin1")
    monDF = pds.read_csv(join(path_here, "ckine/data/MonomericMutSingleCellData.csv"), encoding="latin1")
    respDF = pds.concat([WTbivDF, monDF])

    respDF.loc[(respDF.Bivalent == 0), "Ligand"] = (respDF.loc[(respDF.Bivalent == 0)].Ligand + " (Mono)").values
    respDF.loc[(respDF.Bivalent == 1), "Ligand"] = (respDF.loc[(respDF.Bivalent == 1)].Ligand + " (Biv)").values

    return respDF


def importSigma(cellType):
    """ Loads and formats the receptor covariance matrix for variance propagation """
    sigma = np.zeros((3, 3))

    momentDF = pds.read_csv(join(path_here, "ckine/data/receptor_moments.csv"), encoding="latin1")
    covDF = pds.read_csv(join(path_here, "ckine/data/receptor_covariances.csv"), encoding="latin1")

    momentDF = momentDF.loc[:, ["Cell Type", "Receptor", "Variance"]]
    momentDF = momentDF.groupby(["Cell Type", "Receptor"]).mean().reset_index()
    print(momentDF)
    momentDF = momentDF.loc[(momentDF["Cell Type"] == cellType)]

    for i in range(0, 3):
        sigma[i, i] = momentDF["Variance"].values[i]

    covDF = covDF.loc[:, ["Cell Type", "CD25:Receptor", "Covariance"]]
    covDF = covDF.groupby(["Cell Type", "CD25:Receptor"]).mean().reset_index()
    covDF = covDF.loc[(covDF["Cell Type"] == cellType)]

    for i in range(1, 3):
        sigma[0, i] = sigma[i, 0] = covDF["Covariance"].values[i - 1]

    return sigma
