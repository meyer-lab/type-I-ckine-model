import os
from os.path import dirname, join
from pathlib import Path
import seaborn as sns
import scipy as sp
import numpy as np
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from ..imports import channels
from ..flow import bead_regression
from ..FCimports import combineWells, compMatrix, applyMatrix, import_gates, apply_gates, importF
from FlowCytometryTools import FCMeasurement, ThresholdGate, PolyGate, QuadGate
from matplotlib import pyplot as plt
from matplotlib import cm

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 8), (3, 4), multz={8: 1})
    subplotLabel(ax)

    Tcell_pathname = path_here + "/data/flow/2019-11-08 monomer IL-2 Fc signaling/CD4 T cells - IL2-060 mono, IL2-060 dimeric"
    NK_CD8_pathname = path_here + "/data/flow/2019-11-08 monomer IL-2 Fc signaling/NK CD8 T cells - IL2-060 mono, IL2-060 dimeric"

    Tcell_sample, _ = importF2(Tcell_pathname, "A")
    NK_CD8_sample, _ = importF2(NK_CD8_pathname, "A")

    Tcell_sample = combineWells(Tcell_sample)
    NK_CD8_sample = combineWells(NK_CD8_sample)

    Tcell_sample = applyMatrix(Tcell_sample, compMatrix('2019-11-08', '1', 'A'))
    NK_CD8_sample = applyMatrix(NK_CD8_sample, compMatrix('2019-11-08', '1', 'B'))

    Tcell_sample = Tcell_sample.transform("tlog", channels=['VL1-H', 'VL4-H', 'BL1-H', 'BL3-H'])  # Tlog transformations
    NK_CD8_sample = NK_CD8_sample.transform("tlog", channels=['RL1-H', 'VL4-H', 'BL1-H', 'BL2-H'])  # Tlog transformations

    cd4_gate = ThresholdGate(6500.0, ['VL4-H'], region='above') & ThresholdGate(8000.0, ['VL4-H'], region='below')
    ax[0] = Tcell_sample.plot(['VL4-H'], gates=cd4_gate, ax=ax[0])  # CD4
    plt.title("Singlet Lymphocytes")
    #ax.set(xlabel= "CD4", ylabel="Events")
    plt.grid()

    sampleCD4 = Tcell_sample.gate(cd4_gate)
    Treg_gate = PolyGate([(4.2e3, 7.2e3), (6.5e03, 7.2e03), (6.5e03, 5.3e03), (4.9e03, 5.3e03), (4.2e03, 5.7e03)], ('VL1-H', 'BL1-H'), region='in', name='treg')
    Thelp_gate = PolyGate([(1.8e03, 3.1e03), (1.8e03, 4.9e03), (6.0e03, 4.9e03), (6.0e03, 3.1e03)], ('VL1-H', 'BL1-H'), region='in', name='thelper')

    _ = sampleCD4.plot(['VL1-H', 'BL1-H'], gates=[Treg_gate, Thelp_gate], gate_colors=['red', 'red'], cmap=cm.jet, ax=ax[1])  # CD4
    plt.title("CD4+ Cells")
    plt.xlabel("CD25")
    plt.ylabel("FOXP3")
    plt.grid()

    #CD8+ Cells
    CD3CD8gate = PolyGate([(7.5e3, 8.4e3), (4.7e3, 8.4e3), (4.7e03, 6.5e03), (7.5e03, 6.5e03)], ('VL4-H', 'RL1-H'), region='in', name='treg')
    _ = NK_CD8_sample.plot(['VL4-H', 'RL1-H'], gates=CD3CD8gate, gate_colors='red', cmap=cm.jet, ax=ax[2])  # CD3, CD8
    plt.title("Singlet Lymphocytes")
    plt.xlabel("CD3")
    plt.ylabel("CD8")
    plt.grid()

    # NK Cells
    NKgate = PolyGate([(4.8e3, 5.1e3), (5.9e3, 5.1e3), (5.9e03, 6.1e03), (4.8e03, 6.1e03)], ('VL4-H', 'BL1-H'), region='in', name='treg')
    CD56brightgate = PolyGate([(4.8e3, 6.3e3), (5.9e3, 6.3e3), (5.9e03, 7.3e03), (4.8e03, 7.3e03)], ('VL4-H', 'BL1-H'), region='in', name='treg')
    _ = NK_CD8_sample.plot(['VL4-H', 'BL1-H'], gates=[NKgate, CD56brightgate], gate_colors=['red', 'red'], cmap=cm.jet, ax=ax[3])  # CD3, CD56
    plt.title("Singlet Lymphocytes")
    plt.xlabel("CD3")
    plt.ylabel("CD56")
    plt.grid()

    # Gating for live cells
    sample1A, unstained, isotype = importF("4-23", "1", "A", 1, "IL2R", None)
    sample2B, unstained, isotype = importF("4-23", "1", "B", 2, "IL2R", None)
    sample3C, unstained, isotype = importF("4-23", "1", "C", 3, "IL2R", None)
    panel1 = sample1A.transform("tlog", channels=['VL6-H', 'VL4-H', 'BL1-H', 'VL1-H', 'BL3-H'])
    panel2 = sample2B.transform("tlog", channels=['VL4-H', 'BL3-H'])
    panel3 = sample3C.transform("tlog", channels=['VL6-H', 'VL4-H', 'BL3-H'])

    cd3cd4_gate = PolyGate([(5.0e03, 7.3e03), (5.3e03, 5.6e03), (8.0e03, 5.6e03), (8.0e03, 7.3e03)], ('VL4-H', 'VL6-H'), region='in', name='cd3cd4')
    _ = panel1.plot(['VL4-H', 'VL6-H'], gates=cd3cd4_gate, gate_colors=['red'], cmap=cm.jet, ax=ax[4])  # CD3, CD4
    plt.title("Singlet Lymphocytes")
    plt.xlabel("CD3")
    plt.ylabel("CD4")
    plt.grid()

    samplecd3cd4 = panel1.gate(cd3cd4_gate)
    thelp_gate = PolyGate([(0.2e03, 6.8e03), (0.2e03, 4.4e03), (3.7e03, 4.4e03), (5.7e03, 5.9e03), (5.7e03, 6.8e03)], ('VL1-H', 'BL1-H'), region='in', name='thelp')
    treg_gate = PolyGate([(3.8e03, 4.4e03), (3.8e03, 3.0e03), (6.5e03, 2.9e03), (6.5e03, 5.0e03), (5.7e03, 5.8e03)], ('VL1-H', 'BL1-H'), region='in', name='treg')
    _ = samplecd3cd4.plot(['VL1-H', 'BL1-H'], gates=[thelp_gate, treg_gate], gate_colors=['red', 'red'], cmap=cm.jet, ax=ax[5])  # CD3, CD4
    plt.title("CD3+CD4+ cells")
    plt.xlabel("CD25")
    plt.ylabel("CD127")
    plt.grid()

    nk_gate = PolyGate([(3.3e3, 5.4e3), (5.3e3, 5.4e3), (5.3e3, 7.3e3), (3.3e3, 7.3e3)], ('VL4-H', 'BL3-H'), region='in', name='nk')
    nkt_gate = PolyGate([(5.6e3, 5.1e3), (7.6e3, 5.1e3), (7.6e3, 7.1e3), (5.6e3, 7.1e3)], ('VL4-H', 'BL3-H'), region='in', name='nkt')
    _ = panel2.plot(['VL4-H', 'BL3-H'], gates=[nk_gate, nkt_gate], gate_colors=['red', 'red'], cmap=cm.jet, ax=ax[6])  # CD56 vs. CD3
    samplenk = panel2.gate(nk_gate)
    samplenkt = panel2.gate(nkt_gate)
    plt.title("Singlet Lymphocytes")
    plt.xlabel("CD3")
    plt.ylabel("CD56")
    plt.grid()

    cd8_gate = PolyGate([(4.2e3, 5.7e3), (8.1e3, 5.7e3), (8.1e3, 8.0e3), (4.2e3, 8.0e3)], ('VL4-H', 'VL6-H'), region='in', name='cd8')
    _ = panel3.plot(['VL4-H', 'VL6-H'], gates=cd8_gate, gate_colors=['red'], cmap=cm.jet, ax=ax[7])  # CD8 vs. CD3
    plt.title("Singlet Lymphocytes")
    plt.xlabel("CD3")
    plt.ylabel("CD8")

    for i, axs in enumerate(ax):
        if i == 0:
            print(" ")
            # weird error replace later, axs is not correct object type
            # axs.set(xlabel='CD4',ylabel='Events')
        elif i == 1:
            axs.set_title('T Cell Gating')
            axs.set(xlabel='CD25', ylabel='FOXP3')
        elif i == 2:
            axs.set_title('CD8+ Cells Gating')
            axs.set(xlabel='CD3', ylabel='CD8')
        elif i == 3:
            axs.set_title('NK Cells Gating')
            axs.set(xlabel='CD3', ylabel='CD56')
        elif i == 4:
            axs.set_title('CD3+CD4+ Gating')
            axs.set(xlabel='CD3', ylabel='CD4')
        elif i == 5:
            axs.set_title('T reg and T Helper Gating')
            axs.set(xlabel='CD25', ylabel='CD127')
        elif i == 6:
            axs.set_title('NK and NKT Gating')
            axs.set(xlabel='CD3', ylabel='CD56')
        elif i == 7:
            axs.set_title('CD3+CD8+ Gating')
            axs.set(xlabel='CD3', ylabel='CD8')
        if i != 0:
            axs.grid()

    receptorPlot(ax[8], ax[9], ax[10])

    return f


def importF2(pathname, WellRow):
    """
    Import FCS files. Variable input: name of path name to file. Output is a list of Data File Names in FCT Format
    Title/file names are returned in the array file --> later referenced in other functions as title/titles input argument
    """
    # Declare arrays and int
    file = []
    sample = []
    z = 0
    # Read in user input for file path and assign to array file
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    for path in pathlist:
        wellID = path.name.split("_")[1]
        if wellID[0] == WellRow:
            file.append(str(path))
    file.sort()
    assert file != []
    # Go through each file and assign the file contents to entry in the array sample
    for entry in file:
        sample.append(FCMeasurement(ID="Test Sample" + str(z), datafile=entry))
        z += 1
    # Returns the array sample which contains data of each file in folder (one file per entry in array)
    return sample, file


def receptorPlot(ax1, ax2, ax3):

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

    # plots log10 of mean on
    celltype_pointplot(ax1, df_stats, "Mean")

    receptor_levels = df_rec
    cell_types = ['T-reg']

    for index, cell_type in enumerate(cell_types):

        alphaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD25')]
        betaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD122')]
        gammaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD132')]

        alphaCounts = alphaLevels['Count'].reset_index(drop=True)
        betaCounts = betaLevels['Count'].reset_index(drop=True)
        d = {'alpha': alphaCounts, 'beta': betaCounts}
        recepCounts = pd.DataFrame(data=d)
        recepCounts = recepCounts.dropna()
        recepCounts = recepCounts[(recepCounts[['alpha', 'beta']] != 0).all(axis=1)]

        hex1 = ax2
        hex1.hexbin(recepCounts['alpha'], recepCounts['beta'], xscale='log', yscale='log', mincnt=1, cmap='viridis')
        hex1.set_xlabel('CD25 (IL2RA)')
        hex1.set_ylabel('CD122(IL2RB)')
        hex1.set_title(cell_type + ' Alpha-Beta correlation')

        alphaCounts = alphaLevels['Count'].reset_index(drop=True)
        gammaCounts = gammaLevels['Count'].reset_index(drop=True)
        d2 = {'alpha': alphaCounts, 'gamma': gammaCounts}
        recepCounts2 = pd.DataFrame(data=d2)
        recepCounts2 = recepCounts2.dropna()
        recepCounts2 = recepCounts2[(recepCounts2[['alpha', 'gamma']] != 0).all(axis=1)]

        hex2 = ax3
        hex2.hexbin(recepCounts2['alpha'], recepCounts2['gamma'], xscale='log', yscale='log', mincnt=1, cmap='viridis')
        hex2.set_xlabel('CD25 (IL2RA)')
        hex2.set_ylabel('CD132 (IL2RG)')
        hex2.set_title(cell_type + ' Alpha-Gamma correlation')

    return


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
    sampleD, _ = importF2(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads", "D")
    sampleE, _ = importF2(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "E")
    sampleF, _ = importF2(path_here + "/data/flow/2019-04-23 Receptor Quant - Beads/", "F")
    sampleI, _ = importF2(path_here + "/data/flow/2019-05-16 Receptor Quant - Beads/", "F")

    recQuant1 = np.array([0., 4407, 59840, 179953, 625180])  # CD25, CD122
    recQuant2 = np.array([0., 7311, 44263, 161876, 269561])  # CD132
    recQuant3 = np.array([4407, 59840, 179953, 625180, 0.0])  # CD127

    _, lsq_cd25 = bead_regression(sampleD, channels['D'], recQuant1)
    _, lsq_cd122 = bead_regression(sampleE, channels['E'], recQuant1, 2, True)
    _, lsq_cd132 = bead_regression(sampleF, channels['F'], recQuant2)
    _, lsq_cd127 = bead_regression(sampleI, channels["I"], recQuant3)

    return lsq_cd25, lsq_cd122, lsq_cd132, lsq_cd127
