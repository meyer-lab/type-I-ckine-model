import os
from os.path import dirname, join
from pathlib import Path
import numpy as np
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from .figureC6 import getReceptors
from ..imports import channels
from ..FCimports import combineWells, compMatrix, applyMatrix, import_gates, apply_gates, importF
from FlowCytometryTools import FCMeasurement, ThresholdGate, PolyGate, QuadGate
from matplotlib import pyplot as plt
from matplotlib import cm

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 8), (3, 4))
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

    print("check")

    for i, axs in enumerate(ax):
        if i == 0:
            print(" ")
            # weird error replace later, axs is not correct object type
            # axs.set(xlabel='CD4',ylabel='Events')
        elif i == 1:
            axs.set_title('CD4+ Cells')
            axs.set(xlabel='CD25', ylabel='FOXP3')
        elif i == 2:
            axs.set_title('Singlet Lymphocytes')
            axs.set(xlabel='CD3', ylabel='CD8')
        elif i == 3:
            axs.set_title('Singlet Lymphocytes')
            axs.set(xlabel='CD3', ylabel='CD56')
        elif i == 4:
            axs.set_title('Singlet Lymphocytes')
            axs.set(xlabel='CD3', ylabel='CD4')
        elif i == 5:
            axs.set_title('CD3+CD4+ cells')
            axs.set(xlabel='CD25', ylabel='CD127')
        elif i == 6:
            axs.set_title('Singlet Lymphocytes')
            axs.set(xlabel='CD23', ylabel='CD56')
        elif i == 7:
            axs.set_title('Singlet Lymphocytes')
            axs.set(xlabel='CD3', ylabel='CD8')
        # axs.grid()

    # receptorPlot()

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


def receptorPlot():

    receptor_levels = getReceptors()
    cell_types = ['T-reg', 'T-helper', 'NK', 'CD8+']

    alphaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD25')]
    betaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD122')]
    gammaLevels = receptor_levels.loc[(receptor_levels['Cell Type'] == cell_type) & (receptor_levels['Receptor'] == 'CD132')]

    print("test")

    return
