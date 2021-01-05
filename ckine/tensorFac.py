"""
This file includes various methods for flow cytometry analysis.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tensorly as tl
import matplotlib.cm as cm
from tensorly.decomposition import non_negative_parafac


def makeTensor(sigDF):
    """Makes tensor of data with dimensions mutein x time point x concentration x cell type"""
    ligands = sigDF.Ligand.unique()
    tps = sigDF.Time.unique()
    concs = sigDF.Dose.unique()
    cellTypes = sigDF.Cell.unique()
    tensor = np.empty((len(ligands), len(tps), len(concs), len(cellTypes)))
    tensor[:] = np.nan
    for i, lig in enumerate(ligands):
        for j, tp in enumerate(tps):
            for k, conc in enumerate(concs):
                for ii, cell in enumerate(cellTypes):
                    entry = sigDF.loc[(sigDF.Ligand == lig) & (sigDF.Time == tp) & (sigDF.Dose == conc) & (sigDF.Cell == cell)].Mean.values
                    if len(entry) >= 1:
                        tensor[i, j, k, ii] = np.mean(entry)
    # Normalize
    for i, _ in enumerate(cellTypes):
        tensor[:, :, :, i][~np.isnan(tensor[:, :, :, i])] /= np.nanmax(tensor[:, :, :, i])

    return tensor


def getMaskTens(tensor):
    """Returns binary mask of tensor marking nan locations, and a tensor copy with NaNs as zeros"""
    masktensor = tensor.copy()
    tensorNoNan = tensor.copy()

    masktensor[np.isnan(masktensor)] = 0
    masktensor[np.invert(np.isnan(masktensor))] = 1
    tensorNoNan = np.nan_to_num(tensor, nan=0)

    return tensorNoNan, masktensor


def factorTensor(Tensor, numComps):
    """Takes Tensor, and mask and returns tensor factorized form"""
    nnTensor, maskTens = getMaskTens(Tensor)
    return non_negative_parafac(nnTensor, rank=numComps, mask=maskTens, n_iter_max=1000, init='svd', random_state=0)


def R2Xplot(ax, tensor, compNum):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(compNum)
    nnTens, maskTens = getMaskTens(tensor)
    for i in range(1, compNum + 1):
        tFac = non_negative_parafac(nnTens, rank=i, mask=maskTens, n_iter_max=1000, init='svd', random_state=0)
        varHold[i - 1] = calcR2X(tensor, tFac)

    ax.scatter(np.arange(1, compNum + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 1), xticks=np.arange(0, compNum + 1))


def calcR2X(tensorIn, tensorFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    return 1.0 - (tErr) / (np.nanvar(tensorIn))


def plot_tFac_Ligs(ax, tFac, respDF):
    """Plots tensor factorization of cells"""
    ligands = respDF.Ligand.unique()
    mutFacs = tFac[1][0]
    bivCol = []
    for i, ligand in enumerate(ligands):
        if ligand[-5::] == "(Biv)":
            bivCol.append("Bivalent")
            ligands[i] = ligand[0:-5]
        elif ligand[-6::] == "(Mono)":
            bivCol.append("Monovalent")
            ligands[i] = ligand[0:-6]

    mutDF = pd.DataFrame({"Ligand": ligands, "Valency": bivCol, "Component 1": mutFacs[:, 0], "Component 2": mutFacs[:, 1], "Component 3": mutFacs[:, 2]})
    sns.scatterplot(x="Component 1", y="Component 2", hue=mutDF.Ligand.tolist(), style=mutDF.Valency.tolist(), data=mutDF, ax=ax[0], s=50)
    sns.scatterplot(x="Component 1", y="Component 3", hue=mutDF.Ligand.tolist(), style=mutDF.Valency.tolist(), data=mutDF, ax=ax[1], legend=False, s=50)

    ax[0].set(title="Ligands", xlim=(0, 1), ylim=(0, 1))
    ax[1].set(title="Ligands", xlim=(0, 1), ylim=(0, 1))
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()
    return handles, labels


def plot_tFac_Time(ax, tFac, respDF):
    """Plots tensor factorization of tps"""
    tps = respDF.Time.unique()
    timeFacs = tFac[1][1]

    markersTimes = ["^", "*", "D"]
    for i in range(0, timeFacs.shape[1]):
        ax.plot(tps, timeFacs[:, i], marker=markersTimes[i], label="Component " + str(i + 1))

    ax.legend()
    ax.set(title="Time", xlabel="Time (hrs)", xlim=(0.5, 4), ylabel="Component", ylim=(0, 1))


def plot_tFac_Conc(ax, tFac, respDF):
    """Plots tensor factorization of Conc"""
    concs = respDF.Dose.unique()
    concFacs = tFac[1][2]

    markersConcs = ["^", "*", "D"]
    for i in range(0, concFacs.shape[1]):
        ax.plot(concs, concFacs[:, i], marker=markersConcs[i], label="Component " + str(i + 1))

    ax.legend()
    ax.set(title="Concentration", xlabel="Concentration (nM)", xlim=(concs[-1], concs[0]), ylabel="Component", ylim=(0, 1), xscale='log')


def plot_tFac_Cells(ax, tFac, respDF):
    """Plots tensor factorization of cells"""
    cells = respDF.Cell.unique()
    cellFacs = tFac[1][3]
    print(cellFacs)

    markersCells = ["P", "*", "D", "s"]
    colors = cm.rainbow(np.linspace(0, 1, len(cells)))
    for i, cell in enumerate(cells):
        ax[0].scatter(cellFacs[i, 0], cellFacs[i, 1], marker=markersCells[i], color=colors[i], label=cell)
        ax[1].scatter(cellFacs[i, 0], cellFacs[i, 2], marker=markersCells[i], color=colors[i], label=cell)

    ax[0].legend(loc='best', fontsize=10)
    ax[0].set(title="Cells", xlabel="Component 1", xlim=(0, 1), ylabel="Component 2", ylim=(0, 1))
    ax[1].set(title="Cells", xlabel="Component 1", xlim=(0, 1), ylabel="Component 3", ylim=(0, 1))
