"""
This creates Figure 7.
"""
import string
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
from .figureCommon import subplotLabel, getSetup
from .figureB6 import calc_dose_response_mutein, plot_dose_response
from ..model import receptor_expression
from ..imports import import_muteins, import_Rexpr, import_samples_2_15

dataMean, _ = import_muteins()
dataMean.reset_index(inplace=True)
data, _, _ = import_Rexpr()
data.reset_index(inplace=True)
unkVec, _ = import_samples_2_15(N=100)


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 4))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    muteinC = dataMean.Concentration.unique()
    dataMean["Concentration"] = np.log10(dataMean["Concentration"])  # logscale for plotting

    axis = 0
    second_group_ligands = ['IL2-088', 'IL2-097']

    pred_data = np.zeros((len(muteinC), len(tps), unkVec.shape[1]))  # make empty array for predicted data at each concentration, tp, and parameter sample

    # loop for each mutein and cell type
    for i, ligand_name in enumerate(second_group_ligands):
        for j, cell_name in enumerate(dataMean.Cells.unique()):

            # organize experimental pstat data
            exp_data = np.zeros((12, 4))
            for k, conc in enumerate(dataMean.Concentration.unique()):
                exp_data[k, :] = np.array(dataMean.loc[(dataMean["Cells"] == cell_name) & (dataMean["Ligand"] == ligand_name) & (dataMean["Concentration"] == conc), "RFU"])

            # calculate predicted dose response
            IL2Ra = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\alpha$'), "Count"].item()
            IL2Rb = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\beta$'), "Count"].item()
            gc = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == '$\\gamma_{c}$'), "Count"].item()
            for l in range(unkVec.shape[1]):
                cell_receptors = receptor_expression(np.array([IL2Ra, IL2Rb, gc]).astype(np.float), unkVec[17, l], unkVec[20, l], unkVec[19, l], unkVec[21, l])
                pred_data[:, :, l] = calc_dose_response_mutein(unkVec[:, l], [1., 1., 5.], tps, muteinC, cell_receptors, exp_data)

            # plot experimental and predicted date with a legend for the last subplot
            axis = i * 8 + j
            if axis == 15:
                sns.scatterplot(x="Concentration", y="RFU", hue="Time", data=dataMean.loc[(dataMean["Cells"] == cell_name) &
                                                                                          (dataMean["Ligand"] == ligand_name)], ax=ax[axis], s=10, palette=cm.rainbow, legend='full')
                ax[axis].legend(loc='lower right', title="time (hours)")
            else:
                sns.scatterplot(x="Concentration", y="RFU", hue="Time", data=dataMean.loc[(dataMean["Cells"] == cell_name)
                                                                                          & (dataMean["Ligand"] == ligand_name)], ax=ax[axis], s=10, palette=cm.rainbow, legend=False)
            plot_dose_response(ax[axis], pred_data, tps, muteinC)
            ax[axis].set(xlabel=("[" + ligand_name + "] (log$_{10}$[nM])"), ylabel="Activity", title=cell_name)

    return f
