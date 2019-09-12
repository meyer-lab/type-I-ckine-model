"""
This creates Figure 8. Comparison of Experimental verus Predicted Activity across IL2 and IL15 concentrations.
"""

import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup
from .figureB6 import organize_expr_pred, mutein_scaling
from ..imports import import_muteins, import_Rexpr, import_samples_2_15

dataMean, _ = import_muteins()
dataMean.reset_index(inplace=True)
data, _, _ = import_Rexpr()
data.reset_index(inplace=True)
unkVec_2_15, _ = import_samples_2_15(N=1)  # use one rate
muteinC = dataMean.Concentration.unique()
tps = np.array([0.5, 1., 2., 4.]) * 60.

mutaff = {
    "IL2-060": [1., 1., 5.],  # Wild-type, but dimer
    "IL2-062": [1., 100., 5.],  # Weaker b-g
    "IL2-088": [10., 1., 5.],  # Weaker CD25
    "IL2-097": [10., 100., 5.]  # Both
}


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 6), (4, 1))

    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])

    ligand_order = ['IL2-060', 'IL2-062', 'IL2-088', 'IL2-097']
    cell_order = ['NK', 'CD8+', 'T-reg', 'Naive Treg', 'Mem Treg', 'T-helper', 'Naive Th', 'Mem Th']

    df = pd.DataFrame(columns=['Cells', 'Ligand', 'Time Point', 'Concentration', 'Activity Type', 'Replicate', 'Activity'])

    # loop for each cell type and mutein
    for _, cell_name in enumerate(cell_order):

        IL2Ra = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\alpha$'), "Count"].item()
        IL2Rb = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\beta$'), "Count"].item()
        gc = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == '$\\gamma_{c}$'), "Count"].item()
        receptors = np.array([IL2Ra, IL2Rb, gc]).astype(np.float)

        for _, ligand_name in enumerate(ligand_order):

            # append dataframe with experimental and predicted activity
            df = organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec_2_15)

    scales = np.squeeze(mutein_scaling(df, unkVec_2_15))  # determine sigmoidal scaling constants

    EC50_df = calculate_EC50s(df, scales, cell_order, ligand_order)  # scale model predictions and calculate EC50s

    # plot EC50s for each time point
    catplot_comparison(ax[0], EC50_df, 30.)
    catplot_comparison(ax[1], EC50_df, 60.)
    catplot_comparison(ax[2], EC50_df, 120.)
    catplot_comparison(ax[3], EC50_df, 240.)

    return f


def catplot_comparison(ax, df, tp):
    """ Construct EC50 catplots for given time point. """
    # make subset dataframe without points where least squares fails
    subset_df = df[df['EC-50'] < 1.9]

    # plot predicted EC50
    sns.catplot(x="Cell Type", y="EC-50", hue="Mutein",
                data=subset_df.loc[(subset_df['Time Point'] == tp) & (subset_df["Data Type"] == 'Predicted')],
                legend=False, ax=ax, marker='^')

    # plot experimental EC50
    sns.catplot(x="Cell Type", y="EC-50", hue="Mutein",
                data=subset_df.loc[(subset_df['Time Point'] == tp) & (subset_df["Data Type"] == 'Experimental')],
                legend=False, ax=ax, marker='o')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right", position=(0, 0.02))
    ax.set_xlabel("")  # remove "Cell Type" from xlabel
    ax.set_ylabel(r"EC-50 (log$_{10}$[nM])")
    ax.set_title(str(tp / 60.) + " hours")

    # set manual legend
    ax.get_legend().remove()
    palette = sns.color_palette().as_hex()
    blue = mpatches.Patch(color=palette[0], label='IL2-060')
    yellow = mpatches.Patch(color=palette[1], label='IL2-062')
    green = mpatches.Patch(color=palette[2], label='IL2-088')
    red = mpatches.Patch(color=palette[3], label='IL2-097')
    circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Experimental')
    triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=6, label='Predicted')
    ax.legend(handles=[blue, yellow, green, red, circle, triangle], bbox_to_anchor=(1.02, 1), loc="upper left")


def calculate_EC50s(df, scales, cell_order, ligand_order):
    """ Scales model predictions to experimental data, then calculates EC-50 for all cell types, muteins, and time points. """

    x0 = [1, 2., 1000.]
    data_types = []
    cell_types = []
    mutein_types = []
    EC50s = np.zeros(len(cell_order) * len(tps) * 4 * 2)  # EC50 for all cell types, tps, muteins, and expr/pred
    pred_data = np.zeros((12, 4))
    expr_data = pred_data.copy()
    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+']]

    for i, cell_name in enumerate(cell_order):
        for j, ligand_name in enumerate(ligand_order):
            for k, conc in enumerate(df.Concentration.unique()):
                for l, tp in enumerate(tps):
                    pred_data[k, l] = df.loc[(df["Cells"] == cell_name) & (df["Ligand"] == ligand_name) & (
                        df["Activity Type"] == 'predicted') & (df["Concentration"] == conc) & (df["Time Point"] == tp), "Activity"]
                    expr_data[k, l] = df.loc[(df["Cells"] == cell_name) & (df["Ligand"] == ligand_name) & (
                        df["Activity Type"] == 'experimental') & (df["Concentration"] == conc) & (df["Time Point"] == tp), "Activity"]

            # scale predicted data
            for m, cell_names in enumerate(cell_groups):
                if cell_name in cell_names:
                    pred_data[:, :] = scales[m, 1] * pred_data[:, :] / (pred_data[:, :] + scales[m, 0])

            # calculate predicted and experimental EC50s for all time points
            for n, _ in enumerate(tps):
                EC50s[(8 * j) + (32 * i) + n] = nllsq_EC50(x0, muteinC.astype(np.float), pred_data[:, n])
                EC50s[(8 * j) + (32 * i) + len(tps) + n] = nllsq_EC50(x0, muteinC.astype(np.float), expr_data[:, n])

            data_types.extend(np.tile(np.array('Predicted'), len(tps)))
            data_types.extend(np.tile(np.array('Experimental'), len(tps)))
            cell_types.extend(np.tile(np.array(cell_name), len(tps) * 2))  # for both experimental and predicted
            mutein_types.extend(np.tile(np.array(ligand_name), len(tps) * 2))

    EC50s = np.log10(EC50s)
    dataframe = {'Time Point': np.tile(tps, len(cell_order) * len(ligand_order) * 2), 'Mutein': mutein_types, 'Cell Type': cell_types, 'Data Type': data_types, 'EC-50': EC50s}
    df = pd.DataFrame(dataframe)

    return df


def nllsq_EC50(x0, xdata, ydata):
    """ Performs nonlinear least squares on activity measurements to determine parameters of Hill equation and outputs EC50. """
    lsq_res = least_squares(residuals, x0, args=(xdata, ydata), bounds=([0., 0., 0.], [10**2., 10**2., 10**16]), jac='3-point')
    return lsq_res.x[0]


def hill_equation(x, x0, solution=0):
    """ Calculates EC50 from Hill Equation. """
    k = x0[0]
    n = x0[1]
    A = x0[2]
    xk = np.power(x / k, n)
    return (A * xk / (1.0 + xk)) - solution


def residuals(x0, x, y):
    """ Residual function for Hill Equation. """
    return hill_equation(x, x0) - y
