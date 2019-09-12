"""
This creates Figure 6.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
from scipy.optimize import least_squares
from .figureCommon import subplotLabel, getSetup, plot_conf_int
from .figureB1 import runIL2simple
from ..model import receptor_expression
from ..imports import import_muteins, import_Rexpr, import_samples_2_15

dataMean, _ = import_muteins()
dataMean.reset_index(inplace=True)
data, _, _ = import_Rexpr()
data.reset_index(inplace=True)
unkVec_2_15, _ = import_samples_2_15(N=5)

mutaff = {
    "IL2-060": [1., 1., 5.],  # Wild-type, but dimer
    "IL2-062": [1., 15., 5.],  # Weaker b-g
    "IL2-088": [13., 1., 5.],  # Weaker CD25
    "IL2-097": [13., 15., 5.]  # Both
}


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((18, 8), (4, 8))

    for ii, item in enumerate(ax):
        if ii < 26:
            subplotLabel(item, string.ascii_uppercase[ii])
        else:
            subplotLabel(item, 'A' + string.ascii_uppercase[ii - 26])

    tps = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
    muteinC = dataMean.Concentration.unique()
    dataMean["Concentration"] = np.log10(dataMean["Concentration"])  # logscale for plotting

    ligand_order = ['IL2-060', 'IL2-062', 'IL2-088', 'IL2-097']
    cell_order = ['NK', 'CD8+', 'T-reg', 'Naive Treg', 'Mem Treg', 'T-helper', 'Naive Th', 'Mem Th']

    df = pd.DataFrame(columns=['Cells', 'Ligand', 'Time Point', 'Concentration', 'Activity Type', 'Replicate', 'Activity'])  # make empty dataframe for all cell types

    # loop for each cell type and mutein
    for _, cell_name in enumerate(cell_order):

        IL2Ra = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\alpha$'), "Count"].values[0]
        IL2Rb = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == 'IL-2R$\\beta$'), "Count"].values[0]
        gc = data.loc[(data["Cell Type"] == cell_name) & (data["Receptor"] == '$\\gamma_{c}$'), "Count"].values[0]
        receptors = np.array([IL2Ra, IL2Rb, gc]).astype(np.float)

        for _, ligand_name in enumerate(ligand_order):

            # append dataframe with experimental and predicted activity
            df = organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec_2_15)

    # determine scaling constants
    scales = mutein_scaling(df, unkVec_2_15)

    plot_expr_pred(ax, df, scales, cell_order, ligand_order, tps, muteinC)

    return f


def plot_expr_pred(ax, df, scales, cell_order, ligand_order, tps, muteinC):
    """ Plots experimental and scaled model-predicted dose response for all cell types, muteins, and time points. """

    pred_data = np.zeros((12, 4, unkVec_2_15.shape[1]))
    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+']]
    ylims = [50000., 30000., 2500., 3500.]

    for i, cell_name in enumerate(cell_order):
        for j, ligand_name in enumerate(ligand_order):

            axis = j * 8 + i

            # plot experimental data
            if axis == 31:
                sns.scatterplot(x="Concentration", y="RFU", hue="Time", data=dataMean.loc[(dataMean["Cells"] == cell_name)
                                                                                          & (dataMean["Ligand"] == ligand_name)], ax=ax[axis], s=10, palette=cm.rainbow, legend='full')
                ax[axis].legend(loc='lower right', title="time (hours)")
            else:
                sns.scatterplot(x="Concentration", y="RFU", hue="Time", data=dataMean.loc[(dataMean["Cells"] == cell_name)
                                                                                          & (dataMean["Ligand"] == ligand_name)], ax=ax[axis], s=10, palette=cm.rainbow, legend=False)

            # scale and plot model predictions
            for k, conc in enumerate(df.Concentration.unique()):
                for l, tp in enumerate(tps):
                    for m in range(unkVec_2_15.shape[1]):
                        pred_data[k, l, m] = df.loc[(df["Cells"] == cell_name) & (df["Ligand"] == ligand_name) & (
                            df["Activity Type"] == 'predicted') & (df["Concentration"] == conc) & (df["Time Point"] == tp) & (df["Replicate"] == (m + 1)), "Activity"]

            for n, cell_names in enumerate(cell_groups):
                if cell_name in cell_names:
                    for o in range(unkVec_2_15.shape[1]):
                        pred_data[:, :, o] = scales[n, 1, o] * pred_data[:, :, o] / (pred_data[:, :, o] + scales[n, 0, o])
                    plot_dose_response(ax[axis], pred_data, tps, muteinC)
                    ax[axis].set(ylim=(0, ylims[n]))
            ax[axis].set(xlabel=("[" + ligand_name + "] (log$_{10}$[nM])"), ylabel="Activity", title=cell_name)


def organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec):
    """ Appends dataframe with experimental and predicted activity for a given cell type and mutein. """

    num = len(tps) * len(muteinC)

    # organize experimental pstat data
    exp_data = np.zeros((12, 4))
    mutein_conc = exp_data.copy()
    for i, conc in enumerate(dataMean.Concentration.unique()):
        exp_data[i, :] = np.array(dataMean.loc[(dataMean["Cells"] == cell_name) & (dataMean["Ligand"] == ligand_name) & (dataMean["Concentration"] == conc), "RFU"])
        mutein_conc[i, :] = conc
    df_exp = pd.DataFrame({'Cells': np.tile(np.array(cell_name), num), 'Ligand': np.tile(np.array(ligand_name), num), 'Time Point': np.tile(tps, 12),
                           'Concentration': mutein_conc.reshape(num,), 'Activity Type': np.tile(np.array('experimental'), num), 'Replicate': np.zeros((num)), 'Activity': exp_data.reshape(num,)})
    df = df.append(df_exp, ignore_index=True)

    # calculate predicted dose response
    pred_data = np.zeros((12, 4, unkVec.shape[1]))
    for j in range(unkVec.shape[1]):
        cell_receptors = receptor_expression(receptors, unkVec[17, j], unkVec[20, j], unkVec[19, j], unkVec[21, j])
        pred_data[:, :, j] = calc_dose_response_mutein(unkVec[:, j], mutaff[ligand_name], tps, muteinC, cell_receptors)
        df_pred = pd.DataFrame({'Cells': np.tile(np.array(cell_name), num), 'Ligand': np.tile(np.array(ligand_name), num), 'Time Point': np.tile(
            tps, 12), 'Concentration': mutein_conc.reshape(num,), 'Activity Type': np.tile(np.array('predicted'), num), 'Replicate': np.tile(np.array(j + 1), num), 'Activity': pred_data[:, :, j].reshape(num,)})
        df = df.append(df_pred, ignore_index=True)

    return df


def mutein_scaling(df, unkVec):
    """ Determines scaling parameters for specified cell groups for across all muteins. """

    cell_groups = [['T-reg', 'Mem Treg', 'Naive Treg'], ['T-helper', 'Mem Th', 'Naive Th'], ['NK'], ['CD8+']]

    scales = np.zeros((4, 2, unkVec.shape[1]))
    for i, cells in enumerate(cell_groups):
        for j in range(unkVec.shape[1]):
            subset_df = df[df['Cells'].isin(cells)]
            scales[i, :, j] = optimize_scale(np.array(subset_df.loc[(subset_df["Activity Type"] == 'predicted') & (subset_df["Replicate"] == (j + 1)), "Activity"]),
                                             np.array(subset_df.loc[(subset_df["Activity Type"] == 'experimental'), "Activity"]))

    return scales


def calc_dose_response_mutein(unkVec, input_params, tps, muteinC, cell_receptors):
    """ Calculates activity for a given cell type at various mutein concentrations and timepoints. """

    total_activity = np.zeros((len(muteinC), len(tps)))

    # loop for each mutein concentration
    for i, conc in enumerate(muteinC):
        active_ckine = runIL2simple(unkVec, input_params, conc, tps=tps, input_receptors=cell_receptors)
        total_activity[i, :] = np.reshape(active_ckine, (-1, 4))  # save the activity from this concentration for all 4 tps

    return total_activity


def plot_dose_response(ax, mutein_activity, tps, muteinC):
    """ Plots predicted activity for multiple timepoints and mutein concentrations. """
    colors = cm.rainbow(np.linspace(0, 1, tps.size))

    for tt in range(tps.size):
        plot_conf_int(ax, np.log10(muteinC.astype(np.float)), mutein_activity[:, tt, :], colors[tt])


def optimize_scale(model_act, exp_act):
    """ Formulates the optimal scale to minimize the residual between model activity predictions and experimental activity measurments for a given cell type. """

    # scaling factors are sigmoidal and linear, respectively
    guess = np.array([100.0, np.mean(exp_act) / np.mean(model_act)])

    def calc_res(sc):
        """ Calculate the residuals. This is the function we minimize. """
        scaled_act = sc[1] * model_act / (model_act + sc[0])
        err = exp_act - scaled_act
        return err.flatten()

    # find result of minimization where both params are >= 0
    res = least_squares(calc_res, guess, bounds=(0.0, np.inf))
    return res.x
