"""
A file that includes functions for calculating mutein activity.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .imports import import_muteins
from .model import receptor_expression, runIL2simple

mutaff = {
    "IL2-060": [1., 1., 5.],  # Wild-type, but dimer
    "IL2-062": [1., 15., 5.],  # Weaker b-g
    "IL2-088": [13., 1., 5.],  # Weaker CD25
    "IL2-097": [13., 15., 5.]  # Both
}

dataMean, _ = import_muteins()
dataMean.reset_index(inplace=True)


def organize_expr_pred(df, cell_name, ligand_name, receptors, muteinC, tps, unkVec):
    """ Appends input dataframe with experimental and predicted activity for a given cell type and mutein. """

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
            tps, 12), 'Concentration': mutein_conc.reshape(num,), 'Activity Type': np.tile(np.array('predicted'), num), 'Replicate': np.tile(np.array(j + 1), num),
            'Activity': pred_data[:, :, j].reshape(num,)})
        df = df.append(df_pred, ignore_index=True)

    return df


def calc_dose_response_mutein(unkVec, input_params, tps, muteinC, cell_receptors):
    """ Calculates activity for a given cell type at various mutein concentrations and timepoints. """

    total_activity = np.zeros((len(muteinC), len(tps)))

    # loop for each mutein concentration
    for i, conc in enumerate(muteinC):
        active_ckine = runIL2simple(unkVec, input_params, conc, tps=tps, input_receptors=cell_receptors)
        total_activity[i, :] = np.reshape(active_ckine, (-1, 4))  # save the activity from this concentration for all 4 tps

    return total_activity


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
