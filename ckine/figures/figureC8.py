"""
This creates Figure 7, tensor factorization of mutant and WT biv and monovalent ligands.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..MBmodel import runFullModel
from sklearn.metrics import r2_score

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 5), (2, 3))
    subplotLabel(ax)

    modelDF = runFullModel()
    print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
    Pred_Exp_plot(ax[0], modelDF)
    R2_Plot_Cells(ax[1], modelDF)
    R2_Plot_Ligs(ax[2], modelDF)

    return f


def Pred_Exp_plot(ax, df):
    """Plots all experimental vs. Predicted Values"""
    sns.scatterplot(x="Experimental", y="Predicted", hue="Cell", style="Valency", data=df, ax=ax)
    ax.set(xlim=(0, 75000), ylim=(0, 75000))


def R2_Plot_Cells(ax, df):
    """Plots all experimental vs. Predicted Values"""
    accDF = pd.DataFrame(columns={"Cell Type", "Valency", "Accuracy"})
    for cell in df.Cell.unique():
        for val in df.Valency.unique():
            preds = df.loc[(df.Cell == cell) & (df.Valency == val)].Predicted.values
            exps = df.loc[(df.Cell == cell) & (df.Valency == val)].Experimental.values
            r2 = r2_score(exps, preds)
            accDF = accDF.append(pd.DataFrame({"Cell Type": [cell], "Valency": [val], "Accuracy": [r2]}))

    sns.barplot(x="Cell Type", y="Accuracy", hue="Valency", data=accDF, ax=ax)
    ax.set(ylim=(0, 1))


def R2_Plot_Ligs(ax, df):
    """Plots all experimental vs. Predicted Values"""
    accDF = pd.DataFrame(columns={"Ligand", "Valency", "Accuracy"})
    for ligand in df.Ligand.unique():
        for val in df.loc[df.Ligand == ligand].Valency.unique():
            preds = df.loc[(df.Ligand == ligand) & (df.Valency == val)].Predicted.values
            exps = df.loc[(df.Ligand == ligand) & (df.Valency == val)].Experimental.values
            r2 = r2_score(exps, preds)
            accDF = accDF.append(pd.DataFrame({"Ligand": [ligand], "Valency": [val], "Accuracy": [r2]}))
    sns.barplot(x="Ligand", y="Accuracy", hue="Valency", data=accDF, ax=ax)
    ax.set(ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
