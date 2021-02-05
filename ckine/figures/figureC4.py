"""
This creates Figure 4, fitting of multivalent binding model to Gc Data.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from .figureCommon import subplotLabel, getSetup
from ..MBmodel import runFullModel, cytBindingModel, getKxStar
from ..imports import getBindDict

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 5), (2, 3))
    subplotLabel(ax)

    #minSolved = minimize(runFullModel, -11)
    # print(minSolved.x)
    modelDF = runFullModel()
    print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
    Pred_Exp_plot(ax[0], modelDF)
    MonVsBivalent(ax[1], modelDF)

    R2_Plot_Cells(ax[2], modelDF)
    R2_Plot_Ligs(ax[3], modelDF)
    plotDoseResponses(ax[4], modelDF, "WT N-term", val=1, cellType="Treg")
    plotDoseResponses(ax[5], modelDF, "WT N-term", val=2, cellType="Treg")
    #plotDoseResponses(ax[5], modelDF, "N88D C-term", val=1, cellType="Treg")

    return f


def Pred_Exp_plot(ax, df):
    """Plots all experimental vs. Predicted Values"""
    sns.scatterplot(x="Experimental", y="Predicted", hue="Cell", style="Valency", data=df, ax=ax, alpha=0.35)
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


def plotDoseResponses(ax, df, mut, val, cellType):
    """Plots all experimental vs. Predicted Values"""
    expData = df.loc[(df.Ligand == mut) & (df.Valency == val) & (df.Cell == cellType)]
    date = expData.loc[0, :].Date.values[0]
    expData = expData.loc[(expData.Date == date)]
    expDataSTAT = expData.Experimental.values
    doseMax, doseMin = np.log10(np.amax(expData.Dose.values)), np.log10(np.amin(expData.Dose.values))
    doseVec = np.logspace(doseMin, doseMax, 100)

    preds = cytBindingModel(mut, val, doseVec, cellType, x=False, date=date)
    ax.scatter(expData.Dose.values, expDataSTAT, label="Experimental")
    ax.plot(doseVec, preds, label="Predicted")
    if val == 1:
        ax.set(title=cellType, xlabel=r"$log_{10}$ Monomeric " + mut + " (nM)", ylabel="pSTAT", xscale="log", xlim=(1e-4, 1e2))
    if val == 2:
        ax.set(title=cellType, xlabel=r"$log_{10}$ Dimeric " + mut + " (nM)", ylabel="pSTAT", xscale="log", xlim=(1e-4, 1e2))


def MonVsBivalent(ax, df):
    """Runs model for all data points and outputs date conversion dict for binding to pSTAT. Can be used to fit Kx"""
    df = df.loc[(df.Valency == 2)]
    dates = df.Date.unique()
    monPredDF = pd.DataFrame(columns={"MonPredict", "Date"})
    for date in dates:
        dfDate = df.loc[(df.Date == date)]
        ligands = dfDate.Ligand.unique()
        concs = dfDate.Dose.unique()
        cellTypes = dfDate.Cell.unique()

        for lig in ligands:
            for conc in concs:
                for cell in cellTypes:
                    predVal = cytBindingModel(lig, 1, conc * 2, cell)
                    monPredDF = monPredDF.append(pd.DataFrame({"MonPredict": predVal, "Date": date}))

    for date in dates:
        for cell in cellTypes:
            expVec = df.loc[(df.Date == date) & (df.Cell == cell)].Experimental.values
            predVec = monPredDF.loc[(df.Date == date) & (df.Cell == cell)].MonPredict.values
            slope = np.linalg.lstsq(np.reshape(predVec, (-1, 1)), np.reshape(expVec, (-1, 1)), rcond=None)[0][0]
            monPredDF.loc[(df.Date == date) & (df.Cell == cell), "MonPredict"] = predVec * slope

    df = pd.concat([df, monPredDF], axis=1)

    accDF = pd.DataFrame(columns={"Ligand", "Prediction Valency", "Accuracy"})
    for ligand in df.Ligand.unique():
        BivPreds = df.loc[(df.Ligand == ligand)].Predicted.values
        MonPreds = df.loc[(df.Ligand == ligand)].MonPredict.values
        exps = df.loc[(df.Ligand == ligand)].Experimental.values
        r2Biv = r2_score(exps, BivPreds)
        r2Mon = r2_score(exps, MonPreds)
        accDF = accDF.append(pd.DataFrame({"Ligand": [ligand], "Prediction Valency": [1], "Accuracy": [r2Mon]}))
        accDF = accDF.append(pd.DataFrame({"Ligand": [ligand], "Prediction Valency": [2], "Accuracy": [r2Biv]}))
    sns.barplot(x="Ligand", y="Accuracy", hue="Prediction Valency", data=accDF, ax=ax)
    ax.set(ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    return df
