"""
This creates Figure 4, fitting of multivalent binding model to Gc Data.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from copy import copy
from .figureCommon import subplotLabel, getSetup, nllsq_EC50
from ..MBmodel import runFullModel, cytBindingModel, getKxStar
from ..imports import getBindDict

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""

    ax, f = getSetup((10, 5), (2, 4), multz={4: 1})
    axlabel = copy(ax)
    del axlabel[5]
    subplotLabel(axlabel)
    ax[5].axis("off")

    # minSolved = minimize(runFullModel, x0=-11, args=[0.5, False])
    # print(minSolved.x)
    modelDF = runFullModel(time=[0.5, 1.0])
    print(r2_score(modelDF.Experimental.values, modelDF.Predicted.values))
    Pred_Exp_plot(ax[0], modelDF)

    R2_Plot_Cells(ax[1], modelDF)
    R2_Plot_Ligs(ax[2], modelDF)
    MonVsBivalent(ax[3], modelDF, ligs=True)

    EC50comp(ax[4], modelDF, time=0.5)
    legend = ax[4].get_legend()
    labels = (x.get_text() for x in legend.get_texts())
    ax[5].legend(legend.legendHandles, labels, loc="upper left", prop={"size": 10})  # use this to place universal legend later
    ax[4].get_legend().remove()
    timePlot(ax[6])

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


def MonVsBivalent(ax, dfAll, ligs=True):
    """Compares accuracy of monovalent vs bivalent predictions"""
    df = dfAll.loc[(dfAll.Valency == 2)].copy()
    dates = df.Date.unique()
    #monPredictCol = pd.DataFrame({"MonPredict": np.zeros(df.shape[0])})
    df["MonPredict"] = np.zeros(df.shape[0])

    for date in dates:
        dfDate = df.loc[(df.Date == date)]
        ligands = dfDate.Ligand.unique()
        concs = dfDate.Dose.unique()
        cellTypes = dfDate.Cell.unique()
        times = dfDate.Time.unique()

        for lig in ligands:
            for conc in concs:
                for cell in cellTypes:
                    predVal = cytBindingModel(lig, 1, conc * 2, cell)
                    for time in times:
                        df.loc[(df.Date == date) & (df.Ligand == lig) & (df.Dose == conc) & (df.Cell == cell) & (df.Time == time), "MonPredict"] = predVal

    for date in dates:
        for cell in cellTypes:
            expVec = df.loc[(df.Date == date) & (df.Cell == cell)].Experimental.values
            predVec = df.loc[(df.Date == date) & (df.Cell == cell)].MonPredict.values
            slope = np.linalg.lstsq(np.reshape(predVec, (-1, 1)), np.reshape(expVec, (-1, 1)), rcond=None)[0][0]
            df.loc[(df.Date == date) & (df.Cell == cell), "MonPredict"] = predVec * slope

    if ligs:
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
    else:
        accDF = pd.DataFrame(columns={"Cell Type", "Prediction Valency", "Accuracy"})
        for cellType in df.Cell.unique():
            BivPreds = df.loc[(df.Cell == cellType)].Predicted.values
            MonPreds = df.loc[(df.Cell == cellType)].MonPredict.values
            exps = df.loc[(df.Cell == cellType)].Experimental.values
            r2Biv = r2_score(exps, BivPreds)
            r2Mon = r2_score(exps, MonPreds)
            accDF = accDF.append(pd.DataFrame({"Cell Type": [cellType], "Prediction Valency": [1], "Accuracy": [r2Mon]}))
            accDF = accDF.append(pd.DataFrame({"Cell Type": [cellType], "Prediction Valency": [2], "Accuracy": [r2Biv]}))
        sns.barplot(x="Cell Type", y="Accuracy", hue="Prediction Valency", data=accDF, ax=ax)
        ax.set(ylim=(0, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    return df


def EC50comp(ax, dfAll, time):
    """Predicts EC50s for each drug for 4 cell types, and plots"""
    x0exp = [4, 2.0, 1000.0]
    x0pred = [4, 2.0, 1000.0]
    df = dfAll.loc[(dfAll.Time == time)].copy()
    Ligands = df.Ligand.unique()
    Cells = df.Cell.unique()

    EC50df = pd.DataFrame(columns=["Cell Type", "Ligand", "EC50", "Exp/Pred"])

    for ligand in Ligands:
        Valencies = df.loc[(df.Ligand == ligand), "Valency"].unique()
        for valency in Valencies:
            for cell in Cells:
                #dates = df.loc[(df.Ligand == ligand) & (df.Cell == cell) & (df.Valency == valency)].Date.values
                dosesExp = df.loc[(df.Ligand == ligand) & (df.Cell == cell) & (df.Valency == valency)].Dose.values
                doseMax, doseMin = np.log10(np.amax(dosesExp)) + 4, np.log10(np.amin(dosesExp))
                dosesPredMB = np.logspace(doseMin, doseMax, 40)
                dosesPred = np.log10(dosesPredMB) + 4
                dosesExp = np.log10(dosesExp) + 4

                expVals = df.loc[(df.Ligand == ligand) & (df.Cell == cell) & (df.Valency == valency)].Experimental.values
                predVals = cytBindingModel(ligand, valency, dosesPredMB, cell)
                EC50exp = nllsq_EC50(x0exp, dosesExp, expVals) - 4
                EC50pred = nllsq_EC50(x0pred, dosesPred, predVals) - 4

                if valency == 1:
                    EC50df = EC50df.append(pd.DataFrame({"Cell Type": [cell], "Ligand": [ligand + " (Mono)"], "EC50": [EC50exp], "Exp/Pred": ["Experimental"]}))
                    EC50df = EC50df.append(pd.DataFrame({"Cell Type": [cell], "Ligand": [ligand + " (Mono)"], "EC50": [EC50pred], "Exp/Pred": ["Predicted"]}))
                else:
                    EC50df = EC50df.append(pd.DataFrame({"Cell Type": [cell], "Ligand": [ligand + " (Biv)"], "EC50": [EC50exp], "Exp/Pred": ["Experimental"]}))
                    EC50df = EC50df.append(pd.DataFrame({"Cell Type": [cell], "Ligand": [ligand + " (Biv)"], "EC50": [EC50pred], "Exp/Pred": ["Predicted"]}))

    EC50df = EC50df.loc[(EC50df["Cell Type"].isin(["Treg", "Thelper"]))]
    sns.scatterplot(x="Ligand", y="EC50", hue="Cell Type", style="Exp/Pred", data=EC50df, ax=ax)
    ax.set(ylabel=r"log$_{10}$EC50 (nM)", ylim=(-2, 6))
    ax.set_xticklabels(EC50df.Ligand.unique(), rotation=45)


def timePlot(ax):
    """Plots all experimental vs. Predicted Values"""
    times = [[0.5], [1.], [2.], [4.]]
    accDF = pd.DataFrame(columns={"Time", "Valency", "Accuracy"})
    for time in times:
        df = runFullModel(time=time, saveDict=False)
        for val in df.Valency.unique():
            preds = df.loc[(df.Time == time[0]) & (df.Valency == val)].Predicted.values
            exps = df.loc[(df.Time == time[0]) & (df.Valency == val)].Experimental.values
            r2 = r2_score(exps, preds)
            accDF = accDF.append(pd.DataFrame({"Time": time, "Valency": [val], "Accuracy": [r2]}))
    sns.barplot(x="Time", y="Accuracy", hue="Valency", data=accDF, ax=ax)
    ax.set(ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
