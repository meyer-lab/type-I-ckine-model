"""
This creates Figure 1 for Single Cell FC data analysis. Examples of PCA loadings/scores plots and comparisons to gating.
"""

import os
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from .figureCommon import subplotLabel, getSetup
from ..imports import importMoments

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12.5, 7.5), (3, 3))
    subplotLabel(ax)

    momentsDF = importMoments()
    moments = np.array(["Mean", "Variance", "Skew"])
    ind = [11, 6, 0]
    for i, dose in enumerate(ind):
        for j, moment in enumerate(moments):
            if (3 * i + j) == 0:
                momentPlot(ax[3 * i + j], momentsDF, moment, "2019-4-18", dose, legend=True)
            else:
                momentPlot(ax[3 * i + j], momentsDF, moment, "2019-4-18", dose, legend=False)

    return f


def momentPlot(ax, df, moment, date, doseInd, legend=False):
    """Plots moments of pSTAT for a given dose for both IL2 and IL15"""
    doses = df.Dose.unique()
    times = df.Time.unique()
    markers = ['.', '^', 'P', 'D']
    for i, time in enumerate(times):
        df1 = df.loc[(df["Date"] == date) & (df["Dose"] == doses[doseInd]) & (df["Time"] == time)]
        sns.stripplot(x="Cell", y=moment, hue="Ligand", data=df1, ax=ax, palette={"IL2": "darkorchid", "IL15": "goldenrod"}, marker=markers[i], dodge=True)

    handles, _ = ax.get_legend_handles_labels()
    handles = handles[0:2]
    for ii, name in enumerate(times):
        handles.append(Line2D([0], [0], color="k", marker=markers[ii], label="Time (hrs) " + str(name), linestyle="None"))

    ax.legend(handles=handles)
    ax.set_ylabel(moment)
    ax.set_title(moment + " " + str(doses[doseInd])[0:6] + " nM")
    ax.set_yscale("log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25)
    if not legend:
        ax.get_legend().remove()
