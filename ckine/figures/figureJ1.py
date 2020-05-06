"""
This creates Figure 4 for Single Cell data analysis. Plots of flow intensity versus receptor quantification.
"""
from .figureCommon import subplotLabel, getSetup
from .figureCommonJulia import getUnkVecPy


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 5), (2, 4))

    subplotLabel(ax)
    a = getUnkVecPy()
    print(a)
    print(type(a))

    return f
