"""
This creates Figure 5.
"""
import string
from .figureCommon import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7.5, 2.5), (1, 4))

    # Add subplot labels
    for ii, item in enumerate(ax):
        subplotLabel(item, string.ascii_uppercase[ii])  # Add subplot labels

    return f
