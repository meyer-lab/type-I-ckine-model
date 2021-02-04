"""
This creates Figure 1, which will primarily represent cartoons.
"""

import os
import numpy as np
import pandas as pds
import seaborn as sns
from .figureCommon import subplotLabel, getSetup
from ..flow import importF, gating, cellData


path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((13, 10), (3, 4))
    subplotLabel(ax)

    return f
