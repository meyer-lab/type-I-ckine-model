"""
This creates Figure 5, where we will exlore MB model's handling of variance..
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from .figureCommon import subplotLabel, getSetup

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (3, 1))

    subplotLabel(ax)

    return f
