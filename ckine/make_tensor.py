"""
Generate a tensor for the different y-values that arise at different timepoints during the model and with various initial conditions.
The initial conditions vary the concentrations of the three ligands to simulate different cell lines.
Cell lines are defined by the number of each receptor subspecies on their surface.
"""
import numpy as np
from .model import runCkineU, runCkineU_IL2
from .imports import import_samples_2_15

rxntfR, _ = import_samples_2_15(N=1)
rxntfR = np.squeeze(rxntfR)


# generate n_timepoints evenly spaced timepoints to 4 hrs
tensor_time = np.linspace(0.0, 240.0, 200)


def n_lig(mut):
    """Function to return the number of cytokines used in building the tensor."""
    # Mutant here refers to a tensor made exclusively of WT IL-2 and mutant affinity IL-2s.
    if mut:
        nlig = 3
    else:
        nlig = 4
    return nlig


def ySolver(matIn, ts, tensor=True):
    """ This generates all the solutions for the Wild Type interleukins across conditions defined in meshprep(). """
    matIn = np.squeeze(matIn)
    rxn = rxntfR.copy()

    if tensor:
        rxn[22:30] = matIn[6:14]  # Receptor expression
    rxn[0:6] = matIn[0:6]  # Cytokine stimulation concentrations in the following order: IL2, 15, 7, 9, 4, 21, and in nM

    temp = runCkineU(ts, rxn)

    return temp


def ySolver_IL2_mut(matIn, ts, mut):
    """ This generates all the solutions of the mutant tensor. """
    matIn = np.squeeze(matIn).copy()
    kfwd, k4rev, k5rev = rxntfR[6], rxntfR[7], rxntfR[8]
    k1rev = 0.6 * 10.0
    k2rev = 0.6 * 144.0
    k11rev = 63.0 * k5rev / 1.5

    if mut == "a":
        k2rev *= 10.0  # 10x weaker binding to IL2Rb
    elif mut == "b":
        k2rev *= 0.01  # 100x more bindng to IL2Rb

    rxntfr = np.array([matIn[0], kfwd, k1rev, k2rev, k4rev, k5rev, k11rev, matIn[6], matIn[7], matIn[8], k1rev * 5.0, k2rev * 5.0, k4rev * 5.0, k5rev * 5.0, k11rev * 5.0])  # IL2Ra, IL2Rb, gc

    yOut = runCkineU_IL2(ts, rxntfr)

    return yOut
