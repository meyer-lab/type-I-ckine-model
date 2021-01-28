"""
Implementation of a simple multivalent binding model.
"""

import os
from os.path import dirname, join
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import pandas as pd
import jax.numpy as jnp
from jax import jit, jacfwd
from jax.config import config
from scipy.optimize import root
from scipy.special import binom
from .imports import import_pstat_all
from scipy.optimize import minimize

path_here = dirname(dirname(__file__))
KxStarP = 3e-11


def Req_func(Req, Rtot, L0fA, AKxStar, f):
    """ Mass balance. Transformation to account for bounds. """
    Phisum = jnp.dot(AKxStar, Req.T)
    return Req + L0fA * Req * (1 + Phisum) ** (f - 1) - Rtot


Req_func_jit = jit(Req_func)
Req_func_J_jit = jit(jacfwd(Req_func, 0))


def polyfc(L0, KxStar, f, Rtot, LigC, Kav):
    """
    The main function. Generate all info for heterogenenous binding case
    L0: concentration of ligand complexes.
    KxStar: detailed balance-corrected Kx.
    f: valency
    Rtot: numbers of each receptor appearing on the cell.
    LigC: the composition of the mixture used.
    Kav: a matrix of Ka values. row = IgG's, col = FcgR's
    """
    # Data consistency check
    Kav = np.array(Kav)
    Rtot = np.array(Rtot, dtype=np.float)
    assert Rtot.ndim <= 1
    LigC = np.array(LigC)
    assert LigC.ndim <= 1
    LigC = LigC / np.sum(LigC)
    assert LigC.size == Kav.shape[0]
    assert Rtot.size == Kav.shape[1]
    assert Kav.ndim == 2

    # Run least squares to get Req
    Req = Req_Regression(L0, KxStar, f, Rtot, LigC, Kav)

    nr = Rtot.size  # the number of different receptors

    Phi = np.ones((LigC.size, nr + 1)) * LigC.reshape(-1, 1)
    Phi[:, :nr] *= Kav * Req * KxStar
    Phisum = np.sum(Phi[:, :nr])

    Lbound = L0 / KxStar * ((1 + Phisum) ** f - 1)
    Rbound = L0 / KxStar * f * Phisum * (1 + Phisum) ** (f - 1)
    vieq = L0 / KxStar * binom(f, np.arange(1, f + 1)) * np.power(Phisum, np.arange(1, f + 1))

    return Lbound, Rbound, vieq


def Req_Regression(L0, KxStar, f, Rtot, LigC, Kav):
    """Run least squares regression to calculate the Req vector"""
    A = np.dot(LigC.T, Kav)
    L0fA = L0 * f * A
    AKxStar = A * KxStar

    # Identify an initial guess just on max monovalent interaction
    x0 = np.max(L0fA, axis=0)
    x0 = np.multiply(1.0 - np.divide(x0, 1 + x0), Rtot)

    # Solve Req by calling least_squares() and Req_func()
    lsq = root(Req_func_jit, x0, jac=Req_func_J_jit, method="lm", args=(Rtot, L0fA, AKxStar, f), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)

    return lsq["x"].reshape(1, -1)


def Req_func2(Req, L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    Psi = Req * Kav * KxStar
    Psi = jnp.pad(Psi, ((0, 0), (0, 1)), constant_values=1)
    Psirs = jnp.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Rbound = L0 / KxStar * jnp.sum(Ctheta.reshape(-1, 1) * jnp.dot(Cplx, Psinorm) * jnp.exp(jnp.dot(Cplx, jnp.log1p(Psirs - 1))), axis=0)
    return Req + Rbound - Rtot


Req_func2_jit = jit(Req_func2)
Req_func2_J_jit = jit(jacfwd(Req_func2, 0))


def polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav):
    """
    The main function to be called for multivalent binding
    :param L0: concentration of ligand complexes
    :param KxStar: Kx for detailed balance correction
    :param Rtot: numbers of each receptor on the cell
    :param Cplx: the monomer ligand composition of each complex
    :param Ctheta: the composition of complexes
    :param Kav: Ka for monomer ligand to receptors
    :return:
        Lbound: a list of Lbound of each complex
        Rbound: a list of Rbound of each kind of receptor
    """
    # Consistency check
    Kav = np.array(Kav)
    assert Kav.ndim == 2
    Rtot = np.array(Rtot, dtype=np.float)
    assert Rtot.ndim == 1
    Cplx = np.array(Cplx)
    assert Cplx.ndim == 2
    Ctheta = np.array(Ctheta)
    assert Ctheta.ndim == 1

    assert Kav.shape[0] == Cplx.shape[1]
    assert Kav.shape[1] == Rtot.size
    assert Cplx.shape[0] == Ctheta.size
    Ctheta = Ctheta / np.sum(Ctheta)

    # Solve Req
    lsq = root(Req_func2_jit, Rtot, jac=Req_func2_J_jit, method="lm", args=(L0, KxStar, Rtot, Cplx, Ctheta, Kav), options={"maxiter": 3000})
    assert lsq["success"], "Failure in rootfinding. " + str(lsq)
    Req = lsq["x"].reshape(1, -1)

    # Calculate the results
    Psi = np.ones((Kav.shape[0], Kav.shape[1] + 1))
    Psi[:, : Kav.shape[1]] *= Req * Kav * KxStar
    Psirs = np.sum(Psi, axis=1).reshape(-1, 1)
    Psinorm = (Psi / Psirs)[:, :-1]

    Lbound = L0 / KxStar * Ctheta * np.expm1(np.dot(Cplx, np.log1p(Psirs - 1))).flatten()
    Rbound = L0 / KxStar * Ctheta.reshape(-1, 1) * np.dot(Cplx, Psinorm) * np.exp(np.dot(Cplx, np.log1p(Psirs - 1)))
    Lfbnd = L0 / KxStar * Ctheta * np.exp(np.dot(Cplx, np.log(Psirs - 1))).flatten()
    assert len(Lbound) == len(Ctheta)
    assert Rbound.shape[0] == len(Ctheta)
    assert Rbound.shape[1] == len(Rtot)
    return Lbound, Rbound, Lfbnd


def cytBindingModel(mut, val, doseVec, cellType, x=False, date=False):
    """Runs binding model for a given mutein, valency, dose, and cell type."""
    recQuantDF = pd.read_csv(join(path_here, "ckine/data/RecQuantitation.csv"))
    mutAffDF = pd.read_csv(join(path_here, "ckine/data/WTmutAffData.csv"))
    Affs = mutAffDF.loc[(mutAffDF.Mutein == mut)]
    Affs = np.power(np.array([Affs["IL2RaKD"].values, Affs["IL2RBGKD"].values]) / 1e9, -1)
    Affs = np.reshape(Affs, (1, -1))
    Affs = np.repeat(Affs, 2, axis=0)

    if doseVec.size == 1:
        doseVec = np.array([doseVec])
    output = np.zeros(doseVec.size)
    recCount = recQuantDF[["Receptor", cellType]]
    recCount = [recCount.loc[(recCount.Receptor == "IL2Ra")][cellType].values, recCount.loc[(recCount.Receptor == "IL2Rb")][cellType].values]
    recCount = np.ravel(np.power(10, recCount))

    for i, dose in enumerate(doseVec):
        if x:
            print(x)
            output[i] = polyc(dose / 1e9, np.power(10, x[0]), recCount, [[val, 0]], [1.0], Affs)[1][0][1]
        else:
            output[i] = polyc(dose / 1e9, KxStarP, recCount, [[val, 0]], [1.0], Affs)[1][0][1]  # IL2RB binding only
    if date:
        convDict = pd.read_csv(join(path_here, "ckine/data/BindingConvDict.csv"))
        output *= convDict.loc[(convDict.Date == date) & (convDict.Cell == cellType)].Scale.values

    return output


def runFullModel(x=False):
    """Runs model for all data points and outputs date conversion dict for binding to pSTAT. Can be used to fit Kx"""
    statDF = import_pstat_all()
    statDF = statDF.loc[(statDF.Ligand != "H16L N-term (Mono)") & (statDF.Ligand != "IL15 (Mono)")]
    statDF = statDF.loc[(statDF.Time == 0.5)]
    dateConvDF = pd.DataFrame(columns={"Date", "Scale", "Cell"})
    masterSTAT = pd.DataFrame(columns={"Ligand", "Date", "Cell", "Dose", "Valency", "Experimental", "Predicted"})
    dates = statDF.Date.unique()
    for ii, date in enumerate(dates):
        statDFdate = statDF.loc[(statDF.Date == date)]
        ligands = statDFdate.Ligand.unique()
        concs = statDFdate.Dose.unique()
        cellTypes = statDFdate.Cell.unique()

        for lig in ligands:
            if lig[-5::] == "(Biv)":
                val = 2
                ligName = lig[0:-6]
            else:
                val = 1
                ligName = lig[0:-7]
            for conc in concs:
                for cell in cellTypes:
                    entry = statDFdate.loc[(statDFdate.Ligand == lig) & (statDFdate.Dose == conc) & (statDFdate.Cell == cell)].Mean.values
                    if len(entry) >= 1:
                        expVal = np.mean(entry)
                        predVal = cytBindingModel(ligName, val, conc, cell, x)
                        masterSTAT = masterSTAT.append(pd.DataFrame({"Ligand": ligName, "Date": date, "Cell": cell, "Dose": conc, "Valency": val, "Experimental": expVal, "Predicted": predVal}))

    for date in dates:
        for cell in masterSTAT.Cell.unique():
            expVec = masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell)].Experimental.values
            predVec = masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell)].Predicted.values
            slope = np.linalg.lstsq(np.reshape(predVec, (-1, 1)), np.reshape(expVec, (-1, 1)), rcond=None)[0][0]
            masterSTAT.loc[(masterSTAT.Date == date) & (masterSTAT.Cell == cell), "Predicted"] = predVec * slope
            dateConvDF = dateConvDF.append(pd.DataFrame({"Date": date, "Scale": slope, "Cell": cell}, index=[ii]))

    dateConvDF.set_index("Date").to_csv(join(path_here, "ckine/data/BindingConvDict.csv"))

    if x:
        print(x)
        print(LA.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values))
        return LA.norm(masterSTAT.Predicted.values - masterSTAT.Experimental.values)
    else:
        return masterSTAT
