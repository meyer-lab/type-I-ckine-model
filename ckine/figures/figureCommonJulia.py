"""
This file contains functions that are used in multiple figures from Julia.
"""
from julia.api import Julia  # nopep8
jl = Julia(compiled_modules=False)  # nopep8
from julia import gcSolver  # nopep8


def getUnkVecPy():
    "Returns initial points for unknown quantities"
    return gcSolver.getUnkVec()
