#!/usr/bin/env python3

import pymc3 as pm
from ckine.fit_visterra import build_model, sampling

if __name__ == "__main__":  # only go into this loop if you're running fit.py directly instead of running a file that calls fit.py
    filename = "IL2_visterra_results"
    M = build_model()
    trace = sampling(M.M)
    pm.backends.text.dump(filename, trace)  # instead of pickling data we dump it into file that can be accessed by read_fit_data.py
