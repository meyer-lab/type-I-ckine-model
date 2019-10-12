"""Fits model to Visterra Data"""
import pymc3 as pm
import theano.tensor as T
import numpy as np
from .model import getTotalActiveSpecies, receptor_expression
from .differencing_op import runCkineDoseOp
from .imports import import_Rexpr, import_pstat


def commonTraf(trafficking=True):
    """ Set the common trafficking parameter priors. """
    kfwd = pm.Lognormal('kfwd', mu=np.log(0.001), sd=0.5, shape=1)

    if trafficking:
        endo = pm.Lognormal('endo', mu=np.log(0.1), sd=0.1, shape=1)
        activeEndo = pm.Lognormal('activeEndo', sd=0.1, shape=1)
        kRec = pm.Lognormal('kRec', mu=np.log(0.1), sd=0.1, shape=1)
        kDeg = pm.Lognormal('kDeg', mu=np.log(0.01), sd=0.2, shape=1)
        sortF = pm.Beta('sortF', alpha=12, beta=80, shape=1)
    else:
        # Assigning trafficking to zero to fit without trafficking
        endo = activeEndo = kRec = kDeg = T.zeros(1, dtype=np.float64)
        sortF = T.ones(1, dtype=np.float64) * 0.5
    return kfwd, endo, activeEndo, kRec, kDeg, sortF


def sampling(M):
    """ This is the sampling that actually runs the model. """
    return pm.sample(init="ADVI", chains=2, model=M)


class pSTAT_activity:  # pylint: disable=too-few-public-methods
    """ Calculating the pSTAT activity residuals for IL2 and IL15 stimulation in distinct cell lines from Visterra. """

    def __init__(self):
        self.ts = np.array([0.5, 1.0, 2.0, 4.0]) * 60.0
        # might have to move these lines into model class and pass the relevant data as function args
        _, self.receptor_data, self.cell_names_receptor = import_Rexpr()
        self.cytokC, self.cell_names_pstat, self.IL2_data, self.IL15_data = import_pstat()

        self.cytokM = np.zeros((self.cytokC.size * 2, 6), dtype=np.float64)
        self.cytokM[0: self.cytokC.size, 0] = self.cytokC
        self.cytokM[self.cytokC.size::, 1] = self.cytokC

    def calc(self, unkVec, scale):
        """ Simulate the STAT5 measurements and return residuals between model prediction and experimental data. """

        # Find amount of active species for each simulation
        Op = runCkineDoseOp(tt=self.ts, condense=getTotalActiveSpecies().astype(np.float64), conditions=self.cytokM)

        output = list()
        print(self.cell_names_pstat)
        for i, name in enumerate(self.cell_names_pstat):
            # plot matching experimental and predictive pSTAT data for the same cell type
            j = self.cell_names_receptor.get_loc(name)

            unkVec = T.set_subtensor(unkVec[16], receptor_expression(self.receptor_data[j, 0], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))  # IL2Ra
            unkVec = T.set_subtensor(unkVec[17], receptor_expression(self.receptor_data[j, 1], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))  # IL2Rb
            unkVec = T.set_subtensor(unkVec[18], receptor_expression(self.receptor_data[j, 2], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))  # gc
            unkVec = T.set_subtensor(unkVec[19], receptor_expression(self.receptor_data[j, 3], unkVec[11], unkVec[14], unkVec[13], unkVec[15]))  # IL15Ra
            actVec = Op(unkVec)

            # account for pSTAT5 saturation and then normalize from 0 to 1
            actVec = actVec / (actVec + scale[i])
            actVec = actVec / T.mean(actVec)

            # concatenate the cell's IL-2 data to the IL-15 data so it matches actVec
            data_cat = np.concatenate((self.IL2_data[(i * 4): ((i + 1) * 4)], self.IL15_data[(i * 4): ((i + 1) * 4)]))
            data_cat = data_cat / np.mean(data_cat)

            newVec = T.reshape(actVec, data_cat.shape)

            output.append(newVec - data_cat)  # copy residuals into list

        return output, self.cell_names_pstat


class build_model:
    """ Build the overall model handling Visterra fitting. """

    def __init__(self, traf=True):
        self.traf = traf
        self.act = pSTAT_activity()
        self.M = self.build()

    def build(self):
        """Build."""
        M = pm.Model()

        with M:
            kfwd, endo, activeEndo, kRec, kDeg, sortF = commonTraf(trafficking=self.traf)

            rxnrates = pm.Lognormal("rxn", sd=0.5, shape=6)  # 6 reverse rxn rates for IL2/IL15
            nullRates = T.ones(4, dtype=np.float64)  # k27rev, k31rev, k33rev, k35rev
            scale = pm.Lognormal("scales", mu=np.log(100.0), sd=1, shape=10)

            # plug in 0 for all receptor rates... will override with true measurements in act.calc()
            Rexpr = T.zeros(4, dtype=np.float64)

            unkVec = T.concatenate((kfwd, rxnrates, nullRates, endo, activeEndo, sortF, kRec, kDeg, Rexpr, nullRates * 0.0))

            Y_act, names = self.act.calc(unkVec, scale)  # fitting the data based on dst15.calc for the given parameters

            for ii, Ya in enumerate(Y_act):
                sd_act = T.std(Ya)  # Add bounds for the stderr to help force the fitting solution
                pm.Deterministic("Y_act_" + names[ii], T.sum(T.square(Ya)))
                pm.Normal("fitD_act_" + names[ii], sd=sd_act, observed=Ya)  # experimental-derived stderr is used

            # Save likelihood
            pm.Deterministic("logp", M.logpt)

        return M
