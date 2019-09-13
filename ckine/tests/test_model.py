"""
Unit test file.
"""
import unittest
import numpy as np
from ..model import getTotalActiveCytokine, runCkineU, runCkineU_IL2, ligandDeg
from ..figures.figureB1 import runIL2simple


class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def setUp(self):
        self.ts = np.array([0.0, 100000.0])
        self.y0 = np.random.lognormal(0.0, 1.0, 28)
        self.args = np.random.lognormal(0.0, 1.0, 17)
        self.tfargs = np.random.lognormal(0.0, 1.0, 13)
        self.fully = np.random.lognormal(0.0, 1.0, 62)

        # Force sorting fraction to be less than 1.0
        self.tfargs[2] = np.tanh(self.tfargs[2]) * 0.9

        self.rxntfR = np.concatenate((self.args, self.tfargs))

    def test_gc(self):
        """ Test to check that no active species is present when gamma chain is not expressed. """
        rxntfR = self.rxntfR.copy()
        rxntfR[24] = 0.0  # set expression of gc to 0.0
        yOut = runCkineU(self.ts, rxntfR)
        self.assertAlmostEqual(getTotalActiveCytokine(0, yOut[1]), 0.0, places=5)  # IL2
        self.assertAlmostEqual(getTotalActiveCytokine(1, yOut[1]), 0.0, places=5)  # IL15
        self.assertAlmostEqual(getTotalActiveCytokine(2, yOut[1]), 0.0, places=5)  # IL7
        self.assertAlmostEqual(getTotalActiveCytokine(3, yOut[1]), 0.0, places=5)  # IL9
        self.assertAlmostEqual(getTotalActiveCytokine(4, yOut[1]), 0.0, places=5)  # IL4
        self.assertAlmostEqual(getTotalActiveCytokine(5, yOut[1]), 0.0, places=5)  # IL21

    def test_endosomalCTK_bound(self):
        """ Test that appreciable cytokine winds up in the endosome. """
        rxntfR = self.rxntfR.copy()
        rxntfR[0:6] = 0.0
        rxntfR[6] = 1.0e-6  # Damp down kfwd
        rxntfR[7:22] = 0.1  # Fill all in to avoid parameter variation
        rxntfR[18] = 10.0  # Turn up active endocytosis
        rxntfR[21] = 0.02  # Turn down degradation
        rxntfR[22:30] = 10.0  # Control expression

        # set high concentration of IL2
        rxntfR_1 = rxntfR.copy()
        rxntfR_1[0] = 1000.0
        # set high concentration of IL15
        rxntfR_2 = rxntfR.copy()
        rxntfR_2[1] = 1000.0
        # set high concentration of IL7
        rxntfR_3 = rxntfR.copy()
        rxntfR_3[2] = 1000.0
        # set high concentration of IL9
        rxntfR_4 = rxntfR.copy()
        rxntfR_4[3] = 1000.0
        # set high concentration of IL4
        rxntfR_5 = rxntfR.copy()
        rxntfR_5[4] = 1000.0
        # set high concentration of IL21
        rxntfR_6 = rxntfR.copy()
        rxntfR_6[5] = 1000.0

        # first element is t=0 and second element is t=10**5
        yOut_1 = runCkineU(self.ts, rxntfR_1)
        yOut_2 = runCkineU(self.ts, rxntfR_2)
        yOut_3 = runCkineU(self.ts, rxntfR_3)
        yOut_4 = runCkineU(self.ts, rxntfR_4)
        yOut_5 = runCkineU(self.ts, rxntfR_5)
        yOut_6 = runCkineU(self.ts, rxntfR_6)

        # make sure endosomal free ligand is positive at equilibrium
        # IL2
        self.assertGreater(yOut_1[1, 56], 1.0)
        self.assertLess(np.sum(yOut_1[1, np.array([57, 58, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL15
        self.assertGreater(yOut_2[1, 57], 1.0)
        self.assertLess(np.sum(yOut_2[1, np.array([56, 58, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL7
        self.assertGreater(yOut_3[1, 58], 1.0)
        self.assertLess(np.sum(yOut_3[1, np.array([56, 57, 59, 60, 61])]), 1.0e-9)  # no other ligand
        # IL9
        self.assertGreater(yOut_4[1, 59], 1.0)
        self.assertLess(np.sum(yOut_4[1, np.array([56, 57, 58, 60, 61])]), 1.0e-9)  # no other ligand
        # IL4
        self.assertGreater(yOut_5[1, 60], 1.0)
        self.assertLess(np.sum(yOut_5[1, np.array([56, 57, 58, 59, 61])]), 1.0e-9)  # no other ligand
        # IL21
        self.assertGreater(yOut_6[1, 61], 1.0)
        self.assertLess(np.sum(yOut_6[1, np.array([56, 57, 58, 59, 60])]), 1.0e-9)  # no other ligand

        # make sure total amount of ligand bound to receptors is positive at equilibrium
        self.assertTrue(np.greater(yOut_1[31:37], 0.0).all())
        self.assertTrue(np.greater(yOut_2[38:44], 0.0).all())
        self.assertTrue(np.greater(yOut_3[45:47], 0.0).all())
        self.assertTrue(np.greater(yOut_4[48:50], 0.0).all())
        self.assertTrue(np.greater(yOut_5[51:53], 0.0).all())
        self.assertTrue(np.greater(yOut_6[54:56], 0.0).all())

    def test_runCkineU_IL2(self):
        """ Make sure IL-2 activity is higher when its IL-2 binds tighter to IL-2Ra (k1rev (rxntfr[2]) is smaller). """
        rxntfr_reg = np.ones(15)
        rxntfr_loose = rxntfr_reg.copy()
        rxntfr_gc = rxntfr_reg.copy()
        rxntfr_gc[9] = 0.0  # set gc expression to 0
        rxntfr_loose[1] = 10.0 ** -5  # "looser" dimerization occurs when kfwd is small

        # find yOut vectors for both rxntfr's
        y_reg = runCkineU_IL2(self.ts, rxntfr_reg)
        y_loose = runCkineU_IL2(self.ts, rxntfr_loose)
        y_gc = runCkineU_IL2(self.ts, rxntfr_gc)

        # get total amount of IL-2 derived active species at end of experiment (t=100000)
        active_reg = getTotalActiveCytokine(0, y_reg[1, :])
        active_loose = getTotalActiveCytokine(0, y_loose[1, :])
        active_gc = getTotalActiveCytokine(0, y_gc[1, :])

        self.assertLess(active_loose, active_reg)  # lower dimerization rate leads to less active complex
        self.assertLess(active_gc, active_reg)  # no gc expression leads to less/no active complex

    def test_ligandDeg_All(self):
        """ Verify that ligand degradation increases when sortF and kDeg increase. """
        # case for IL2
        y = runCkineU_IL2(self.ts, np.ones(15))
        sortF, kDeg = 0.5, 1.0
        reg = ligandDeg(y[1, :], sortF, kDeg, 0)
        high_sortF = ligandDeg(y[1, :], 0.9, kDeg, 0)
        high_kDeg = ligandDeg(y[1, :], sortF, kDeg * 10, 0)
        low_kDeg = ligandDeg(y[1, :], sortF, kDeg * 0.1, 0)

        self.assertGreater(high_sortF, reg)
        self.assertGreater(high_kDeg, reg)
        self.assertGreater(reg, low_kDeg)

        # case for IL15
        y = runCkineU(self.ts, self.rxntfR)
        reg = ligandDeg(y[1, :], sortF, kDeg, 1)
        high_kDeg = ligandDeg(y[1, :], sortF, kDeg * 10, 1)
        self.assertGreater(high_kDeg, reg)

    def test_IL2_endo_binding(self):
        """ Make sure that the runIL2simple works and that increasing the endosomal reverse reaction rates causes tighter binding (less ligand degradation). """
        rxntfR = self.rxntfR.copy()
        inp_normal = np.array([1.0, 1.0, 5.0])
        inp_tight = np.array([1.0, 1.0, 1.0])  # lower reverse rates in the endosome

        out_norm = runIL2simple(rxntfR, inp_normal, 1.0, ligandDegradation=True)
        out_tight = runIL2simple(rxntfR, inp_tight, 1.0, ligandDegradation=True)

        self.assertLess(out_tight, out_norm)  # tighter binding will have a lower rate of ligand degradation since all free ligand is degraded
