"""
Test that the fitting code can at least build the likelihood model.
"""
import unittest


class TestFit(unittest.TestCase):
    """Class to test fitting."""

    def test_fitIL2_15(self):
        """ Test that the IL2/15 model can build. """
        from ..fit_visterra import build_model
        M = build_model()
        M.build()
