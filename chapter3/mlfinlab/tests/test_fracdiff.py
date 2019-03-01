
import numpy as np
import pandas as pd
import unittest
from statsmodels.tsa.stattools import adfuller
from mlfinlab.fracdiff.fracdiff import frac_diff_ffd


class TestFractionalDifference(unittest.TestCase):
	def test_adf_stat(self):
		close = pd.read_csv('test_data/SPY.csv', index_col=0, parse_dates=True)[['Close']]
		close = close['1993':]

		fracs = frac_diff_ffd(close.apply(np.log), differencing_amt=0.47, threshold=1e-5)
		result = adfuller(fracs, maxlag=2, regression='C', autolag='AIC', store=False, regresults=False)
		self.assertEqual(abs(round(result[0], 3)), 3.704)

