import numpy as np
import pandas as pd


class MeanVarianceOptimisationModified:

    def __init__(self):
        return

    def allocate(self, asset_prices, covariance, solution='inverse_variance', resample_returns_by='B'):
        '''

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        :param solution: (str) the type of solution/algorithm to use to calculate the weights
        :param resample_returns_by: (str) specifies how to resample the returns - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        '''

        if not isinstance(asset_prices, pd.DataFrame):
            asset_prices = pd.DataFrame(asset_prices)

        # Calculate returns
        asset_returns = self._calculate_returns(asset_prices, resample_returns_by=resample_returns_by)
        assets = asset_prices.columns

        self.weights = []
        if solution == 'inverse_variance':
            self.weights = self._inverse_variance(asset_returns=asset_returns, covariance=covariance)
        self.weights = pd.DataFrame(self.weights)
        self.weights.index = assets
        self.weights = self.weights.T

    def _calculate_returns(self, asset_prices, resample_returns_by):
        '''

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        :param resample_returns_by: (str) specifies how to resample the returns - weekly, daily, monthly etc.. Defaults to
                                  'B' meaning daily business days which is equivalent to no resampling
        :return: (pd.Dataframe) stock returns
        '''

        asset_returns = asset_prices.pct_change()
        asset_returns = asset_returns.dropna(how='all')
        asset_returns = asset_returns.resample(resample_returns_by).mean()
        return asset_returns

    def _inverse_variance(self, asset_returns, covariance):
        '''

        :param asset_prices: (pd.Dataframe/np.array) the matrix of historical asset prices (daily close)
        :return: (np.array) array of portfolio weights
        '''

        cov = pd.DataFrame(covariance)
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp