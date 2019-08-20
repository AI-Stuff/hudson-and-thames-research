import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



def _get_price_diff(prices, first_fill=np.nan):
	"""
	Computes the change in price between ticks
	:param prices: (pd.Series) a series of prices
	:param first_fill: (float) optional value to replace the first change
	:return: (pd.Series) a series of price changes 

	"""
	price_change = prices.diff()
	if first_fill: 
		price_change.iloc[0] = first_fill
	return price_change

def _get_price_ratio(prices, first_fill=np.nan):
	ratio = pd.Series(index=prices.index[1:], data=prices.values[1:]/prices.values[:-1])
	return ratio

def tick_rule(tick_prices):
	"""
	Applies the tick rule to classify trades as buy-initiated or sell-initiated
	"""
	price_change = _get_price_diff(tick_prices)
	aggressor = pd.Series(index=tick_prices.index, data=0)

	aggressor.iloc[0] = 1.
	aggressor[price_change < 0] = -1.
	aggressor[price_change > 0] = 1.

	prev_idx = price_change.index[0]
	for idx, val in price_change.iloc[1:].items():
		if val == 0: aggressor[idx] = aggressor[prev_idx]
		prev_idx = idx
	return aggressor

def roll_model(tick_prices):
	"""
	Estimates 1/2*(bid-ask spread) and unobserved noise based on price sequences
	"""
	price_change = _get_price_diff(tick_prices)
	autocorr = price_change.autocorr(lag=1)
	spread_squared = np.max([-autocorr, 0])
	spread = np.sqrt(spread_squared)
	noise = price_change.var() + 2 * autocorr
	return spread, noise

def high_low_estimator(high, low, window):
	"""
	Estimates volatility using Parkinson's method
	"""
	log_high_low = np.log(high / low)
	volatility = log_high_low.rolling(window=window).mean() / np.sqrt(8. / np.pi)
	return volatility

class CorwinShultz:
	"""
	A class that encapsulates all the functions for Corwin and Shultz estimator
	"""

	@staticmethod
	def get_beta(high, low, sample_length):
		log_high_low = np.log(high / low) ** 2
		sum_neighbors = log_high_low.rolling(window=2).sum()
		beta = sum_neighbors.rolling(window=sample_length).mean()
		return beta

	@staticmethod
	def get_gamma(high, low):
		high_over_2_bars = high.rolling(window=2).max()
		low_over_2_bars = low.rolling(window=2).min()
		gamma = np.log(high_over_2_bars / low_over_2_bars) ** 2
		return gamma

	@staticmethod
	def get_alpha(beta, gamma):
		denominator = 3 - 2 ** 1.5
		beta_term = (np.sqrt(2) - 1) * np.sqrt(beta) / denominator
		gamma_term = np.sqrt(gamma / denominator)
		alpha = beta_term - gamma_term
		alpha[alpha < 0] = 0
		return alpha

	@staticmethod
	def get_becker_parkinson_volatility(beta, gamma):
		k2 = np.sqrt(8 / np.pi)
		denominator = 3 - 2 ** 1.5
		beta_term = (2 ** (-1.5) -1) * np.sqrt(beta) / (k2 * denominator)
		gamma_term = np.sqrt(gamma / (k2 ** 2 * denominator))
		volatility = beta_term + gamma_term
		volatility[volatility < 0] = 0
		return volatility



def corwin_shultz_spread(high, low, sample_length=1):
	beta = CorwinShultz.get_beta(high, low, sample_length)
	gamma = CorwinShultz.get_gamma(high, low)
	alpha = CorwinShultz.get_alpha(beta, gamma)
	spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
	n =  spread.shape[0]
	start_ind = pd.Series(index=spread.index, data=high.index[0 : n])
	return spread, start_ind

def becker_parkinson_volatility(high, low, sample_length=1):
	beta = CorwinShultz.get_beta(high, low, sample_length)
	gamma = CorwinShultz.get_gamma(high, low)
	volatility = CorwinShultz.get_becker_parkinson_volatility(beta, gamma)
	return volatility

def kyles_lambda(tick_prices, tick_volumes, regressor=LinearRegression()):
	price_change = _get_price_diff(tick_prices)
	tick_sings = tick_rule(tick_prices)
	net_order_flow = tick_sings * tick_volumes
	X = net_order_flow.values[1:].reshape(-1, 1)
	y = price_change.dropna().values
	lambda_ = regressor.fit(X, y)
	return lambda_.coef_[0]

def dollar_volume(tick_prices, tick_volumes):
	return (tick_prices * tick_volumes).sum()

def amihuds_lambda(close, dollar_volume, regressor=LinearRegression()):
	log_close = np.log(close)
	abs_change = np.abs(_get_price_diff(log_close))
	X = dollar_volume.values[1:].reshape(-1, 1)
	y = abs_change.dropna()
	lambda_ = regressor.fit(X, y)
	return lambda_.coef_[0]

def hasbroucks_lambda(close, hasbroucks_flow, regressor=LinearRegression()):
	ratio = _get_price_ratio(close)
	log_ratio = np.log(ratio)
	X = hasbroucks_flow.values[1:].reshape(-1, 1)
	y = log_ratio
	lambda_ = regressor.fit(X, y)
	return lambda_.coef_[0]


def hasbroucks_flow(tick_prices, tick_volumes, tick_sings):
	return (np.sqrt(tick_prices * tick_volumes) * tick_sings).sum()
	
def vpin(buy_volumes, sell_volumes, volume, num_bars):
	abs_diff = (buy_volumes - sell_volumes).abs()
	vpin = abs_diff.rolling(window=num_bars).mean() / volume
	return vpin




