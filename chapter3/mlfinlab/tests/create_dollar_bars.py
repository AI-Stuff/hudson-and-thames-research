

import numpy as np
import pandas as pd
from tqdm import tqdm
from mlfinlab.util.utils import cprint


def read_kibot_ticks(fp):
	# read tick data from http://www.kibot.com/support.aspx#data_format
	cols = list(map(str.lower ,['Date' ,'Time' ,'Price' ,'Bid' ,'Ask' ,'Size']))
	df = (pd.read_csv(fp, header=None)
		  .rename(columns=dict(zip(range(len(cols)) ,cols)))
		  .assign(dates=lambda df: (pd.to_datetime(df['date' ] +df['time'],
												   format='%m/%d/%Y%H:%M:%S')))
		  .assign(v=lambda df: df['size']) # volume
		  .assign(dv=lambda df: df['price' ] *df['size']) # dollar volume
		  .drop(['date' ,'time'] ,axis=1)
		  .set_index('dates')
		  .drop_duplicates())
	return df


def mad_outlier(y, thresh=3.):
	'''
	compute outliers based on mad
	# args
		y: assumed to be array with shape (N,1)
		thresh: float()
	# returns
		array index of outliers
	'''
	median = np.median(y)
	diff = np.sum((y - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh


def dollar_bars(df, dv_column, m):
	'''
	compute dollar bars

	# args
		df: pd.DataFrame()
		dv_column: name for dollar volume data
		m: int(), threshold value for dollars
	# returns
		idx: list of indices
	'''
	t = df[dv_column]
	ts = 0
	idx = []
	for i, x in enumerate(tqdm(t)):
		ts += x
		if ts >= m:
			idx.append(i)
			ts = 0
			continue
	return idx


def dollar_bar_df(df, dv_column, m):
	idx = dollar_bars(df, dv_column, m)
	return df.iloc[idx].drop_duplicates()


def main():
	print("hello!")
	# sys.path.append('../util')

	file_prefix = 'IVE_tickbidask'
	asset_name = 'IVE'

	# file_prefix = 'WDC_tickbidask'
	# asset_name = 'WDC'

	# load the tick data
	file_name = 'data/raw/' + file_prefix + '.txt'
	df = read_kibot_ticks(file_name)
	cprint(df)

	# save the data interim directory
	file_name = 'data/interim/' + asset_name + '.parq'
	df.to_parquet(file_name)

	# determine outliers and drop them
	print("Detect and eliminate outliers")
	mad = mad_outlier(df.price.values.reshape(-1, 1))
	df = df.loc[~mad]
	file_name = 'data/processed/' + asset_name + '.parq'
	df.to_parquet(file_name)
	cprint(df)

	# compute dollar bars
	print("Compute dollar bars")
	dollar_M = 1_000_000  # arbitrary
	dv_bar_df = dollar_bar_df(df, 'dv', dollar_M)
	cprint(dv_bar_df)

	# create dollar bars
	print("Form dollar bars")
	dbars = dollar_bar_df(df, 'dv', 1_000_000).drop_duplicates().dropna()
	cprint(dbars)

	file_name = 'data/processed/DOL_' + asset_name + '.parq'
	dbars.to_parquet(file_name)
	print("Goodbye!")


if __name__ == '__main__':
	main()
