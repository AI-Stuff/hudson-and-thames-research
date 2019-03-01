import numpy as np
import pandas as pd
from tqdm import tqdm
from mlfinlab.util.multiprocess import MultiProcessingFunctions


class CoreFunctions:
    """ The class holds functions in Chapter 2 and 3 of AFML """

    def __init__(self):
        pass

    @staticmethod
    def get_daily_vol(close, lookback=100):
        """
        Snippet 3.1, page 44, Daily Volatility Estimates

        As argued in the previous section, in practice we want to set profit taking and stop-loss limits
        that are a function of the risks involved in a bet. Otherwise, sometimes we will be aiming
        too high (tao ≫ sigma_t_i,0), and sometimes too low (tao ≪ sigma_t_i,0 ), considering
        the prevailing volatility. Snippet 3.1 computes the daily volatility at intraday estimation points,
        applying a span of lookback days to an exponentially weighted moving standard deviation.

        See the pandas documentation for details on the pandas.Series.ewm function.

        Note: This function is used to compute dynamic thresholds for profit taking and stop loss limits.

        :param close: (data frame) Closing prices
        :param lookback: (int) lookback period to compute volatility
        :return: (series) of daily volatility value
        """
        print('Calculating daily volatility for dynamic thresholds')
        # daily vol re-indexed to close
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = (pd.Series(close.index[df0 - 1],
                         index=close.index[close.shape[0] - df0.shape[0]:]))
        try:
            df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns

        except Exception as e:
            print(f'error: {e} \n please confirm no duplicate indices')
            print('adjusting shape of close.loc[df0.index]')
            cut = close.loc[df0.index].shape[0] - close.loc[df0.values].shape[0]
            df0 = close.loc[df0.index].iloc[:-cut] / close.loc[df0.values].values - 1

        df0 = df0.ewm(span=lookback).std()
        df0.rename(columns={'price': 'dailyVol'}, inplace=True)
        return df0

    @staticmethod
    def get_t_events(raw_price, threshold):
        """
        Snippet 2.4, page 39, The Symmetric CUSUM Filter.

        The CUSUM filter is a quality-control method, designed to detect a shift in the
        mean value of a measured quantity away from a target value. The filter is set up to
        identify a sequence of upside or downside divergences from any reset level zero.

        We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

        One practical aspect that makes CUSUM filters appealing is that multiple events are not
        triggered by gRaw hovering around a threshold level, which is a flaw suffered by popular
        market signals such as Bollinger Bands. It will require a full run of length threshold for raw_price
        to trigger an event.

        Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine
        whether the occurrence of such events constitutes actionable intelligence.

        Below is an implementation of the Symmetric CUSUM filter.

        :param raw_price: (series) of close prices.
        :param threshold: (float) when the abs(change) is larger than the threshold, the
        function captures it as an event.
        :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
        """
        print('Applying Symmetric CUSUM filter.')

        t_events = []
        s_pos = 0
        s_neg = 0

        # log returns
        diff = np.log(raw_price).diff().dropna()

        # Get event time stamps for the entire series
        for i in tqdm(diff.index[1:]):
            # ToDo: I don't understand why this try catch is needed?
            try:
                pos, neg = float(s_pos + diff.loc[i]), float(s_neg + diff.loc[i])
            except Exception as error:
                print(error)
                print(s_pos + diff.loc[i], type(s_pos + diff.loc[i]))
                print(s_neg + diff.loc[i], type(s_neg + diff.loc[i]))
                break

            s_pos = max(0., pos)
            s_neg = min(0., neg)

            if s_neg < -threshold:
                s_neg = 0
                t_events.append(i)

            elif s_pos > threshold:
                s_pos = 0
                t_events.append(i)

        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps

    @staticmethod
    def add_vertical_barrier(t_events, close, num_days=1):
        """
        Snippet 3.4 page 49, Adding a Vertical Barrier

        For each index in t_events, it finds the timestamp of the next price bar at or immediately after
        a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

        This function creates a series that has all the timestamps of when the vertical barrier is reached.

        :param t_events: (series) series of events (symmetric CUSUM filter)
        :param close: (series) close prices
        :param num_days: (int) maximum number of days a trade can be active
        :return: (series) timestamps of vertical barriers
        """
        t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
        t1 = t1[t1 < close.shape[0]]
        t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at end
        return t1

    @staticmethod
    def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
        """ This function applies the triple-barrier labeling method. It works
        on a set of datetime index values (molecule).  This allows the program
        to parallelize the processing.

        Snippet 3.2 page 45
        :param close: (series) close prices
        :param events: (series) of indices that signify "events" (see get_t_events function
        for more details)
        :param pt_sl: (array) element 0, indicates the profit taking level; element 1 is stop loss level
        :param molecule: (an array) a set of datetime index values for processing
        :return:
        """
        # apply stop loss/profit taking, if it takes place before t1 (end of event)
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)
        if pt_sl[0] > 0:
            pt = pt_sl[0] * events_['trgt']
        else:
            pt = pd.Series(index=events.index)  # NaNs
        if pt_sl[1] > 0:
            sl = -pt_sl[1] * events_['trgt']
        else:
            sl = pd.Series(index=events.index)  # NaNs
        for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
            df0 = close[loc:t1]  # path prices
            df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
            out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
            out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking
        return out

    @staticmethod
    def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False, side=None):
        """
        Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels

        This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

        :param close: (series) Close prices
        :param t_events: (series) of t_events. These are timestamps that will seed every triple barrier.
            These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
            Eg: CUSUM Filter
        :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level.
            A non-negative float that sets the width of the two barriers. A 0 value means that the respective
            horizontal barrier (profit taking and/or stop loss) will be disabled.
        :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width
            of the barrier. In this program this is daily volatility series.
        :param min_ret: (float) The minimum target return required for running a triple barrier search.
        :param num_threads: (int) The number of threads concurrently used by the function.
        :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
            We pass a False when we want to disable vertical barriers.
        :param side: (series) Side of the bet (long/short) as decided by the primary model
        :return: (data frame) of events
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
        """

        # 1) Get target
        target = target.loc[t_events]
        target = target[target > min_ret]  # min_ret

        # 2) Get vertical barrier (max holding period)
        if vertical_barrier_times is False:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

        # 3) Form events object, apply stop loss on vertical barrier
        if side is None:
            # side_, pt_sl_ = pd.Series(1., index=target.index), [pt_sl[0], pt_sl[0]]
            side_ = pd.Series(1., index=target.index)
            pt_sl_ = [pt_sl[0], pt_sl[0]]
        else:
            # side_, pt_sl_ = side.loc[target.index], pt_sl[:2]
            side_ = side.loc[target.index]
            pt_sl_ = pt_sl[:2]

        events = (pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
                  .dropna(subset=['trgt']))

        # Apply Triple Barrier
        df0 = MultiProcessingFunctions.mp_pandas_obj(func=CoreFunctions.apply_pt_sl_on_t1,
                                                     pd_obj=('molecule', events.index),
                                                     num_threads=num_threads,
                                                     close=close,
                                                     events=events,
                                                     pt_sl=pt_sl_)

        events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan

        if side is None:
            events = events.drop('side', axis=1)

        return events

    @staticmethod
    def get_bins(events, close):
        """
        Snippet 3.7, page 51, Labeling for Side & Size with Meta Labels

        Compute event's outcome (including side information, if provided).
        events is a DataFrame where:

        Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
        a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
        The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
        to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

        :param events: (data frame)
                    -events.index is event's starttime
                    -events['t1'] is event's endtime
                    -events['trgt'] is event's target
                    -events['side'] (optional) implies the algo's position side
                    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
        :param close: (series) close prices
        :return: (data frame) of meta-labeled events
        """
        # Todo: This function doesnt match the book. Whats up with the commented out lines?

        # 1) prices aligned with events
        events_ = events.dropna(subset=['t1'])
        # px=events_.index.union(events_['t1'].values).drop_duplicates()

        # start of change
        px = events_.index.union(events_['t1'].values)
        px = px.drop_duplicates()
        # end of change

        px = close.reindex(px, method='bfill')
        # 2) create out object
        out = pd.DataFrame(index=events_.index)
        out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

        if 'side' in events_:
            out['ret'] *= events_['side']  # meta-labeling

        # out['bin']=np.sign(out['ret'])
        out['bin'] = [1 if out['ret'].iloc[i] >= 0. else -1 for i in np.arange(len(out))]

        if 'side' in events_:
            out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
        return out

    @staticmethod
    def drop_labels(events, min_pct=.05):
        """ The function recursively eliminates rare observations.

        Snippet 3.8 page 54
        :param events: (data frame) events
        :param min_pct: (float) a fraction used to decide if the observation occurs less than
        that fraction
        :return: (data frame) of events
        """
        # apply weights, drop labels with insufficient examples
        while True:
            df0 = events['bin'].value_counts(normalize=True)

            if df0.min() > min_pct or df0.shape[0] < 3:
                break

            print('dropped label: ', df0.argmin(), df0.min())
            events = events[events['bin'] != df0.argmin()]

        return events
