import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import funcs as f
import seaborn as sns
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from skopt.plots import plot_objective
import scipy.stats as stat

data = pd.read_pickle('datas.pkl')# read data
dataPeriod = 30# minute

lot = 1000
slippage_pips = 1
spread_pips =.002# 0.2銭(=%) 楽天

def SMA(values,n):# n: hours
	return pd.Series(values).rolling(int(n*60/dataPeriod)).mean()

def BBANDS(data, n_lookback, n_std):
	"""Bollinger bands indicator"""
	hlc3 = (data.High + data.Low + data.Close) / 3
	mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
	upper = mean + n_std*std
	lower = mean - n_std*std
	return upper, lower


close = data.Close.values
sma10 = SMA(data.Close, 10)
sma20 = SMA(data.Close, 20)
sma50 = SMA(data.Close, 50)
sma100 = SMA(data.Close, 100)
upper, lower = BBANDS(data, 20, 2)

# Design matrix / independent features:

# Price-derived features
data['X_SMA10'] = (close - sma10) / close
data['X_SMA20'] = (close - sma20) / close
data['X_SMA50'] = (close - sma50) / close
data['X_SMA100'] = (close - sma100) / close

data['X_DELTA_SMA10'] = (sma10 - sma20) / close
data['X_DELTA_SMA20'] = (sma20 - sma50) / close
data['X_DELTA_SMA50'] = (sma50 - sma100) / close

# Indicator features
data['X_MOM'] = data.Close.pct_change(periods=2)
data['X_BB_upper'] = (upper - close) / close
data['X_BB_lower'] = (lower - close) / close
data['X_BB_width'] = (upper - lower) / close
data['X_Sentiment'] = ~data.index.to_series().between('2017-09-27', '2017-12-14')

# Some datetime features for good measure
data['X_day'] = data.index.dayofweek
data['X_hour'] = data.index.hour

data = data.dropna().astype(float)

def get_X(data):# 特徴量だけを抜き出す
	"""Return model design matrix X"""
	return data.filter(like='X').values


def get_y(data):# 7日前の特徴量が0近傍奈良ゼロ，一定以上なら1，以下なら-1という評価をすべての列に対して実行
	"""Return dependent variable y"""
	y = data.Close.pct_change(48*2).shift(-48*2)  # Returns after roughly one days
	y[y.between(-.004, .004)] = 0             # Devalue returns smaller than 0.4%
	y[y > 0] = 1
	y[y < 0] = -1
	return y


def get_clean_Xy(df):
	"""Return (X, y) cleaned of NaN values"""
	X = get_X(df)
	y = get_y(df).values
	isnan = np.isnan(y)# NaNの行だけTrue
	X = X[~isnan]# NaNじゃない行だけを抽出
	y = y[~isnan]
	return X, y

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Openの価格で常に取引

N_TRAIN = 400


class MLTrainOnceStrategy(Strategy):
	price_delta_int = 4# 0.4%　ロスカット

	def init(self):        
		# Init our model, a kNN classifier
		self.clf = KNeighborsClassifier(9)
		self.price_delta = self.price_delta_int*0.001

		# Train the classifier in advance on the first N_TRAIN examples
		df = self.data.df.iloc[:N_TRAIN]
		X, y = get_clean_Xy(df)
		self.clf.fit(X, y)

		# Plot y for inspection
		self.I(get_y, self.data.df, name='y_true')

		# Prepare empty, all-NaN forecast indicator
		self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

	def next(self):
		# Skip the training, in-sample data
		if len(self.data) < N_TRAIN:
			return

		# Proceed only with out-of-sample data. Prepare some variables
		high, low, close = self.data.High, self.data.Low, self.data.Close
		current_time = self.data.index[-1]

		# Forecast the next movement
		X = get_X(self.data.df.iloc[-1:])
		forecast = self.clf.predict(X)[0]

		# Update the plotted "forecast" indicator
		self.forecasts[-1] = forecast

		# If our forecast is upwards and we don't already hold a long position
		# place a long order for 20% of available account equity. Vice versa for short.
		# Also set target take-profit and stop-loss prices to be one price_delta
		# away from the current closing price.
		upper, lower = close[-1] * (1 + np.r_[1, -1]*self.price_delta) # 現在の価格におけるロスカットの上限下限

		if forecast == 1 and not self.position.is_long:
			self.buy(size=.2, tp=upper, sl=lower)
		elif forecast == -1 and not self.position.is_short:
			self.sell(size=.2, tp=lower, sl=upper)

		# Additionally, set aggressive stop-loss on trades that have been open 
		# for more than two days
		for trade in self.trades:
			if current_time - trade.entry_time > pd.Timedelta('7 days'):
				if trade.is_long:
					trade.sl = max(trade.sl, low)
				else:
					trade.sl = min(trade.sl, high)


bt = Backtest(data, MLTrainOnceStrategy, commission=spread_pips, margin=1)
stats,heatmap = bt.optimize(
	price_delta_int=range(4,120,20),
	max_tries=200,
	random_state=0,
	return_heatmap=True
) # バックテストを実行
print(stats)
print(heatmap.sort_values().iloc[-3:])
# bt.plot()
# PnL: ポートフォリオの価値((トレード損益)+(今回のポートフォリオ価値ー前回のポートフォリオ価値))
# トレード損益：ポジションの決済金額ーポジション構築金額
# ReturnPct: 損益率.これを検定するべき
plt.hist(stats["_trades"]["ReturnPct"])#損益ヒストグラム
print(stats["_trades"])
print("k-means機械学習の損益率のシャピロウィルクテストP値 =",stat.shapiro(stats["_trades"]["ReturnPct"]))
print("k-means機械学習の損益率のウィルコクソンの順位和検定P値 =",stat.wilcoxon(stats["_trades"]["ReturnPct"]))
plt.show()
