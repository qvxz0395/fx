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
spread_pips =.002# 0.2銭 楽天

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
	y = data.Close.pct_change(48).shift(-48*7)  # Returns after roughly one days
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
	price_delta = .004  # 0.4%　ロスカット

	def init(self):        
		# Init our model, a kNN classifier
		self.clf = KNeighborsClassifier(9)

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


bt = Backtest(data[-10000:], MLTrainOnceStrategy, commission=0, margin=1)
stats = bt.run()
print(stats)
plt.hist(stats["_trades"]["ReturnPct"])#損益ヒストグラム
plt.show()
print("shapiro =",stat.shapiro(stats["_trades"]["ReturnPct"]))
bt.plot()
class SmaCross(Strategy): # 今回はサンプルとして良く採用される単純移動平均線（SMA）の交差を売買ルールに。
	n1= 250 #hours
	n2  = 2668 #hours
	def init(self): # 初期設定（移動平均線などの値を決める）
		price = self.data.Close
		self.ma1 = self.I(SMA, price, self.n1) # 短期の移動平均線
		self.ma2 = self.I(SMA, price, self.n2) # 長期の移動平均線

	def next(self): # ヒストリカルデータの行ごとに呼び出される処理
		if crossover(self.ma1, self.ma2): # ma1がma2を上回った時（つまりゴールデンクロス）
			self.buy() # 買い
		elif crossover(self.ma2, self.ma1): # ma1がma2を下回った時（つまりデッドクロス）
			self.sell() # 売り

bt = Backtest(
	data,
	SmaCross,
	cash=lot,
	commission=spread_pips,
	margin=1,
	exclusive_orders=True)

stats = bt.run() # バックテストを実行
print(stats)
# bt.plot()
# PnL: ポートフォリオの価値((トレード損益)+(今回のポートフォリオ価値ー前回のポートフォリオ価値))
# トレード損益：ポジションの決済金額ーポジション構築金額
# ReturnPct: 損益率.これを検定するべき
plt.hist(stats["_trades"]["ReturnPct"])#損益ヒストグラム
print(stats["_trades"])
print("shapiro normal dist =",stat.shapiro(stats["_trades"]["ReturnPct"]))
plt.show()
stats["_trades"].to_csv("test.csv")
'''
# print(stats) # バックテストの結果を表示
periods = dict({	"n1min":1,
				"n1max":24*30*2,
				"n2min":5,
				"n2max":24*30*6
				})


status_skopt, heatmap, optimize_result = bt.optimize(
	n1=[periods["n1min"],periods["n1max"]],
	n2=[periods["n2min"],periods["n2max"]],
	maximize="Equity Final [$]", 
	constraint=lambda p: p.n1 < p.n2,
	method="skopt",
	max_tries=200,
	return_heatmap=True,
	return_optimization=True) # 最適化バックテスト実行

print(heatmap.sort_values().iloc[-3:])
print(status_skopt["_trades"])
# display heatmap
# _ = plot_objective(optimize_result, n_points=10)
plt.hist(status_skopt["_trades"]["PnL"])
plt.show()
print("shapiro normal dist =",stat.shapiro(status_skopt["_trades"]["PnL"]))
# simuparams = str()

status_skopt["_trades"].to_csv("n1:",str(heatmap.sort_values().iloc[-1,0])+"_n2:",str(heatmap.sort_values().iloc[-1,1]),"_SmaCross.csv")
# bt.plot()

'''