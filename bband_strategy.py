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
	hlc3 = (data.High + data.Low + data.Close) / 3
	return pd.Series(hlc3).rolling(int(n*60/dataPeriod)).mean()

def BBANDS(data, n_lookback, n_std):
	"""Bollinger bands indicator"""
	hlc3 = (data.High + data.Low + data.Close) / 3

	mean, std = hlc3.rolling(int(n_lookback*60/dataPeriod)).mean(), hlc3.rolling(int(n_lookback*60/dataPeriod)).std()
	upper = mean + n_std*std
	lower = mean - n_std*std
	return upper, lower

class BBand_forward_Strategy(Strategy):
	n_lookback = 24
	n_std1 = 1
	n_std2 = 2

	def init(self):
		price = self.data.df
		self.upper1,self.lower1 = self.I(BBANDS,price,self.n_lookback,self.n_std1)
		self.upper2,self.lower2 = self.I(BBANDS,price,self.n_lookback,self.n_std2)
	
	def next(self):
		if self.position:
			if self.position.is_long:
				if crossover(self.upper1, self.data.Close):# 利確 over 2 sigma
					self.position.close()
			if self.position.is_short:
				if crossover(self.data.Close, self.lower1):# 利確
					self.position.close()
		else:
			if crossover(self.data.Close, self.upper2):# long signal over 2 sigma
				self.buy()
			elif crossover(self.lower2, self.data.Close):# short signal lower 2 sigma
				self.sell()

bt = Backtest(
	data,
	BBand_forward_Strategy,
	cash=lot,
	commission=spread_pips,
	margin=1,
	exclusive_orders=True)
'''
stats = bt.run() # バックテストを実行
print(stats)
bt.plot()
# PnL: ポートフォリオの価値((トレード損益)+(今回のポートフォリオ価値ー前回のポートフォリオ価値))
# トレード損益：ポジションの決済金額ーポジション構築金額
# ReturnPct: 損益率.これを検定するべき
plt.hist(stats["_trades"]["ReturnPct"])#損益ヒストグラム
print(stats["_trades"])
print("shapiro normal dist =",stat.shapiro(stats["_trades"]["ReturnPct"]))
plt.show()

'''

stats,heatmap = bt.optimize(
	n_lookback=range(24*1,24*30,5),
	maximize='Equity Final [$]',
	max_tries=200,
	random_state=0,
	return_heatmap=True)
print(stats)
print(heatmap.sort_values().iloc[-3:])
plt.hist(stats["_trades"]["ReturnPct"])#損益ヒストグラム
print(stats["_trades"])
print("BBand順張りの損益率のシャピロウィルクテストP値 =",stat.shapiro(stats["_trades"]["ReturnPct"]))
print("BBand順張りの損益率のウィルコクソンの順位和検定P値 =",stat.wilcoxon(stats["_trades"]["ReturnPct"]))
plt.show()
