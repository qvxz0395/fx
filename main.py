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

lot = 100000
slippage_pips = 1
spread_pips =0#.002# 0.2銭 楽天

def SMA(values,n):# n: hours
	return pd.Series(values).rolling(int(n*60/dataPeriod)).mean()

# Openの価格で常に取引

class SmaCross(Strategy): # 今回はサンプルとして良く採用される単純移動平均線（SMA）の交差を売買ルールに。
	n1= 24*10 #hours
	n2  = 24*30 #hours
	def init(self): # 初期設定（移動平均線などの値を決める）
		price = self.data.Close
		self.ma1 = self.I(SMA, price, self.n1) # 短期の移動平均線
		self.ma2 = self.I(SMA, price, self.n2) # 長期の移動平均線

	def next(self): # ヒストリカルデータの行ごとに呼び出される処理
		if crossover(self.ma1, self.ma2): # ma1がma2を上回った時（つまりゴールデンクロス）
			self.buy(size=1) # 買い
		elif crossover(self.ma2, self.ma1): # ma1がma2を下回った時（つまりデッドクロス）
			self.sell(size=1) # 売り

bt = Backtest(
	data,
	SmaCross,
	cash=lot,
	commission=spread_pips,
	margin=1,
	exclusive_orders=True)

stats = bt.run() # バックテストを実行
# bt.plot()
plt.hist(stats["_trades"]["PnL"])#損益ヒストグラム
print(len(stats["_trades"]["PnL"]))
print("shapiro normal dist =",stat.shapiro(stats["_trades"]["PnL"]))
plt.show()
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
plt.hist(status_skopt["_trades"]["Size"])
plt.show()
# simuparams = str()

status_skopt["_trades"].to_csv("n1:",str(heatmap.sort_values().iloc[-1,0])+"_n2:",str(heatmap.sort_values().iloc[-1,1]),"_SmaCross.csv")
# bt.plot()

'''