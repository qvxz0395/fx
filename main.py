import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import funcs as f
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

data = pd.read_pickle('datas.pkl')# read data
dataPeriod = 30# minute

# sankou https://kabu-fx-frontier.com/2020/09/05/pl-simulatino/
lot = 1000
pips_yen = 0.005
slippage_pips = 1
spread_pips =1
cost_pips = slippage_pips+spread_pips

def SMA(values,n):# n: hours
	return pd.Series(values).rolling(int(n*60/dataPeriod)).mean()

class SmaCross(Strategy): # 今回はサンプルとして良く採用される単純移動平均線（SMA）の交差を売買ルールに。
	n1= 10 #hours
	n2  = 30 #hours
	def init(self): # 初期設定（移動平均線などの値を決める）
		price = self.data.Close
		self.ma1 = self.I(SMA, price, self.n1) # 短期の移動平均線
		self.ma2 = self.I(SMA, price, self.n2) # 長期の移動平均線

	def next(self): # ヒストリカルデータの行ごとに呼び出される処理
		if crossover(self.ma1, self.ma2): # ma1がma2を上回った時（つまりゴールデンクロス）
			self.buy() # 買い
		elif crossover(self.ma2, self.ma1): # ma1がma2を下回った時（つまりデッドクロス）
			self.position.close() # 売り

bt = Backtest(
	data,
	SmaCross,
	cash=lot,
	commission=pips_yen,
	margin=1,
	exclusive_orders=True)

# stats = bt.run() # バックテストを実行
# print(stats) # バックテストの結果を表示
periods = dict({	"n1min":1,
				"n1max":13,
				"n1step":3,
				"n2min":1,
				"n2max":25,
				"n2step":3
				})

r_n1 = range(periods["n1min"],periods["n1max"],periods["n1step"])
r_n2 = range(periods["n2min"],periods["n2max"],periods["n2step"])
output2 = bt.optimize(n1=r_n1,n2=r_n2,maximize="Equity Final [$]", constraint=lambda p: p.n1 < p.n2) # 最適化バックテスト実行

print(output2)
print("test range n1=",r_n1, "n2=", r_n2)
print("best param =",output2["_strategy"])


simuparams = str()
for period in periods.values(): simuparams += str(period).zfill(4)
simuparams += str(output2["_strategy"])

output2['_trades'].to_csv(simuparams+".csv")
# bt.plot()
