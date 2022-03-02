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
pips_yen = 0.01
slippage_pips = 1
spread_pips =1
cost_pips = slippage_pips+spread_pips
evalPeriod = 6 # month
windPeriod = 3 # month


def SMA(values,n):
	return pd.Series(values).rolling(int(n*24*60/dataPeriod)).mean()

class SmaCross(Strategy): # 今回はサンプルとして良く採用される単純移動平均線（SMA）の交差を売買ルールに。
	def init(self): # 初期設定（移動平均線などの値を決める）
		price = self.data.Close
		self.ma1 = self.I(SMA, price, 5) # 短期の移動平均線
		self.ma2 = self.I(SMA, price, 25) # 長期の移動平均線

	def next(self): # ヒストリカルデータの行ごとに呼び出される処理
		if crossover(self.ma1, self.ma2): # ma1がma2を上回った時（つまりゴールデンクロス）
			self.buy() # 買い
		elif crossover(self.ma2, self.ma1): # ma1がma2を下回った時（つまりデッドクロス）
			self.sell() # 売り

bt = Backtest(data[int(-1*evalPeriod*30*24*60/dataPeriod):], SmaCross, cash=lot, commission=0, exclusive_orders=True)

stats = bt.run() # バックテストを実行
print(stats) # バックテストの結果を表示

bt.plot()
