import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('datas.pkl')# read data
dataPeriod = 30# minute

# calc moving average
meanPeriod = [5,25,75]# day
for i in meanPeriod:
	df["mean"+str(i)] = df["open"].rolling(int(i*24*60/dataPeriod),center=True).mean()
	df["std"+str(i)] = df["open"].rolling(int(i*24*60/dataPeriod),center=True).std()# sample standard deviation
	df["bband+2σ"+str(i)] = df["mean"+str(i)] + (2*df["std"+str(i)])# Bollinger Band
	df["bband-2σ"+str(i)] = df["mean"+str(i)] - (2*df["std"+str(i)])# ref: https://www.moneypartners.co.jp/support/tech/bolb.html



print(df)

plt.plot(df.index,df["open"],label = "open")
for i in meanPeriod:
	plt.plot(df.index,df["mean"+str(i)],label ="mean"+str(i))
	plt.plot(df.index,df["bband+2σ"+str(i)],label = "bband+2σ"+str(i))
	plt.plot(df.index,df["bband-2σ"+str(i)],label = "bband-2σ"+str(i))
plt.legend()
plt.show()