import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('datas.pkl')# read data
dataPeriod = 30# minute

# calc moving average
meanPeriod = [5,25,75]# day
for i in meanPeriod:
	df["mean"+str(i)] = df["mean"].rolling(int(i*24*60/dataPeriod)).mean()

bbandPeriod = [5,20]
for i in bbandPeriod:
	df["std"+str(i)] = df["mean"].rolling(int(i*24*60/dataPeriod)).std()# sample standard deviation
	df["bband+2σ"+str(i)] = df["mean"].rolling(int(i*24*60/dataPeriod)).mean() + (2*df["std"+str(i)])# Bollinger Band
	df["bband-2σ"+str(i)] = df["mean"].rolling(int(i*24*60/dataPeriod)).mean() - (2*df["std"+str(i)])# ref: https://www.moneypartners.co.jp/support/tech/bolb.html

# Ichimoku cloud
# https://www.moneypartners.co.jp/support/tech/ichimoku.html

df["ITLine"] = (df['mean'].rolling(int(9*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2
df["IBLine"] = (df['mean'].rolling(int(26*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2
df["IPSpan1"] = ((df["ITLine"] + df ["IBLine"])/2).shift(int(26*24*60/dataPeriod))
df["IPSpan2"] = ((df['mean'].rolling(int(52*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2).shift(int(26*24*60/dataPeriod))
df["ILSpan"] = df["close"].shift(-1*int(26*24*60/dataPeriod))

print(df)

#plot Ichimoku cloud
# for i in ["mean","ITLine","IBLine","IPSpan1","IPSpan2","ILSpan"]:
	# plt.plot(df["datetime"],df[i],label = i)

bbandLabels = ["bband+2σ"+str(i) for i in bbandPeriod] + ["bband-2σ"+str(i) for i in bbandPeriod] 
for i in ["mean"]+bbandLabels:
	plt.plot(df["datetime"],df[i],label =i)
plt.legend()
plt.show()