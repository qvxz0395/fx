import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('datas.pkl')# read data
dataPeriod = 30# minute

# calc moving average
meanPeriod = [5,25,75]# day
for i in meanPeriod:
	df["mean"+str(i)] = df["open"].rolling(int(i*24*60/dataPeriod),center=True).mean()



plt.plot(df.index,df["open"],label = "open")
for i in meanPeriod:
	plt.plot(df.index,df["mean"+str(i)],label ="mean"+str(i))
plt.legend()
plt.show()