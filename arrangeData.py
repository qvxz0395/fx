import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

data = pd.read_csv("USDJPY_M30_201310240000_202112140800.csv",sep="\t")
data = data.rename(columns={"<DATE>":"date","<TIME>":"time","<OPEN>":"Open","<HIGH>":"High","<LOW>":"Low","<CLOSE>":"Close","<TICKVOL>":"Tickvol","<VOL>":"Volume","<SPREAD>":"Spread"}) # rename column name

data["datetime"] = data.apply(lambda row:dt.datetime.strptime(row['date'] +' '+ row['time'], '%Y.%m.%d %H:%M:%S'),axis=1)# add datetime column
data["datetime"] = pd.to_datetime(data["datetime"])
# data["mean"] = data.filter(items = ["open","high","low","close"]).mean(axis = "columns")#add mean column
data = data.drop(columns=["date","time","Tickvol","Spread","Volume"])# delete columns
data = data.set_index("datetime")
print(data)

data.to_pickle('datas.pkl')