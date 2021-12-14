import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
df = pd.read_pickle('datas.pkl')

print(df)

plt.plot(df.index,df["open"])
plt.show()