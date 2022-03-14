import numpy as np
import scipy as sp
import scipy.stats as stats
from matplotlib import pyplot as plt
 
height = np.array([
    51.0, 45.9, 48.8, 54.0, 53.5,
    48.0, 44.5, 46.0, 50.3, 48.0
])
 
# 検定統計量
n = len(height)                            # サンプル数
k = n -1                                   # 自由度
u_var = np.var(height, ddof=1)             # 不偏分散
statistical_sample_mean = np.mean(height)  # 標本平均
print("標本平均:",statistical_sample_mean)
# 仮説
statistical_population_mean = 50.2         # 母平均
 
t, p = stats.ttest_1samp(height, popmean=statistical_population_mean)
 
print("母平均が{0}のt値：{1}".format(statistical_population_mean, str(t)))
print("母平均が{0}である確率(p値)：{1}".format(statistical_population_mean, str(p)))
 
# グラフ描画
x = np.linspace(-4, 4, 200)
flg, ax = plt.subplots(1, 1)
 
# t分布を描画
ax.plot(x, stats.t.pdf(x, k), linestyle="-", label="k="+str(k))
 
# t分布に今回の確率分布を表示させる
ax.plot(t, p, "x", color="red", markersize=7, markeredgewidth=2, alpha=0.8, label="experiment")
 
# t分布の95%信頼区間から外れた領域を描画する
bottom, up = stats.norm.interval(alpha=0.95, loc=0, scale=1)
plt.fill_between(x, stats.t.pdf(x, k), 0, where=(x>=up)|(x<=bottom), facecolor="black", alpha=0.1)
 
plt.xlim(-6, 6)
plt.ylim(0, 0.4)
 
plt.legend()
plt.show()