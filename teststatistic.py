import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
# http://www.turbare.net/transl/scipy-lecture-notes/intro/scipy.html#statistics-and-random-numbers-scipy-stats
a = np.random.normal(size=1000)# 正規分布に従う平均0，標準偏差1，サイズ1000の乱数生成
bins = np.arange(-4,5)# 階数を-4~4に
print(bins)
histogram = np.histogram(a,bins=bins,density=True)[0]# 正規分布乱数を階数でヒストグラムに

print(histogram)
bins = 0.5*(bins[1:] + bins[:-1])# 階数の中央値
print(bins)
b= stats.norm.pdf(bins)# 理想的な正規分布の値

plt.plot(bins,histogram,label = "random_hist")
plt.plot(bins,b,label="normal_hist")
# plt.hist(a)
plt.legend()
plt.show()
