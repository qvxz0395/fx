import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
# http://www.turbare.net/transl/scipy-lecture-notes/intro/scipy.html#statistics-and-random-numbers-scipy-stats
# wilcoxonの符号順位和検定の妥当性　http://lbm.ab.a.u-tokyo.ac.jp/~omori/kensyu/nonpara.htm
a = np.random.normal(size=50)# 正規分布に従う平均0，標準偏差1，サイズ1000の乱数生成
t = np.random.standard_t(16,size = 50)
print("shapiro t dist=",stats.shapiro(t))# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html#scipy.stats.shapiro
print("shapiro normal dist =",stats.shapiro(a))# 棄却域p=0．05以下で正規分布とは言い難い(1-pの確率で正規分布ではない)．それ以上で正規分布かもしれない．
bins = np.arange(-4,5)# 階数を-4~4に
print(bins)
histogram = np.histogram(t,bins=bins,density=True)[0]# 正規分布乱数を階数でヒストグラムに

bins = 0.5*(bins[1:] + bins[:-1])# 階数の中央値
print(bins)
b= stats.norm.pdf(bins)# 理想的な正規分布の値

# stats.probplot(t,plot=plt)# Q-Q plot 直線状で正規分布

# plt.plot(bins,histogram,label = "random_hist")
# plt.plot(bins,b,label="normal_hist")
# rng = np.random.default_rng()
# plt.hist(stats.norm.rvs(loc=5, scale=3, size=1000, random_state=rng))
# plt.hist(a)
# plt.legend()
# plt.show()

# 正規分布を仮定しない検定(Wilcoxonの符号順位和検定，母集団が対称な分布であることを仮定している．棄却率はT検定とほぼ遜色ない)

print("wilcoxon t =",stats.wilcoxon(a-0.21))

