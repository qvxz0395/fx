import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def add_meanPeriod(df,day,dataPeriod=30):
	df["mean"+str(i)] = df["mean"].rolling(int(day*24*60/dataPeriod)).mean()
	return df

def add_bband(df,day,dataPeriod=30):
	df["std"+str(day)] = df["mean"].rolling(int(day*24*60/dataPeriod)).std()# sample standard deviation
	df["bband+2σ"+str(day)] = df["mean"].rolling(int(day*24*60/dataPeriod)).mean() + (2*df["std"+str(day)])# Bollinger Band
	df["bband-2σ"+str(day)] = df["mean"].rolling(int(day*24*60/dataPeriod)).mean() - (2*df["std"+str(day)])# ref: https://www.moneypartners.co.jp/support/tech/bolb.html
	return df

def add_ichimoku(df,dataPeriod=30):# https://www.moneypartners.co.jp/support/tech/ichimoku.html
	df["ITLine"] = (df['mean'].rolling(int(9*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2
	df["IBLine"] = (df['mean'].rolling(int(26*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2
	df["IPSpan1"] = ((df["ITLine"] + df ["IBLine"])/2).shift(int(26*24*60/dataPeriod))
	df["IPSpan2"] = ((df['mean'].rolling(int(52*24*60/dataPeriod)).max()+df['mean'].rolling(int(9*24*60/dataPeriod)).min())/2).shift(int(26*24*60/dataPeriod))
	df["ILSpan"] = df["close"].shift(-1*int(26*24*60/dataPeriod))
	return df