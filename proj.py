# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 14:37:10 2021

@author: new laptop --2318887
"""
import numpy as np
import matplotlib as plt
import pandas as pd

dataset=pd.read_csv('LifeExpectancyData.csv')

X = dataset.loc[:,dataset.columns.difference(['Life expectancy '],sort=False)].values
y = dataset.iloc[:,19:20].values





















