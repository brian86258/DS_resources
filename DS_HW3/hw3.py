import pandas as pd
import numpy as np

raw_data=pd.read_csv('Automobile_data.csv')

# 如果要用map 一定要把所有變數範圍都包含
# raw_data['normalized-losses']=raw_data['normalized-losses'].replace('?',np.NaN)
# raw_data=raw_data.replace('?',np.NaN)
# print(raw_data.head())
# print(raw_data.info(max_cols=5))

wrong=raw_data[raw_data['normalized-losses']=='?']
empty=raw_data[raw_data['normalized-losses'].isnull()]
print(wrong.index)
print(empty.index)
# print(wrong)
wrong=raw_data[raw_data['num-of-cylinders']=='?']
empty=raw_data[raw_data['num-of-cylinders'].isnull()]
print(wrong.index)
print(empty.index)
# print(raw_data['num-of-cylinders'])

# null_columns=raw_data.columns[raw_data.isnull().any()]
# print(null_columns)
# print(raw_data[null_columns].isnull().sum())
# #  NaN ,'', 以python來看 都是 null
# empty=raw_data[raw_data['curb-weight'].isnull()]
# print(empty['curb-weight'])