import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_csv('car_data.csv')
# ax = df.plot(x='Make',y='city mpg',color='DarkBlue',label='City MPG')
# df.plot(x='Make',y='highway MPG',color='Red',label='Highway MPG',ax=ax)


# # plt.show()
# plt.savefig('./mpg.jpg')


# print(df[df['Make'].isin(['Ferrari','BMW'])]['MSRP'])
# print(df[df['Make'].isin(['Ferrari','BMW'])])
print(df.describe())