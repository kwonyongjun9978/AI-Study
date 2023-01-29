import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#1. 데이터
path='./_data/stock/' #데이터 위치 표시
df1 = pd.read_csv(path+'아모레퍼시픽 주가.csv', index_col=0,
                  header=0, encoding='cp949', sep=',', thousands=',').loc[::-1]
# print(df1)
# print(df1.shape)
'''
[2220 rows x 16 columns]
(2220, 16)
'''

df2 = pd.read_csv(path+'삼성전자 주가.csv', index_col=0,
                  header=0, encoding='cp949', sep=',',  thousands=',').loc[::-1]
# print(df2)
# print(df2.shape)
'''
[1980 rows x 16 columns]
(1980, 16)
'''

#수치형 데이터로 변경
# for i in range(len(df1.index)): #모든 str -> int 변경
#     for j in range(len(df1.iloc[i])):
#         df1.iloc[i,j] = int(df1.iloc[i,j].replace(',', ''))

# for i in range(len(df2.index)): #모든 str -> int 변경
#     for j in range(len(df2.iloc[i])):
#         df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))      
        
#데이터를 오름차순으로 변경
df1=df1.sort_values(['일자'], ascending=[True])
df2=df2.sort_values(['일자'], ascending=[True])  
print(df1)
print(df2)    

# 아모레 x 추출
amore_x = df1[['고가', '저가', '종가', '등락률', '거래량', '개인', '기관', '외인(수량)', '외국계', '외인비']]
print(amore_x)
print(amore_x.shape) #(2220, 10)

# 삼성전자 x ,y 추출
samsung_x = df2[['고가', '저가', '종가', '등락률', '거래량', '개인', '기관', '외인(수량)', '외국계', '외인비']]
samsung_y = df2[['시가']].to_numpy()
print(samsung_x)
print(samsung_y)
print(samsung_x.shape) #(1980, 10)
print(samsung_y.shape) #(1980, 1)  

samsung_x = MinMaxScaler().fit_transform(samsung_x)
amore_x = MinMaxScaler().fit_transform(amore_x)

def split_data(dataset, timesteps):
    tmp = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        tmp.append(subset)
    return np.array(tmp)
amore_x = split_data(amore_x, 5)
samsung_x = split_data(samsung_x, 5)
print(amore_x.shape) #(2216, 5, 10)
print(samsung_x.shape) #(1976, 5, 10)

samsung_y = samsung_y[4:, :] 
print(samsung_y.shape) #(1976, 1)

# 예측에 사용할 데이터 추출 (마지막 값)
amore_x_predict = amore_x[-1].reshape(-1, 5, 10)
samsung_x_predict = samsung_x[-1].reshape(-1, 5, 10)
print(samsung_x_predict.shape) # (1, 5, 10)
print(amore_x_predict.shape) # (1, 5, 10)

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test, amore_x_train, amore_x_test  = train_test_split(
    samsung_x, samsung_y, amore_x, train_size=0.75, random_state=444)

print(samsung_x_train.shape, samsung_x_test.shape)  # (1383, 5, 5) (593, 5, 5)
print(samsung_y_train.shape, samsung_y_test.shape) # (1383, 1) (593, 1)
print(amore_x_train.shape, amore_x_test.shape)  # (1383, 5, 5) (593, 5, 5)


