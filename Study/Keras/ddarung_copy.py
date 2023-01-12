import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

path = './_data/ddarung/'    

train = pd.read_csv(path + 'train.csv', index_col=0)  
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)  # 답안지

train.info() # 결측치 확인

# 변수 간 상관관계 확인
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), annot=True, fmt='.2f')  # annot_kws={'size':10, }, cmap = 'rainbow'

train.isna().sum()
train[train['hour_bef_temperature'].isna()] # 결측치 위치 확인
# 0시, 18시인데 두 시간대의 온도는 매우 다를 것이므로 온도 컬럼 전체의 중앙값/평균값 같은 걸로 대체하는건 무리가 있다고 판단.


# 보간법
train.interpolate(inplace=True)
test.interpolate(inplace=True)


# 시간별 평균값 확인해서 결측값을 메꾸기
train.groupby('hour')['hour_bef_temperature'].mean() 
train['hour_bef_temperature'].fillna({934:14.788136, 1035:20.926667}, inplace=True)
# 저장하기 위해서 inplace 옵션 True

# 마찬가지로 test 결측치 처리. 이 때 결측값을 대체할 값은 train에서 가져온다는 점 주의

model = RandomForestRegressor(criterion = 'mse')
# 따릉이 대회의 평가지표는 RMSE. RMSE는 MSE 평가지표에 루트를 씌운 것이므로 MSE를 평가척도로 정함.

X_train = train.drop(['count'], axis=1)
Y_train = train['count']

model.fit(X_train, Y_train) # 모델 학습
y_predict = model.predict(test)

def RMSE(Y_train, y_predict) :
    return np.sqrt(mean_squared_error(Y_train, y_predict))



#제출
y_submit = model.predict(test)

# to.csv() 를 사용해서 submission_0105.csv를 완성하시오.
submission['count'] = y_submit
submission.to_csv(path + 'submission_01.csv')