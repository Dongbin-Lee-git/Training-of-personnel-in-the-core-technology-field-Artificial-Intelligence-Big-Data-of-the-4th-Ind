
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
#한글 폰트 설정
import matplotlib as mpl
mpl.rc('font', family='NanumGothic')

#font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#rc('font', family=font_name)
#plt.rcParams['axes.unicode_minus'] = False

path = "부문별 발생현황(인천_전체_최근5년).xlsx"  # 파일 확장자 xlsx
raw_data = pd.read_excel(path, index_col=[0,1], skiprows=2)
raw_data.head()

print(raw_data.columns)
#print(raw_data.index)
print(raw_data.index.names)

raw_data.info()

#raw_data.shape
raw_data.describe()

category_list = []
for i, cat in enumerate(raw_data.index):
    if i % 3 == 0:
        category_list.append(cat[0]) 
category_list
#category_list = list(set(tmp))
#category_list

tup = raw_data.index[:3]
event_list = []
for i in range(3):
    event_list.append(tup[i][1]) 
event_list

processed_data = raw_data.T # 행과 열을 뒤바꿈. transpose
processed_data.head()

processed_data.columns = processed_data.columns.droplevel(0)
processed_data.columns
processed_data.head()

processed_data.columns = np.arange(27)
processed_data.head()


# 전체 사고: 발생건수, 부상자 수(사망자 수 제외)

total_stat = processed_data[(x for x in processed_data.columns[:3] if x != 1)]
total_stat.columns = [event_list[0], event_list[2]]
total_stat.name = category_list[0]
total_stat.head()
#total_stat.plot.bar()
#total_stat.plot()

last_accident = total_stat.iloc[0, 0]
#last_accident = total_stat.iat[0, 0]

cur_accident = total_stat.iloc[4, 0]
no_accidents = np.abs(cur_accident - last_accident)

ratio_accident = no_accidents/last_accident * 100
print('발생건수 감소 비율 {0:.2f}%, 감소건수 {1:,}건'.format(ratio_accident, no_accidents))

last_injured = total_stat.iat[0, 1]
cur_injured = total_stat.iat[4, 1]
no_injured = np.abs(cur_injured - last_injured)

ratio_injured = no_injured/last_injured * 100
print('부상자 수 감소 비율 {0:.2f}%, 감소자 수 {1:,}명'.format(ratio_injured, no_injured))


# # 유형별 발생건수 

# - 사고건수는 감소 추세. 전체 사고 중 1. 보행자 사고, 2. 사업용 자동차 사고가 차지하는 비중이 높음.
# - 노인사고비율은 해마다 증가 추세.

pd.options.mode.chained_assignment = None

event_df1 = processed_data[(x for x in processed_data.columns[6:] if x % 3 == 0)]
event_df1.columns = category_list[2:]
event_df1.name = event_list[0] #발생 건수

event_df1.loc['평균'] = event_df1.mean(axis=0).values
event_df1


# 평균 항목은 사고유형 비율을 찾기 위한 목적
# 이 셀 이후에는 해당 행을 삭제
event_df1.drop(['평균'], inplace=True) 
event_df1

index_list = processed_data.index
sum_list = event_df1.loc[index_list].sum(axis=1)

rnd_f = lambda x: np.round(x, 1)

event_df1['보행자사고비율'] = rnd_f((event_df1['보행자사고'][index_list]/sum_list) * 100)
event_df1['사업용자동차사고비율'] = rnd_f((event_df1['사업용자동차사고'][index_list]/sum_list) * 100)
event_df1['노인사고비율'] = rnd_f((event_df1['노인사고'][index_list]/sum_list) * 100)
event_df1['음주운전사고비율'] = rnd_f((event_df1['음주운전사고'][index_list]/sum_list) * 100)

event_df1

col_list = ['보행자사고비율', '사업용자동차사고비율', '노인사고비율', '음주운전사고비율']
dfx1 = event_df1[col_list]
dfx1.columns = col_list
dfx1.name = event_list[0] # 발생 건수
#dfx1.plot.bar()
dfx1.plot()

col_list = ['보행자사고비율', '사업용자동차사고비율', '노인사고비율', '음주운전사고비율']
labels = ['passenger', 'commercial', 'old ages', 'drunken']
fracs1 = []
for r in event_df1.loc['2018년'][col_list].values:
    fracs1.append(r)
#fracs1    
explode = (0, 0.25, 0, 0)
plt.pie(fracs1, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)
plt.title('type of accidents')
plt.show()


# # 유형별 부상자 수
# 유형별 부상자수 통계: 전체 사고 중 1. 사업용 자동차 사고, 2. 보행자 사고, 3.음주운전 사고 순(발생건수와 비교해 1, 2위 순위가 바뀜).

event_df2 = processed_data[(x for x in processed_data.columns[6:] if x % 3 == 2)]
event_df2.columns = category_list[2:]
event_df2.name = event_list[2] # 부상자 수
index_list = processed_data.index

sum_list = event_df2.loc[index_list].sum(axis=1)

event_df2['사업용자동차부상자비율'] = rnd_f((event_df2['사업용자동차사고'][index_list]/sum_list) * 100)
event_df2['보행자부상자비율'] = rnd_f((event_df2['보행자사고'][index_list]/sum_list) * 100)
event_df2['음주운전사고비율'] = rnd_f((event_df2['음주운전사고'][index_list]/sum_list) * 100)
event_df2

col_list2 = ['사업용자동차부상자비율', '보행자부상자비율', '음주운전사고비율']
dfx2 = event_df2[col_list2]
dfx2.columns = col_list2
dfx2.name = event_list[2] #부상자 수
#dfx2.plot.bar()
dfx2.plot()

# # 유형별 사망자수

# 유형별 사망자수 통계: 전체 사고 중 1. 보행자 사고, 2. 노인사고, 3. 사업용 자동차 사고 사고 순(노인사고의 사망자 수 통계가 높게 나타남).

event_df3 = processed_data[(x for x in processed_data.columns[6:] if x % 3 == 1)]
event_df3.columns = category_list[2:]
event_df3.name = event_list[1] # 사망자 수
index_list = processed_data.index

sum_list = event_df3.loc[index_list].sum(axis=1)

event_df3['보행자사고사망자비율'] = rnd_f((event_df3['보행자사고'][index_list]/sum_list) * 100)
event_df3['노인사고사망자비율'] = rnd_f((event_df3['노인사고'][index_list]/sum_list) * 100)
event_df3['사업용자동차사망자비율'] = rnd_f((event_df3['사업용자동차사고'][index_list]/sum_list) * 100)

event_df3

col_list3 = ['보행자사고사망자비율', '노인사고사망자비율','사업용자동차사망자비율']
dfx3 = event_df3[col_list3]
dfx3.columns = col_list3
dfx3.name = event_list[1] #사망자 수
#dfx3.plot.bar()
dfx3.plot()

# # 전체 사고 발생 건수 예측 : 단일 feature 사용

event_df1 = processed_data[(x for x in processed_data.columns if x % 3 == 0)]
event_df1.columns = category_list
event_df1

#'전체사고'와 '음주운전사고'의 상관관계: 0.925 -> 매우 높은 편 
corr = event_df1[['전체사고', '음주운전사고']].corr(method='pearson')
corr

#'전체사고'와 '이륜차사고'의 상관관계: 0.189 -> 매우 낮은 편 
corr2 = event_df1[['전체사고', '이륜차사고']].corr(method='pearson')
corr2


# '전체사고'와 '음주운전사고'의 높은 상관관계를 추이 예측에 활용해 보자. 즉, 2014~2017년도 '전체사고'와 '음주운전사고' 발생건수를 선형회귀(linear regression)를 적용해 지도학습(supervised learning) 방식으로 학습한다. 2018년 '음주운전사고' 발생건수를 사용해 2018년 '전체사고' 발생건수를 예측하고, 정확도를 알아보자.

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt

# np.c_ : 여러 개의 ndarray를 second axis를 따라 합침.
# 아래 예에서 [1,2,3]
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

x = np.c_[a, b]
print(a.shape, b.shape, x.shape)


# 선형회귀 모델링을 만들 때, train dataset(X)은 2차원 이상이어야 함. 단, label(y)은 1차원이어도 됨. 아래 예는 X가 1차원이기 때문에 에러 발생.

X = np.arange(10)
y = np.arange(10)
lr = linear_model.LinearRegression()
#lr.fit(x, y)

print('before', X.ndim)
#이를 위해 numpy의 reshape(-1,1) 메소드를 적용해 2차원 배열(ndarray)로 변환. 
#''-1'은 차원을 잘 모르니 numpy가 알아서 정하라는 뜻. 
# 두 번째 파라미터 1은 label(y) 차원과 일치시키기 위해 지정.

X = X.reshape(-1,1)
print('after', X.ndim)

z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])
#z.reshape(-1, 1)
z.reshape(-1, 3)


# 기계학습으로 적용하기 위해 수치 데이터(numpy의 ndarray)로 변환해야 함.

Xsample = np.c_[event_df1['음주운전사고'][:4]]  #ndarray로 변환
ysample = np.c_[event_df1['전체사고'][:4]]     #ndarray로 변환
type(ysample)

#선형회귀 모델
lr = linear_model.LinearRegression()
model = lr.fit(Xsample, ysample)

# 학습한 coefficient 출력 
print(lr.intercept_[0], lr.coef_[0][0])

#실제 전체 사고발생건수는 7,632건 : 절대오차 = |7673-7632| = 41
drunk_data = np.c_[event_df1.loc['2018년']['음주운전사고']]
y_predict = lr.predict(drunk_data)

#drunk_data = event_df1.loc['2018년']['음주운전사고']
#y_predict = lr.predict(drunk_data.reshape(-1,1))

y_predict[0][0]

#선형회귀 모델 평가
print(model.score(Xsample, ysample)) 

#RMSE(root mean square error) - 오차의 합이기 때문에 값이 작을수록 좋음. 
y_predict = lr.predict(Xsample)
print(sqrt(mean_squared_error(ysample, y_predict)))

# # 전체 사고 발생 건수 예측 : 다중 feature 사용
# 한 개의 feature('음주운전사고')만을 사용해 '전체사고' 발생건수를 예측했다. 이번에는 2개 이상의 feature를 고려할 경우 '전체사고' 발생건수 예측 정확도가 어떻게 달라지는지 살펴보자.

import seaborn as sns

# feature 간의 상관계수 행렬
corr = event_df1[category_list].corr(method='pearson')
corr
#전체사고는 사업용자동차사고 0.994, 보행자사고 0.991, 사망사고 0.950, 음주운전사고 0.925와 상관관계가 높음.
#어린이사고는 음주운전사고 0.973, 자전거사고 0.954, 보행자사고 0.912와 상관관계가 높음.

show_cols = ['total', 'fatal', 'child', 'elderly', 'pedestrian', 
             'bicycle', 'motorcycle', 'commercial', 'drunk']

# corr 행렬 heapmap 출력
plt.rcParams['figure.figsize']=[12,8]
plt.rc('font', family='NanumGothic')
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values,
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=show_cols,
            xticklabels=show_cols)

plt.tight_layout()
plt.show()

# 상관관계 분석을 통해 음주운전사고 외에 사업용자동차사고, 보행자사고, 사망사고 등 3개의 feature를 더 포함

col_list = ['음주운전사고', '사망사고', '보행자사고', '사업용자동차사고']
Xsample2 = np.c_[event_df1[col_list][:4]]  #ndarray로 변환
ysample2 = np.c_[event_df1['전체사고'][:4]] #ndarray로 변환

ysample2

#선형회귀 모델
lr2 = linear_model.LinearRegression()
model2 = lr2.fit(Xsample2, ysample2)

#실제 전체 사고발생건수는 7,632건 : : 절대오차 = |7906-7632| = 274로 크게 증가
event4_data = np.c_[event_df1.loc['2018년'][col_list]]

y_predict = lr2.predict(event4_data.reshape(-1,4))
y_predict[0][0]

# # 전체 사고 발생 건수 예측 : 다중 feature 사용 + 스케일링

# 4개의 feature간 값의 범위가 차이가 있기 때문. 사망사고 발생건수는 103-144건 이지만,
# 사업용자동차사고 건수는 1,701-2,118건 사이로 큼.
# 이렇게 학습에 반영할 feature의 값의 크기가 서로 다르면,
# 학습에 미치는 영향이 편향(bias, 큰 값을 갖는 feature의 영향이 더 크게 반영)될 수 있음.
# 이를 해결하는 방법은 feature의 값의 범위를 scaling하는 것이 필요.
# 대표적인 scaling 방식은 데이터 샘플에서 평균만큼 빼고, 이를 다시 표준편차로 나누는 방식.

event_df1 = processed_data[(x for x in processed_data.columns if x % 3 == 0)]
event_df1.columns = category_list
event_df1

def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df

scaled_event = standard_scaling(event_df1, event_df1.columns[1:]) #전체 사고는 예측값이기 때문에 스케일링 대상에서 제외
scaled_event

col_list = ['음주운전사고', '사망사고', '보행자사고', '사업용자동차사고']
Xsample3 = np.c_[scaled_event[col_list][:4]]  #ndarray로 변환
ysample3 = np.c_[scaled_event['전체사고'][:4]] #ndarray로 변환

ysample3

#선형회귀 모델
lr3 = linear_model.LinearRegression()
model3 = lr3.fit(Xsample3, ysample3)

#실제 전체 사고발생건수는 7,632건  : 절대오차 = |7661-7632| = 29 (41에서 29로 감소)

event4_data = np.c_[scaled_event.loc['2018년'][col_list]]
event4_data
#drunk_data.shape
y_predict = lr3.predict(event4_data.reshape(-1,4))
y_predict[0][0]