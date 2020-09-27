import numpy as np
import pandas as pd

# reindex : index를 새롭게 설정하면, 이 index에 맞게 pandas 객체의 값(values)을 재배치. index에 해당되는 값이 없을 경우 NaN 할당. 

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
obj2 = obj.reindex(['a','b','c','d','e'])
obj2

# ffill : 값을 채워넣어야 할 경우 사용. 누락된 값(missing value)을 이전 항목의 값으로 채움. 

obj2.ffill()
obj3 = pd.Series(['blue','purple','yellow'], index=[0,2,4])
obj3
obj4 = obj3.reindex(range(6)) 
#obj3.reindex(np.arange(6)) - 갯수가 많으면 numpy 사용!
obj4.ffill()
obj3.reindex(range(6), method='ffill')

# DataFrame에 대한 reindexing : 행과 열 모두 기존 index를 새롭게 변경 가능.
# - 행은 리스트 형태로 전달하면 reindexing
# - 열은 columns 속성에 지정해야 reindexing

frame = pd.DataFrame(np.arange(9).reshape((3,3)),
                     index=['a','c','d'],
                     columns=['Ohio','Texas','California'])
frame
new_idx = ['a','b','c','d']
frame2 = frame.reindex(new_idx)
frame2

# ffill을 사용하는 대신 reindex할 때 빈 값을 채우는 옵션을 설정할 수도 있다.

frame22 = frame.reindex(new_idx, fill_value=0)
frame22
frame2.ffill(axis=0)
states = ['Texas','Utah','California']
frame3 = frame.reindex(columns=states)
frame3
frame3.ffill(axis=1)

# drop : 행 또는 열을 삭제한 새로운 pandas 객체를 생성

obj = pd.Series(np.arange(5.), index=['a','b','c','d','e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d','c'])
#obj - 원본 Series 객체는 변경 사항 없음.

# DataFrame은 행과 열에서 값을 삭제할 수 있음.

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio', 'Colorado', 'Utah', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data
data.drop(['Colorado', 'Ohio'])  # 행 삭제. default(axis=0 생략 가능).
#data.drop(['Colorado', 'Ohio'], axis=0)
#data.drop(['Colorado', 'Ohio'], axis='rows')
data.drop('two', axis=1) # 열 삭제
#data.drop('two', axis='columns') # 열 삭제
data.drop(['two', 'four'], axis='columns')

# 행 또는 열을 삭제한 새로운 객체를 만들지 않고 원본 객체를 직접 변경하려면 inplace=True 설정.

obj.drop('c', inplace=True)
obj

