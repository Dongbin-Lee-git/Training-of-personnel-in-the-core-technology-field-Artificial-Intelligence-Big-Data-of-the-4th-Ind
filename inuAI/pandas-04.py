import numpy as np
import pandas as pd

# Pandas 객체의 indexing은 python 배열과 달리 정수일 필요는 없음. 다양한 형태의 indexing이 가능.

obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj
obj['b']
obj[1]
obj[['b', 'a', 'd']]
#obj[[1, 0, 3]]
obj[obj < 2]

# 레이블을 사용한 index slicing의 경우 python과 달리 endpoint를 포함. 숫자 index slicing의 경우 endpoint 제외.
obj['b':'c']
obj[1:3]

# DataFrame 객체의 경우 indexing을 통해 한 개 이상의 column 값을 가져올 수 있음.

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio', 'Colorado', 'Utah', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data
data['two']

# 2개 이상 컬럼을 선택할 경우 대괄호('[ ]')로 묶어 리스트로 표시해야 함.

data[['three', 'one']]

# 정수 slicing으로 행을 선택할 경우 endpoint는 제외. boolean 배열로도 행을 선택할 수 있음.

data[:2]
data[data['three']>5] #컬럼 'three'의 값을 기준으로 행을 선택
data < 5 #Boolean DataFrame을 생성하여 True인 항목만 선택
data[data < 5] = 0
data

#  loc과 iloc 으로 선택

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio', 'Colorado', 'Utah', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data

# loc은 이름으로 선택할 때, iloc을 정수 index로 선택할 때 사용

data.loc['Colorado', ['two', 'three']]
#data.iloc[1, [1,2]]
#data.iat[1, 1] #배열 인덱스는 사용할 수 없음
#data.iat[1, 2]
data.iloc[2, [3, 0, 1]]
data.iloc[2]
data.iloc[[1,2], [3,0,1]]

# loc, iloc은 slicing 등 다양한 indexing도 사용 가능

data.loc[:'Utah', 'two']
data

# 필터링을 통해 선택된 항목들에 대해 다시 조건식을 적용할 수 있음. 아래 셀에 한 줄짜리 코드를 두 줄로 나눈 코드를 보면 적용 과정을 쉽게 이해할 수 있음. 대개 한 줄짜리 코드보다는 디버깅이 쉽도록 두 줄로 나눠 작성.

data.iloc[:,:3][data.three > 5]
tmp_df = data.iloc[:,:3]
tmp_df[tmp_df.three > 5]

