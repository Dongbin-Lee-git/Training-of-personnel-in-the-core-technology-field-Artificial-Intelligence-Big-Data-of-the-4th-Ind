# 논리 연산 결과를 이용한 배열 원소 선택
import numpy as np

# 아래 예에서 배열 names의 원소는 모두 7개이며, 각 원소는 배열 data의 행(row)에 대응된다고 가정.

names = np.array(['Kim', 'Lee', 'Park', 'Kim', 'Park', 'Lee', 'Lee'])
data = np.random.randn(7,4)
names
data

# 배열 data에서 'Kim'에 대응되는 행(row)인 0, 3을 찾고 싶다. 'Lee'는 행 1,5,6, 'Park'은 행 2,4.

names == 'Kim'
data[names == 'Kim']
data[names == 'Kim', 2:]
#data[names == 'Kim', 3]
names != 'Kim'
#~(names == 'Kim')
data[~(names == 'Kim')]
cond = names == 'Kim'
data[~cond]

# 논리 AND('&') 및 논리 OR('|') 연산

mask = (names == 'Kim') | (names == 'Park')
mask
data[mask]
data[data<0] = 0  # 모든 음수 값을 0으로 바꿈
data
data[names != 'Lee'] = 7
data

