# 배열=행렬: 행렬 전치(transpose) 및 행렬 내적(inner product)
import numpy as np

arr = np.arange(15).reshape((3,5))
arr
arr.T

arr = np.random.randn(6,3)
arr

np.dot(arr.T, arr)  # 3x6 dot 6x3 = 3x3
#np.dot(arr, arr.T) # 6x3 dot 3x6 = 6x6

# 배열 원소에 대한 통계 연산(sum, mean, std 등)을 해 보자!

arr = np.random.randn(5,4)
arr
arr.mean()
#np.mean(arr)
arr.sum()

arr.mean(axis=1) # 행(row) 단위로 평균을 구함
arr.sum(axis=0) # 열(column) 단위로 평균을 구함

# 1차 배열의 경우 누적 합(cumsum), 누적 곱(cumprod)은 같은 크기의 배열을 생성.

arr = np.array([1,2,3,4,5,6,7,8])
arr.cumsum()
#arr.cumprod()
arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
arr
arr.cumsum(axis=0)
arr.cumprod(axis=1)

# 이제 numpy의 where 구문을 사용해서 구현해 보자!

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
           for x, y, c in zip(xarr, yarr, cond)]
result
result = np.where(cond, xarr, yarr)
result

arr = np.random.randn(4,4)
arr

arr > 0

np.where(arr>0, 2, -2)

np.where(arr>0, 2, arr) # where 구문의 parameter는 scalar이거나 배열

