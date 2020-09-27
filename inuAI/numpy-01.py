# numpy는 대규모 배열 처리에 효율적. 처리 속도도 python에 비해 10~100배 이상 빠름.
import numpy as np
np.arange(10) #numpy 1차원 배열

range(10) #python 1차원 배열

my_arr = np.arange(1000000)  #numpy 1차원 배열 - 원소 개수 100만개
my_list = list(range(1000000)) #python 1차원 배열

# numpy 기초: n차원 배열(ndarray) - 원소의 type은 같음. ndarray는 같은 형을 갖는 원소로 이루어진 배열.

np.random.randn() # 크기를 지정하지 않으면 1개의 랜덤 숫자(float) 반환.

data = np.random.randn(2,3)  # 2x3 실수 배열, randn :정규분포로부터 랜덤 숫자(float) 생성
data

type(data)
#data.shape  # 몇 차원 배열인가? 
#data.dtype  # 배열의 원소의 type은?
#data.ndim # 몇 차원 배열인가?

data * 10

data + data

# randn의 n은 정규 분포(normal distribution) : 평균-0, 표준편차-1

arr = np.random.randn(100)
arr2 = np.random.randn(1000)
arr3 = np.random.randn(10000)

print(arr.mean(), arr.std())
print(arr2.mean(), arr2.std())
print(arr3.mean(), arr3.std())

# numpy 배열 ndarray 생성  

#array 함수 사용: python의 리스트를 1차원 배열로 변환
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

#ndarray를 생성할 때 type을 추론 -> 7.5 -> 모든 원소의 type은 float64
#ndarray에 속한 모든 원소의 타입은 같아야 함.
arr1.dtype

arr1.shape
#arr1.ndim   #몇 차원 배열인가?

#array 함수 사용: python의 다중 리스트를 2차원 배열로 변환
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2

arr2.shape
#arr2.ndim
#arr2.dtype # ndarray를 생성할 때 type을 추론 -> 모든 원소의 type은 int32

# np.linspace, np.logspace, np.diag함수를 사용한 ndarray 생성 

np.linspace(0, 10, 25) #0~10사이의 값 25개 생성

np.logspace(0,10,10, base=np.e)

np.diag([1,2,3])

# numpy 배열을 생성하기 위한 함수들: ones, zeros, empty, eye, identity 등

np.zeros(10)
np.ones((3,6))
np.empty((2,3,2))
np.eye(3,3)
np.identity(4)

# ndarray 차원 변환 : reshape

arr = np.arange(32).reshape((8,4))
arr

