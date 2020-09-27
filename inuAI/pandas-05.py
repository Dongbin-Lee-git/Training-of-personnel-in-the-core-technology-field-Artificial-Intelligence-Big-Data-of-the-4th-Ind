import pandas as pd
import numpy as np

# Pandas 객체간 산술 연산을 할 때, 일치하지 않는 index가 있으면 이 index는 연산 결과에 합쳐진다.
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], 
               index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], 
               index=['a', 'c', 'e', 'f', 'g'])
s1
s2

# - a,c,e는 2개 객체에 공통 index이므로 덧셈 연산 실행.
# - d,f,g가 각 객체에 독립 index이므로 연산 결과는 NaN.

s1 + s2
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)),
                   columns=list('bcd'), 
                   index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                   columns=list('bde'), 
                   index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df3 = df1+df2
df3

# 일치하지 않는 index를 갖는 pandas 객체 연산에서 연산 후 ffill을 사용해 값을 채워넣는 건 바람직하지 않음. 연산하기 전에 초기값을 설정하는 것이 바람직.

df3.ffill(axis=1)
df3 = df1.add(df2, fill_value=0)
df3
df3.ffill(axis=1)

# 서로 다른 index를 갖는 객체 간 산술 연산 결과 NaN 대신 특정 값을 지정하고자 할 경우

df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                  columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                  columns=list('abcde'))
df2
df2.loc[1, 'b'] = np.nan  # 1행, column='b'의 값을 NaN으로 변경
df2
df1
df1 + df2
df1.add(df2, fill_value=0) 

# 산술 연산 메소드는 피연산자(operand)의 순서를 바꾸는(reverse) 연산이 존재. div <-> rdiv. add <-> radd, sub <-> rsub, mul <-> rmul, floordiv <-> rfloordiv, pow <-> rpow 등. 아래 연산 결과에서 1/0와 같은 연산 결과는 inf로 표시. inf는 infinte(연산 불능).

1 / df1
df1.rdiv(1)  # 1/df1 과 같은 연산.
#df1.div(1) # df/1 은 div, 1/df는 rdiv 메소드를 사용

# 함수, lambda, sorting, rank

frame = pd.DataFrame(np.random.randn(4, 3),
                columns=list('bde'), 
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame

# Pandas 객체에도 numpy의 universal function을 적용할 수 있다.
#np.abs(frame)
np.round(np.abs(frame), 3)

# 간단한 기능은 lambda 함수를 사용해 행 또는 열에 적용.

f = lambda x: x.max()-x.min()
frame.apply(f) # 각 열에 적용(default)
#frame.apply(f, axis='rows')
frame.apply(f, axis='columns') # 각 행에 적용
frame.sum()  # sum이나 mean 같은 일반 함수는 apply 메소드를 사용하지 않아도 됨.
#frame.sum(axis='columns')

# apply 메소드에는 scalar 값(한 가지 값) 대신, Series 객체를 전달할 수 있다. 이 경우 lambda 대신 함수를 정의해야 한다.

def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)
#frame.apply(f, axis='columns')


# python 함수를 Pandas 객체의 각 원소에 적용할 수 있다. 이 경우 apply 대신 applymap을 사용한다.

format = lambda x: '%.2f' % x  # 소수점 2째자리까지 출력
frame.applymap(format)
frame['e'].map(format)

# sort_index(axis) - index(axis=0, default) 또는 column(axis=1)을 기준으로 행 또는 열을 정렬.
# 
# sort_values() - 값을 기준으로 정렬. 2차 행렬은 정렬 기준이 되는 column을 지정할 수 있음.

obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj
obj.sort_index()
obj.sort_values()
frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                     index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame
frame.sort_index() # index를 기준으로 정렬
#frame.sort_index(axis=0)

frame.sort_index(axis=1) # column 이름을 기준으로 정렬
frame.sort_index(axis=1, ascending=False) # column을 내림차순 정렬
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values() # NaN은 내림차순, 오름차순 상관없이 마지막에 배치.

#obj.sort_values(ascending=False) 

frame = pd.DataFrame({'b':[4, 7, -3, 2], 'a':[0, 1, 0, 1]})
frame
#frame.sort_values(by='a') # 2차 행렬일 때 정렬 기준인 by 옵션을 반드시 설정해야 함.
frame.sort_values(by='b')
frame.sort_values(by=['a','b']) # by를 리스트로 설정 가능. 리스트 순에 따라 정렬

# 통계 및 요약

df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                  [np.nan, np.nan], [0.75, -1.3]],
                  index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df


# numpy와 달리 Pandas의 통계 관련 메소드는 NaN을 제외하고 연산을 수행
df.sum()
#df.sum(axis=0)
df.sum(axis='columns')
#df.sum(axis='columns', skipna=False)
df.mean(axis='columns', skipna=False)
df.idxmin()
#df.idxmax()
df.cumsum()
df.describe()
df.quantile()

# unique 및 vlaue_counts 함수

obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])

# 중복되지 않은 값만을 넘파이 배열(ndarray) 객체로 반환. 발생 순서대로 찾은 결과(정렬되지 않음). set 타입으로 변환해도 같은 결과를 얻을 수 있음.

uniques = obj.unique()
uniques
set(obj.values)
obj.value_counts()
pd.value_counts(obj.values, sort=False)
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
                     'Qu2': [2, 3, 1, 2, 3],
                     'Qu3': [1, 5, 2, 4, 4]})
data
result = data.apply(pd.value_counts).fillna(0)
result
result2 = data.apply(pd.value_counts)
result2