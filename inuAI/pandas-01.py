# # Pandas 자료 구조 : Series

import pandas as pd
import numpy as np
# Series = (1차원 배열의)value + (배열 원소와 연관된)index 
# - index를 지정하지 않으면, 정수 0~N-1이 index로 할당됨. N은 (1차원) 배열의 원소 개수

obj = pd.Series([4, 7, -5, 3])
obj

#len(obj)
obj.values

obj.index 

obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2

# index를 직접 지정하면 RangeIndex 타입이 아니라 Index 타입으로 바뀜

obj2.index
#type(obj2.index)

# 한 개의 index를 지정하면 value 만 가져오지만, 여러 개 index를 지정하면 Series 객체를 반환한다.

obj2['a']
#type(obj2['a'])

obj2[['c', 'a', 'd']]
#type(obj2[['c', 'a', 'd']])

# Series 객체에 산술 연산을 적용해도 값만 바뀔 뿐 index-value 관계는 그대로 유지됨.

obj2
obj2[obj2 > 5]
#type(obj2[obj2 > 0])
obj2 * 2
np.exp(obj2)
#np.round(np.exp(obj2), 2)
'b' in obj2
#'b' in obj2.index
#'e' in obj2
#-5 in obj2.values


# Pandas의 Series 객체는 python의 dictionary 타입과 유사. 
# - python의 dictionary 객체를 사용하여 Pandas의 Series 객체를 생성할 수 있음.
# - dictionary 객체의 key가 Series 객체의 index가 됨.
# - dictionary는 keys(), values()와 같이 메소드를 호출하지만, Series에서는 index, values 속성을 사용

sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}
#sdata.keys()
#sdata.values()
obj3 = pd.Series(sdata)
obj3
#obj3.index
#obj3.values

# Series 객체의 index를 직접 지정하고 싶으면, index 속성을 직접 설정하면 됨.

states = ['California', 'Ohio', 'Oregon', 'Texas']

# Series 객체에 index로 새롭게 추가된 'California'가 원본 dictionary 객체에 정의되지 않았기 때문에, Series 객체의 value는 NaN(Not a Number)로 출력됨.반면 dictionary 객체에 정의된 key 'Utah'는 Series객체의 index에서 빠졌기 때문에 출력에서 제외됨.

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4


# isnull, notnull 메소드: Series 객체에 어떤 index의 값이 정의되지 않았는지 확인할 때 사용. 

pd.isnull(obj4)
#obj4[pd.isnull(obj4)]
#isnull, notnull은 아래처럼 Series 객체의 메소드로 호출할 수도 있음.
#obj4.isnull()   
#obj4.notnull()

pd.notnull(obj4)

# Series 객체끼리 산술 연산 결과는 index를 기준으로 정렬해서 출력.

obj3
obj4
obj3+obj4

# Series 객체 및 Series 객체의 index에 name 속성을 설정할 수 있음.
obj4.name = 'population'  #values에 대한 이름 지정
obj4.index.name = 'state' #index에 대한 이름 지정
obj4

# Series 객체의 index는 새 리스트를 대입해서 변경 가능. 단, 새 리스트의 원소 개수는 이전 index 개수와 정확히 일치해야 함. 

obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
#obj.index = ['Bob', 'Steve', 'Jeff'] 
obj

