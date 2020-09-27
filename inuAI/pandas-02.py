
# coding: utf-8

# # Pandas 자료 구조: DataFrame

# DataFrame은 데이터베이스의 Table 또는 Excel의 spreadsheet와 유사. 2차원 행렬 구조.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}


# In[3]:


frame = pd.DataFrame(data)
frame


# In[4]:


frame.head()  # 처음 5개 행 출력
#frame.head(2) # 처음 2개 행 출력
#frame.tail() # 마지막 5개 행 출력


# In[7]:


frame.columns
#frame.index
#frame.values


# columns 속성을 통해 컬럼 순서를 리스트로 지정하면, 원본 data와 달리 순서를 바꿀 수 있다. Series 객체와 마찬가지로 index 속성을 지정하지 않으면 자동으로 정수 index를 생성한다. 

# In[8]:


pd.DataFrame(data, columns=['year', 'state', 'pop'])


# In[9]:


frame = pd.DataFrame(data, 
                     columns=['year', 'state', 'pop'], 
                     index=['one', 'two', 'three', 'four', 'five', 'six'])
frame


# DataFrame 원소 추출(참조): 열 단위(columnwise), 행 단위(rowwise) 및 한 개의 원소를 참조할 수 있다. 한 개의 열 또는 한 개의 행에 속한 원소들을 통째로 추출한 결과는 Series 객체이다. 물론 범위 지정(slicing)을 통해 여러 개의 열 또는 행도 추출할 수 있다. 참고로 DataFrame이 2차원 구조만 다룰 수 있는 것처럼 보이지만, 계층적 구조를 통해 더 복잡한 차원의 데이터 구조도 충분히 다룰 수 있다. 

# In[12]:


frame['state']  #strongly recommended
#frame.state  #별로 추천하고 싶지 않음.
#type(frame['state'])


# In[11]:


frame.loc['one']
#type(frame.loc['one'])


# In[36]:


frame.iloc[2]


# 원본 데이터에 없던 column을 지정하면, 해당 column의 값은 NaN이 할당됨.

# In[15]:


new_cols = ['year', 'state', 'pop', 'debt']
old_index =['one', 'two', 'three', 'four', 'five', 'six']
frame2 = pd.DataFrame(data, columns=new_cols, index=old_index)
frame2


# DataFrame의 column에 속한 값을 출력하려면 df.column_name 또는 df['column_name']을 사용

# In[39]:


frame2['state']
#frame2.state


# 행을 출력하는 방법은 loc 또는 iloc을 사용

# In[41]:


frame2.loc['three']  
#frame2.iloc[2]


# 할당문을 사용해 column에 원하는 값을 지정할 수 있음.

# In[42]:


frame2


# In[43]:


frame2['debt'] = 16.5
frame2


# In[17]:


frame2['debt'] = np.arange(6.)
#frame2['debt'] = np.arange(6)  #소수점을 포함하지 않으면?
#frame2['debt'] = np.random.randn(6)
frame2


# 리스트(또는 배열)을 column에 할당할 경우 리스트 크기가 DataFrame 크기와 같아야 함.
# - index와 column 값을 갖는 Series 객체를 DataFrame에 지정할 때, index에 해당하는 column 값을 지정하지 않은 경우 NaN을 할당.

# In[45]:


val = pd.Series([-1.2, -1.5, -1.7], index=['two','four','five'])


# In[46]:


frame2['debt']=val
frame2


# DataFrame에 존재하지 않는 column에 값을 지정하면 해당되는 column을 새롭게 추가. 

# In[20]:


frame2['eastern'] = frame2.state == 'Ohio'
frame2


# - column을 없애려면 del 메소드 사용 : del DataFrame['column_name']
# - 행을 없애려면 drop 메소드 사용 : dataframe_name.drop(['index_name'])

# In[21]:


del frame2['eastern']
#frame2


# In[19]:


frame2.drop(['six'])


# In[23]:


frame2.columns


# nested dictionary {...{...}}의 경우에도 DataFrame을 생성할 수 있다.
# - 바깥쪽 dictionary의 key는 column이 되며, 안쪽 dictionary의 key는 row가 된다.

# In[21]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}


# In[22]:


frame3 = pd.DataFrame(pop)
frame3  # 출력은 index를 기준으로 정렬되지 않았음.


# In[23]:


frame3.index


# In[26]:


frame3.T # 전치(transpose)


# In[24]:


frame3 = pd.DataFrame(pop, index=[2000,2001,2002]) # 출력은 index를 기준으로 정렬.
#pd.DataFrame(pop, index=[2001,2002,2003])
frame3


# In[44]:


frame3['Ohio'][:-1] #마지막 인덱스는 제외
#frame3['Ohio'][:2]


# 조금 더 복잡한 형태로 dictionary 객체를 사용해 DataFrame을 생성하는 방법을 알아보자. 기존 DataFrame(frame3)의 컬럼으로부터 추출한 값을 dictionary 객체의 key-value 쌍에서 value에 직접 지정해서 DataFrame을 생성할 수 있다.

# In[25]:


pdata = {'Ohio':frame3['Ohio'][:-1],
         'Nevada':frame3['Nevada'][:2]}
pd.DataFrame(pdata)


# DataFrame의 index와 column에 name속성 지정

# In[26]:


frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# In[27]:


frame3.values

