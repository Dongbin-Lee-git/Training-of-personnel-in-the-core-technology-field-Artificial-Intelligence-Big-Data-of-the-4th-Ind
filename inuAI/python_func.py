# # python 함수

# 함수 정의 및 함수 호출
# - 형식 인자(formal parameter)를 선언하거나 생략할 수 있음(a, b, c)
# - 형식 인자 중 초기값을 갖는 인자를 non-default argument라고 함(c=1.0)
# - non-default argument는 다른 인자(default argument)보다 뒤에 나타나야 함.

def my_func(a, b, c=1.0):
    if c != 0:
        k = (a+b)/c
    else:
        k = a+b
    return k

print(my_func(5, 6, c=0.7))
print(my_func(3.14, 7, c=0))
print(my_func(3.14, 7))

#어떤 형식 매개변수(formal parameter)에 어떤 값이 전달됐는지 알기 쉽도록 선언할 수 있음. 
#my_func(a=10, b=5, c=1)
my_func(c=1, b=5, a=10)


# 변수의 범위(scope): 함수에서 정의한 변수(lis)는 함수 밖에서는 참조할 수 없다.

def foo():
    lis = []
    for i in range(10):
        lis.append(i)
    print(lis)

foo()
#lis - 함수에서 정의한 변수를 참조(access)할 수 없음


# python 함수는 한 개의 값이 아니라 여러 개 값을 반환(return) 할 수 있다.

def bar():
    x=1; y=2; z=3    # 한 줄에 여러 개 문장을 쓸 때는 ;로 구분. 
    return x, y, z

a, b, c = bar()
print(a, b, c)


# 함수의 다양한 사용 예 : 예제 인용 - Chapter 3, Python for Data Analysis(2nd Ed.)
# - 아래 예제처럼 설문조사 응답결과 raw data의 형식이 일정치 않은 경우(대소문자 혼용, 특수문자 사용 등)

states = ['Alabama', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 
         'south   carolina##', 'West  virginia?']

import re  # 정규표현식(regular expression) 라이브러리 사용

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()  # 공백문자 제거
        value = re.sub('[!#?]','', value) # 특수문자 제거
        value = value.title() # 첫글자를 대문자로 변환
        value = re.sub('[ ]+',' ', value) # 2개 이상 공백문자가 있을 경우 1개만 남김
        result.append(value)
    return result

clean_strings(states)

#원본 리스트는 바뀌지 않음.
#states = clean_strings(states) #원본 리스트 내용도 바꾸려면
#states

# 결과는 같지만, 적용할 함수들을 list에 담아놓고 순서대로 적용.

def remove_specialchars(value):
    return re.sub('[?!#]', '', value)
def redundant_spaces(value):
    return re.sub('[ ]+',' ', value)

apply_ops = [str.strip, remove_specialchars, str.title, redundant_spaces]

def clean_strings2(strings, ops):
    result = []
    for value in strings:
        for func in ops:
            value = func(value)
        result.append(value)
    return result 

clean_strings2(states, apply_ops)

states = ['Alabama', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 
         'south carolina##', 'West  virginia?']

# 이번에는 map 함수를 이용해 구현해 보자!

def clean_strings3(strings, ops):
    result = []
    for value in map(ops, strings):
        result.append(value)
    return result

states = clean_strings3(states, str.strip)
states = clean_strings3(states, remove_specialchars)
states = clean_strings3(states, str.title)
states = clean_strings3(states, redundant_spaces)
states

# lambda function
# - 함수 이름 대신 lambda 키워드 사용. 한 문장으로 이루어진 간단한 함수.
# - 독립적으로 사용하기보다는 함수의 형식 인자를 람다 함수로 정의하는 형태를 주로 사용.

def boo(x):
    return x*2

mboo = lambda x: x*2

boo_lst = list(range(1,4))
def apply_to_list(lst, f):
    return [f(x) for x in lst]
apply_to_list(boo_lst, mboo)

apply_to_list(boo_lst, lambda x:x*2)

slist = ['foot', 'a', 'python', 'but', 'he', 'longlong', 'short']

slist.sort(key=lambda x: len(x))
slist

