# 유용한 함수들: enumerate, sorted, zip, reversed, list notation

# 1. enumerate

my_list = ['python', 'numpy', 'pandas', 'matplotlib']

for index, name in enumerate(my_list):
    print('번호={0}, 이름={1}'.format(index, name))
my_dict = {}
for key, value in enumerate(my_list):
    my_dict[key] = value
my_dict

# 2. sorted : sort와 사용 방식이 다름.

lis = [9, 5, 1, 4, 2, 8, 6, 3, 7]
#lis.sort()
#lis
sorted(lis)
lis2 = 'hello world'
sorted(lis2)

# 3. zip

lis3 = ['no.3', 'no.2', 'no.1', 'no.4']
lis4 = ['KIM', 'CHOI', 'PARK', 'LEE']
new_list = zip(sorted(lis3), sorted(lis4))
zipped = list(new_list)
for i, (a, b) in enumerate(zipped):
    print('{0}: {1}\t = {2}'.format(i, a, b))

# 압축(zip)된 리스트를 풀 때는 리스트 이름 앞에 *(asterisk)를 붙임. 압축한 원소 개수만큼 돌려받기 위한 변수들을 선언.
rank, last_name = zip(*zipped)
rank
last_name

# 압축(zip)할 때 2개 리스트의 원소 개수가 맞지 않으면, 작은 쪽을 기준으로 맞춤.

lis5 = [True, False]
another_list = zip(sorted(lis3), lis5)
list(another_list)

# 4. reversed

lis6 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#lis6 = range(10)
list(reversed(lis6))

# 5. list notation - 가장 단순하게 리스트를 만드는 방법

name_list = ['PARK', 'KIM', 'LEE', 'HONG', 'CHOI', 'RYU']
[s.lower() for s in name_list if len(s)==3 ]
length = [len(x) for x in name_list]
#length = {len(x) for x in name_list}
length

list(map(len, name_list))
#set(map(len, name_list))

my_dic = {v:i for i, v in enumerate(name_list)}
my_dic

# nested list : 실습코드 인용 - Chapter 3, Python for Data Analysis(2nd Ed.)

all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
           ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]

# 문제: 영어, 스페인어 이름에 'e'가 2개 이상 포함된 이름을 구해보자

name_of_interest = []
for names in all_data:
    enough_es = [name for name in names if name.count('e')>=2]
    name_of_interest.extend(enough_es)
name_of_interest

# 아래처럼 하나의 문장으로 압축해서 나타낼 수 있음. 
# - 첫 번째 for문의 names가 이어서 나오는 두 번째 for문에서 사용됨.

result = [[name for names in all_data for name in names if name.count('e')>=2]]
result

list_of_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

simple_list = [x for tup in list_of_tuples for x in tup]
simple_list

simple_list = []
for tup in list_of_tuples:
    for x in tup:
        simple_list.append(x)
simple_list

other_list = [[x for x in tup] for tup in list_of_tuples]
other_list

