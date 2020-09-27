import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

path = "2014년역사별승하차현황.csv"  # 파일 확장자 csv
raw_data = pd.read_csv(path, encoding='CP949')
raw_data.head()

raw_data.info()

raw_data.shape

raw_data.describe()

station_list = list(raw_data['역명'].unique())
print('역사 수 = ', len(station_list))
print('역사 이름 = ', station_list[:5])
len(raw_data['월'].unique())

monthly_passengers = raw_data.groupby('월').sum()
#monthly_passengers
#monthly_passengers.plot.bar()

id_max = monthly_passengers['이용인원'].idxmax()
val_max = monthly_passengers['이용인원'].max()
id_min = monthly_passengers['이용인원'].idxmin()
val_min = monthly_passengers['이용인원'].min()

print()
print('최대 이용 월 및 이용인원 =  {0:>2}월 {1:,}명'.format(id_max, val_max))
print('최소 이용 월 및 이용인원 =  {0:>2}월 {1:,}명'.format(id_min, val_min))

val1 = monthly_passengers['이용인원'].mean()
val2 = monthly_passengers['승차인원'].mean()
val3 = monthly_passengers['하차인원'].mean()
print()
print('월평균 이용인원 = {0:,} 명'.format(val1))
print('월평균 승차인원 = {0:,} 명'.format(val2))
print('월평균 하차인원 = {0:,} 명'.format(val3))

tmp = raw_data.loc[:, ['역명', '월', '승차인원', '하차인원']]
monthly_passengers2 = tmp.groupby('월').sum()
#monthly_passengers
monthly_passengers2.plot.bar()

#monthly_passengers['이용인원']
labels = ['winter', 'spring', 'summer', 'fall']
fracs1 = []
fracs1.append(monthly_passengers.iloc[[0,1,11], 0].sum()) #12, 1, 2월
fracs1.append(monthly_passengers.iloc[[2,3,4], 0].sum())  #3, 4, 5월
fracs1.append(monthly_passengers.iloc[[5,6,7], 0].sum()) #6, 7, 8월
fracs1.append(monthly_passengers.iloc[[8,9,10], 0].sum()) #9, 10, 11월
explode = (0, 0.25, 0, 0)

plt.pie(fracs1, explode=explode, labels=labels, autopct='%.0f%%', shadow=True)
plt.title('season passengers')
plt.show()

grouped = raw_data.groupby(raw_data['역명'])
#grouped.max()
#grouped.min()
#grouped.mean()

grouped = raw_data.groupby(['역명', '월'])
type(grouped)
#grouped = raw_data.groupby([raw_data['역명'], raw_data['월']])
#means = grouped.mean()
#means.head(12)

#groupby 객체에 속한 원소들을 순환문을 통해 가져올 수 있음.
#반환되는 값은 소괄호로 묶인 tuple 형태
i = 0
for (k1, k2), group in raw_data.groupby(['역명', '월']):
    print((k1, k2), group) 
    i+= 1
    if (i>5): break

def station_statistics(sta, grouped):
    new_df = grouped[sta] #new_df : DataFrame 객체 
    passenger_sr = new_df.loc[ :, '이용인원'] #passenger_sr : Series 객체
    max_val = passenger_sr.max()
    id_max = passenger_sr.idxmax()
    avg_val = passenger_sr.mean()
    min_val = passenger_sr.min()
    id_min = passenger_sr.idxmin()   
    print()
    print('역명 >>>', sta)
    print('월 평균 이용인원 = {0:,}명'.format(np.round(avg_val,1)))
    print('최대 이용 월 및 이용인원 = {0:>2}월 {1:,}명'.format(id_max, max_val))
    print('최소 이용 월 및 이용인원 = {0:>2}월 {1:,}명'.format(id_min, min_val))

pieces = dict(list(raw_data.groupby(['역명'])))  
#print(type(pieces['간석오거리'].values))

for sta in station_list:
    station_statistics(sta, pieces)

pieces = dict(list(raw_data.groupby(['역명', '월'])))  
pieces[('간석오거리', 1)]

def plot_station_statistics(sta, grouped):
    sta1 = grouped[sta].loc[:, '이용인원']
    sta1_avg = sta1.mean()
    sta1_max = sta1.max()
    sta1_min = sta1.min()

    x = np.arange(12)+1
    plt.plot(x, sta1, 'y', label=sta)
    plt.plot([0., len(x)], [sta1_avg, sta1_avg], 'k--', label='평균이용인원')
    plt.plot([0., len(x)], [sta1_max, sta1_max], 'r--')
    plt.plot([0., len(x)], [sta1_min, sta1_min], 'b--')

    plt.xlabel('월')
    plt.ylabel('이용인원(명)')
    plt.title('월별 이용 인원')
    plt.legend(loc='best')

pieces = dict(list(raw_data.groupby(['역명'])))  
plot_station_statistics('간석오거리', pieces)

pieces = dict(list(raw_data.groupby(['역명'])))  
sta1 = pieces['간석오거리']
sta2 = pieces['인천터미널']
sta3 = pieces['작전']
sta4 = pieces['계산']
sta1_reduced = sta1.loc[:, '이용인원']
sta2_reduced = sta2.loc[:, '이용인원']
sta3_reduced = sta3.loc[:, '이용인원']
sta4_reduced = sta4.loc[:, '이용인원']

x = np.arange(12)+1
plt.plot(x, sta1_reduced, 'y', label='간석오거리')
plt.plot(x, sta2_reduced, 'b', label='인천터미널')
plt.plot(x, sta3_reduced, 'r', label='작전')
plt.plot(x, sta4_reduced, 'g', label='계산')

plt.title('월별 이용 인원')
plt.legend(loc='upper right')

top3_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='이용인원', ascending=False)[:3])
#plist3 = top3_station.loc[:, ['역명', '이용인원']]
#plist3

#value_counts는 Series객체를 반환
top3_sr = top3_station['역명'].value_counts()
for (i, s) in top3_sr.iteritems():
    print('역명={}\t, 빈도={}'.format(i, s))
    
#for s in top3_sr.index:
#    print('역명={0}\t, 빈도={1}'.format(s, top3_sr[s]))

top5_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='이용인원', ascending=False)[:5])
top5_sr = top5_station['역명'].value_counts()
for (i, s) in top5_sr.iteritems():
    print('역명={}\t, 빈도={}'.format(i, s))

#이용인원 하위 3개 전철역 
# 국제업무지구, 귤현, 문학경기장(1,2,7,11,12월), 
# 센트럴파크(3,4,5,6,9월), 부평삼거리(10월), 인천대입구(8월)
low3_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='이용인원')[:3])
#plist32 = low3_station.loc[:, ['역명', '이용인원']]
#plist32

low3_sr = low3_station['역명'].value_counts()
for (i, s) in low3_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))

#이용인원 하위 5개 전철역 
low5_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='이용인원')[:5])

low5_sr = low5_station['역명'].value_counts()
for (i, s) in low5_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))

pieces = dict(list(raw_data.groupby(['역명'])))  
sta12 = pieces['국제업무지구']
sta22 = pieces['귤현']
sta32 = pieces['문학경기장']
sta42 = pieces['센트럴파크']
sta52 = pieces['부평삼거리']
sta62 = pieces['인천대입구']
sta12_reduced = sta12.loc[:, '이용인원']
sta22_reduced = sta22.loc[:, '이용인원']
sta32_reduced = sta32.loc[:, '이용인원']
sta42_reduced = sta42.loc[:, '이용인원']
sta52_reduced = sta52.loc[:, '이용인원']
sta62_reduced = sta62.loc[:, '이용인원']

x = np.arange(12)+1
plt.plot(x, sta12_reduced, 'b', label='국제업무지구')
plt.plot(x, sta22_reduced, 'r', label='귤현')
plt.plot(x, sta32_reduced, 'y', label='문학경기장')
plt.plot(x, sta42_reduced, 'g', label='센트럴파크')
plt.plot(x, sta52_reduced, 'b--', label='부평삼거리')
plt.plot(x, sta62_reduced, 'r--', label='인천대입구')

plt.legend(loc='best')
plt.title('이용인원 수 기준 하위 6개 전철역')

plot_station_statistics('문학경기장', pieces)

raw_data['승하차인원차'] = raw_data['승차인원']- raw_data['하차인원']
raw_data.head()

id_max = raw_data['승하차인원차'].idxmax()
print('승하차인원차 최대인 역사')
print(raw_data.iloc[id_max, :])

print()
id_min = raw_data['승하차인원차'].idxmin()
print('승하차인원차 최소인 역사')
print(raw_data.iloc[id_min, :])

print()
avg_diff = raw_data['승하차인원차'].mean()
print('승하차인원차 평균 = {0:,}명'.format(np.round(avg_diff, 1)))

def monthly_diff_statistics(mon, grouped):
    new_df = grouped[mon] #new_df : DataFrame 객체 
    diff_sr = new_df.loc[ :, '승하차인원차'] #passenger_sr : Series 객체
    max_val = diff_sr.max()
    id_max = diff_sr.idxmax()
    sta_max = new_df.loc[id_max,'역명']
    
    avg_val = diff_sr.mean()
    
    min_val = diff_sr.min()
    id_min = diff_sr.idxmin() 
    sta_min = new_df.loc[id_min,'역명']
    
    print()
    print('월 >>>', mon)
    print('월 평균 승하차인원차 = {0:,}명'.format(np.round(avg_val,1)))
    print('최대 승하차인원차 = {0:>2}역 {1:,}명'.format(sta_max, max_val))
    print('최소 승하차인원차 = {0:>2}역 {1:,}명'.format(sta_min, min_val))
    return avg_val

pieces2 = dict(list(raw_data.groupby(['월'])))  
#pieces2[1].head()

avg_diff_list = []
for mon in range(1,13):
    avg_mon = monthly_diff_statistics(mon, pieces2)
    avg_diff_list.append(np.round(avg_mon, 1))
    
avg_diff_list

tmp_df = raw_data.loc[:, ['월', '승하차인원차']]
monthly_diff_df = tmp_df.groupby('월').sum()
#monthly_diff_df
monthly_diff_df.plot.bar()

monthly_diff_df.columns = ['평균']
#monthly_diff_df.columns
monthly_diff_df['평균'] = avg_diff_list
#monthly_diff_df
monthly_diff_df.plot.bar()

# 승하차인원차 하위 3개 전철역 
low_diff2 = raw_data.groupby("월").apply(lambda x: x.sort_values(by='승하차인원차')[:3])
low_diff2_sr = low_diff2['역명'].value_counts()
for (i, s) in low_diff2_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))

# 승하차인원차 상위 3개 전철역 
high_diff2 = raw_data.groupby("월").apply(lambda x: x.sort_values(by='승하차인원차', ascending=False)[:3])
high_diff2_sr = high_diff2['역명'].value_counts()
for (i, s) in high_diff2_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))

# 승차인원 하위 3개 전철역 - 이용인원과 유사
low32_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='승차인원')[:3])
low32_sr = low32_station['역명'].value_counts()
for (i, s) in low32_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))
#plist34 = low32_station.loc[:, ['역명', '승차인원']]
#plist34


# 승차인원 상위 3개 전철역 - 이용인원과 같음
top32_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='승차인원', ascending=False)[:3])
top32_sr = top32_station['역명'].value_counts()
for (i, s) in top32_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))
#plist32 = top32_station.loc[:, ['역명', '승차인원']]
#plist32


# 하차인원 하위 3개 전철역 - 이용인원과 유사
low33_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='하차인원')[:3])
low33_sr = low33_station['역명'].value_counts()
for (i, s) in low33_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))
#plist35 = low33_station.loc[:, ['역명', '하차인원']]
#plist35


# 하차인원 상위 3개 전철역 -  이용인원과 유사
top33_station = raw_data.groupby("월").apply(lambda x: x.sort_values(by='하차인원', ascending=False)[:3])
top33_sr = top33_station['역명'].value_counts()
for (i, s) in top33_sr.iteritems():
    print('역명={0}\t, 빈도={1}'.format(i, s))
#plist3 = top33_station.loc[:, ['역명', '하차인원']]
#plist3

corr = raw_data[['이용인원', '승차인원', '하차인원', '승하차인원차']].corr(method='pearson')
print(corr)

import seaborn as sns
import matplotlib.pyplot as plt

cols_view = ['total', 'passengers', 'drop-offs', 'difference']
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values,
                cbar=True,
                annot=True,
                square=True,
                fmt='.1f',
                annot_kws={'size': 10},
                yticklabels=cols_view,
                xticklabels=cols_view)

plt.tight_layout()
plt.show()
sns.set(style='whitegrid', context='notebook')
sns.pairplot(raw_data[['이용인원', '승차인원', '하차인원', '승하차인원차']], height=2.5)
plt.show()