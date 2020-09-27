# # python 파일 입출력
# 파일 읽기 : 경로(path) 지정과 open(path)
path = 'examples/segismundo.txt'
f = open(path)

lines = [x.rstrip() for x in open(path)]
for line in lines:
    print(line, end='\n')
f.close()

# with를 사용하면 file을 닫는 과정(할당된 자원을 OS에게 반납)을 생략할 수 있다.

with open(path) as f:
    lines = [x.rstrip() for x in f]

# 파일 내용 읽어오기 : read, seek, tell 메소드

f = open(path, 'r')  # r-read only, w, x, a, r+, b, t
f.read(10)  # 디폴트 인코딩 방식은 utf-8
f.close()
f2 = open(path, 'rb') # b-binary, r-read only
f2.read(10) # byte 단위 : 10개 byte를 읽어 옴. \xc3, \xb1이 각각 1byte
f2.close()

# 파일 저장
with open('tmp.txt', 'w') as handle:
    handle.writelines(x for x in open(path) if len(x)>1)
with open('tmp.txt') as f:
    lines = f.readlines()
lines
#for line in lines:
#   print(line, end='\n')
# byte와 unicode

with open(path) as f:
    chars = f.read(10)  # utf-8
chars

with open(path, 'rb') as f:
    data = f.read(10)  # binary 
data
data.decode('utf8')  # unicode 인코딩을 utf-8 인코딩으로 decode

with open(path, 'rb') as f:
    data2 = f.read()
data2.decode('utf8')