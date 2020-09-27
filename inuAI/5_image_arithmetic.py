import cv2
import numpy as np

# * line 1: a는 1행 2열 배열이고, unsinged integer 타입이다.
# * line 2: b는 a와 동일한 구조와 타입을 갖는 배열이다.

a = np.array([[200, 50]], dtype=np.uint8)
b = np.array([[100,100]], dtype=np.uint8)

print(a.shape)
print(b.shape)

# * line 2: cv2.add를 사용하면 첫 번째 요소의 값이 200 + 100 = 300이지만, 255로 설정

add_result = cv2.add(a, b)
print(add_result)
print(add_result.shape)

# * Subract 연산결과. 연산결과 음수가 발생하면 0으로 설정

sub_result = cv2.subtract(b, a)
print(sub_result)

# * Multiply 연산결과. 255를 넘어가는 결과는 255로 설정

mul_result = cv2.multiply(a, b)
print(mul_result)

# * Divide 연산결과.

div_result = cv2.divide(a, b)
print(div_result)

