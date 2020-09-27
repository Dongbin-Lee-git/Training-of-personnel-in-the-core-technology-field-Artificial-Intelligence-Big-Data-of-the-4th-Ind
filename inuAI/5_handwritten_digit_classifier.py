import torch
from torch import nn, optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
print(type(X))
print(X.shape)

y = digits.target
print(type(y))
print(y.shape)
print(y)

# #### 이미지로 손글씨 확인하기

i = 12

num = X[i]
print(num.shape)
num = num.reshape((8,8))
print(num.shape)
print(num)
plt.imshow(num, cmap='gray')
plt.show()

# ## 손글씨 숫자를 분류하는 Logistic regression model

# ### 학습과 테스트 데이터 분류

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

print('Type: x_train, y_train, x_test, y_test')
print(type(x_train), type(y_train), type(x_test), type(y_test))

print('Shape: x_train, y_train, x_test, y_test')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# #### 학습을 위한 모델, Loss, Optimizer 설정

# In[4]:

net = nn.Linear(64, 10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# #### Tensor로 변환

x_train = torch.tensor(x_train, dtype=torch.float).to('cuda')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
x_test = torch.tensor(x_test, dtype=torch.float).to('cuda')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')

# #### 학습 시작

losses = []
net.train()
net.to('cuda')

for epoc in range(600):

  optimizer.zero_grad()

  y_pred = net(x_train)
  loss = loss_fn(y_pred, y_train)
  loss.backward()

  optimizer.step()
  losses.append(loss.item())
  print(epoc, ' Loss: ', loss.item())

# #### 학습 과정 중의 loss visualization

plt.plot(losses)

# #### 테스트 시작

net.eval()

with torch.no_grad():

  test_result = net(x_test)
  pred = torch.argmax(test_result, dim=1)

  num_correct = (pred == y_test).sum().item()
  print('Accuracy: ', num_correct*100.0/len(y_test), '%')

