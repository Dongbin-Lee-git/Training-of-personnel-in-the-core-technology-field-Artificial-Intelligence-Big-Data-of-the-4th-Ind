# ## Overfit 유도: Multi-layer Perceptron을 이용한 Digit classifier

# ### 손글씨 숫자 데이터 분류

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

# ### 학습과 테스트 데이터 분류

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)

print('Type: x_train, y_train, x_test, y_test')
print(type(x_train), type(y_train), type(x_test), type(y_test))

print('Shape: x_train, y_train, x_test, y_test')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# #### Tensor로 변환

x_train = torch.tensor(x_train, dtype=torch.float).to('cuda')
y_train = torch.tensor(y_train, dtype=torch.long).to('cuda')
x_test = torch.tensor(x_test, dtype=torch.float).to('cuda')
y_test = torch.tensor(y_test, dtype=torch.long).to('cuda')

# ### TensorDataset과 Dataloader를 이용

# #### Dataset을 만든다. (비유: 약수물을 정수기 통에 담는다.)

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

print(train_dataset)
print(test_dataset)


# #### Dataloader를 만든다. (비유: 정수기 물통에 냉수 파이프와 온수 파이프를 설치한다)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=True)

print(train_dataloader)
print(test_dataloader)

# #### 학습을 위한 모델, Loss, Optimizer 설정

# ##### 다층 nn.Linear를 이용한 모델

k = 100

net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, 10)
)


# ##### Loss함수: Cross Entropy
# ##### Optimizer: Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# #### 학습과 테스트를 동시에

train_losses = []
test_losses = []
net.to('cuda')

for epoc in range(100):

  batch_loss = 0.0
  net.train()

  for x_train, y_train in train_dataloader:
    
    optimizer.zero_grad()

    y_pred = net(x_train)
    # print('y_pred size: ', y_pred.size())
    loss = loss_fn(y_pred, y_train)
    loss.backward()

    optimizer.step()
    batch_loss += loss.item()
  train_losses.append(batch_loss)
  print(epoc, ' Loss: ', batch_loss)

  net.eval()
  with torch.no_grad():
    for x_test, y_test in test_dataloader:
      test_result = net(x_test)
      test_loss = loss_fn(test_result, y_test)
      test_losses.append(test_loss.item())


# #### 학습 과정 중의 train과 loss visualization

plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend(loc='upper right')
plt.ylim((0,2.0))
plt.show()

