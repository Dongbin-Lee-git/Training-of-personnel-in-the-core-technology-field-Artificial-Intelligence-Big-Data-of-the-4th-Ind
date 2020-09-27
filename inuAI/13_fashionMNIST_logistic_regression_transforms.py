# ## FashionMNIST : logistic regression + batch normalization

# ### Fashion MNINST 학습데이터 분석

from torchvision.datasets import FashionMNIST
from torchvision import transforms

fashion_train = FashionMNIST('./', train=True, download=True, transform=transforms.ToTensor())
print('fashion_train type: ',type(fashion_train))
print('fashion_train length: ', len(fashion_train))
print('')
print('fashion_train[0] type: ', type(fashion_train[0]))
print('fashion_train[0] length: ', len(fashion_train[0]))
print('')
print('fashion_train[0][0] type: ', type(fashion_train[0][0]))
print('fashion_train[0][0] info: ',fashion_train[0][0])
print('')
print('fashion_train[0][1] type: ', type(fashion_train[0][1]))
print('fashion_train[0][1]: ', fashion_train[0][1])

# ### Fashion MNIST 테스트데이터 분석

fashion_test = FashionMNIST('./', train=False, download=True, transform=transforms.ToTensor())
print('fashion_test type: ',type(fashion_test))
print('fashion_test length: ', len(fashion_test))
print('')
print('fashion_test[0] type: ', type(fashion_test[0]))
print('fashion_test[0] length: ', len(fashion_test[0]))
print('')
print('fashion_test[0][0] type: ', type(fashion_test[0][0]))
print('fashion_test[0][0] info: ',fashion_test[0][0])
print('')
print('fashion_test[0][1] type: ', type(fashion_test[0][1]))
print('fashion_test[0][1]: ', fashion_test[0][1])

# ### Dataloader를 이용

from torch.utils.data import DataLoader
import torch

fashion_train_dataloader = DataLoader(fashion_train, batch_size=64, shuffle=True)
fashion_test_dataloader = DataLoader(fashion_test, batch_size=20000, shuffle=True)

for xi, yi in fashion_train_dataloader:
  xi = xi.view(-1, 784)
  print(xi.size(), xi.device, xi.dtype)
  # print(xi, yi)
  break;

# #### 학습을 위한 모델, Loss, Optimizer 설정

# ##### 다층 nn.Linear를 이용한 모델

import torch
from torch import nn, optim

net = nn.Sequential(
    nn.Linear(784, 392),
    nn.BatchNorm1d(392),
    nn.ReLU(),
    nn.Linear(392, 196),
    nn.BatchNorm1d(196),
    nn.ReLU(),
    nn.Linear(196, 98),
    nn.BatchNorm1d(98),
    nn.ReLU(),
    nn.Linear(98, 49),
    nn.BatchNorm1d(49),
    nn.ReLU(),
    nn.Linear(49, 24),
    nn.BatchNorm1d(24),
    nn.ReLU(),
    nn.Linear(24, 10),
)

# ##### Loss함수: Cross Entropy
# ##### Optimizer: Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# #### 학습 시작

losses = []
net.to('cuda')

for epoc in range(200):

  batch_loss = 0.0
  net.train()

  for x_train, y_train in fashion_train_dataloader:

    x_train = x_train.to(torch.device('cuda'))
    y_train = y_train.to(torch.device('cuda'))

    optimizer.zero_grad()

    y_pred = net(x_train.view((-1,784)))
 
    loss = loss_fn(y_pred, y_train)
    loss.backward()

    optimizer.step()
    batch_loss += loss.item()
  losses.append(batch_loss)
  print(epoc, ' Loss: ', batch_loss)

  net.eval()

  with torch.no_grad():
    for x_test, y_test in fashion_test_dataloader:

      x_test = x_test.to(torch.device('cuda'))
      y_test = y_test.to(torch.device('cuda'))

      test_result = net(x_test.view((-1, 784)))
      pred = torch.argmax(test_result, dim=1)

      num_correct = (pred == y_test).sum().item()
      print('Accuracy: ', num_correct*100.0/len(y_test), '%')

# #### 학습 과정 중의 loss visualization

import matplotlib.pyplot as plt
plt.plot(losses)

