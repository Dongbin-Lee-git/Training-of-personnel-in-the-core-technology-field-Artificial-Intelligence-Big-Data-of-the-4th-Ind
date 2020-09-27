# ## FashionMNIST를 위한 Convolutional Neural Network

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

# ### TensorDataset과 Dataloader를 이용

# #### Dataset을 만든다. (비유: 약수물을 정수기 통에 담는다.)

from torch.utils.data import DataLoader
import torch

fashion_train_dataloader = DataLoader(fashion_train, batch_size=64, shuffle=True)
fashion_test_dataloader = DataLoader(fashion_test, batch_size=20000, shuffle=True)

for x_train, y_train in fashion_train_dataloader:
  print('x_train size: ', x_train.size())
  break

for x_test, y_test in fashion_test_dataloader:
  print('x_test size: ', x_test.size())
  break

# #### 학습을 위한 모델, Loss, Optimizer 설정

# ##### 다층 nn.Linear를 이용한 모델

import torch
from torch import nn, optim

class FashionCNN(nn.Module):

  def __init__(self):
    super().__init__()
        
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
        
    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
        
    self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
    self.drop = nn.Dropout2d(0.25)
    self.fc2 = nn.Linear(in_features=600, out_features=120)
    self.fc3 = nn.Linear(in_features=120, out_features=10)
        
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    out = self.drop(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out

# ##### Loss함수: Cross Entropy
# ##### Optimizer: Adam

net = FashionCNN()
net.to(torch.device('cuda'))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# #### 학습 시작

losses = []

for epoc in range(200):

  batch_loss = 0.0
  net.train()
  for x_train, y_train in fashion_train_dataloader:

    x_train = x_train.to(torch.device('cuda'))
    y_train = y_train.to(torch.device('cuda'))

    optimizer.zero_grad()

    y_pred = net(x_train)
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

      test_result = net(x_test)
      pred = torch.argmax(test_result, dim=1)

      num_correct = (pred == y_test).sum().item()
      print('Accuracy: ', num_correct*100.0/len(y_test), '%')


# #### 학습 과정 중의 loss visualization

import matplotlib.pyplot as plt
plt.plot(losses)

