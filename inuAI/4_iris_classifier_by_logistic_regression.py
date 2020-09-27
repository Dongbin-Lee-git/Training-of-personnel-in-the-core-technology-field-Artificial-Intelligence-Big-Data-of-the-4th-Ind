import torch
from torch import nn, optim
from sklearn.datasets import load_iris

# ### IRIS 데이터를 로딩

iris = load_iris()

print('Flower names: ', iris.target_names)
print('Feature names: ', iris.feature_names)

# ### 데이터 개수를 확인

flower_features = iris.data
flower_classificaiton = iris.target

print('flower features len: ', len(flower_features))
print('flower classification len: ', len(flower_classificaiton))

# ### 데이터를 실제로 확인

print(flower_features[0])
print(flower_features[50])
print(flower_features[100])


# ### 분류데이터를 실제로 확인. 0, 1, 2로 꽃 종류를 구분

print(flower_classificaiton[0])
print(flower_classificaiton[50])
print(flower_classificaiton[100])

# ## Cross Entroy Loss 예제

# ### prediction [6, 3, 0.1]에 대해서, ground truth와 비교하여 LOSS계산

import torch.nn

loss_fn = nn.CrossEntropyLoss()

prediction = torch.tensor([[6.0, 3.0, 0.1]], dtype=torch.float)
gt = torch.tensor([0])

print('prediction size: ', prediction.size())
print('ground truth size: ', gt.size())

# ### 정답이 0일 때, Prediction이 이를 맞추었기 때문에 Loss값이 작다 

loss_res = loss_fn(prediction, gt)
print('loss for 0: ', loss_res)

# ### 정답이 1이므로 Loss값이 크다.

gt = torch.tensor([1])
loss_res = loss_fn(prediction, gt)
print('loss for 1: ', loss_res)


# ### 정답이 2인데, prediction에서는 이의 가능성을 가장 낮게 평가했으므로, Loss가 가장 크다.

gt = torch.tensor([2])
loss_res = loss_fn(prediction, gt)
print('loss for 2: ', loss_res)

# ### 정답을 3으로 했으나, 이는 범위를 벗어나기 때문에 오류

gt = torch.tensor([3])
loss_res = loss_fn(prediction, gt)
print('loss for 3: ', loss_res)

# ### 4개의 값을 이용하여 꽃 종류를 맞추는 모델

net = nn.Linear(4, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.25)

# ### Train과 Test 데이터로 분리

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(flower_features, flower_classificaiton, test_size=0.1)
print(type(x_train), x_train.dtype)

# ### 학습과정

print('x_train: ', x_train.size(), x_train.dtype)
print('y_train: ', y_train.size(), y_train.dtype)

net = net.train()
for i in range(1000):
  optimizer.zero_grad()
  X = net(x_train)
  loss = loss_fn(X, y_train)
  loss.backward()
  optimizer.step()

# ### 테스트 과정

net.eval()

with torch.no_grad():
  prediction_res = net(x_test)
  print('Prediction size: ', prediction_res.size())

print('Prediction: ', prediction_res)
print('Ground truth: ', y_test)

# ### 테스트데이터에 대해서 정답율 체크

pred = torch.argmax(prediction_res, dim=1) 
print(pred)

num_correct = (pred == y_test).sum().item()
print('Accuracy: ', num_correct*100 / len(y_test), '%')