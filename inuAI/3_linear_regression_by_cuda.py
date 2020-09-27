import torch
from torch import nn, optim
import matplotlib.pyplot as plt

basic = torch.ones(13,dtype=torch.float32)
x1_gas = torch.tensor([0,73,31,39,78,22,96,82,24,81,61,28,91],dtype=torch.float32)
x2_gas = torch.tensor([11,88,81,2,73,88,8,64,80,45,67,34,25], dtype=torch.float32)

toxic = torch.FloatTensor([34, 411, 306, 85, 376,309, 217,357,289,298,324, 159,258])
toxic = toxic.view((-1,1))
toxic = toxic.to('cuda')
# print(toxic.dtype)

basic = basic.view((-1,1))
# print(basic.size())
x1_gas = x1_gas.view((-1,1))
# print(x1_gas.size())
x2_gas = x2_gas.view((-1,1))
# print(x2_gas.size())

X = torch.cat([basic, x1_gas, x2_gas], dim=1)
X = X.to('cuda')
# print('X size: ', X.size())
# print(X)

net = nn.Linear(in_features=3, out_features=1, bias=False)
net = net.to('cuda')
optimizer = optim.SGD(net.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

losses = []

for epoch in range(15):

    optimizer.zero_grad()

    y_pred = net(X)
    loss = loss_fn(y_pred, toxic)
    loss.backward()

    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.show()

print(list(net.parameters()))

