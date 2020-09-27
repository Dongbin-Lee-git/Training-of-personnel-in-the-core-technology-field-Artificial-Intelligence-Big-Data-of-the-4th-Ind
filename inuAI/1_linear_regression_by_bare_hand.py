import torch
import matplotlib.pyplot as plt

basic = torch.ones(13,dtype=torch.float32)
x1_gas = torch.tensor([0,73,31,39,78,22,96,82,24,81,61,28,91],dtype=torch.float32)
x2_gas = torch.tensor([11,88,81,2,73,88,8,64,80,45,67,34,25], dtype=torch.float32)

toxic = torch.FloatTensor([34, 411, 306, 85, 376,309, 217,357,289,298,324, 159,258])
# print(toxic.dtype)

basic = basic.view((-1,1))
# print(basic.size())
x1_gas = x1_gas.view((-1,1))
# print(x1_gas.size())
x2_gas = x2_gas.view((-1,1))
# print(x2_gas.size())

X = torch.cat([basic, x1_gas, x2_gas], dim=1)
# print('X size: ', X.size())
# print(X)
w = torch.randn(3, requires_grad=True)
# print('w: ', w)

losses = []

for epoch in range(10000):

    w.grad = None

    y_pred = torch.mv(X,w)
    # print('y_pred.size(): ', y_pred.size())
    # print('y_pred: ', y_pred)
    # print('w: ', w)
    loss = torch.mean((toxic - y_pred)**2)
    loss.backward()

    w.data = w.data - 0.000005 * w.grad.data
    # print(f'{epoch}: {loss.item()}')
    losses.append(loss.item())

print(w)

plt.plot(losses)
plt.show()

