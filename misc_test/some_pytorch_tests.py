import torch
from torch import optim
import numpy as np

#########################Manual computation
x = np.array([1,2,3,4,5])
y_true = np.array([2,4,6,8, 10])
w = np.array(0.5)


def forward(x, weigths):
    return x * weigths

def loss(y_true, y_predicted):
    loss = (y_true - y_predicted)**2
    return loss.mean()

def grad(x, y_pred, y_true):
    grads = 2*(y_true - y_pred) * -1 * x #manual computations
    return grads.mean()

def optimize(weigths, grads, learning_rate):
    weigths  -= grads * learning_rate
    return weigths

for i in range(50):
    
    y_pred = forward(x, w)
    loss_ = loss(y_true, y_pred)
    grads = grad(x, y_pred, y_true)
    w = optimize(w, grads, 0.01)
    # if i%5 == 0:
    #     print(f'Loss: {loss_:0.3f}, weight: {w:0.3f}, y:{y_true}, y_pred:{y_pred}')

##########################Computation with pytorch backpropagation

print()
print()
print()
x = torch.tensor(np.array([1,2,3,4,5], dtype=float))
y_true = torch.tensor(np.array([2,4,6,8, 10], dtype=float))
w = torch.tensor(np.array(0.5), requires_grad=True)
learning_rate = 0.01

for i in range(50):
    
    y_pred = forward(x, w)
    
    loss_ = loss(y_true, y_pred)
    loss_.backward()

    #We don't want to make optimization as a part of pytorch computational graph
    with torch.no_grad(): # pytorch dosn't allowed inplace operation such us += for tensor which requires grads 
        w = optimize(w, w.grad, learning_rate)
    
    w.grad.zero_()
    #w = optimize(w, grads = w.grad, learning_rate=learning_rate)
        
    # gradients after backward are accumulated therfore there need to be set to zero
    #w.grad.zero_() # in pytorch underscore at the end is "inplace" operation
    
    # if i%5 == 0:
    #     print(f'Loss: {loss_:0.3f}, weight: {w:0.3f}, y:{y_true}, y_pred:{y_pred}')


##############################################################
import torch.nn as nn

print()
print()
print()
x = torch.tensor(np.array([1,2,3,4,5], dtype=float))
y_true = torch.tensor(np.array([2,4,6,8, 10], dtype=float))
w = torch.tensor(np.array(0.5), requires_grad=True)
learning_rate = 0.01

optimize = torch.optim.SGD([w], learning_rate)

for i in range(50):
    
    y_pred = forward(x, w)
    loss_ = nn.MSELoss()(y_true, y_pred)
    loss_.backward()
    
    optimize.step()
    optimize.zero_grad()
    
    # if i%5 == 0:
    #     print(f'Loss: {loss_:0.3f}, weight: {w:0.3f}, y:{y_true}, y_pred:{y_pred}')
        
########################################################FUlly pyTorch


import torch.nn as nn

print()
print()
print()
x = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32)
y_true = torch.tensor([[2],[4],[6],[8], [10]], dtype=torch.float32)
learning_rate = 0.01

model = nn.Linear(1, 1, bias = False)

optimize = torch.optim.SGD(model.parameters(), learning_rate)


for i in range(50):
    
    y_pred = model(x)
    loss_ = nn.MSELoss()(y_true, y_pred)
    loss_.backward()
    optimize.step()
    optimize.zero_grad()
    
    if i%5 == 0:
        print(f'Loss: {loss_:0.3f}, weight: {list(model.parameters())}, y:{y_true}, y_pred:{y_pred}')