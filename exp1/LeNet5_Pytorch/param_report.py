import torch
from LeNet5 import LeNet5

model = LeNet5()
model.load_state_dict(torch.load('best.pt'))
print('Successfully loaded from ckpt!\n')

keys = model.state_dict().keys()
print(keys)

c1_weight = model.c1.c1.c1.weight
print(c1_weight)

c1_bias = model.c1.c1.c1.bias
print(c1_bias)


torch.set_printoptions(profile="full")
text = open('parameter_2.txt', 'w')
param = list(model.parameters())
print(param, file=text)
text.close()

