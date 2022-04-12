import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torchvision import transforms


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 20, kernel_size=(5, 5), stride=1, padding=0, bias=True)),   # 1x28x28 --> 20x24x24
            ('relu1', nn.ReLU()),
            ('mp1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # 20x12x12
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(20, 50, kernel_size=(5, 5), stride=1, padding=0, bias=True)),  # 20x12x12 --> 50x8x8
            ('relu2', nn.ReLU()),
            ('mp2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))  # 50x4x4
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class FC1(nn.Module):
    def __init__(self):
        super(FC1, self).__init__()

        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(800, 500, bias=True)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.fc1(img)
        return output


class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(500, 10, bias=True)),
            ('sig4', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.fc2(img)
        return output


# LeNet5 Architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = C1()
        self.c2 = C2()
        self.fc1 = FC1()
        self.fc2 = FC2()

    def forward(self, img):
        # (20,12,12)
        opt_c1 = self.c1(img)
        np.save('./intermediate_layer_param/opt_c1.npy', opt_c1.detach().numpy())

        # (50,4,4)
        opt_c2 = self.c2(opt_c1)
        np.save('./intermediate_layer_param/opt_c2.npy', opt_c2.detach().numpy())

        opt_flatten = opt_c2.view(img.size(0), -1)   # 50x4x4=800 --> (b, 800)

        # (500,)
        opt_fc1 = self.fc1(opt_flatten)  # 800x500
        np.save('./intermediate_layer_param/opt_fc1.npy', opt_fc1.detach().numpy())

        # (10,)
        opt_fc2 = self.fc2(opt_fc1)  # 500x10
        np.save('./intermediate_layer_param/opt_fc2.npy', opt_fc2.detach().numpy())

        return opt_fc2


if __name__ == '__main__':

    model = LeNet5()
    model.load_state_dict(torch.load('best.pt'))
    print('Successfully loaded from ckpt!\n')

    img_5_arr = np.genfromtxt('img_5_norm.txt')
    print(img_5_arr)

    img_5_tensor_2 = torch.tensor(img_5_arr)
    # dim: 2 [28,28]--> dim :3 [1,28,28]
    img_5_tensor_3 = torch.unsqueeze(img_5_tensor_2, 0)
    # dim: 3 [1,28,28]--> dim :4 [1,1,28,28]
    img_5_tensor_4 = torch.unsqueeze(img_5_tensor_3, 0)

    img_5_tensor_4 = img_5_tensor_4.to(torch.float32)
    print(type(img_5_tensor_4))
    print(img_5_tensor_4.shape)
    print(img_5_tensor_4)

    logits = model(img_5_tensor_4)
    print(logits)

    pred = logits.argmax(dim=1)
    print(pred)


