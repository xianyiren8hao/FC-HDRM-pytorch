import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import optim


class FC_HDRM_Net(nn.Module):
    def __init__(self):
        super(FC_HDRM_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.afun = nn.Sigmoid()
        self.loss = CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), 0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.afun(self.fc1(x))
        x = self.afun(self.fc2(x))
        x = self.fc3(x)
        return x
