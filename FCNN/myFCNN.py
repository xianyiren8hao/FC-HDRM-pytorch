import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.autograd import Variable


class FC_HDRM_Net(nn.Module):
    def __init__(self):
        super(FC_HDRM_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.afun = nn.Sigmoid()
        self.loss = CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), 0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.afun(self.fc1(x))
        x = self.afun(self.fc2(x))
        x = self.fc3(x)
        return x


def Save_wfile(obj, path):
    torch.save(obj.myNet.state_dict(), path)
    obj.LE_state.setText("权重文件保存成功！")


def Load_wfile(obj, path):
    file = torch.load(path)
    obj.myNet.load_state_dict(file)
    obj.LE_state.setText("权重文件读取成功！")


def Train_once(obj):
    obj.inputs = Variable(obj.sampleTsr.unsqueeze(0))
    obj.labels = Variable(obj.sampleTar.unsqueeze(0))
    obj.myNet.opt.zero_grad()
    obj.outputs = obj.myNet(obj.inputs)
    error = obj.myNet.loss(obj.outputs, obj.labels)
    error.backward()
    obj.myNet.opt.step()
    obj.outlabs = torch.zeros(10)
    obj.outlabs[obj.sampleTar] = 1.0
    obj.out_cpu = obj.outputs.cpu()
    obj.out_cpu = obj.out_cpu.detach().squeeze(0)
    obj.out_cpu = F.softmax(obj.out_cpu, dim=0)


def Test_trainset(obj):
    obj.all_num = torch.zeros(10, dtype=torch.int).cuda()
    obj.cor_num = torch.zeros(10, dtype=torch.int).cuda()
    for i, datas in enumerate(obj.trainLoader, 0):
        print("正在处理：{}".format(i + 1))
        obj.inputs, obj.labels = datas
        obj.inputs = Variable(obj.inputs.cuda())
        obj.labels = Variable(obj.labels.cuda())
        obj.outputs = obj.myNet(obj.inputs)
        # print(obj.outputs)
        obj.predict = torch.max(obj.outputs, 1).indices
        # print(obj.predict)
        obj.correct = (obj.predict == obj.labels)
        for j in range(10):
            obj.all_num[j] += (obj.labels == j).sum()
            obj.cor_num[j] += (obj.correct * (obj.labels == j)).sum()
    obj.total_all = obj.all_num.sum()
    obj.total_cor = obj.cor_num.sum()
    obj.rate_num = 100.0 * obj.cor_num / obj.all_num
    obj.rate_total = 100.0 * obj.total_cor / obj.total_all
    obj.all_num = obj.all_num.cpu().numpy().tolist()
    obj.cor_num = obj.cor_num.cpu().numpy().tolist()
    obj.rate_num = obj.rate_num.cpu().numpy().tolist()
    obj.total_all = obj.total_all.cpu().numpy().tolist()
    obj.total_cor = obj.total_cor.cpu().numpy().tolist()
    obj.rate_total = obj.rate_total.cpu().numpy().tolist()
    print("处理完成！")


def Test_testset(obj):
    obj.all_num = torch.zeros(10, dtype=torch.int).cuda()
    obj.cor_num = torch.zeros(10, dtype=torch.int).cuda()
    for i, datas in enumerate(obj.testLoader, 0):
        print("正在处理：{}".format(i + 1))
        obj.inputs, obj.labels = datas
        obj.inputs = Variable(obj.inputs.cuda())
        obj.labels = Variable(obj.labels.cuda())
        obj.outputs = obj.myNet(obj.inputs)
        obj.predict = torch.max(obj.outputs.data, 1).indices
        obj.correct = (obj.predict == obj.labels)
        for j in range(10):
            obj.all_num[j] += (obj.labels == j).sum()
            obj.cor_num[j] += (obj.correct * (obj.labels == j)).sum()
    obj.total_all = obj.all_num.sum()
    obj.total_cor = obj.cor_num.sum()
    obj.rate_num = 100.0 * obj.cor_num / obj.all_num
    obj.rate_total = 100.0 * obj.total_cor / obj.total_all
    obj.all_num = obj.all_num.cpu().numpy().tolist()
    obj.cor_num = obj.cor_num.cpu().numpy().tolist()
    obj.rate_num = obj.rate_num.cpu().numpy().tolist()
    obj.total_all = obj.total_all.cpu().numpy().tolist()
    obj.total_cor = obj.total_cor.cpu().numpy().tolist()
    obj.rate_total = obj.rate_total.cpu().numpy().tolist()
    print("处理完成！")


def Train_all(obj):
    for i, datas in enumerate(obj.trainLoader, 0):
        print("正在训练：{}".format(i + 1))
        obj.inputs, obj.labels = datas
        obj.inputs = Variable(obj.inputs.cuda())
        obj.labels = Variable(obj.labels.cuda())
        obj.myNet.opt.zero_grad()
        obj.outputs = obj.myNet(obj.inputs)
        error = obj.myNet.loss(obj.outputs, obj.labels)
        error.backward()
        obj.myNet.opt.step()
    print("训练完成！")

