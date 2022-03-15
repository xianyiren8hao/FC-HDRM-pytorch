from PyQt5.QtCore import QThread, pyqtSignal
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from PIL import Image
from FCNN.myFCNN import FC_HDRM_Net


class FC_HDRM_Work(QThread):
    sig_dsok = pyqtSignal()
    sig_samok = pyqtSignal()
    sig_wslok = pyqtSignal()
    sig_nnstep = pyqtSignal(str)
    sig_nnok = pyqtSignal()

    def __init__(self):
        super(FC_HDRM_Work, self).__init__()
        # 状态变量 - 线程控制
        self.thr_state = 0
        # 状态变量 - 数据读取
        self.dset_on = False
        self.dset_path = ""
        self.dset_trans = Compose([ToTensor()])
        self.dset_train = None
        self.dset_test = None
        self.dload_train = None
        self.dload_test = None
        # 状态变量 - 样本显示
        self.sam_index = None
        self.sam_count = None
        self.sam_trans = Compose([ToPILImage()])
        self.sam_fromS = False
        self.sam_tsr = None
        self.sam_img = None
        self.sam_tar = None
        self.sam_str = ""
        # 状态变量 - 网络训练
        self.nn_net = FC_HDRM_Net().cuda()
        self.nn_wsave = False
        self.nn_wpath = ""
        self.nn_onlyonce = False
        self.nn_needback = False
        self.nn_trainset = False
        self.nn_ifstatis = False
        self.nn_inputs = None
        self.nn_inlabs = None
        self.nn_outputs = None
        self.nn_onelabs = None
        self.nn_outcpu = None
        # 状态变量 - 数据统计
        self.sta_predict = None
        self.sta_correct = None
        self.sta_lab_num = None
        self.sta_lab_cor = None
        self.sta_lab_rate = None
        self.sta_all_num = None
        self.sta_all_cor = None
        self.sta_all_rate = None

    def run(self):
        # 0:初始化状态
        # 1:数据读取
        # 2:样本展示
        # 3:存取权重文件
        # 4:神经网络运行
        if self.thr_state == 1:
            self.Dset_init()
            self.sig_dsok.emit()
        elif self.thr_state == 2:
            self.Sam_get()
            self.sig_samok.emit()
        elif self.thr_state == 3:
            self.Wfile_sl()
            self.sig_wslok.emit()
        elif self.thr_state == 4:
            self.NN_opa()
            self.sig_nnok.emit()

    def Dset_init(self):
        self.dset_train = MNIST(
            root=self.dset_path, train=True, transform=self.dset_trans, download=True)
        self.dset_test = MNIST(
            root=self.dset_path, train=False, transform=self.dset_trans, download=True)
        self.dload_train = DataLoader(
            self.dset_train, batch_size=1000, shuffle=True, drop_last=False)
        self.dload_test = DataLoader(
            self.dset_test, batch_size=1000, shuffle=False, drop_last=False)

    def Sam_get(self):
        # 获取样本的PIL图像
        if self.sam_index == 0:
            self.sam_tsr, self.sam_tar = self.dset_train[self.sam_count]
        elif self.sam_index == 1:
            self.sam_tsr, self.sam_tar = self.dset_test[self.sam_count]
        self.sam_img = self.sam_trans(self.sam_tsr)
        # 获取的数据转入cuda中
        self.sam_tsr = self.sam_tsr.cuda()
        self.sam_tar = torch.tensor(self.sam_tar).cuda()
        # 将PIL图像转成QPixmap
        self.sam_img = self.sam_img.resize((280, 280), Image.NEAREST)
        self.sam_img = self.sam_img.toqpixmap()
        self.sam_str = self.dset_train.classes[self.sam_tar]

    def Wfile_sl(self):
        if self.nn_wsave:
            torch.save(self.nn_net.state_dict(), self.nn_wpath)
        else:
            file = torch.load(self.nn_wpath)
            self.nn_net.load_state_dict(file)

    def NN_opa(self):
        # 确定迭代器
        if self.nn_onlyonce:
            loader_iter = enumerate([0], 0)
        elif self.nn_trainset:
            loader_iter = enumerate(self.dload_train, 0)
        else:
            loader_iter = enumerate(self.dload_test, 0)
        # 准备数据统计
        if self.nn_ifstatis:
            self.sta_lab_num = torch.zeros(10, dtype=torch.int).cuda()
            self.sta_lab_cor = torch.zeros(10, dtype=torch.int).cuda()
        # 进入主循环
        for i, datas in loader_iter:
            # 打印消息
            if self.nn_onlyonce:
                m_str = "正在进行单次训练……"
            elif self.nn_trainset:
                if self.nn_needback:
                    m_str = "正在训练训练集：{}".format(i + 1)
                else:
                    m_str = "正在测试训练集：{}".format(i + 1)
            else:
                if self.nn_needback:
                    m_str = "正在训练测试集：{}".format(i + 1)
                else:
                    m_str = "正在测试测试集：{}".format(i + 1)
            self.sig_nnstep.emit(m_str)
            # 载入初始数据
            if self.nn_onlyonce:
                self.nn_inputs = Variable(self.sam_tsr.unsqueeze(0))
                self.nn_inlabs = Variable(self.sam_tar.unsqueeze(0))
            else:
                self.nn_inputs, self.nn_inlabs = datas
                self.nn_inputs = Variable(self.nn_inputs.cuda())
                self.nn_inlabs = Variable(self.nn_inlabs.cuda())
            # 前向与后向
            if self.nn_onlyonce or self.nn_needback:
                self.nn_net.opt.zero_grad()
            self.nn_outputs = self.nn_net(self.nn_inputs)
            if self.nn_onlyonce or self.nn_needback:
                NNerror = self.nn_net.loss(self.nn_outputs, self.nn_inlabs)
                NNerror.backward()
                self.nn_net.opt.step()
            # 数据统计
            if self.nn_ifstatis:
                self.sta_predict = torch.max(self.nn_outputs, 1).indices
                self.sta_correct = (self.sta_predict == self.nn_inlabs)
                for j in range(10):
                    self.sta_lab_num[j] += (self.nn_inlabs == j).sum()
                    self.sta_lab_cor[j] += (self.sta_correct * (self.nn_inlabs == j)).sum()
        # 后续处理
        if self.nn_onlyonce:
            self.nn_onelabs = torch.zeros(10)
            self.nn_onelabs[self.sam_tar] = 1.0
            self.nn_onelabs = self.nn_onelabs.numpy().tolist()
            self.nn_outcpu = self.nn_outputs.cpu()
            self.nn_outcpu = self.nn_outcpu.detach().squeeze(0)
            self.nn_outcpu = F.softmax(self.nn_outcpu, dim=0)
            self.nn_outcpu = self.nn_outcpu.numpy().tolist()
        elif self.nn_ifstatis:
            self.sta_all_num = self.sta_lab_num.sum()
            self.sta_all_cor = self.sta_lab_cor.sum()
            self.sta_lab_rate = 100.0 * self.sta_lab_cor / self.sta_lab_num
            self.sta_all_rate = 100.0 * self.sta_all_cor / self.sta_all_num
            self.sta_lab_num = self.sta_lab_num.cpu().numpy().tolist()
            self.sta_lab_cor = self.sta_lab_cor.cpu().numpy().tolist()
            self.sta_lab_rate = self.sta_lab_rate.cpu().numpy().tolist()
            self.sta_all_num = self.sta_all_num.cpu().numpy().tolist()
            self.sta_all_cor = self.sta_all_cor.cpu().numpy().tolist()
            self.sta_all_rate = self.sta_all_rate.cpu().numpy().tolist()
