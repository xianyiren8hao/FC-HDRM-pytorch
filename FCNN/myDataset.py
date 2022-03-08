import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from torch.utils.data import DataLoader
from PIL import Image


def DS_init(obj):
    ds_root = obj.LE_root.text()
    obj.transf = Compose([ToTensor(), Normalize(0.5, 0.5)])
    obj.MNIST_train = MNIST(
        root=ds_root, train=True, transform=obj.transf, download=True)
    obj.MNIST_test = MNIST(
        root=ds_root, train=False, transform=obj.transf, download=True)
    obj.LE_traininfo.setText(str(obj.MNIST_train.data.size()))
    obj.LE_testinfo.setText(str(obj.MNIST_test.data.size()))
    obj.trainLoader = DataLoader(
        obj.MNIST_train, batch_size=100, shuffle=True, drop_last=False)
    obj.testLoader = DataLoader(
        obj.MNIST_test, batch_size=100, shuffle=False, drop_last=False)
    obj.LE_state.setText("数据集载入成功！")


def Sample_show(obj, dsindex, dscount):
    if dsindex == 0:
        obj.sampleTsr, obj.sampleTar = obj.MNIST_train[dscount]
    elif dsindex == 1:
        obj.sampleTsr, obj.sampleTar = obj.MNIST_test[dscount]
    trans = Compose([Normalize(-1.0, 2.0), ToPILImage()])
    obj.sampleImg = trans(obj.sampleTsr)
    obj.sampleTsr = obj.sampleTsr.cuda()
    obj.sampleTar = torch.tensor(obj.sampleTar).cuda()
    obj.sampleImg = obj.sampleImg.resize((280, 280), Image.NEAREST)
    obj.sampleImg = obj.sampleImg.toqpixmap()
    obj.L_image.setPixmap(obj.sampleImg)
    obj.sampleStr = obj.MNIST_train.classes[obj.sampleTar]
    obj.LE_sampletar.setText(obj.sampleStr)
