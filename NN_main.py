from PyQt5.QtWidgets import QMainWindow, QApplication
from UI.NNwindow import Ui_NNWindow
from FCNN.myDataset import DS_init, Sample_show
from FCNN.myFCNN import\
    FC_HDRM_Net, Save_wfile, Load_wfile, Train_once,\
    Test_trainset, Test_testset, Train_all


class MyMainWindow(QMainWindow, Ui_NNWindow):
    def __init__(self, parent=None):
        # 窗口初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 部分组件设置
        self.CB_setchoice.addItem("训练集")
        self.CB_setchoice.addItem("测试集")
        self.CB_setchoice.setCurrentIndex(0)
        # 状态变量 - 数据集
        self.dataset_on = False
        self.transf = None
        self.MNIST_train = None
        self.MNIST_test = None
        self.trainLoader = None
        self.testLoader = None
        # 状态变量 - 样本显示
        self.sampleTsr = None   # CUDA
        self.sampleImg = None   # CPU
        self.sampleTar = None   # CUDA
        self.sampleStr = ""     # CPU
        # 状态变量 - 神经网络
        self.myNet = FC_HDRM_Net().cuda()   # CUDA
        self.inputs = None      # CUDA
        self.labels = None      # CUDA
        self.outputs = None     # CUDA
        self.outlabs = None     # CPU
        self.out_cpu = None     # CPU
        # 状态变量 - 数据统计
        self.predict = None
        self.correct = None
        self.all_num = None
        self.cor_num = None
        self.total_all = None
        self.total_cor = None
        self.rate_num = None
        self.rate_total = None

    def DSload_clicked(self):
        DS_init(self)
        if not self.dataset_on:
            self.dataset_on = True
            self.CB_setchoice.setEnabled(True)
            self.Schoice_activated(self.CB_setchoice.currentIndex())
            self.SB_imagecount.setEnabled(True)
            self.PB_nntrainone.setEnabled(True)
            self.PB_nntrainall.setEnabled(True)
            self.PB_nnttrain.setEnabled(True)
            self.PB_nnttest.setEnabled(True)

    def Schoice_activated(self, dsindex):
        if dsindex == 0:
            self.SB_imagecount.setMaximum(len(self.MNIST_train) - 1)
        elif dsindex == 1:
            self.SB_imagecount.setMaximum(len(self.MNIST_test) - 1)
        dscount = self.SB_imagecount.value()
        Sample_show(self, dsindex, dscount)

    def Icount_vchanged(self, dscount):
        dsindex = self.CB_setchoice.currentIndex()
        Sample_show(self, dsindex, dscount)

    def Savewf_clicked(self):
        save_path = self.LE_savewfile.text()
        Save_wfile(self, save_path)

    def Loadwf_clicked(self):
        load_path = self.LE_loadwfile.text()
        Load_wfile(self, load_path)

    def Ntone_clicked(self):
        Train_once(self)
        self.LE1_set(self.outlabs.numpy().tolist())
        self.LE2_set(self.out_cpu.numpy().tolist())
        self.LE_state.setText("单次训练完成！")

    def Ttrain_clicked(self):
        Test_trainset(self)
        self.Finish_test()
        self.LE_state.setText("测训练集完成！")

    def Ttest_clicked(self):
        Test_testset(self)
        self.Finish_test()
        self.LE_state.setText("测测试集完成！")

    def Finish_test(self):
        self.LE1_seti(self.all_num)
        self.LE2_seti(self.cor_num)
        self.LE3_set(self.rate_num)
        self.LE_allnum1.setText("%d" % self.total_all)
        self.LE_allnum2.setText("%d" % self.total_cor)
        self.LE_allnum3.setText("%.2f%%" % self.rate_total)

    def Ntall_clicked(self):
        Train_all(self)
        self.LE_state.setText("一轮训练完成！")
        
    def LE1_set(self, num_list):
        self.LE_zero1.setText("%.4f" % num_list[0])
        self.LE_one1.setText("%.4f" % num_list[1])
        self.LE_two1.setText("%.4f" % num_list[2])
        self.LE_three1.setText("%.4f" % num_list[3])
        self.LE_four1.setText("%.4f" % num_list[4])
        self.LE_five1.setText("%.4f" % num_list[5])
        self.LE_six1.setText("%.4f" % num_list[6])
        self.LE_seven1.setText("%.4f" % num_list[7])
        self.LE_eight1.setText("%.4f" % num_list[8])
        self.LE_nine1.setText("%.4f" % num_list[9])

    def LE2_set(self, num_list):
        self.LE_zero2.setText("%.4f" % num_list[0])
        self.LE_one2.setText("%.4f" % num_list[1])
        self.LE_two2.setText("%.4f" % num_list[2])
        self.LE_three2.setText("%.4f" % num_list[3])
        self.LE_four2.setText("%.4f" % num_list[4])
        self.LE_five2.setText("%.4f" % num_list[5])
        self.LE_six2.setText("%.4f" % num_list[6])
        self.LE_seven2.setText("%.4f" % num_list[7])
        self.LE_eight2.setText("%.4f" % num_list[8])
        self.LE_nine2.setText("%.4f" % num_list[9])

    def LE3_set(self, num_list):
        self.LE_zero3.setText("%.2f%%" % num_list[0])
        self.LE_one3.setText("%.2f%%" % num_list[1])
        self.LE_two3.setText("%.2f%%" % num_list[2])
        self.LE_three3.setText("%.2f%%" % num_list[3])
        self.LE_four3.setText("%.2f%%" % num_list[4])
        self.LE_five3.setText("%.2f%%" % num_list[5])
        self.LE_six3.setText("%.2f%%" % num_list[6])
        self.LE_seven3.setText("%.2f%%" % num_list[7])
        self.LE_eight3.setText("%.2f%%" % num_list[8])
        self.LE_nine3.setText("%.2f%%" % num_list[9])
    
    def LE1_seti(self, num_list):
        self.LE_zero1.setText("%d" % num_list[0])
        self.LE_one1.setText("%d" % num_list[1])
        self.LE_two1.setText("%d" % num_list[2])
        self.LE_three1.setText("%d" % num_list[3])
        self.LE_four1.setText("%d" % num_list[4])
        self.LE_five1.setText("%d" % num_list[5])
        self.LE_six1.setText("%d" % num_list[6])
        self.LE_seven1.setText("%d" % num_list[7])
        self.LE_eight1.setText("%d" % num_list[8])
        self.LE_nine1.setText("%d" % num_list[9])

    def LE2_seti(self, num_list):
        self.LE_zero2.setText("%d" % num_list[0])
        self.LE_one2.setText("%d" % num_list[1])
        self.LE_two2.setText("%d" % num_list[2])
        self.LE_three2.setText("%d" % num_list[3])
        self.LE_four2.setText("%d" % num_list[4])
        self.LE_five2.setText("%d" % num_list[5])
        self.LE_six2.setText("%d" % num_list[6])
        self.LE_seven2.setText("%d" % num_list[7])
        self.LE_eight2.setText("%d" % num_list[8])
        self.LE_nine2.setText("%d" % num_list[9])


# 主程序
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
