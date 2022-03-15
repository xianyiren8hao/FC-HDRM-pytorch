from PyQt5.QtWidgets import QMainWindow, QApplication
from UI.NNwindow import Ui_NNWindow
from FCNN.myThread import FC_HDRM_Work


class MyMainWindow(QMainWindow, Ui_NNWindow):
    def __init__(self, parent=None):
        # 窗口初始化
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        # 部分组件设置
        self.CB_setchoice.addItem("训练集")
        self.CB_setchoice.addItem("测试集")
        self.CB_setchoice.setCurrentIndex(0)
        # 构建神经网络及其外围模型
        self.myWork = FC_HDRM_Work()
        # 连接特殊信号
        self.myWork.sig_dsok.connect(self.DSload_finished)
        self.myWork.sig_samok.connect(self.SamShow_finished)
        self.myWork.sig_wslok.connect(self.Wsl_finished)
        self.myWork.sig_nnstep.connect(self.NN_onestep)
        self.myWork.sig_nnok.connect(self.NN_finished)

    def Interact_ena(self, ena=True):
        ena1 = ena and self.myWork.dset_on
        self.PB_dsload.setEnabled(ena)
        self.PB_savewfile.setEnabled(ena)
        self.PB_loadwfile.setEnabled(ena)
        self.CB_setchoice.setEnabled(ena1)
        self.SB_imagecount.setEnabled(ena1)
        self.PB_nntrainone.setEnabled(ena1)
        self.PB_nntrainall.setEnabled(ena1)
        self.PB_nnttrain.setEnabled(ena1)
        self.PB_nnttest.setEnabled(ena1)

    def DSload_clicked(self):
        self.LE_state.setText("正在载入数据集……")
        self.Interact_ena(False)
        self.myWork.dset_path = self.LE_root.text()
        self.myWork.thr_state = 1
        self.myWork.start()

    def DSload_finished(self):
        self.LE_traininfo.setText(str(self.myWork.dset_train.data.size()))
        self.LE_testinfo.setText(str(self.myWork.dset_test.data.size()))
        if not self.myWork.dset_on:
            self.myWork.dset_on = True
        self.Schoice_activated(self.CB_setchoice.currentIndex())
        self.LE_state.setText("数据集载入成功！")

    def Schoice_activated(self, dsindex):
        self.Interact_ena(False)
        if dsindex == 0:
            self.SB_imagecount.setMaximum(len(self.myWork.dset_train) - 1)
        elif dsindex == 1:
            self.SB_imagecount.setMaximum(len(self.myWork.dset_test) - 1)
        self.myWork.sam_index = dsindex
        self.myWork.sam_count = self.SB_imagecount.value()
        self.myWork.thr_state = 2
        self.myWork.start()

    def Icount_vchanged(self, dscount):
        self.Interact_ena(False)
        self.myWork.sam_index = self.CB_setchoice.currentIndex()
        self.myWork.sam_count = dscount
        self.myWork.sam_fromS = True
        self.myWork.thr_state = 2
        self.myWork.start()

    def SamShow_finished(self):
        self.L_image.setPixmap(self.myWork.sam_img)
        self.LE_sampletar.setText(self.myWork.sam_str)
        self.Interact_ena(True)
        if self.myWork.sam_fromS:
            self.myWork.sam_fromS = False
            self.SB_imagecount.setFocus()

    def Savewf_clicked(self):
        self.LE_state.setText("权重文件正在保存……")
        self.Interact_ena(False)
        self.myWork.nn_wsave = True
        self.myWork.nn_wpath = self.LE_savewfile.text()
        self.myWork.thr_state = 3
        self.myWork.start()

    def Loadwf_clicked(self):
        self.LE_state.setText("权重文件正在读取……")
        self.Interact_ena(False)
        self.myWork.nn_wsave = False
        self.myWork.nn_wpath = self.LE_loadwfile.text()
        self.myWork.thr_state = 3
        self.myWork.start()

    def Wsl_finished(self):
        self.Interact_ena(True)
        if self.myWork.nn_wsave:
            self.LE_state.setText("权重文件保存成功！")
        else:
            self.LE_state.setText("权重文件读取成功！")

    def Ntone_clicked(self):
        self.LE_state.setText("正在准备单次训练……")
        self.Interact_ena(False)
        self.myWork.nn_onlyonce = True
        self.myWork.thr_state = 4
        self.myWork.start()

    def Ttrain_clicked(self):
        self.LE_state.setText("正在准备测试训练集……")
        self.Interact_ena(False)
        self.myWork.nn_onlyonce = False
        self.myWork.nn_trainset = True
        self.myWork.nn_needback = False
        self.myWork.nn_ifstatis = True
        self.myWork.thr_state = 4
        self.myWork.start()

    def Ttest_clicked(self):
        self.LE_state.setText("正在准备测试测试集……")
        self.Interact_ena(False)
        self.myWork.nn_onlyonce = False
        self.myWork.nn_trainset = False
        self.myWork.nn_needback = False
        self.myWork.nn_ifstatis = True
        self.myWork.thr_state = 4
        self.myWork.start()

    def Ntall_clicked(self):
        self.LE_state.setText("正在准备训练训练集……")
        self.Interact_ena(False)
        self.myWork.nn_onlyonce = False
        self.myWork.nn_trainset = True
        self.myWork.nn_needback = True
        self.myWork.nn_ifstatis = False
        self.myWork.thr_state = 4
        self.myWork.start()

    def NN_onestep(self, m_str):
        self.LE_state.setText(m_str)

    def NN_finished(self):
        if self.myWork.nn_onlyonce:
            self.LE1_set(self.myWork.nn_onelabs)
            self.LE2_set(self.myWork.nn_outcpu)
            f_str = "单次训练完成！"
        else:
            if self.myWork.nn_ifstatis:
                self.LE1_seti(self.myWork.sta_lab_num)
                self.LE2_seti(self.myWork.sta_lab_cor)
                self.LE3_set(self.myWork.sta_lab_rate)
                self.LE_allnum1.setText("%d" % self.myWork.sta_all_num)
                self.LE_allnum2.setText("%d" % self.myWork.sta_all_cor)
                self.LE_allnum3.setText("%.2f%%" % self.myWork.sta_all_rate)
            if self.myWork.nn_trainset:
                if self.myWork.nn_needback:
                    f_str = "训练训练集完成！"
                else:
                    f_str = "测试训练集完成！"
            else:
                if self.myWork.nn_needback:
                    f_str = "训练测试集完成！"
                else:
                    f_str = "测试测试集完成！"
        self.Interact_ena(True)
        self.LE_state.setText(f_str)

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
