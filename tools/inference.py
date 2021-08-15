from mmdet.apis import init_detector, inference_detector
import mmcv
import sys
import time
import numpy as np
import collections
# Form implementation generated from reading ui file './untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.init_detection()
        self.init_logo()
        self.init_slots()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1047, 1029)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(90, 40, 811, 81))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.verticalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(80, 140, 821, 632))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setMaximumSize(QtCore.QSize(512, 512))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("4992.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(80, 800, 821, 151))
        self.textEdit.setObjectName("textEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(80, 780, 64, 15))
        self.label_2.setObjectName("label_2")
        self.layoutWidget.raise_()
        self.verticalLayoutWidget.raise_()
        self.textEdit.raise_()
        self.label_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1047, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.close)
        self.pushButton_2.clicked.connect(self.label.update)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_4.setText(_translate("MainWindow", "open image file "))
        self.pushButton_2.setText(_translate("MainWindow", "Detection"))
        self.pushButton_3.setText(_translate("MainWindow", "show_result"))
        self.pushButton.setText(_translate("MainWindow", "Exit"))
        self.label_2.setText(_translate("MainWindow", "result"))
    def init_slots(self):
        self.pushButton_4.clicked.connect(self.button_image_open)
        self.pushButton_2.clicked.connect(self.detection)
        self.pushButton_3.clicked.connect(self.show_result)
    def init_logo(self):
        pix=QtGui.QPixmap('result.jpg')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)
    def button_image_open(self):
        self.img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        self.label.setPixmap(QtGui.QPixmap(self.img_name))
    def init_detection(self):
        config_file = '../configs/pascal_voc/cascade_food_voc.py'
        checkpoint_file = './work_dirs/cascade_food_voc/epoch_24.pth'
        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
    def detection(self):
#        config_file = '/home/guozebin/mmdetection/configs/pascal_voc/cascade_food_voc.py'
#        checkpoint_file = '/home/guozebin/mmdetection/tools/work_dirs/cascade_food_voc/epoch_24.pth'
        # build the model from a config file and a checkpoint file
        #model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.result = inference_detector(self.model, self.img_name)
        self.model.show_result(self.img_name, self.result, out_file='result.jpg')
        self.label.setPixmap(QtGui.QPixmap('result.jpg'))
    def show_result(self):
        Calories_dict = {1: '100', 2: '90', 3: '30', 4: '20', 5: '15', 6: '90', 7: '50', 8: '8', 9: '45', 10: '25'}
        class_names = ('pingguo', 'xiangjiao', 'fanqie', 'huanggua', 'xigua', 'li', 'juzi',
                       'caomei', 'putao', 'mihoutao')
        bbox_result = self.result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        inds = bboxes[:, -1] > 0.5
        food_list = inds * np.add(labels, 1)
        m = dict(collections.Counter(food_list))
        Calor = []
        str=''
        self.textEdit.setText(str)
        for i, j in zip(m.keys(), m.values()):
            if i == 0:
                continue
            print('Class: {}  Calories: {}  number: {}'.format(class_names[i - 1], int(Calories_dict[i]) * j, j))
            str='Class: {}  Calories: {}  number: {}'.format(class_names[i - 1], int(Calories_dict[i]) * j, j)
            self.textEdit.append(str)
            Calor.append(int(Calories_dict[i]) * j)
        print('Total calories:{}'.format(sum(Calor)))
        str1='Total calories:{}'.format(sum(Calor))
        self.textEdit.append(str1)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())

#def Calories_show(result,score_thr):
#    Calories_dict = {1: '100', 2: '90', 3: '30', 4: '20', 5: '15', 6: '90', 7: '50', 8: '8', 9: '45', 10: '25'}
#    class_names=('pingguo', 'xiangjiao', 'fanqie', 'huanggua', 'xigua', 'li', 'juzi',
#               'caomei', 'putao', 'mihoutao' )
#    bbox_result = result
#    bboxes = np.vstack(bbox_result)
#    labels = [
#        np.full(bbox.shape[0], i, dtype=np.int32)
#        for i, bbox in enumerate(bbox_result)
#    ]
#    labels = np.concatenate(labels)
#    inds= bboxes[:,-1]>score_thr
#    food_list=inds*np.add(labels,1)
#    m = dict(collections.Counter(food_list))
#    Calor = []
#    for i, j in zip(m.keys(), m.values()):
#        if i==0:
#            continue
#        print('Class: {}  Calories: {}  number: {}'.format(class_names[i-1], int(Calories_dict[i]) * j, j))
#        Calor.append(int(Calories_dict[i]) * j)
#    print('Total calories:{}'.format(sum(Calor)))
#
## Specify the path to model config and checkpoint file
#config_file = '/home/guozebin/mmdetection/configs/pascal_voc/cascade_food_voc.py'
#checkpoint_file = '/home/guozebin/mmdetection/tools/work_dirs/cascade_food_voc/epoch_24.pth'
#
## build the model from a config file and a checkpoint file
#model = init_detector(config_file, checkpoint_file, device='cuda:0')
#
## test a single image and show the results
#img = '/home/guozebin/mmdetection/tools/14256401205096.jpg'
#a=time.time()# or img = mmcv.imread(img), which will only load it once
#result = inference_detector(model, img)
#Calories_show(result,score_thr=0.5)
#
#
#
#print(time.time()-a)
## visualize the results in a new window
#model.show_result(img, result)
## or save the visualization results to image files
#model.show_result(img, result, out_file='result.jpg')
#
## test a video and show the results
##video = mmcv.VideoReader('video.mp4')
##for frame in video:
##    result = inference_detector(model, frame)
##    model.show_result(frame, result, wait_time=1)
#