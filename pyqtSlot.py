
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton
from untitled import Ui_MainWindow

class Slot(QDialog,Ui_MainWindow):

    def __init__(self):

        super(Slot, self).__init__()
        self.setupUi(self)

    def connectSlot(self):
        # connect the signal and slot
        self.openFile.clicked.connect(Slot.openFileSlot)

    def openFileSlot(self):
        qFileDialog = QFileDialog()
        filename, tmp = QFileDialog.getOpenFileName(qFileDialog, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename is ' ':
            return
        #opencv read pic
        self.img = cv2.imread(filename, -1)
        if self.img.size == 1:
            return
        self.refreshShow()

    def imageProcessSlot(self):
        return

    def faceDetectSlot(self):
        return

    def refreshShow(self):
        #get the size and channel of pic , invert image to Qimage
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        #show the qimage
        self.srcImage.setPixmap(QPixmap.fromImage(self.qImg))
