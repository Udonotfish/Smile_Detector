import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from untitled import Ui_MainWindow

from pyqtSlot import Slot

if __name__ == '__main__':

    app = QApplication(sys.argv)

    MainWindow = QMainWindow()

    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)

    Slot.connectSlot(ui)

    MainWindow.show()

    sys.exit(app.exec_())
