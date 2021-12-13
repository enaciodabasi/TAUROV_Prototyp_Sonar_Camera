import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QFormLayout, QPushButton, QLabel, QLineEdit, QTextEdit
from PyQt5.QtGui import QIntValidator, QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import PyQt5.QtWidgets
from PyQt5 import QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np

from brping import Ping1D

import cv2 as cv

import sys

globx = 0
globy = 0
globDistance = 0
glob_camera_values = [99, 199, 299]
glob_sensor_values = []
#mysonar = Ping1D()
#myping.connect_serial("/dev/ttyUSB0", 115200)
class MainWindow(QWidget):

    def __init__(self, parent=None):

        super().__init__(parent)

        m_mainLayout = QHBoxLayout()

        #Video Display and Mission Buttons
        m_partone = QVBoxLayout()
        m_missionbuttonslayout = QVBoxLayout()

        self.m_videolabel = QLabel(self)
        self.m_videolabel.setFixedSize(640, 480)
        self.m_videolabel.setStyleSheet("border: 2px solid black;")
        m_partone.addWidget(self.m_videolabel)

        m_executemissionpushbutton = QPushButton()
        m_executemissionpushbutton.setText("Execute Mission")
        m_missionbuttonslayout.addWidget(m_executemissionpushbutton)

        m_stopmissionbutton =QPushButton()
        m_stopmissionbutton.setText("Stop")
        m_missionbuttonslayout.addWidget(m_stopmissionbutton)

        m_switchmaskview = QPushButton()
        m_switchmaskview.setText("Switch to Mask View")
        m_missionbuttonslayout.addWidget(m_switchmaskview)

        m_partone.addLayout(m_missionbuttonslayout)
        m_mainLayout.addLayout(m_partone)

        # Color inputs, output values etc.

        m_parttwolayout = QVBoxLayout()
        m_valueslayout = QFormLayout()
        m_colorinputslayout = QFormLayout()

        colorinputintvalidator = QIntValidator()
        colorinputintvalidator.setBottom(0)
        colorinputintvalidator.setTop(255)

        self.lowhueledit = QLineEdit()
        self.lowhueledit.setValidator(colorinputintvalidator)
        self.lowsatledit = QLineEdit()
        self.lowsatledit.setValidator(colorinputintvalidator)
        self.lowvalledit = QLineEdit()
        self.lowvalledit.setValidator(colorinputintvalidator)
        self.highhueledit = QLineEdit()
        self.highhueledit.setValidator(colorinputintvalidator)
        self.highsatledit = QLineEdit()
        self.highsatledit.setValidator(colorinputintvalidator)
        self.highvalledit = QLineEdit()
        self.highvalledit.setValidator(colorinputintvalidator)


        m_colorinputslayout.addRow(QLabel("Low Hue:"), self.lowhueledit)
        m_colorinputslayout.addRow(QLabel("Low Saturation:"), self.lowsatledit)
        m_colorinputslayout.addRow(QLabel("Low Value:"), self.lowvalledit)
        m_colorinputslayout.addRow(QLabel("High Hue:"), self.highhueledit)
        m_colorinputslayout.addRow(QLabel("High Saturation:"), self.highsatledit)
        m_colorinputslayout.addRow(QLabel("High Value:"), self.highvalledit)
        m_parttwolayout.addLayout(m_colorinputslayout)

        self.xvalueledit = QLineEdit()
        self.xvalueledit.setReadOnly(True)
        self.yvalueledit = QLineEdit()
        self.yvalueledit.setReadOnly(True)
        self.distancefromcameraledit = QLineEdit()
        self.distancefromcameraledit.setReadOnly(True)
        self.sonarvalueledit = QLineEdit()
        self.sonarvalueledit.setReadOnly(True)

        m_valueslayout.addRow(QLabel("X-Position:"), self.xvalueledit)
        m_valueslayout.addRow(QLabel("Y-Position:"), self.yvalueledit)
        m_valueslayout.addRow(QLabel("Distance from the camera:"), self.distancefromcameraledit)
        m_valueslayout.addRow(QLabel("Distance from the BR Sonar:"), self.sonarvalueledit)
        m_parttwolayout.addLayout(m_valueslayout)

        m_mainLayout.addLayout(m_parttwolayout)

        # Data plotting

        self.m_plotlayout = QVBoxLayout()

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        #x1 = [1, 2, 3]
        #y1 = [1.2, 2.3, 3.4]
        #y2 = [1.1, 2.2, 3.3]

        #self.PlotCameraAndSensorData(y1, y2)
        #self.sc.axes.legend(loc="best")
        self.m_plotlayout.addWidget(self.sc)

        m_showplotBtn = QPushButton()
        m_showplotBtn.setText("Show Data Graph")
        self.m_plotlayout.addWidget(m_showplotBtn)

        m_mainLayout.addLayout(self.m_plotlayout)

        m_executemissionpushbutton.clicked.connect(self.OnExecuteButton)
        m_stopmissionbutton.clicked.connect(self.closeEvent)
        m_showplotBtn.clicked.connect(self.PlotCameraAndSensorData)

        self.setLayout(m_mainLayout)
        self.setWindowTitle("Deneme")

    def closeEvent(self, event):
        self.thread.stop()

        #event.accept()

    def PlotCameraAndSensorData(self):

        #secs = np.arange(1, len(glob_camera_values)+1, 1)

        secs = [1, 2, 3]
        y1 = [10, 20, 30]
        y2 = [15, 25, 35]
        self.sc.axes.plot(secs, glob_camera_values, label="camera distance")
        self.sc.axes.plot(secs, y2, label="sonar distance")
        self.sc.axes.legend(loc="best")
        self.m_plotlayout.addWidget(self.sc)

    @pyqtSlot(np.ndarray)
    def UpdateImage(self, frame):

        global globDistance

        qt_pixmap = self.ConvertCVToQT(frame)
        self.m_videolabel.setPixmap(qt_pixmap)
        self.xvalueledit.setText(str(globx))
        self.yvalueledit.setText(str(globy))
        self.distancefromcameraledit.setText(str(globDistance))

        global glob_camera_values
        global glob_sensor_values
        glob_camera_values.append(globDistance)

    def ConvertCVToQT(self, frame):

        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot()
    def OnExecuteButton(self):

        self.thread = ColoredObjectTracker(int(self.lowhueledit.text()),
                                           int(self.lowsatledit.text()),
                                           int(self.lowvalledit.text()),
                                           int(self.highhueledit.text()),
                                           int(self.highsatledit.text()),
                                           int(self.highvalledit.text()))

        self.thread.change_pixmap_signal.connect(self.UpdateImage)
        self.thread.start()

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, edgecolor="black")
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel("Time [s]")
        self.axes.set_ylabel("Distance [cm]")

        super(MplCanvas, self).__init__(fig)

class ColoredObjectTracker(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, lh=0, ls=0, lv=0, hh=0, hs=0, hv=0):
        super().__init__()
        self._run_flag = True

        self.lh = lh
        self.ls = ls
        self.lv = lv
        self.hh = hh
        self.hs = hs
        self.hv = hv

    def run(self):

        cap = cv.VideoCapture(0, cv.CAP_DSHOW)

        while self._run_flag:

            ret, frame = cap.read()

            if ret:

                hsv = cv.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, (self.lh, self.ls, self.lv), (self.hh, self.hs, self.hv))
                mask = cv.erode(mask, None, iterations=2)
                mask = cv.dilate(mask, None, iterations=2)

                contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
                center = None

                if len(contours) > 0:

                    c = max(contours, key=cv.contourArea)
                    ((self.x,self.y), self.r) = cv.minEnclosingCircle(c)
                    M = cv.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    if self.r > 10:
                        cv.circle(frame, (int(self.x), int(self.y)), int(self.r), (0, 255, 255), 2)
                        cv.circle(frame, center, 5, (0, 0, 255), -1)
                    global globx
                    global globy
                    global globDistance
                    globx = int(self.x)
                    globy = int(self.y)
                    globDistance = int((640*6.0)/((self.r)))
                    print(globDistance)


                self.change_pixmap_signal.emit(frame)


        cap.release()
        cv.destroyAllWindows()

    def stop(self):

        self._run_flag = False
        self.wait()


#myping = Ping1D()
#myping.connect_serial("/dev/ttyUSB0", 115200)

def GetValueFromBrSonar(mysonar):

        data = mysonar.get_distance()

        if data is not None:
            return data["distance"]
        else:
            return 0

if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
