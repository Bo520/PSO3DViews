from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.lines import Line2D

from Ui_PSO3DViews import Ui_PSO3DViews
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, pyqtSignal, Qt, QSize
import sys
import time
import numpy as np
import sip

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定字体为开题，防止图像中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from PSO import PSO
from threading import Thread
import math


class Figure_Canvas(FigureCanvas):
    def __init__(self, parent=None, ):
        self.fig = Figure()
        super(Figure_Canvas, self).__init__(self.fig)
        self.ax = self.fig.add_subplot(111)


class PSO3DViews(QMainWindow, Ui_PSO3DViews):
    signalFinishCalculate = pyqtSignal()
    c1 = 2  # 学习因子
    c2 = 2
    pop_size = 100  # 粒子群个体数
    dim = 3  # 变量数
    omega = 0.9  # 惯性因子
    times = 50  # 迭代次数
    x_min = 0
    x_max = 15
    y_min = 0
    y_max = 15
    z_min = 0
    z_max = 15
    playRate = 1  # 播放速率，即每秒刷新次数
    first_draw = True  # 第一次画图
    first_play = True  # 第一次播放动画
    caculate_result_history_cnt = 0  # 计算历史的统计
    caculate_result = []  # 存放历史记录
    caculate_result_history_position = []  # 存放历史记录
    caculate_result_history_best_index = []  # 存放历史记录

    fitnessFunction = '(2 * x1 ** 2 - 3 * x2 ** 2 - 4 * x1 + 5 * x2 + x3) * 100'  # 优化函数，用户可以自定义
    result = []
    position_history = [[]]  # 记录粒子历史移动信息，用于绘图
    best_position_index_history = []  # 记录粒子历史全局最优的索引信息，用于绘图

    def __init__(self, parent=None):
        super(PSO3DViews, self).__init__(parent)
        self.setupUi(self)
        self.statusbar.hide()
        self.pushButton_start_play.setEnabled(False)
        self.center()  # 主窗口居中
        self.pushButton_start_calculate.clicked.connect(self.runPSO)  # 开始计算PSO
        self.signalFinishCalculate.connect(self.handleFininshCalculate)  # 完成计算
        self.pushButton_start_play.clicked.connect(self.pathUpdate)  # 点击按钮更新图像
        self.comboBox_result_history.currentIndexChanged.connect(self.reloadHistory)  # 重新载入画板
        self.path3dFigureLayout = QGridLayout(self.groupBox)  # 画板布局
        self.path3dFigureLayout.setContentsMargins(0, 0, 0, 0)
        self.drawTimer = QTimer()  # 计时器
        # 设置groupbox背景图
        self.bgiLabel = QLabel()
        self.bgiLabel.setScaledContents(True)
        self.path3dFigureLayout.addWidget(self.bgiLabel)
        pix = QPixmap('img/bgi.jpg')
        picSize = QSize(self.bgiLabel.width(), self.bgiLabel.height())
        print(picSize)
        pix = pix.scaled(picSize, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.bgiLabel.setPixmap(pix)

    def Init_Widgets(self):
        self.PrepareData()
        self.Prepare3DPathCanvas()

    def PrepareData(self):
        self.x = np.array(self.position_history[0])
        self.X = self.x[:, 0]
        self.y = np.array(self.position_history[0])
        self.Y = self.y[:, 1]
        self.z = np.array(self.position_history[0])
        self.Z = self.z[:, 2]  # 初始数据为第一组数据

    def Prepare3DPathCanvas(self):
        self.path3dFigure = Figure_Canvas()  # 创建画布
        self.path3dFigureLayout.addWidget(self.path3dFigure)
        self.path3dFigure.ax.remove()
        self.ax3d = self.path3dFigure.fig.gca(projection='3d')
        self.ax3d.set_xlim(self.x_min, self.x_max)
        self.ax3d.set_title("三维空间PSO演示")
        self.ax3d.set_ylim(self.y_min, self.y_max)
        self.ax3d.set_zlim(self.z_min, self.z_max)
        self.ax3d.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
        self.ax3d.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
        self.ax3d.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
        self.Surf = self.ax3d.scatter(self.X, self.Y, self.Z, cmap='rainbow', c='g', s=10, marker='o')
        self.ax3d.scatter(self.X[0],
                          self.Y[0],
                          self.Z[0], c='g', label='其他点', s=10, marker='o')
        self.ax3d.scatter(self.X[self.best_position_index_history[0]],
                          self.Y[self.best_position_index_history[0]],
                          self.Z[self.best_position_index_history[0]], c='r', label='全局最优', s=40,
                          marker='X')
        self.ax3d.legend(loc="upper right", frameon=True)

    def runPSO(self):
        print("yes")
        self.c1 = self.doubleSpinBox_c1.value()  # 学习因子
        self.c2 = self.doubleSpinBox_c2.value()
        self.pop_size = self.spinBox_pop_size.value()  # 粒子群个体数
        self.dim = 3  # 变量数
        self.omega = self.doubleSpinBox_omega.value()  # 惯性因子
        self.times = self.spinBox_times.value()
        self.x_min = self.doubleSpinBox_x1_min.value()
        self.x_max = self.doubleSpinBox_x1_max.value()
        self.y_min = self.doubleSpinBox_x2_min.value()
        self.y_max = self.doubleSpinBox_x2_max.value()
        self.z_min = self.doubleSpinBox_x3_min.value()
        self.z_max = self.doubleSpinBox_x3_max.value()
        if (self.textEdit.toPlainText() != ''):
            self.fitnessFunction = self.textEdit.toPlainText()
        print(self.fitnessFunction)
        newPSO = PSO(self.pop_size, self.dim, self.omega, self.c1, self.c2, self.x_max, self.x_min, self.y_max,
                     self.y_min, self.z_max, self.z_min, self.fitnessFunction)
        newPSO.initial()
        solveThread = Thread(target=newPSO.solving, args=(self.times,))
        solveThread.start()  # 启动计算线程
        solveThread.join()
        # 保存历史记录
        self.caculate_result_history_cnt += 1
        self.caculate_result.append(newPSO.returnbest())
        self.caculate_result_history_position.append(newPSO.rerturn_position_history())
        self.caculate_result_history_best_index.append(newPSO.return_best_position_index_history())
        # 在combox记录历史信息
        self.comboBox_result_history.addItem("计算结果" + str(self.caculate_result_history_cnt) + str(
            time.strftime("(生成于：%Y-%m-%d %H:%M:%S)", time.localtime())))
        self.comboBox_result_history.setCurrentIndex(self.caculate_result_history_cnt - 1)

        self.position_history = newPSO.rerturn_position_history()
        self.best_position_index_history = newPSO.return_best_position_index_history()
        self.result = newPSO.returnbest()
        self.signalFinishCalculate.emit()  # 发送完成计算的信号

    def handleFininshCalculate(self):
        self.pushButton_start_play.setEnabled(True)

    def pathUpdate(self):
        self.playRate = self.doubleSpinBox_play_rate.value()
        self.playRate = (1 / self.playRate) * 1000
        self.drawTimer.start(int(self.playRate))
        self.ts = time.time()  # 开始时间
        print("ts:{}".format(self.ts))
        self.result2dFigure = Figure_Canvas()  # 绘制结果对比图
        self.resultX = []
        self.resultY = []
        self.resultX_min = 1
        self.resultX_max = self.times
        self.resultY_min = min(np.array(self.result)[:, 0])
        self.resultY_max = max(np.array(self.result)[:, 0])
        self.resultX.append(1)
        self.resultY.append(self.result[0][0])
        self.result2dFigure.ax.plot(self.resultX, self.resultY)
        self.result2dFigure.ax.set_xlim(self.resultX_min, self.resultX_max)
        self.result2dFigure.ax.set_ylim(self.resultY_min - 0.02 * abs(self.resultY_min),
                                        self.resultY_max + 0.02 * abs(self.resultY_max))
        self.result2dFigure.ax.set_xlabel("迭代次数")
        self.result2dFigure.ax.set_ylabel("最优值")
        self.result2dFigure.ax.set_title("PSO优化结果动态展示")
        self.result2dFigure.draw()
        # 设置3d图的尺寸策略
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.path3dFigure.sizePolicy().hasHeightForWidth())
        self.path3dFigure.setSizePolicy(sizePolicy)
        # 设置2d图的尺寸策略
        self.path3dFigureLayout.addWidget(self.result2dFigure)
        self.first_play = False
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.result2dFigure.sizePolicy().hasHeightForWidth())
        self.result2dFigure.setSizePolicy(sizePolicy)

        self.drawTimer.timeout.connect(self.pathUpdateTimer)

    def pathUpdateTimer(self):
        dt = time.time() - self.ts
        dt = math.ceil(dt)
        # print("dt:{} yyy".format(dt))
        if dt < self.times:
            self.x = np.array(self.position_history[dt])
            self.X = self.x[:, 0]
            self.y = np.array(self.position_history[dt])
            self.Y = self.y[:, 1]
            self.z = np.array(self.position_history[dt])
            self.Z = self.z[:, 2]
            self.ax3d.clear()
            self.ax3d.set_xlim(self.x_min, self.x_max)
            self.ax3d.set_ylim(self.y_min, self.y_max)
            self.ax3d.set_zlim(self.z_min, self.z_max)
            self.ax3d.set_title("三维空间PSO演示")
            self.ax3d.set_zlabel('x1')
            self.ax3d.set_ylabel('x2')
            self.ax3d.set_xlabel('x3')
            self.ax3d.scatter(self.X, self.Y, self.Z, c='g', s=10, marker='o')
            print([self.best_position_index_history[dt]])
            self.ax3d.scatter(self.X[0],
                              self.Y[0],
                              self.Z[0], c='g', label='其他点', s=10, marker='o')
            self.ax3d.scatter(self.X[self.best_position_index_history[dt]],
                              self.Y[self.best_position_index_history[dt]],
                              self.Z[self.best_position_index_history[dt]], c='r', label='全局最优', s=40,
                              marker='X')
            self.ax3d.legend(loc="upper right", frameon=True)
            self.path3dFigure.draw()
            # 绘制结果图
            self.result2dFigure.ax.cla()
            self.resultX.append(dt + 1)
            self.resultY.append(self.result[dt][0])
            self.result2dFigure.ax.plot(self.resultX, self.resultY)
            self.result2dFigure.ax.set_xlim(self.resultX_min, self.resultX_max)
            self.result2dFigure.ax.set_ylim(self.resultY_min - 0.02 * abs(self.resultY_min),
                                            self.resultY_max + 0.02 * abs(self.resultY_max))
            self.result2dFigure.ax.set_xlabel("迭代次数")
            self.result2dFigure.ax.set_ylabel("最优值")
            self.result2dFigure.ax.set_title("PSO优化结果动态展示")
            self.result2dFigure.draw()

    def center(self):  # 定义一个函数使得窗口居中显示
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def reloadHistory(self):  # 用户选择历史数据，更新画布
        index = self.comboBox_result_history.currentIndex()
        print(index)
        self.position_history = self.caculate_result_history_position[index]
        self.best_position_index_history = self.caculate_result_history_best_index[index]
        self.result = self.caculate_result[index]
        self.groupBox.show()
        if (self.first_draw == False):
            self.path3dFigureLayout.removeWidget(self.path3dFigure)
            sip.delete(self.path3dFigure)
        else:
            self.path3dFigureLayout.removeWidget(self.bgiLabel)
            sip.delete(self.bgiLabel)
        if (self.first_play == False):
            self.path3dFigureLayout.removeWidget(self.result2dFigure)
            sip.delete(self.result2dFigure)
            self.drawTimer.stop()
            self.first_play = True
        self.Init_Widgets()
        self.first_draw = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = PSO3DViews()
    ui.setWindowIcon(QIcon("img/head.jpg"))
    ui.show()
    sys.exit(app.exec_())
