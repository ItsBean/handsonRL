import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class QChartDrawer:
    def __init__(self, env, qnet, gamma):
        self.env = env
        self.qnet = qnet
        self.gamma = gamma
        self.file_count = 0

    # given q_chart (4,12*4),and draw the 3d-bar graph
    def draw(self, q_chart):
        # setup the figure and axes
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111, projection='3d')

        # fake data
        _x = np.arange(4)  # row
        _y = np.arange(48)  # column
        # 4 * 48 = 160 +32 = 192
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = q_chart.ravel()  # x + y
        bottom = np.zeros_like(top)
        width = depth = 1

        # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'] * 6

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color='y')
        ax1.set_title(f'Shaded q-chart-{self.file_count}')
        # try to save the plt to ./tmp/q_chart{count}.png, if directory doesn't exist, create it:
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        plt.savefig("./tmp/q_chart{}.png".format(self.file_count))
        self.file_count += 1
        plt.show()


# q_chart to nparray:
#q_chart = np.array(q_chart)

q_chart_drawer = QChartDrawer(None, None, None)
from load_csv import load_csv

data = load_csv()
print(data.shape)
# data has 300 data if I input arrow down ,then the next data will be drawn:
# get input
#instruction = input("input arrow down to draw next data:")
# if input is q ,then exit,if input is arrow down, then draw the next data:
while len(data) > 0:
    q_chart = data[0]
    q_chart_drawer.draw(q_chart)
    data = np.delete(data, 0, 0)
    #print(len(data))
    #instruction = input("input arrow down to draw next data:")

    # q_chart_drawer.draw(q_chart)
