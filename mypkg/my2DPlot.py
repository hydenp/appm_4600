import matplotlib.pyplot as plt
import numpy as np


class my2DPlot:
    def __init__(self, f, a, b):
        x = np.arange(a, b, 0.01)
        y = f(x)
        self.p = plt.plot(x, y)

    def show(self):
        # method to show the plot
        plt.show()

    def dotted(self):
        # make the first plot's line dotted
        self.p[0].set_linestyle('dotted')

    def labels(self, x, y):
        # add axiis labels
        plt.xlabel(x)
        plt.ylabel(y)

    def addPlot(self, f):
        # add a plot on top of the existing plot
        x = self.p[0].get_data()[0]
        y = f(x)
        self.p = plt.plot(x, y)

    def color(self, colorName):
        self.p[-1].set_color(colorName)

    def logy(self):
        # set the yscale to log
        plt.yscale('log')

    def logx(self):
        # set the xscale to log
        plt.xscale('log')

    def savefig(self, fileName):
        # save the figure to the specified filename
        plt.savefig(fileName)
