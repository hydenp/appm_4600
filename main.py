import numpy as np
import mypkg


if __name__ == '__main__':
    plt = mypkg.my2DPlot(lambda x : np.sin(x) + 1,0.,10.)
    plt.labels('x','y')
    plt.addPlot(lambda x : np.cos(x) + 1)
    plt.dotted()
    plt.color('black')
    plt.logy()
    plt.logx()
    plt.savefig('figure.pdf')
    plt.show()
