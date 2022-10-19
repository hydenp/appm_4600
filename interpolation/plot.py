import numpy as np
from matplotlib import pyplot as plt


def plot_int(x_eval_pts, f_exact: np.array, y_eval_pts: np.array, N: int, save=False, file_prefix='') -> None:
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 7)

    # plot interpolation with exact
    ax[0].plot(x_eval_pts, f_exact, label='exact')
    ax[0].plot(x_eval_pts, y_eval_pts, 'r--', label='interpolation')
    ax[0].legend(loc='upper right')
    ax[0].set_title(f'Interpolation with N={N}')

    # plotting error
    err = abs(y_eval_pts - f_exact)
    ax[1].set_title('Error')
    ax[1].semilogy(x_eval_pts, err)

    if save:
        plt.savefig(f'{file_prefix}-n-{N}')
        plt.cla()
    else:
        plt.show()
