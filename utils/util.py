import math
import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

P = np.array([[0, 0, 0, 1.],
              [0.14901961, 0.87058824, 0.45098039, 1.],
              [0.78431373, 0.96078431, 0.21176471, 1.],
              [0.87058824, 0.65098039, 0.14901961, 1.],
              [1.0, 0.34117647, 0.16862745, 1.],
              [0, 0.8, 0, 1.0],
              [0, 0, 0, 0.0]])
cmap = ListedColormap(P)

# cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen", "orange", "sandybrown", "skyblue"])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def most_common(lst):
    return max(set(lst), key=lst.count)

# Referred to https://stackoverflow.com/questions/59638155/how-to-set-0-to-white-at-a-uneven-color-ramp
def plot_class_band(gt_labels, pred_labels, save_file, acc=""):
    single_len = 100000
    num_splits = len(pred_labels) // single_len + 1
    # gt_labels = gt_labels + [6] * (num_splits * single_len - len(gt_labels))
    # pred_labels = pred_labels + [6] * (num_splits * single_len - len(pred_labels))
    for idx in range(1):
        gt_label_splits = gt_labels[idx*single_len : (idx+1)*single_len]
        pred_label_splits = pred_labels[idx*single_len : (idx+1)*single_len]

        len_labels = len(gt_label_splits)
        x, y = np.meshgrid(np.linspace(0, len_labels, len_labels+1), np.linspace(0, 1, 2))
        z_gt = np.array([gt_label_splits])
        z_pred = np.array([pred_label_splits])

        label_width = 0.005 * len_labels

        plt.subplot(2 * num_splits, 1, 2 * idx + 1)
        plt.pcolormesh(x, y, z_gt, cmap="viridis")
        plt.axis("off")
        if idx == 0:
            plt.title(os.path.basename(save_file).split(".")[0] + f"--{acc}")
        plt.margins(x=0)
        plt.margins(y=0)

        plt.subplot(2 * num_splits, 1, 2 * idx + 2)
        plt.pcolormesh(x, y, z_pred, cmap="viridis")
        plt.axis("off")
        plt.margins(x=0)
        plt.margins(y=0)
        plt.gcf().set_size_inches(label_width, 2)
    plt.savefig(save_file)
    plt.clf()
    # plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    kwargs.update({"cmap": "YlOrBr"})
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_loss(train_iter_loss, val_iter_loss, save_file):
    train_iters = [x[0] for x in train_iter_loss]
    train_losses = [x[1] for x in train_iter_loss]
    val_iters = [x[0] for x in val_iter_loss]
    val_losses = [x[1] for x in val_iter_loss]
    with plt.style.context(["classic", "classic"]):
   # with plt.style.context(["science", "no-latex"]):
        plt.plot(train_iters, train_losses, 'r', label="Train")  # plotting t, a separately
        plt.plot(val_iters, val_losses, 'b', label="Validation")  # plotting t, b separately
        plt.legend()
        plt.savefig(save_file)
        plt.clf()
        plt.close()