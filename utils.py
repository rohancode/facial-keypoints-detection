import warnings
import cv2
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

np.random.seed(42)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)


def plot_keypoint_df(df, path, size=128):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flat
    df = df.sample(8)
    df.reset_index(inplace=True, drop=True)
    for i in range(df.shape[0]):
        img = load_image(os.path.join(path, df.newfile[i]), size)
        axes[i].imshow(img)
        for j in range(len(df.pts_x[i])):
            cir = patches.Circle(
                (float(df.pts_x[i][j]) * size,
                 float(df.pts_y[i][j]) * size), 0.7,
                color='red')
            axes[i].add_patch(cir)
    plt.show()


def plot_keypoint_img(images, labels, name=None):
    fig, axes = plt.subplots(2, 4, figsize=(25, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        x = labels[i][0::2]
        y = labels[i][1::2]
        for j in range(98):
            cir = patches.Circle((float(x[j]), float(y[j])), 0.5, color='red')
            ax.add_patch(cir)
    if name is not None:
        fig.savefig(name)


def plot_sample_img(data, dic, from_=0, to=16, figsize=(9, 6), n_i=4, n_j=4):
    ims = [open_image(data.id[i]) for i in range(from_, to)]
    label = [data.label[i] for i in range(from_, to)]
    fig, axes = plt.subplots(n_i, n_j, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        display_img(ims[i], dic=dic, ax=ax, label=label[i])
    plt.tight_layout(pad=0.1)


def display_img(im, dic, prediction=None, figsize=None, ax=None, alpha=None, label=0, pred=False):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    if pred:
        ax.text(1, 10, 'pred: %s (%.2f)' % (dic[np.argmax(prediction)], np.max(prediction)),
                color='w', backgroundcolor='k', alpha=0.8)
    ax.text(1, 30, '%s' % dic[label], color='k',
            backgroundcolor='w', alpha=0.8)
    ax.set_axis_off()
    return ax


def open_image(fn):

    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None:
                raise OSError('File not recognized by opencv: %d', fn)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


# def correct_pts(x):
#     x = [input_size - 0.001 if p >= input_size else p for p in x]
#     x = [0.0 if p <= 0.0 else p for p in x]
#     return x


def correction(x):
    x = [1.0 if p > 1.0 else p for p in x]
    x = [0.0 if p < 0.0 else p for p in x]
    return x


def combine_list(l1, l2):
    assert len(l1) == len(l2), print('List dont have equal length')
    l = np.empty(len(l1)*2)
    l[0::2] = l1
    l[1::2] = l2
    return l


def load_image(path, input_size):
    img = cv2.imread(path)
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
