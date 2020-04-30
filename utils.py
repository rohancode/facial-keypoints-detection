import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

np.random.seed(42)


def plot_bbox_face(df, size=128, batch=1):

    fig, axes = plt.subplots(1, 4, figsize=(20, 20))
    index = batch * 4
    for _, ax in enumerate(axes.flat):
        im = np.array(Image.open(img_path / df.filename[index - 1]),
                      dtype=np.uint8)
        im = im / 255
        im = cv2.resize(im, (size, size))
        t = df[df.filename == df.filename[index - 1]]
        t.reset_index(inplace=True)

        for i in range(t.shape[0]):
            ax.imshow(im)
            for j in range(len(t.pts_x[i])):
                cir = patches.Circle(
                    (float(t.pts_x[i][j]) * size,
                     float(t.pts_y[i][j]) * size), 0.7,
                    color='red')
                ax.add_patch(cir)
        ax.text(-2, -3, 'File ID : ' + str(t.filename[0]),
                withdash=True, color='black', fontsize=15)
        index -= 1
    plt.show()


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


def correct_pts(x):
    x = [input_size - 0.001 if p >= input_size else p for p in x]
    x = [0.0 if p <= 0.0 else p for p in x]
    return x


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
