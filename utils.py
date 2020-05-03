import cv2
import os
import ast
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

np.random.seed(42)


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


def str_to_list(df):
    df.pts_x = df.pts_x.apply(lambda x: ast.literal_eval(x))
    df.pts_y = df.pts_y.apply(lambda x: ast.literal_eval(x))
    return df


def load_image(path, input_size):
    img = cv2.imread(path)
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
