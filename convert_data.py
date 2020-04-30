import cv2
import os
import glob
import errno
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# making folders to save cropped images
if not os.path.exists('./data/train_images/'):
    os.mkdir('./data/train_images/')
if not os.path.exists('./data/test_images/'):
    os.mkdir('./data/test_images/')


def normalize(row):
    row[1] = [np.round((px - row[4])/row[8], 4) for px in row[1]]
    row[2] = [np.round((py - row[5])/row[8], 4) for py in row[2]]
    return row


def crop_save(row, path):
    """save cropped image"""
    filepath = './data/WFLW_images/'+row[0]

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    img = cv2.imread(filepath)
    img = img[row[5]:row[5]+row[8], row[4]:row[4]+row[8]]
    t, b, l, r = 0, 0, 0, 0
    if img.shape[0] != row[8]:
        b = (row[8]-img.shape[0])
    if img.shape[1] != row[8]:
        r = (row[8]-img.shape[1])
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT)
    cv2.imwrite(path+row[-1], img)


def read_file(path):
    """Read text file and extract filename,keypoints and bbox"""
    with open(path, 'r') as file:
        lines = [line.rstrip('\n') for line in file.readlines()]

    img_list, bbox = [], []
    pts_x = []
    pts_y = []
    for line in lines:
        l = line.split()
        img_list.append(l[-1])
        bbox.append(l[196:200])
        pts_x.append(l[:196][0::2])
        pts_y.append(l[:196][1::2])
    return img_list, pts_x, pts_y, bbox


def process(path, njobs=6):

    img_list, pts_x, pts_y, bbox = read_file(path)

    df = pd.DataFrame()
    df['filename'] = img_list
    df['pts_x'] = pts_x
    df['pts_y'] = pts_y
    df['bbox'] = bbox

    df.pts_x = df.pts_x.apply(lambda x: list(map(float, x)))
    df.pts_y = df.pts_y.apply(lambda x: list(map(float, x)))
    df.bbox = df.bbox.apply(lambda x: list(map(int, x)))

    # Expanding the bbox size so that whole face gets cropped
    df.bbox = df.bbox.apply(
        lambda x: [int(x[0]-0.1*x[0]), int(x[1]-0.1*x[1]), int(x[2]+0.10*x[2]), int(x[3]+0.10*x[3])])

    df['x'] = df.bbox.apply(lambda x: int(x[0]))
    df['y'] = df.bbox.apply(lambda x: int(x[1]))
    df['height'] = df.bbox.apply(lambda x: int((x[3]-x[1])))
    df['width'] = df.bbox.apply(lambda x: int((x[2]-x[0])))
    df['hw'] = df.apply(lambda x: max(x.height, x.width), axis=1)

    df = np.array(Parallel(n_jobs=njobs)(delayed(normalize)(i)
                                         for i in df.values))

    df = pd.DataFrame(data=df, columns=[
        'filename', 'pts_x', 'pts_y', 'bbox', 'x', 'y', 'height', 'width', 'hw'])

    # Creating filename  for cropped images
    df['newfile'] = list(range(0, df.shape[0]))
    df['newfile'] = df.newfile.apply(lambda x: str(x)+'.jpg')

    print("Saving Images...")
    # Saving Images
    _ = Parallel(n_jobs=njobs)(delayed(crop_save)(i, f'./data/{folder}_images/')
                               for i in df.values)

    print("Saving csv file...")
    # Saving dataframe
    df.to_csv(f'./data/{folder}.csv', index=False)


for file in glob.glob('./data/WFLW_annotations/list_98pt_rect_attr_train_test/*.txt'):

    folder = file.split('_')[-1].split('.')[0]
    print("Processing ", folder)
    process(file, njobs=6)
