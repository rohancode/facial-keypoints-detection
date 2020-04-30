import cv2

import pandas as pd
import numpy as np
import albumentations as albu

from tensorflow.keras.utils import Sequence
from util import load_image
np.random.seed(42)


class DataGenerator(Sequence):

    def __init__(self, df, bs, input_size=128, is_valid=False):

        self.df = df
        self.bs = bs
        self.input_size = input_size
        self.is_valid = is_valid
        self.aug = albu.Compose([albu.HorizontalFlip(p=0.5),
                                 albu.RandomBrightnessContrast(p=0.3),
                                 albu.ShiftScaleRotate(shift_limit=0.05,
                                                       scale_limit=0.1,
                                                       rotate_limit=30, p=1)],
                                p=1, keypoint_params=albu.KeypointParams('xy'))

    def __len__(self):

        return np.ceil(self.df.shape[0] / self.bs).astype(int)

    def on_epoch_end(self):

        if self.is_valid == False:
            self.df = shuffle(self.df, random_state=42)
            self.df.reset_index(inplace=True, drop=True)

    def __getitem__(self, idx):

        x_batch, y_batch = [], []
        start = idx * self.bs
        end = (idx + 1) * self.bs
        file_batch = self.df.newfile[start:end]
        points_batch = list(self.df.pts[start:end])
        for i, filename in enumerate(file_batch):
            image = load_image(filename, self.input_size)
            points = [p * self.input_size for p in points_batch[i]]
            # Augmentation
            if not self.is_valid:
                image = transform['keypoints']
                points = transform['image']

            x_batch.append(image)
            y_batch.append(points)

        x_batch = np.array(x_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)

        return x_batch, y_batch
