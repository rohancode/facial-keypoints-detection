import os
import cv2
import pandas as pd
import numpy as np
import albumentations as albu
from tensorflow.keras.utils import Sequence
from utils import load_image, combine_list

seed = 42
np.random.seed(seed)


class DataGenerator(Sequence):

    def __init__(self, df, bs, path='./data', input_size=128, is_valid=False):

        self.df = df
        self.bs = bs
        self.path = path
        self.input_size = input_size
        self.is_valid = is_valid
        self.augment = albu.Compose([albu.HorizontalFlip(p=0.5),
                                     albu.RandomBrightnessContrast(p=0.3),
                                     albu.ShiftScaleRotate(shift_limit=0.05,
                                                           scale_limit=0.1,
                                                           rotate_limit=30, p=0.8)],
                                    p=1, keypoint_params=albu.KeypointParams('xy'))

    def __len__(self):

        return np.ceil(self.df.shape[0] / self.bs).astype(int)

    def on_epoch_end(self):

        if self.is_valid == False:
            self.df = shuffle(self.df, random_state=seed)
            self.df.reset_index(inplace=True, drop=True)

    def __getitem__(self, idx):

        x_batch, y_batch = [], []
        start = idx * self.bs
        end = (idx + 1) * self.bs

        file_batch = self.df.newfile[start:end]
        points_batch = list(self.df.pts[start:end])

        for i, filename in enumerate(file_batch):

            image = load_image(os.path.join(
                self.path, filename), self.input_size)
            points = [p * self.input_size for p in points_batch[i]]
            # Augmentation
            if not self.is_valid:
                points = list(zip(iter(points[0::2]), iter(points[1::2])))
                transform = self.augment(image=image, keypoints=points)
                image = transform['image']
                points = np.array(transform['keypoints'])
                points = combine_list(points[:, 0], points[:, 1])

            x_batch.append(image)
            y_batch.append(points)

        x_batch = np.array(x_batch, np.float32) / 255.
        y_batch = np.array(y_batch, np.float32)

        return x_batch, y_batch
