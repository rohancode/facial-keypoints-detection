import warnings
import ast
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils import combine_list, correction, str_to_list
from model import KeypointModel
from generator import DataGenerator

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


PATH = "./data"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
TRAIN_FOLDER = "train_images"
TEST_FOLDER = "test_images"

WEIGHT_FILENAME = "face_keypoint_mobile.h5"
INPUT_SIZE = 128
BATCH_SIZE = 64


def train():

    # Reading train and test csv file
    train_df = pd.read_csv(os.path.join(PATH, TRAIN_CSV))
    test_df = pd.read_csv(os.path.join(PATH, TEST_CSV))

    train_df, test_df = str_to_list(train_df), str_to_list(test_df)

    train_df['pts'] = train_df.apply(
        lambda x: combine_list(x.pts_x, x.pts_y), axis=1)
    test_df['pts'] = test_df.apply(
        lambda x: combine_list(x.pts_x, x.pts_y), axis=1)

    train_df.pts = train_df.pts.apply(lambda x: correction(x))
    test_df.pts = test_df.pts.apply(lambda x: correction(x))

    print(f"train shape : {train_df.shape} and test shape : {test_df.shape}")

    train_generator = DataGenerator(train_df,
                                    BATCH_SIZE,
                                    path=os.path.join(PATH, TRAIN_FOLDER),
                                    is_valid=False)

    test_generator = DataGenerator(test_df,
                                   BATCH_SIZE*2,
                                   path=os.path.join(PATH, TEST_FOLDER),
                                   is_valid=True)

    # Initialize  Model
    print("Loading Model ...")
    model = KeypointModel()
    print(model.summary(110))

    learning_rate = 0.001
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mae', metrics=['mse'])

    cbks = [ModelCheckpoint(f"./weights/{WEIGHT_FILENAME}", monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                              mode='min', min_delta=0.0001, min_lr=1e-5),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                          restore_best_weights=False)]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        verbose=1,
        callbacks=cbks,
        validation_data=test_generator,
        validation_steps=len(test_generator))


if __name__ == "__main__":
    train()
