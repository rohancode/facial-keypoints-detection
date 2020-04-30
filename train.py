import ast
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from generator import DataGenerator
from model import MobileNet
from utils import combine_list, load_img

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

PATH = "./data"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
WEIGHT_FILENAME = "face_keypoint_mobile.h5"
INPUT_SIZE = 128


def str2int(df):
    df.pts_x = df.pts_x.apply(lambda x: ast.literal_eval(x))
    df.pts_y = df.pts_y.apply(lambda x: ast.literal_eval(x))
    return df


def main():

    train_df = pd.read_csv(os.path.join(PATH, TEST_CSV))
    test_df = pd.read_csv(os.path.join(PATH, TRAIN_CSV))

    print(f"train shape : {train_df.shape} and test shape : {test_df.shape}")

    train_df, test_df = str2int(train_df), str2int(train_df)

    train_generator = DataGenerator(train_data, 64, False)
    valid_generator = DataGenerator(test_data, 64, True)

    model = MobileNet()

    learning_rate = 0.001
    adam = k.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mae', metrics=['mse'])

    cbks = callbacks.ModelCheckpoint(f"./weights/{WEIGHT_FILENAME}", monitor='val_loss', verbose=1,
                                        save_best_only=True, mode='min'),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                                          mode='min', min_delta=0.0001, min_lr=1e-5),
            callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                                            mode='min', baseline=None, restore_best_weights=False)]

    model.fit_generator(
        generator = train_generator,
        steps_per_epoch = np.ceil(float(train_df.shape[0]) / float(bs)),
        epochs = ep, verbose = 1,
        callbacks = cbks,
        validation_data = valid_generator,
        validation_steps = np.ceil(float(valid_df.shape[0]) / float(bs)))
