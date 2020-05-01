import os
import argparse
from tensorflow import lite


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Convert keras model to tflite")

    parser.add_argument('--model_filename',
                        help='name of the h5 file',
                        default="face_keypoint_mobile.h5",
                        type=str)

    parser.add_argument("--model_folder",
                        help="path of folder containing model file",
                        default="./weights",
                        type=str)

    parser.add_argument('--tflite_filename',
                        help='name of the tflite file',
                        default="face_keypoint_mobile.tflite",
                        type=str)

    parser.add_argument("--tflite_folder",
                        help="path of folder where converted model is to be saved",
                        default="./weights",
                        type=str)

    return parser.parse_args()


def convert_model(model_path, tflite_path):

    converter = lite.TFLiteConverter.from_keras_model_file(model_path)
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)


if __name__ == "__main__":

    args = parse_args()

    model_path = os.path.join(args.model_folder, args.model_filename)
    tflite_path = os.path.join(args.tflite_folder, args.tflite_filename)

    print("Converting model...")
    convert_model(model_path, tflite_path)
