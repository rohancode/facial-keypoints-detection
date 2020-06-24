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

face_xml_path = './data/haarcascade_frontalface_default.xml'
landmark_path = './data/nat_folder/baseline_NTA/data/train/landmarks.csv'
train_images_path = './data/nat_folder/baseline_NTA/data/train/images/'

split_train_images_path = './data/train_images/'
split_test_images_path = './data/test_images/'

split_train_landmark_new_path = './data/train.csv'
split_test_landmark_new_path = './data/test.csv'

face_cascade = cv2.CascadeClassifier(face_xml_path)

def facecrop(img):
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    facesize = []
    for (x, y, w, h) in faces:
        facesize.append(w*h)
    extra_ratio = 0.2
    i = np.argmax(facesize)
    x = faces[i][0]
    y = faces[i][1]
    w = faces[i][2]
    h = faces[i][3]

    y_new = y-int(extra_ratio*h)
    if y_new < 0:
        y_new = 0
    h_new = h+int(2*extra_ratio*h)
    x_new = x-int(extra_ratio*w)
    if x_new < 0:
        x_new = 0
    w_new = w+int(2*extra_ratio*w)

    img = img[y_new:y_new+h_new, x_new:x_new+w_new]

    return img, x_new, y_new, w_new, h_new

def make_landmark(train):
    landmark = pd.read_csv(landmark_path)
    if train == True:
        landmark = landmark.iloc[:1600]
    else:
        landmark = landmark.iloc[1600:]

    image_names = list(landmark['filename'])


    fx_set = []
    fy_set = []
    fw_set = []
    fh_set = []
    hw_set = []
    bbox_set = []
    for filename in image_names:
        try:
            img = cv2.imread(os.path.join(train_images_path,filename))
            f_img, fx, fy, fw, fh = facecrop(img)
            fx_set.append(fx)
            fy_set.append(fy)
            fw_set.append(fw)
            fh_set.append(fh)
            hw_set.append(max(fw,fh))
            bbox_set.append([fx, fy, fh, fw])


        except:
            img = cv2.imread(os.path.join(train_images_path,filename))
            fx_set.append(0)
            fy_set.append(0)
            fw_set.append(0)
            fh_set.append(0)
            hw_set.append(0)
            bbox_set.append([0, 0, 0, 0])

    landmark['bbox'] = bbox_set
    landmark['x'] = fx_set
    landmark['y'] = fy_set
    landmark['height'] = fh_set
    landmark['width'] = fw_set
    landmark['hw'] = hw_set

    landmark = landmark[landmark['height'] != 0]

    cname_x = [x for x in landmark.columns if 'X' in x]
    cname_y = [x for x in landmark.columns if 'Y' in x]
    df = pd.DataFrame()
    pts_x = []
    pts_y = []
    for i, r in landmark.iterrows():
        x = []
        y = []
        for c in cname_x:
            x.append(r[c])
        pts_x.append(x)
        for c in cname_y:
            y.append(r[c])
        pts_y.append(y)

    landmark['pts_x'] = pts_x
    landmark['pts_y'] = pts_y

    landmark.drop(cname_x, axis=1, inplace=True)
    landmark.drop(cname_y, axis=1, inplace=True)

    landmark.pts_x = landmark.pts_x.apply(lambda x: list(map(float, x)))
    landmark.pts_y = landmark.pts_y.apply(lambda x: list(map(float, x)))
    landmark.bbox = landmark.bbox.apply(lambda x: list(map(int, x)))

    landmark.bbox = landmark.bbox.apply(
        lambda x: [int(x[0]-0.1*x[0]), int(x[1]-0.1*x[1]), int(x[2]+0.10*x[2]), int(x[3]+0.10*x[3])])

    landmark['x'] = landmark.bbox.apply(lambda x: int(x[0]))
    landmark['y'] = landmark.bbox.apply(lambda x: int(x[1]))
    landmark['height'] = landmark.bbox.apply(lambda x: int((x[3]-x[1])))
    landmark['width'] = landmark.bbox.apply(lambda x: int((x[2]-x[0])))
    landmark['hw'] = landmark.apply(lambda x: max(x.height, x.width), axis=1)

    return landmark

def normalize(row):
    row[7] = [np.round((px - row[2])/row[6], 4) for px in row[7]]
    row[8] = [np.round((py - row[3])/row[6], 4) for py in row[8]]
    return row

def crop_save(row, path):
    filepath = train_images_path+row[0]

    img = cv2.imread(filepath)
    img = img[row[3]:row[3]+row[5], row[2]:row[2]+row[5]]
    t, b, l, r = 0, 0, 0, 0
    if img.shape[0] != row[5]:
        b = (row[5]-img.shape[0])
    if img.shape[1] != row[5]:
        r = (row[5]-img.shape[1])
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT)
    cv2.imwrite(path+row[-1], img)

landmark = make_landmark(True)
landmark = np.array(Parallel(n_jobs=6)(delayed(normalize)(i)
                                     for i in landmark.values))
landmark = pd.DataFrame(data=landmark, columns=[
        'filename', 'bbox', 'x', 'y', 'height', 'width', 'hw', 'pts_x', 'pts_y'])

landmark['newfile'] = list(range(0, landmark.shape[0]))
landmark['newfile'] = landmark.newfile.apply(lambda x: str(x)+'.jpg')

_ = Parallel(n_jobs=6)(delayed(crop_save)(i, split_train_images_path)
                               for i in landmark.values)
landmark.to_csv(split_train_landmark_new_path, index=False)

landmark = make_landmark(False)
landmark = np.array(Parallel(n_jobs=6)(delayed(normalize)(i)
                                     for i in landmark.values))
landmark = pd.DataFrame(data=landmark, columns=[
        'filename', 'bbox', 'x', 'y', 'height', 'width', 'hw', 'pts_x', 'pts_y'])

landmark['newfile'] = list(range(0, landmark.shape[0]))
landmark['newfile'] = landmark.newfile.apply(lambda x: str(x)+'.jpg')

_ = Parallel(n_jobs=6)(delayed(crop_save)(i, split_test_images_path)
                               for i in landmark.values)
landmark.to_csv(split_test_landmark_new_path, index=False)
