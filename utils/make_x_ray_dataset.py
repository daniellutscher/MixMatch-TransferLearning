# Adapted from:
#https://www.kaggle.com/paultimothymooney/predicting-pathologies-in-x-ray-images
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import cv2
import zlib
import itertools
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler


def proc_images():
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    NoFinding = "No Finding" #0
    Consolidation="Consolidation" #1
    Infiltration="Infiltration" #2
    Pneumothorax="Pneumothorax" #3
    Edema="Edema" # 7
    Emphysema="Emphysema" #7
    Fibrosis="Fibrosis" #7
    Effusion="Effusion" #4
    Pneumonia="Pneumonia" #7
    Pleural_Thickening="Pleural_Thickening" #7
    Cardiomegaly="Cardiomegaly" #7
    NoduleMass="Nodule" #5
    Hernia="Hernia" #7
    Atelectasis="Atelectasis"  #6
    RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly","Hernia"]
    x = [] # images as arrays
    y = [] # labels
    WIDTH = 256
    HEIGHT = 256
    for img in images:
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        symbol = "|"

        #import ipdb; ipdb.set_trace()
        if symbol in finding:
            continue
        else:
            if NoFinding in finding:
                finding = 0
                #y.append(finding)
                #x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Consolidation in finding:
                finding = 1
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Infiltration in finding:
                finding = 2
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Pneumothorax in finding:
                finding = 3
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Edema in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Emphysema in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Fibrosis in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Effusion in finding:
                finding = 4
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Pneumonia in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Pleural_Thickening in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Cardiomegaly in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif NoduleMass in finding:
                finding = 5
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Hernia in finding:
                finding = 7
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            elif Atelectasis in finding:
                finding = 6
                y.append(finding)
                x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            else:
                continue
    return np.array(x), y


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.pardir(os.getcwd()))
    data_dir = os.path.join(base_dir, 'dataset')
    image_dir = os.path.join(data_dir, 'x_ray_images')

    images = glob(os.path.join(image_dir, '*.png'))
    labels = pd.read_csv(os.path.join(image_dir, 'sample_labels.csv'))

    #drop unused columns
    labels = labels[['Image Index','Finding Labels',
                     'Follow-up #','Patient ID',
                     'Patient Age','Patient Gender']]

    #create new columns for each decease
    pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia',
                      'Nodule','Pneumothorax','Atelectasis','Pleural_Thickening',
                      'Mass','Edema','Consolidation','Infiltration',
                      'Fibrosis','Pneumonia']
    for pathology in pathology_list :
        labels[pathology] = labels['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
    #remove Y after age
    labels['Age']=labels['Patient Age'].apply(lambda x: x[:-1]).astype(int)

    X, y = proc_images()


    dict_characters = {1: 'Consolidation', 2: 'Infiltration',
        3: 'Pneumothorax', 4:'Effusion', 5: 'Nodule Mass', 6: 'Atelectasis', 7: "Other Rare Classes"}


    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify = y)

    print(' --- NORMALIZATION VALUES --- ')
    print(f'TRAIN MEAN: {np.mean(X_train, axis=(0,1,2))/255}')
    print(f'TRAIN STD: {np.std(X_train, axis=(0,1,2))/255}')

    print(f'TRAIN MEAN: {np.mean(X_test, axis=(0,1,2))/255}')
    print(f'TRAIN STD: {np.std(X_test, axis=(0,1,2))/255}\n')




    print('SAVING ORIGINAL DATASET.\n')
    # images
    np.save(os.path.join(data_dir, 'xray_x_train_unbalanced.npy'), X_train)
    np.save(os.path.join(data_dir, 'xray_x_test_unbalanced.npy'), X_test)

    # labels
    np.save(os.path.join(data_dir, 'xray_y_train_unbalanced.npy'), Y_train)
    np.save(os.path.join(data_dir, 'xray_y_test_unbalanced.npy'), Y_test)

    print('UPSAMPLING ORIGINAL DATASET TO IMPROVE IMBALANCES\n')
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
    print(f'ORIGINAL CLASS BALACE: {class_weight}')

    # Make Data 1D for compatability upsampling methods
    X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
    X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
    X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
    X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)

    ros = RandomOverSampler(ratio='auto')
    X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
    X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

    Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 8)
    Y_testRosHot = to_categorical(Y_testRos, num_classes = 8)

    for i in range(len(X_trainRos)):
        height, width, channels = 256,256,3
        X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)

    for i in range(len(X_testRos)):
        height, width, channels = 256,256,3
        X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)


    class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
    print(f'NEW CLASS BALACE: {class_weight}')
















