import pandas as pd
import numpy as np
import os
from glob import glob
import random
import cv2
import zlib
import itertools
import sklearn
import argparse
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder


def proc_images(images, labels, diagnosis_labels_mapping, HEIGHT, WIDTH):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """

    x = [] # images as arrays
    y = [] # labels

    for img in images:
        base = os.path.basename(img)

        # Read and resize image
        full_size_image = cv2.imread(img)
        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        symbol = "|" # multi-label classification is not supported right now

        if symbol not in finding:
            for diagnosis, label in diagnosis_labels_mapping.items():
                if diagnosis in finding:
                    y.append(label)
                    x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT),
                                        interpolation=cv2.INTER_CUBIC))
                    break

    return np.array(x), y


def save_dataset(data_dir, X, y, train=True, balanced=True):

    train_string = 'train' if train else 'test'
    balanced_string = '' if balanced else '_unbalanced'

    np.save(os.path.join(data_dir,
            f'xray_x_{train_string}{balanced_string}.npy'), X)
    np.save(os.path.join(data_dir,
            f'xray_y_{train_string}{balanced_string}.npy'), y)


def main(args):
    images = glob(os.path.join(args.image_dir, '*.png'))
    labels = pd.read_csv(os.path.join(args.image_dir, 'sample_labels.csv'))


    print('PREPROCESSING IMAGES\n')
    X, y = proc_images(images, labels, args.diagnosis_labels_mapping,
                        args.height, args.width)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                        test_size=args.test_size, stratify = y)

    print(' --- NORMALIZATION VALUES --- ')
    print(f'TRAIN MEAN: {np.mean(X_train, axis=(0,1,2))/255}')
    print(f'TRAIN STD: {np.std(X_train, axis=(0,1,2))/255}')

    print(f'TEST MEAN: {np.mean(X_test, axis=(0,1,2))/255}')
    print(f'TEST STD: {np.std(X_test, axis=(0,1,2))/255}\n')


    print('SAVING ORIGINAL DATASET.\n')
    _ = save_dataset(args.data_dir, X = X_train, y = Y_train,
                     train=True, balanced=False)
    _ = save_dataset(args.data_dir, X = X_test, y = Y_test,
                     train=False, balanced=False)


    print('UPSAMPLING ORIGINAL DATASET TO IMPROVE IMBALANCES\n')
    class_weighting = cw.compute_class_weight('balanced',
                                                        np.unique(y),
                                                        y)
    print(f'ORIGINAL CLASS BALACE: {class_weighting}')

    # save original dimensions (except nr of samples) for reshaping later
    X_train_shape = list(X_train.shape[1:])
    X_test_shape = list(X_test.shape[1:])

    # Do the oversampling
    ros = RandomOverSampler(ratio='auto')
    X_train_balanced, Y_train_balanced = ros.fit_sample(
                              X = np.reshape(X_train, [X_train.shape[0], -1]),
                              y = Y_train)
    X_test_balanced, Y_test_balanced = ros.fit_sample(
                              X = np.reshape(X_test, [X_test.shape[0], -1]),
                              y = Y_test)

    # Reshape into original dimensions
    X_train_balanced = np.reshape(X_train_balanced,
                                 [len(X_train_balanced)] + X_train_shape)
    X_test_balanced = np.reshape(X_test_balanced,
                                 [len(X_test_balanced)] + X_test_shape)

    class_weight = cw.compute_class_weight('balanced',
                                           np.unique(Y_train_balanced),
                                           Y_train_balanced)
    print(f'NEW CLASS BALACE: {class_weight}\n')

    print('SAVING BALANCED DATASET.')
    _ = save_dataset(args.data_dir, X=X_train_balanced, y=Y_train_balanced,
                     train=True, balanced=True)
    _ = save_dataset(args.data_dir, X=X_test_balanced, y=Y_test,
                     train=False, balanced=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--height', type=int, default=256,
                    help='Height of preprocessed output image. Default is 256.')
    parser.add_argument('--width', type=int, default=256,
                help='Width of preprocessed output image. Default is 256.')
    parser.add_argument('--test_size', type=float, default=0.2,
                help='Size of test split in percent. Default is 0.2.')
    args = parser.parse_args()


    # set dataset directories
    args.data_dir = os.path.join(os.getcwd(), 'dataset')
    args.image_dir = os.path.join(args.data_dir, 'x_ray_images')

    args.diagnosis_labels_mapping = {'Consolidation': 1,
                                     'Infiltration': 2,
                                     'Pneumothorax':3,
                                     'Effusion': 4,
                                     'Nodule': 5,
                                     'Atelectasis': 6,
                                     'Edema': 7,
                                     'Emphysema': 7,
                                     'Fibrosis': 7,
                                     'Pneumonia': 7,
                                     'Pleural_Thickening': 7,
                                     'Cardiomegaly': 7,
                                     'Hernia': 7
                                    }

    main(args)
