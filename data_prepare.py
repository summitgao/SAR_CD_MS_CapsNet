from __future__ import absolute_import, division

from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import scipy.io as sio
import numpy as np


def load_data(image_file, label_file):

    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    image = image_data['nrmap']
    label = label_data['im_gt']

    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image, label


def one_hot_transform(x, length):
    ont_hot_array = np.zeros([1, length])
    ont_hot_array[0, int(x)-1] = 1
    return ont_hot_array


def readdata(image_file, label_file, train_nsamples=1000, validation_nsamples=1000,
             windowsize=9, istraining=True, shuffle_number=None, batchnumber=5000, times=0):

    image, label = load_data(image_file, label_file)
    shape = np.shape(image)
    halfsize = int((windowsize - 1) / 2)
    number_class = np.max(label)
    Mask = np.zeros([shape[0], shape[1]])
    Mask[halfsize:shape[0] - halfsize, halfsize:shape[1] - halfsize] = 1
    label = label * Mask
    not_zero_raw, not_zero_col = label.nonzero()
    number_samples = len(not_zero_raw)
    number_samples11111 = len(not_zero_col)
    test_nsamples = number_samples - train_nsamples - validation_nsamples
    if train_nsamples + validation_nsamples >= number_samples:
        raise ValueError('train_nsamples + validation_nsamples bigger than total samples')

    if istraining:

        shuffle_number = np.arange(number_samples)
        np.random.shuffle(shuffle_number)
        print('shuffle_number',shuffle_number)
        shape1 = np.shape(shuffle_number)
        print('shuffle_number.size:',shape1[0])
        train_image = np.zeros([train_nsamples, windowsize, windowsize, shape[2]], dtype=np.float32)
        validation_image = np.zeros([validation_nsamples, windowsize, windowsize, shape[2]], dtype=np.float32)

        train_label = np.zeros([train_nsamples, number_class], dtype=np.uint8)
        validation_label = np.zeros([validation_nsamples, number_class], dtype=np.uint8)

        for i in range(train_nsamples):
            train_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[i]] - halfsize):(not_zero_raw[shuffle_number[i]] + halfsize + 1),
                                            (not_zero_col[shuffle_number[i]] - halfsize):(not_zero_col[shuffle_number[i]] + halfsize + 1), :]
            train_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[i]],
                                                  not_zero_col[shuffle_number[i]]], number_class)

        for i in range(validation_nsamples):
            validation_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[i+train_nsamples]] - halfsize):(not_zero_raw[shuffle_number[i+train_nsamples]] + halfsize + 1),
                                                 (not_zero_col[shuffle_number[i+train_nsamples]] - halfsize):(not_zero_col[shuffle_number[i+train_nsamples]] + halfsize + 1), :]
            validation_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[i+train_nsamples]],
                                                       not_zero_col[shuffle_number[i+train_nsamples]]], number_class)

        return [train_image, train_label, validation_image, validation_label], shuffle_number

    else:
        n_batch = test_nsamples // batchnumber
        if times > n_batch:

            return None

        if n_batch == times:

            batchnumber_test = test_nsamples - n_batch * batchnumber
            test_image = np.zeros([batchnumber_test, windowsize, windowsize, shape[2]], dtype=np.float32)
            test_label = np.zeros([batchnumber_test, number_class], dtype=np.uint8)

            for i in range(batchnumber_test):
                test_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] - halfsize):(not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] + halfsize + 1),
                                               (not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] - halfsize):(not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] + halfsize + 1), :]
                test_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]],
                                                     not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]]], number_class)

            return [test_image, test_label]

        if times < n_batch:

            test_image = np.zeros([batchnumber, windowsize, windowsize, shape[2]], dtype=np.float32)
            test_label = np.zeros([batchnumber, number_class], dtype=np.uint8)

            for i in range(batchnumber):
                test_image[i, :, :, :] = image[(not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] - halfsize):(not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] + halfsize + 1),
                                               (not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] - halfsize):(not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]] + halfsize + 1), :]
                test_label[i, :] = one_hot_transform(label[not_zero_raw[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]],
                                                     not_zero_col[shuffle_number[batchnumber*times+i+train_nsamples+validation_nsamples]]], number_class)

            return [test_image, test_label]
