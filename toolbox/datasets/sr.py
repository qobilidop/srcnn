import os

import h5py
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import list_pictures
from scipy.misc import imresize


def load_data(train_dir='Train', test_dir='Test/Set5'):
    x_train, y_train = generate(train_dir)
    x_test, y_test = generate(test_dir, save_path='test.h5', stride=21)
    return (x_train, y_train), (x_test, y_test)


def generate(directory, save_path='train.h5', size_input=33, size_label=21,
             scale=3, stride=14):
    # settings
    directory = os.path.join(os.path.dirname(__file__),
                             '../../data', directory)

    # initialization
    data = []
    label = []
    padding = abs(size_input - size_label) // 2

    # generate data
    filepaths = list_pictures(directory, ext='bmp')

    for path in filepaths:
        image = load_img(path)
        image = image.convert('YCbCr')
        image = img_to_array(image)[:, :, 1]

        im_label = modcrop(image, scale)
        hei, wid = im_label.shape
        size = np.array(im_label.shape)
        im_input = imresize(imresize(im_label, size // scale, 'bicubic'),
                            size, 'bicubic')

        for x in range(0, hei - size_input, stride):
            for y in range(0, wid - size_input, stride):
                subim_input = im_input[x: x + size_input, y: y + size_input]
                subim_label = im_label[x + padding: x + padding + size_label,
                              y + padding: y + padding + size_label]
                data += [subim_input]
                label += [subim_label]

    order = np.random.permutation(len(data))
    data = np.stack(data)[order]
    label = np.stack(label)[order]

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('dat', data=data)
        f.create_dataset('lab', data=label)

    return data, label


def modcrop(image, scale):
    size = np.array(image.shape)
    size -= size % scale
    image = image[:size[0], :size[1]]
    return image
